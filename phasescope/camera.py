"""Camera back-ends and phase shifters.

Camera sources
--------------
``SimulatedInterferometer``  synthetic, live, phase-shiftable interferogram of a
                             "material sample" (no hardware needed).
``OpenCVCamera``             any webcam / USB / DirectShow / V4L2 camera.
``ThorlabsCamera``           Thorlabs scientific cameras (Zelux, Kiralux,
                             Quantalux, CS-series) through the Thorlabs
                             Scientific Camera SDK (``thorlabs_tsi_sdk``).
``StaticImageSource``        replays an image loaded from disk.

Phase shifters
--------------
``SimulatedShifter``     shifts the simulated interferometer.
``ManualShifter``        you move the reference yourself and capture each step.
``SerialPiezoShifter``   piezo controller with an ASCII serial protocol
                         (defaults suit Thorlabs MDT693B / MDT694B).

All sources return 2-D numpy arrays (grey scale); colour frames are averaged.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from abc import ABC, abstractmethod

import numpy as np


class CameraError(RuntimeError):
    """Camera or phase-shifter problem with a plain-language message.

    ``kind`` lets the GUI offer specific help: "sdk_missing" (Thorlabs Python SDK
    not installed), "dll" (Thorlabs native DLLs not found), "no_camera" (no camera
    detected), "not_found" (requested camera not available; ``available`` lists
    the serial numbers found), "port_missing" / "port_busy" (serial piezo port;
    ``available`` lists the serial ports), or "" for anything else.
    """

    def __init__(self, message: str = "", kind: str = "", available: tuple = ()) -> None:
        super().__init__(message)
        self.kind = kind
        self.available = tuple(available)


# --------------------------------------------------------------------------
# Camera sources
# --------------------------------------------------------------------------

class CameraSource(ABC):
    name = "camera"
    supports_exposure = False
    supports_gain = False
    exposure_range_ms = (0.01, 1000.0)
    gain_range = (0.0, 0.0)
    gain_decimals = 1  # decimals to show for the gain value
    gain_unit = ""

    def open(self) -> None:  # pragma: no cover - trivial default
        pass

    def close(self) -> None:  # pragma: no cover - trivial default
        pass

    @abstractmethod
    def grab(self) -> np.ndarray | None:
        """Return the next frame as a 2-D array, or None on timeout."""

    def set_exposure_ms(self, value: float) -> None:
        raise CameraError(f"{self.name}: exposure control not supported")

    def set_gain(self, value: float) -> None:
        raise CameraError(f"{self.name}: gain control not supported")

    def get_exposure_ms(self) -> float | None:
        return None

    def get_gain(self) -> float | None:
        return None

    @property
    def bit_depth(self) -> int:
        return 8

    def describe(self) -> str:
        return self.name


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        if frame.shape[2] == 1:
            return frame[:, :, 0]
        if frame.dtype in (np.uint8, np.uint16, np.float32):
            import cv2
            code = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            return cv2.cvtColor(frame, code)  # keeps the bit depth
        return frame[:, :, :3].mean(axis=2).astype(np.float32)
    return frame


class SimulatedInterferometer(CameraSource):
    """Two-beam interferometer looking at a synthetic sample.

    The sample contains a thin-film step, a polished dome, a pit and a
    slowly drifting particle so the live view visibly changes.  ``phase_shift``
    is the reference-arm phase (set by :class:`SimulatedShifter`).
    """

    name = "Simulated interferometer"
    supports_exposure = True
    supports_gain = True
    exposure_range_ms = (0.05, 200.0)
    gain_range = (0.0, 24.0)
    gain_unit = "dB"

    def __init__(self, width: int = 640, height: int = 480, wavelength_nm: float = 632.8,
                 tilt_fringes: tuple[float, float] = (55.0, 18.0), noise: float = 0.01,
                 fps: float = 30.0, animate: bool = True, vibration_rad: float = 0.0,
                 seed: int | None = None) -> None:
        self.width, self.height = int(width), int(height)
        self.wavelength_nm = wavelength_nm
        self.tilt_fringes = tilt_fringes
        self.noise = noise
        self.fps = fps
        self.animate = animate
        self.vibration_rad = vibration_rad
        self.phase_shift = 0.0
        self.exposure_ms = 10.0
        self.gain_db = 0.0
        self._rng = np.random.default_rng(seed)
        self._t0 = time.perf_counter()
        self._last = 0.0
        self._lock = threading.Lock()
        yy, xx = np.mgrid[:self.height, :self.width].astype(np.float32)
        self._x, self._y = xx, yy
        self._static = self._static_height()
        r2 = ((xx - self.width / 2) ** 2 + (yy - self.height / 2) ** 2) / (0.6 * max(self.width, self.height)) ** 2
        self._envelope = np.exp(-r2).astype(np.float32)  # Gaussian illumination

    # sample topography in nm
    def _static_height(self) -> np.ndarray:
        x, y, w, h = self._x, self._y, self.width, self.height

        def edge(d, width=2.0):  # smooth step (a few pixels wide) so the phase stays unwrappable
            with np.errstate(over="ignore"):
                return 1.0 / (1.0 + np.exp(-d / width))

        film = 220.0 * edge(x - 0.62 * w) * edge(0.16 * h - np.abs(y - 0.30 * h))
        dome = 650.0 * np.exp(-(((x - 0.30 * w) / (0.13 * w)) ** 2 + ((y - 0.62 * h) / (0.17 * h)) ** 2))
        pit = -160.0 * edge(0.07 * min(w, h) - np.hypot(x - 0.70 * w, y - 0.72 * h), 1.5)
        bow = 60.0 * (((x - w / 2) / w) ** 2 + ((y - h / 2) / h) ** 2)
        return (film + dome + pit + bow).astype(np.float32)

    def height_map(self, t: float | None = None) -> np.ndarray:
        """Ground-truth sample height (nm) at time ``t`` (s since start)."""
        if t is None:
            t = time.perf_counter() - self._t0
        hmap = self._static
        if self.animate:
            cx = self.width * (0.5 + 0.30 * np.cos(0.15 * t))
            cy = self.height * (0.5 + 0.30 * np.sin(0.15 * t))
            s = 0.04 * min(self.width, self.height)
            hmap = hmap + 300.0 * np.exp(-((self._x - cx) ** 2 + (self._y - cy) ** 2) / (2 * s * s))
        return hmap

    def grab(self) -> np.ndarray:
        if self.fps > 0:  # emulate the camera frame rate
            wait = self._last + 1.0 / self.fps - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        self._last = time.perf_counter()
        with self._lock:
            shift = self.phase_shift
        fx, fy = self.tilt_fringes
        carrier = 2 * np.pi * (fx * self._x / self.width + fy * self._y / self.height)
        phase = 4 * np.pi * self.height_map() / self.wavelength_nm + carrier + shift
        if self.vibration_rad:
            phase = phase + self._rng.normal(0, self.vibration_rad)
        visibility = 0.85 * self._envelope
        intensity = 0.5 * self._envelope * (1 + visibility * np.cos(phase))
        scale = (self.exposure_ms / 10.0) * 10 ** (self.gain_db / 20.0)
        intensity = intensity * scale + self._rng.normal(0, self.noise, intensity.shape)
        return np.clip(intensity * 4095, 0, 4095).astype(np.uint16)  # 12-bit camera

    def set_phase_shift(self, value: float) -> None:
        with self._lock:
            self.phase_shift = float(value)

    def set_exposure_ms(self, value: float) -> None:
        self.exposure_ms = float(np.clip(value, *self.exposure_range_ms))

    def set_gain(self, value: float) -> None:
        self.gain_db = float(np.clip(value, *self.gain_range))

    def get_exposure_ms(self) -> float:
        return self.exposure_ms

    def get_gain(self) -> float:
        return self.gain_db

    @property
    def bit_depth(self) -> int:
        return 12

    def describe(self) -> str:
        return f"{self.name} {self.width}x{self.height}, 12-bit"


class StaticImageSource(CameraSource):
    """Serves the same image repeatedly (offline analysis of a saved frame)."""

    name = "Image file"

    def __init__(self, image: np.ndarray, label: str = "image", fps: float = 10.0) -> None:
        self.image = _to_gray(np.asarray(image))
        self.label = label
        self.fps = fps

    def grab(self) -> np.ndarray:
        time.sleep(1.0 / self.fps)
        return self.image.copy()

    @property
    def bit_depth(self) -> int:
        return 16 if self.image.dtype == np.uint16 else 8

    def describe(self) -> str:
        return f"{self.label} ({self.image.shape[1]}x{self.image.shape[0]})"


class OpenCVCamera(CameraSource):
    """Generic camera through OpenCV (webcams, USB, DirectShow, V4L2 ...)."""

    name = "OpenCV camera"
    supports_gain = False

    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None) -> None:
        self.index = int(index)
        self.width, self.height = width, height
        self.cap = None

    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise CameraError("OpenCV is not installed: pip install opencv-python-headless") from exc
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise CameraError(f"Camera #{self.index} could not be opened. Check that it is plugged in, "
                              "try another number, and close other programs that use it.",
                              kind="not_found")
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap = cap

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def grab(self) -> np.ndarray | None:
        if self.cap is None:
            raise CameraError("camera is not open")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return _to_gray(frame)

    def describe(self) -> str:
        return f"{self.name} #{self.index}"


_TSI_NATIVE = "Native_64_lib" if sys.maxsize > 2 ** 32 else "Native_32_lib"
_TSI_DEFAULT_DLL_DIRS = tuple(
    "\\".join((root, "Thorlabs", "Scientific Imaging", "Scientific Camera Support",
               "Scientific Camera Interfaces", "SDK", "Native Toolkit", "dlls", _TSI_NATIVE))
    for root in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
) if sys.platform.startswith("win") else ()


class ThorlabsCamera(CameraSource):
    """Thorlabs scientific camera through the Thorlabs TSI Python SDK.

    Install the SDK that ships with ThorCam (``Scientific Camera Interfaces``)::

        pip install thorlabs_tsi_camera_python_sdk_package.zip

    On Windows the native DLLs must be findable: set the environment variable
    ``THORLABS_TSI_DLL_DIR`` to the folder containing ``thorlabs_tsi_camera_sdk.dll``
    (e.g. ``...\\Scientific Camera Interfaces\\SDK\\Native Toolkit\\dlls\\Native_64_lib``).
    Without the variable the default ThorCam install folders are tried.
    """

    name = "Thorlabs camera (TSI SDK)"
    supports_exposure = True
    gain_decimals = 0  # the TSI SDK gain is an integer index

    DEFAULT_DLL_DIRS = _TSI_DEFAULT_DLL_DIRS
    _DLL_HELP = ("The Thorlabs camera driver (DLL) could not be loaded. Set the environment variable "
                 "THORLABS_TSI_DLL_DIR to the '...\\SDK\\Native Toolkit\\dlls\\" + _TSI_NATIVE
                 + "' folder of ThorCam's Scientific Camera Interfaces and restart PhaseScope.")

    def __init__(self, serial: str | None = None, exposure_ms: float = 10.0) -> None:
        self.serial = serial
        self._initial_exposure_ms = exposure_ms
        self.sdk = None
        self.cam = None
        self._shape = (0, 0)
        self._bit_depth = 12

    @classmethod
    def _configure_dll_path(cls) -> None:
        dll_dir = os.environ.get("THORLABS_TSI_DLL_DIR")
        if not dll_dir:
            dll_dir = next((d for d in cls.DEFAULT_DLL_DIRS if os.path.isdir(d)), None)
        if dll_dir and os.path.isdir(dll_dir):
            path = os.environ.get("PATH", "")
            if dll_dir not in path.split(os.pathsep):
                os.environ["PATH"] = dll_dir + os.pathsep + path
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(dll_dir)
                except OSError:
                    pass

    @classmethod
    def _new_sdk(cls):
        """Create a ``TLCameraSDK``; failures become a :class:`CameraError` with a ``kind``."""
        cls._configure_dll_path()
        try:
            from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
        except ImportError as exc:
            if "dll" in str(exc).lower():
                raise CameraError(cls._DLL_HELP, kind="dll") from exc
            raise CameraError("The Thorlabs camera software for Python is not installed. Install "
                              "thorlabs_tsi_camera_python_sdk_package.zip from the ThorCam folder "
                              "with pip (see README).", kind="sdk_missing") from exc
        except OSError as exc:
            raise CameraError(cls._DLL_HELP, kind="dll") from exc
        try:
            return TLCameraSDK()
        except Exception as exc:
            text = str(exc)
            low = text.lower()
            if isinstance(exc, OSError) or "library" in low or "dll" in low:
                raise CameraError(cls._DLL_HELP, kind="dll") from exc
            if "in use" in low or "already" in low:
                raise CameraError("The Thorlabs camera software is already in use by this program – "
                                  "disconnect the open Thorlabs camera first.") from exc
            raise CameraError(f"The Thorlabs camera software could not start: {text}") from exc

    @classmethod
    def list_cameras(cls, sdk=None) -> list[str]:
        """Serial numbers of the connected cameras.

        Pass an already open ``TLCameraSDK`` (e.g. the ``sdk`` of an open
        ThorlabsCamera): the SDK allows only one instance per process.
        """
        if sdk is not None:
            return [str(s) for s in sdk.discover_available_cameras()]
        sdk = cls._new_sdk()
        try:
            return [str(s) for s in sdk.discover_available_cameras()]
        finally:
            sdk.dispose()

    @staticmethod
    def _poll_timeout_ms(exposure_ms: float) -> int:
        return int(max(1000, 2 * exposure_ms + 500))

    def _pick_serial(self, available: list[str]) -> str:
        if not self.serial:
            return available[0]
        wanted = str(self.serial).strip()
        if wanted in available:
            return wanted
        if wanted.isdigit() and len(wanted) <= 2 and int(wanted) < len(available):
            return available[int(wanted)]  # small numbers are an index, like for OpenCV
        raise CameraError(f"Thorlabs camera {wanted} was not found. Connected: "
                          f"{', '.join(available)}.", kind="not_found", available=tuple(available))

    def open(self) -> None:
        sdk = self._new_sdk()
        try:
            available = [str(s) for s in sdk.discover_available_cameras()]
            if not available:
                raise CameraError("No Thorlabs camera was detected. Check the USB cable and close "
                                  "ThorCam (only one program can use the camera at a time).",
                                  kind="no_camera")
            serial = self._pick_serial(available)
            try:
                cam = sdk.open_camera(serial)
            except Exception as exc:
                raise CameraError(f"Thorlabs camera {serial} could not be opened – close ThorCam or "
                                  f"any other program that uses it. ({exc})") from exc
            try:
                self._configure(cam)
            except Exception as exc:
                try:
                    cam.dispose()
                except Exception:
                    pass
                raise CameraError(f"Thorlabs camera {serial} could not be started: {exc}") from exc
        except Exception:
            sdk.dispose()
            raise
        self.sdk, self.cam, self.serial = sdk, cam, serial

    def _configure(self, cam) -> None:
        try:
            rng = cam.exposure_time_range_us
            self.exposure_range_ms = (rng.min / 1000.0, rng.max / 1000.0)
        except Exception:
            pass
        exposure_ms = float(np.clip(self._initial_exposure_ms, *self.exposure_range_ms))
        cam.exposure_time_us = int(exposure_ms * 1000)
        cam.frames_per_trigger_zero_for_unlimited = 0  # continuous mode
        cam.image_poll_timeout_ms = self._poll_timeout_ms(exposure_ms)
        try:
            g = cam.gain_range
            if g.max > g.min:
                self.supports_gain = True
                self.gain_range = (float(g.min), float(g.max))
        except Exception:
            pass
        self._bit_depth = int(cam.bit_depth)
        self._shape = (cam.image_height_pixels, cam.image_width_pixels)
        cam.arm(2)
        cam.issue_software_trigger()

    def close(self) -> None:
        if self.cam is not None:
            try:
                self.cam.disarm()
            finally:
                self.cam.dispose()
                self.cam = None
        if self.sdk is not None:
            self.sdk.dispose()
            self.sdk = None

    def grab(self) -> np.ndarray | None:
        if self.cam is None:
            raise CameraError("camera is not open")
        frame = self.cam.get_pending_frame_or_null()
        if frame is None:
            return None
        return np.copy(frame.image_buffer).reshape(self._shape)

    def set_exposure_ms(self, value: float) -> None:
        lo, hi = self.exposure_range_ms
        value = float(np.clip(value, lo, hi))
        if self.cam is None:
            self._initial_exposure_ms = value
            return
        self.cam.exposure_time_us = int(value * 1000)
        self.cam.image_poll_timeout_ms = self._poll_timeout_ms(value)

    def get_exposure_ms(self) -> float | None:
        return self.cam.exposure_time_us / 1000.0 if self.cam is not None else None

    def set_gain(self, value: float) -> None:
        if not self.supports_gain:
            raise CameraError("this camera has no adjustable gain")
        self.cam.gain = int(np.clip(value, *self.gain_range))

    def get_gain(self) -> float | None:
        return float(self.cam.gain) if (self.cam is not None and self.supports_gain) else None

    @property
    def bit_depth(self) -> int:
        return self._bit_depth

    def describe(self) -> str:
        h, w = self._shape
        return f"Thorlabs {self.serial} {w}x{h}, {self._bit_depth}-bit"


# --------------------------------------------------------------------------
# Phase shifters
# --------------------------------------------------------------------------

class PhaseShifter(ABC):
    name = "shifter"
    automatic = True  # False: the user moves the reference and captures each step

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    @abstractmethod
    def set_phase(self, delta_rad: float) -> None:
        """Move the reference so the interferogram phase changes by ``delta_rad``."""


class SimulatedShifter(PhaseShifter):
    name = "Simulated (camera simulator)"

    def __init__(self, camera: SimulatedInterferometer) -> None:
        self.camera = camera

    def set_phase(self, delta_rad: float) -> None:
        self.camera.set_phase_shift(delta_rad)


class ManualShifter(PhaseShifter):
    name = "Manual (capture each step yourself)"
    automatic = False

    def set_phase(self, delta_rad: float) -> None:  # the user does it
        pass


def _natural_key(text: str) -> tuple:
    import re
    return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text))


def list_serial_ports() -> list[tuple[str, str]]:
    """``[(device, description)]`` of the serial ports on this computer ([] without pyserial)."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    try:
        ports = list_ports.comports()
    except Exception:
        return []
    out = []
    for p in ports:
        desc = str(getattr(p, "description", "") or "")
        out.append((str(p.device), "" if desc in ("n/a", p.device) else desc))
    return sorted(out, key=lambda d: _natural_key(d[0]))


_ESCAPES = {"r": "\r", "n": "\n", "t": "\t", "0": "\0", "\\": "\\"}
_HEX = "0123456789abcdefABCDEF"


def _as_typed(template: str) -> str:
    """A command template for messages, with control characters shown as escapes."""
    return str(template).replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def decode_command(template: str) -> str:
    r"""Turn the escapes ``\r \n \t \0 \\ \xHH`` typed in a command template into characters.

    Real control characters (e.g. an actual carriage return) are kept as they are.
    Raises :class:`CameraError` for an unknown or incomplete escape.
    """
    out, i, n = [], 0, len(template)
    while i < n:
        ch = template[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= n:
            raise CameraError(f"The piezo command '{_as_typed(template)}' ends with a lone backslash. "
                              "Use \\r for a carriage return or \\\\ for a backslash.")
        esc = template[i + 1]
        if esc in _ESCAPES:
            out.append(_ESCAPES[esc])
            i += 2
        elif esc == "x" and all(c in _HEX for c in template[i + 2:i + 4]) and i + 4 <= n:
            out.append(chr(int(template[i + 2:i + 4], 16)))
            i += 4
        else:
            shown = template[i + 1:i + 4] if esc == "x" else esc
            raise CameraError(f"The piezo command '{_as_typed(template)}' contains the unknown escape "
                              f"'\\{shown}'. Allowed: \\r (carriage return), \\n (line feed), \\t, "
                              "\\0, \\xHH and \\\\ (backslash).")
    return "".join(out)


class SerialPiezoShifter(PhaseShifter):
    r"""Piezo controller driven by ASCII commands over a serial port.

    The voltage for a phase shift ``delta`` is
    ``offset_v + delta / (2 pi) * volts_per_2pi``.  Calibrate ``volts_per_2pi``
    with the Carré algorithm (it reports the real step) - for a reflection
    set-up one fringe (2 pi) corresponds to a mirror motion of lambda / 2.

    The default command template matches the Thorlabs MDT693B / MDT694B
    open-loop piezo controllers (``xvoltage=<V>``, 115200 baud); check your
    controller manual and adapt ``command`` if needed.  The template may be
    given as typed by the user, with ``\r`` / ``\n`` written as backslash
    escapes (see :func:`decode_command`), or with the real characters.
    """

    name = "Serial piezo controller"

    def __init__(self, port: str = "COM3", baudrate: int = 115200,
                 command: str = "xvoltage={volts:.3f}\r", volts_per_2pi: float = 4.0,
                 offset_v: float = 10.0, max_v: float = 75.0) -> None:
        self.port, self.baudrate, self.command = port, baudrate, command
        self.volts_per_2pi, self.offset_v, self.max_v = volts_per_2pi, offset_v, max_v
        self.ser = None

    def command_bytes(self, delta_rad: float) -> bytes:
        """The bytes sent for a phase shift; :class:`CameraError` for a bad command template."""
        shown = _as_typed(self.command)
        template = decode_command(str(self.command))
        if "{volts" not in template:
            raise CameraError(f"The piezo command '{shown}' has no {{volts}} placeholder – write e.g. "
                              "xvoltage={volts:.3f}\\r so the voltage can be inserted.")
        try:
            text = template.format(volts=self.voltage_for(delta_rad))
        except (KeyError, IndexError, ValueError, AttributeError, TypeError) as exc:
            raise CameraError(f"The piezo command '{shown}' is not valid ({exc}). Use {{volts}} or "
                              "{volts:.3f} where the voltage goes and nothing else in braces.") from exc
        try:
            return text.encode("ascii")
        except UnicodeEncodeError as exc:
            raise CameraError(f"The piezo command '{shown}' contains non-ASCII characters; the "
                              "controller only understands plain ASCII text.") from exc

    def connect(self) -> None:
        self.command_bytes(0.0)  # check the template before touching the port
        try:
            import serial  # pyserial
        except ImportError as exc:
            raise CameraError("pyserial is required: pip install pyserial") from exc
        try:
            self.ser = serial.serial_for_url(self.port, baudrate=self.baudrate, timeout=0.5,
                                             write_timeout=2.0)
        except (serial.SerialException, OSError, ValueError) as exc:
            raise self._open_error(exc) from exc
        try:
            self.set_phase(0.0)
        except Exception:
            self.disconnect()
            raise

    def _open_error(self, exc: Exception) -> CameraError:
        ports = tuple(d for d, _ in list_serial_ports())
        listed = (f" Available ports: {', '.join(ports)}." if ports else
                  " No serial ports were found – is the controller plugged in and switched on?")
        low = str(exc).lower()
        busy = ("permissionerror", "access is denied", "permission denied", "busy", "errno 13",
                "errno 16", "exclusively lock")
        missing = ("filenotfounderror", "no such file", "cannot find", "errno 2]", "does not exist")
        if isinstance(exc, PermissionError) or any(m in low for m in busy):
            return CameraError(f"{self.port} is in use by another program – close Kinesis/APT or "
                               "any terminal.", kind="port_busy", available=ports)
        if (isinstance(exc, FileNotFoundError) or any(m in low for m in missing)
                or (ports and self.port.upper() not in {q.upper() for q in ports}
                    and "://" not in self.port)):
            return CameraError(f"Port {self.port} does not exist.{listed}", kind="port_missing",
                               available=ports)
        return CameraError(f"Could not open {self.port}: {exc}", available=ports)

    def disconnect(self) -> None:
        if self.ser is not None:
            try:
                self.ser.close()
            finally:
                self.ser = None

    def voltage_for(self, delta_rad: float) -> float:
        v = self.offset_v + delta_rad / (2 * np.pi) * self.volts_per_2pi
        return float(np.clip(v, 0.0, self.max_v))

    def set_phase(self, delta_rad: float) -> None:
        if self.ser is None:
            raise CameraError("piezo controller is not connected")
        data = self.command_bytes(delta_rad)
        try:
            self.ser.write(data)
            self.ser.flush()
            self.ser.reset_input_buffer()  # discard the echo / prompt
        except (OSError, ValueError) as exc:  # SerialException is an OSError
            raise CameraError(f"Lost contact with the piezo controller on {self.port}: {exc}") from exc
