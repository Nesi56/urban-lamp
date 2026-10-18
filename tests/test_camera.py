import sys
import types

import numpy as np
import pytest

from phasescope import camera as C
from phasescope.camera import (ManualShifter, SerialPiezoShifter, SimulatedInterferometer,
                               SimulatedShifter, StaticImageSource, _to_gray)


def test_simulator_frames_and_controls():
    cam = SimulatedInterferometer(width=64, height=48, fps=0, seed=0)
    frame = cam.grab()
    assert frame.shape == (48, 64) and frame.dtype == np.uint16 and frame.max() <= 4095
    cam.set_exposure_ms(1000)  # clamped to the allowed range -> saturates
    assert cam.get_exposure_ms() == cam.exposure_range_ms[1]
    assert (cam.grab() == 4095).mean() > 0.3


def test_simulated_shifter_moves_fringes():
    cam = SimulatedInterferometer(width=64, height=48, fps=0, animate=False, noise=0, seed=0)
    a = cam.grab().astype(float)
    SimulatedShifter(cam).set_phase(np.pi)
    b = cam.grab().astype(float)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] < -0.5  # half a fringe -> contrast inverted


def test_static_source_and_gray_conversion():
    rgb = np.zeros((10, 12, 3), np.uint8)
    rgb[..., 1] = 200
    src = StaticImageSource(rgb, fps=1000)
    assert src.grab().shape == (10, 12)
    assert _to_gray(np.zeros((5, 6, 4), np.uint16)).shape == (5, 6)


def test_piezo_voltage_mapping_and_command():
    sh = SerialPiezoShifter(volts_per_2pi=4.0, offset_v=10.0, max_v=12.0)
    assert sh.voltage_for(0.0) == pytest.approx(10.0)
    assert sh.voltage_for(np.pi) == pytest.approx(12.0)
    assert sh.voltage_for(2 * np.pi) == pytest.approx(12.0)  # clipped to max_v
    assert sh.command.format(volts=sh.voltage_for(np.pi / 2)) == "xvoltage=11.000\r"
    assert not ManualShifter.automatic


def test_piezo_over_loopback_port():
    pytest.importorskip("serial")
    sh = SerialPiezoShifter(port="loop://")
    sh.connect()
    sh.set_phase(np.pi)
    sh.disconnect()


# -- errors with a kind -------------------------------------------------------


def test_camera_error_kind_and_available():
    e = C.CameraError("plain")
    assert str(e) == "plain" and e.kind == "" and e.available == ()
    e = C.CameraError("missing", kind="not_found", available=["1", "2"])
    assert isinstance(e, RuntimeError) and e.kind == "not_found" and e.available == ("1", "2")


def test_gain_display_attributes():
    assert (C.CameraSource.gain_decimals, C.CameraSource.gain_unit) == (1, "")
    assert C.SimulatedInterferometer.gain_unit == "dB" and C.SimulatedInterferometer.gain_decimals == 1
    assert C.ThorlabsCamera.gain_decimals == 0 and C.ThorlabsCamera.gain_unit == ""
    assert C.OpenCVCamera.gain_decimals == 1


class _Range:
    def __init__(self, lo, hi):
        self.min, self.max = lo, hi


class _FakeTLCamera:
    def __init__(self):
        self.exposure_time_us = 0
        self.frames_per_trigger_zero_for_unlimited = 1
        self.image_poll_timeout_ms = 0
        self.exposure_time_range_us = _Range(40, 2_000_000)
        self.gain_range = _Range(0, 48)
        self.gain = 0
        self.bit_depth = 12
        self.image_height_pixels, self.image_width_pixels = 4, 6
        self.armed = self.disposed = False

    def arm(self, n):
        self.armed = True

    def issue_software_trigger(self):
        pass

    def disarm(self):
        self.armed = False

    def dispose(self):
        self.disposed = True

    def get_pending_frame_or_null(self):
        return types.SimpleNamespace(image_buffer=np.arange(24, dtype=np.uint16))


def _fake_tsi_sdk(monkeypatch, cameras=("11111",), fail=None):
    """Install a fake ``thorlabs_tsi_sdk`` package; returns the list of SDK instances created."""
    created = []

    class TLCameraSDK:
        def __init__(self):
            if fail is not None:
                raise fail
            self.disposed = False
            self.cam = None
            created.append(self)

        def discover_available_cameras(self):
            return list(cameras)

        def open_camera(self, serial):
            self.cam = _FakeTLCamera()
            return self.cam

        def dispose(self):
            self.disposed = True

    pkg = types.ModuleType("thorlabs_tsi_sdk")
    mod = types.ModuleType("thorlabs_tsi_sdk.tl_camera")
    mod.TLCameraSDK = TLCameraSDK
    pkg.tl_camera = mod
    monkeypatch.setitem(sys.modules, "thorlabs_tsi_sdk", pkg)
    monkeypatch.setitem(sys.modules, "thorlabs_tsi_sdk.tl_camera", mod)
    monkeypatch.delenv("THORLABS_TSI_DLL_DIR", raising=False)
    return created


def test_thorlabs_sdk_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "thorlabs_tsi_sdk", None)
    monkeypatch.delenv("THORLABS_TSI_DLL_DIR", raising=False)
    with pytest.raises(C.CameraError) as err:
        C.ThorlabsCamera().open()
    assert err.value.kind == "sdk_missing"


@pytest.mark.parametrize("fail", [OSError("[WinError 126] The specified module could not be found"),
                                  Exception("Unable to load library - are the thorlabs tsi camera sdk "
                                            "libraries discoverable?")])
def test_thorlabs_dll_failure(monkeypatch, fail):
    _fake_tsi_sdk(monkeypatch, fail=fail)
    with pytest.raises(C.CameraError) as err:
        C.ThorlabsCamera().open()
    assert err.value.kind == "dll" and "THORLABS_TSI_DLL_DIR" in str(err.value)


def test_thorlabs_no_camera_and_not_found(monkeypatch):
    created = _fake_tsi_sdk(monkeypatch, cameras=())
    with pytest.raises(C.CameraError) as err:
        C.ThorlabsCamera().open()
    assert err.value.kind == "no_camera" and created[-1].disposed

    created = _fake_tsi_sdk(monkeypatch, cameras=("11111", "22222"))
    with pytest.raises(C.CameraError) as err:
        C.ThorlabsCamera("33333").open()
    assert err.value.kind == "not_found" and err.value.available == ("11111", "22222")
    assert "33333" in str(err.value) and created[-1].disposed


def test_thorlabs_open_poll_timeout_and_list(monkeypatch):
    created = _fake_tsi_sdk(monkeypatch, cameras=("11111", "22222"))
    cam = C.ThorlabsCamera("22222", exposure_ms=10.0)
    cam.open()
    try:
        tl = cam.cam
        assert cam.serial == "22222" and tl.armed and tl.exposure_time_us == 10_000
        assert tl.image_poll_timeout_ms == 1000  # max(1000, 2 * 10 + 500)
        cam.set_exposure_ms(800.0)
        assert tl.exposure_time_us == 800_000 and tl.image_poll_timeout_ms == 2100
        assert cam.supports_gain and cam.gain_range == (0.0, 48.0)
        assert cam.grab().shape == (4, 6)
        # the SDK allows one instance per process: reuse the open one
        assert C.ThorlabsCamera.list_cameras(sdk=cam.sdk) == ["11111", "22222"]
        assert len(created) == 1
    finally:
        cam.close()
    assert created[0].disposed
    cam = C.ThorlabsCamera("1")  # a small number is an index into the detected cameras
    cam.open()
    assert cam.serial == "22222"
    cam.close()
    assert C.ThorlabsCamera.list_cameras() == ["11111", "22222"] and created[-1].disposed


def test_opencv_camera_not_found(monkeypatch):
    cv2 = pytest.importorskip("cv2")

    class Closed:
        def __init__(self, *args):
            pass

        def isOpened(self):
            return False

    monkeypatch.setattr(cv2, "VideoCapture", Closed)
    with pytest.raises(C.CameraError) as err:
        C.OpenCVCamera(7).open()
    assert err.value.kind == "not_found" and "#7" in str(err.value)


# -- serial piezo --------------------------------------------------------------

def test_command_template_escapes_as_typed():
    typed = SerialPiezoShifter(command=r"xvoltage={volts:.3f}\r", offset_v=10.0)  # backslash + r
    assert typed.command_bytes(0.0) == b"xvoltage=10.000\r"
    real = SerialPiezoShifter(command="xvoltage={volts:.3f}\r", offset_v=10.0)  # real CR
    assert real.command_bytes(0.0) == b"xvoltage=10.000\r"
    assert C.decode_command(r"a\\b\n\x41") == "a\\b\nA"
    assert SerialPiezoShifter(command=r"V {volts:.1f}\r\n", offset_v=5.0).command_bytes(0.0) == b"V 5.0\r\n"


@pytest.mark.parametrize("template", ["xvoltage={volts:.3f}\\", r"xvoltage={volts}\q", r"x\x4",
                                      "xvoltage=\\r", "xvoltage={volt}", "xvoltage={volts:.3z}",
                                      "{volts} {", "spannung={volts} µV"])
def test_bad_command_templates_raise_camera_error(template):
    sh = SerialPiezoShifter(port="loop://", command=template)
    with pytest.raises(C.CameraError):
        sh.connect()  # checked before the port is opened
    assert sh.ser is None


def test_serial_port_errors(monkeypatch):
    serial = pytest.importorskip("serial")
    with pytest.raises(C.CameraError) as err:
        SerialPiezoShifter(port="/dev/phasescope-no-such-port").connect()
    assert err.value.kind == "port_missing" and "does not exist" in str(err.value)
    assert isinstance(err.value.available, tuple)

    def busy(*args, **kwargs):
        raise serial.SerialException("could not open port 'COM3': PermissionError(13, 'Access is "
                                     "denied.', None, 5)")

    monkeypatch.setattr(serial, "serial_for_url", busy)
    with pytest.raises(C.CameraError) as err:
        SerialPiezoShifter(port="COM3").connect()
    assert err.value.kind == "port_busy" and "in use" in str(err.value)

    def missing(*args, **kwargs):
        raise serial.SerialException("could not open port 'COM7': FileNotFoundError(2, 'The system "
                                     "cannot find the file specified.', None, 2)")

    monkeypatch.setattr(serial, "serial_for_url", missing)
    with pytest.raises(C.CameraError) as err:
        SerialPiezoShifter(port="COM7").connect()
    assert err.value.kind == "port_missing" and str(err.value).startswith("Port COM7 does not exist.")


def test_list_serial_ports(monkeypatch):
    ports = C.list_serial_ports()
    assert isinstance(ports, list) and all(isinstance(d, str) and isinstance(t, str) for d, t in ports)
    monkeypatch.setitem(sys.modules, "serial", None)
    monkeypatch.setitem(sys.modules, "serial.tools", None)
    assert C.list_serial_ports() == []
