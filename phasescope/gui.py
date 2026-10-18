"""PhaseScope main window."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import unicodedata
from dataclasses import asdict, replace

import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import __version__
from .acquisition import AcquisitionThread, FramePacket
from .camera import (CameraError, CameraSource, ManualShifter, OpenCVCamera, PhaseShifter, SerialPiezoShifter,
                     SimulatedInterferometer, SimulatedShifter, StaticImageSource, ThorlabsCamera,
                     list_serial_ports)
from .phase import (ALGORITHMS, OUTPUT_MODES, UNWRAP_METHODS, PhaseConfig, PhaseResult, analyse, analyse_stack,
                    fringe_quality, skimage_available)
from .processing import REGISTRY, Param, Pipeline, operations
from .widgets import COLORMAPS, STATUS_COLORS, ImagePanel, ParamDialog, StatusPill, Surface3DPanel

# --------------------------------------------------------------------------
# Labels (internal keys never change, only what the user reads)
# --------------------------------------------------------------------------

SOURCES = {
    "sim": "Simulator (no hardware)",
    "opencv": "USB camera / webcam",
    "thorlabs": "Thorlabs camera",
}
SOURCE_TIPS = {
    "sim": "Built-in simulated interferometer - try everything without hardware",
    "opencv": "Any USB / webcam / DirectShow camera through OpenCV",
    "thorlabs": "Thorlabs scientific camera (Zelux, Kiralux, Quantalux, CS series) through the Thorlabs SDK",
}
SOURCE_SHORT = {"sim": "Simulator", "opencv": "USB camera", "thorlabs": "Thorlabs camera"}
SHIFTERS = {
    "sim": "Simulated (demo)",
    "manual": "Manual",
    "serial": "Serial piezo",
}
SHIFTER_TIPS = {
    "sim": "Shifts the simulated camera - only works with the simulator",
    "manual": "You move the reference yourself; PhaseScope tells you where and you press Space",
    "serial": "Piezo controller with a serial / USB port (Thorlabs MDT693B / MDT694B …)",
}
PHASE_METHODS = {"off": "Off", "ftp": "Fourier transform (live, single frame)",
                 **{k: a.name for k, a in ALGORITHMS.items()}}
METHOD_HINTS = {
    "off": "Phase is off - the analysis shows the processed camera image.",
    "ftp": "A height map from every live frame. Needs straight tilt fringes 5–20 px apart.",
    "3step": "3 frames, 120° apart. Fastest, but sensitive to phase-shifter errors.",
    "4step": "4 frames, 90° apart. The recommended precise measurement.",
    "5step": "5 frames, 90° apart. Tolerates a slightly mis-calibrated phase shifter.",
    "carre": "4 frames with any equal step. Also reports the real step - use it to calibrate a piezo.",
    "nstep": "N frames spread over 360°. More frames average out noise.",
}
MODES = {"image": "Live image", "live_phase": "Live phase", "measure": "Measure"}
DETREND_LABELS = {"none": "Nothing", "piston": "Offset only", "tilt": "Offset + tilt (plane)",
                  "quadratic": "Offset + tilt + curvature"}
UNWRAP_LABELS = {"none": "None (show wrapped phase)", "itoh_rows": "Path-following (rows first)",
                 "itoh_cols": "Path-following (columns first)",
                 "least_squares": "Least-squares (recommended)",
                 "weighted_ls": "Weighted least-squares (slower)",
                 "quality_guided": "Quality-guided (robust)"}
OUTPUT_LABELS = {"reflection": "Surface height (reflection)", "transmission": "Thickness (transmission)",
                 "index": "Refractive-index change Δn", "opd": "Optical path difference", "phase": "Phase (rad)"}
OUTPUT_FORMULAS = {"reflection": "h = φ·λ / 4π", "transmission": "t = φ·λ / (2π·(n sample − n medium))",
                   "index": "Δn = φ·λ / (2π·t)", "opd": "OPD = φ·λ / 2π", "phase": "raw phase in radians"}
PHYSICAL_LABELS = {"reflection": "Height map", "transmission": "Thickness map", "index": "Δn map",
                   "opd": "Optical path map", "phase": "Phase map"}
CHANNELS = {"processed": "Processed image", "raw": "Raw camera", "physical": "Height map",
            "wrapped": "Wrapped phase", "unwrapped": "Unwrapped phase",
            "modulation": "Fringe contrast"}
IMAGE_CHANNELS = ("processed", "raw")
AUTO_CMAP = {"processed": "gray", "raw": "gray", "wrapped": "CET-C2 (cyclic)", "unwrapped": "viridis",
             "physical": "viridis", "modulation": "inferno"}
WAVELENGTHS = (("HeNe laser 632.8 nm", 632.8), ("Green 532 nm", 532.0), ("Violet 405 nm", 405.0),
               ("Red diode 650 nm", 650.0), ("Near-IR 785 nm", 785.0))

SERIAL_PARAMS = (
    Param("port", "Serial port", "str", "COM3", tooltip="e.g. COM3 on Windows, /dev/ttyUSB0 on Linux"),
    Param("baudrate", "Baud rate", "int", 115200, 300, 1000000),
    Param("command", "Command template", "str", "xvoltage={volts:.3f}\\r",
          tooltip="{volts} is replaced by the voltage; \\r and \\n are allowed"),
    Param("volts_per_2pi", "Volts per 2π phase", "float", 3.2, 0.001, 150.0, 0.1, decimals=3,
          tooltip="Reflection: 2π = λ/2 mirror travel. Calibrate with the Carré method."),
    Param("offset_v", "Offset voltage (V)", "float", 20.0, 0.0, 150.0, 1.0),
    Param("max_v", "Maximum voltage (V)", "float", 75.0, 1.0, 150.0, 1.0),
)

IMAGE_FILTER = "Images (*.png *.tif *.tiff *.bmp *.jpg *.jpeg *.npy);;All files (*)"
MEASUREMENT_FILES = ("_raw.tif", "_map.tif", "_maps.npz", "_view.png", "_info.txt")  # one measurement
SDK_ZIP = (r"C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support\Scientific Camera Interfaces"
           r"\SDK\Python Toolkit\thorlabs_tsi_camera_python_sdk_package.zip")


# --------------------------------------------------------------------------
# File helpers
# --------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """The image in ``path`` (the first page of a multi-page TIFF, see load_pages)."""
    name = os.path.basename(path)
    if os.path.getsize(path) == 0:
        raise IOError(f"{name} is empty")
    if path.lower().endswith(".npy"):
        img = np.load(path, allow_pickle=False)
    else:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"{name} is not an image PhaseScope can read")
    return _as_image(img, name)


def load_pages(path: str) -> list[np.ndarray]:
    """All images in ``path``: every page of a multi-page TIFF (e.g. a phase-step stack), else one."""
    if path.lower().endswith((".tif", ".tiff")) and os.path.getsize(path) > 0:
        name = os.path.basename(path)
        pages = None
        try:
            if hasattr(cv2, "imdecodemulti"):  # OpenCV >= 4.7, also for non-ASCII paths
                ok, pages = cv2.imdecodemulti(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            elif path.isascii():
                ok, pages = cv2.imreadmulti(path, flags=cv2.IMREAD_UNCHANGED)
            else:
                ok = False
        except cv2.error:
            ok = False
        if ok and pages is not None and len(pages) > 1:
            return [_as_image(page, f"{name} (page {i + 1})") for i, page in enumerate(pages)]
    return [load_image(path)]


def _as_image(img, name: str) -> np.ndarray:
    """Check and convert a decoded file to a 2-D real-valued image."""
    img = np.asarray(img)
    if img.dtype == bool:
        img = img.astype(np.uint8)
    if img.dtype.kind not in "uif":
        raise IOError(f"{name}: unsupported data type {img.dtype} (needs real-valued numbers)")
    if img.ndim == 3 and img.shape[2] in (1, 3, 4):
        img = img[:, :, 0] if img.shape[2] == 1 else img[:, :, :3].mean(axis=2).astype(np.float32)
    if img.ndim != 2 or min(img.shape) < 2:
        raise IOError(f"{name}: expected a 2-D image, got an array of shape {img.shape}")
    return img


def _imwrite(path: str, arr: np.ndarray) -> None:
    ok, buf = cv2.imencode(os.path.splitext(path)[1], arr)  # works with non-ASCII paths on Windows
    if not ok:
        raise IOError(f"could not write {path}")
    buf.tofile(path)


def save_array(path: str, data: np.ndarray) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, data)
    elif ext in (".csv", ".txt"):
        np.savetxt(path, data, delimiter=",", fmt="%.6g")
    elif ext in (".tif", ".tiff"):
        _imwrite(path, data if data.dtype in (np.uint8, np.uint16) else data.astype(np.float32))
    else:  # 8/16-bit display image for png / jpg / bmp
        if data.dtype in (np.uint8, np.uint16) and ext == ".png":
            arr = data
        else:
            finite = data[np.isfinite(data)]
            lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
            arr = np.clip((np.nan_to_num(data, nan=lo) - lo) / max(hi - lo, 1e-12) * 255, 0, 255).astype(np.uint8)
        _imwrite(path, arr)


def _safe_name(text: str) -> str:
    """Sample name for file names: letters, marks and digits of any script, '_', '-' and '.'.

    Every run of other characters becomes one '-'."""
    out, gap = [], False
    for ch in unicodedata.normalize("NFC", text.strip()):
        if unicodedata.category(ch)[0] in "LMN" or ch in "_-.":
            if gap and out:
                out.append("-")
            out.append(ch)
            gap = False
        else:
            gap = True
    return "".join(out).strip("-.")[:40].rstrip("-.")


def _natural_key(text: str) -> list:
    """Sort 'step 2' before 'step 10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def _check_written(ok: bool, path: str) -> None:
    if not ok:
        raise IOError(f"could not write {path}")


def _set_row_visible(form: QtWidgets.QFormLayout, field: QtWidgets.QWidget, visible: bool) -> None:
    """Show/hide a QFormLayout row (QFormLayout.setRowVisible needs Qt >= 6.4)."""
    field.setVisible(visible)
    label = form.labelForField(field)
    if label is not None:
        label.setVisible(visible)


def _muted(text: str = "") -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(f"color: {STATUS_COLORS['muted']}")
    return lab


def _wrap(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w


class _Signaller(QtCore.QObject):
    """Delivers results from plain Python threads to the GUI thread."""
    done = QtCore.Signal(object)


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, source: str | None = None, device: str = "", use_gl: bool = True,
                 autostart: bool = True, restore_settings: bool = True) -> None:
        super().__init__()
        self.setWindowTitle(f"PhaseScope {__version__}")
        self._persist = restore_settings
        self.settings = QtCore.QSettings("PhaseScope", "PhaseScope") if restore_settings else None

        self.pipeline = Pipeline()
        self.source: CameraSource | None = None
        self.source_key = "sim"
        self.thread: AcquisitionThread | None = None
        self.shifter: PhaseShifter | None = None
        self.phase_cfg = PhaseConfig()
        self.serial_settings = {p.name: p.default for p in SERIAL_PARAMS}
        self.last_packet: FramePacket | None = None
        self.last_result: PhaseResult | None = None
        self.manual_frames: list[np.ndarray] = []
        self._sync = True  # True while widgets are updated programmatically
        self._stale_threads: list[AcquisitionThread] = []
        self._state = "disconnected"
        self._last_error = ""
        self._last_frame_t = 0.0
        self._last_result_t = 0.0
        self._last_pill_t = 0.0
        self._recording = False
        self._progress_note = ""
        self._hold = False
        self._capture: dict | None = None
        self._ae: dict | None = None
        self._meter_history: list[str] = []
        self._meter_level: str | None = None
        self._live_notice: tuple[str, str] | None = None
        self._fallback: tuple[str, str] | None = None  # (camera, shifter) the user wanted when we fell back to sim
        self._seq_id = -1  # id of the phase-stepping sequence the window is waiting for
        self._seq_method = ""
        self._seq_started = 0.0
        self._seq_context = None  # where the recorded frames came from (camera, region, processing)
        self._dropped_seq: set[int] = set()  # cancelled runs whose late maps are ignored
        self._pause_after_seq = False
        self._connected_device = ""
        self._closing: list = []  # (thread, camera) pairs: camera closed when its thread ends
        self._closed_ids: set[int] = set()
        self._calc_busy = False
        self._calc_kind = ""  # kind of the running calculation ('manual' or 'reanalyse')
        self._recalc_error = ""  # why the last recalculation of the shown map failed
        self._result_files: list[str] | None = None  # files the shown map was computed from
        self._orphan = None  # (thread, seq id, context) of a camera thread that ended while its map was calculated
        self._seq_progress = (0, 0)  # steps recorded / in the sequence
        self._calc_queue: list = []  # manual-stepping analyses waiting (never dropped)
        self._calc_reanalyse = False  # a recalculation is wanted after the running job
        self._packet_thread = None  # thread that produced last_packet (frame indices restart per thread)
        self._result_context = None
        self._headline_text = ""
        self._job_serial = 0  # background calculations: only the newest one is shown
        self._calc = _Signaller()
        self._calc.done.connect(self._calculation_done)
        self._levels_source = None
        self._result_source = None
        self._source_serial = 0
        self._view_mode = "both"
        self._shortcut_actions: list[tuple[str, QtGui.QAction]] = []
        self._exposure_by_source: dict[str, dict] = {}
        self._detector = _Signaller()
        self._detector.done.connect(self._detect_done)

        self._build_actions()
        self._build_menus()
        self._build_central(use_gl)
        self._build_docks()
        self._build_toolbars()
        self._build_menubar()
        self._build_statusbar()
        self._default_state = self.saveState()

        saved_source = self._restore_settings()
        if source is None:
            source = saved_source or "sim"
        self.source_combo.setCurrentIndex(list(SOURCES).index(source if source in SOURCES else "sim"))
        if device:
            self.device_combo.setEditText(device)
        self._source_ui_changed()
        self._sync = False
        self._phase_widgets_changed()
        self._refresh_pipeline_list()
        self._set_state("disconnected")
        self._update_analysis(force=True)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(25)
        self._slow_timer = QtCore.QTimer(self)
        self._slow_timer.timeout.connect(self._slow_update)
        self._slow_timer.start(500)
        if autostart:
            fallback = source != "sim"
            QtCore.QTimer.singleShot(0, lambda: self.connect_camera(fallback_to_sim=fallback, quiet=fallback))

    # ======================================================================
    # Actions & menus
    # ======================================================================
    def _action(self, text, slot=None, shortcut=None, checkable=False, tip=None, group="General",
                icon=None):
        act = QtGui.QAction(text, self)
        if icon is not None:
            act.setIcon(self.style().standardIcon(icon))
        if shortcut:
            seqs = [QtGui.QKeySequence(s) for s in (shortcut if isinstance(shortcut, (list, tuple)) else [shortcut])]
            act.setShortcuts(seqs)
            act.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            self._shortcut_actions.append((group, act))
        act.setCheckable(checkable)
        self._set_tip(act, tip or text)
        if slot is not None:
            act.triggered.connect(slot)
        return act

    @staticmethod
    def _shortcut_text(act: QtGui.QAction) -> str:
        return ", ".join(s.toString(QtGui.QKeySequence.SequenceFormat.NativeText) for s in act.shortcuts())

    def _set_tip(self, obj, tip: str) -> None:
        if isinstance(obj, QtGui.QAction) and obj.shortcuts():
            tip = f"{tip}  ({self._shortcut_text(obj)})"
        obj.setToolTip(tip)
        obj.setStatusTip(tip)

    def _build_actions(self) -> None:
        sp = QtWidgets.QStyle.StandardPixmap
        A = self._action
        self.act_live = A("Start live", self._toggle_live, "F5", checkable=True,
                          tip="Start / pause the live camera", group="Camera", icon=sp.SP_MediaPlay)
        self.act_stop = A("Pause", self.pause_live, "F6", tip="Pause the live camera", group="Camera",
                          icon=sp.SP_MediaStop)
        self.act_connect = A("Connect", lambda: self.connect_camera(), tip="Connect the selected camera",
                             group="Camera")
        self.act_disconnect = A("Disconnect", self.disconnect_camera, tip="Disconnect the camera", group="Camera")
        self.act_auto_exposure = A("Auto exposure", self.start_auto_exposure, "Ctrl+E",
                                   tip="Set the exposure so the brightest pixels sit at about 75 % of full scale",
                                   group="Camera")
        self.act_true_brightness = A("True brightness in the live view", self._true_brightness_toggled,
                                     checkable=True, tip="Show the live image on the camera's full scale, so a dark "
                                     "image looks dark (untick to stretch the contrast)", group="View")
        self.act_true_brightness.setChecked(True)
        self.act_show_clipped = A("Mark clipped pixels in red", lambda on: (
            None if on else self.live_panel.set_mask_overlay(None)), checkable=True,
            tip="Paint saturated pixels red on the live image", group="View")
        self.act_show_clipped.setChecked(True)
        self.act_live_hist = A("Show histogram under the live image", lambda on: self._refresh_live_overlays(),
                               checkable=True, group="View")
        self.act_live_hist.setChecked(True)

        # measurement modes (one click / one key)
        self.mode_group = QtGui.QActionGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_actions = {}
        tips = {"image": "Just the (processed) camera image - no phase",
                "live_phase": "Live height map from every frame (Fourier method, needs tilt fringes)",
                "measure": "Precise phase-stepping measurement (Acquire / Capture step)"}
        for key, keyseq in (("image", "F2"), ("live_phase", "F3"), ("measure", "F4")):
            act = A(MODES[key], lambda _=False, k=key: self.set_mode(k), keyseq, checkable=True,
                    tip=tips[key], group="Measurement")
            self.mode_group.addAction(act)
            self.mode_actions[key] = act

        self.act_acquire = A("Acquire", self.acquire_sequence, ["Ctrl+P", "Ctrl+Return"],
                             tip="Record one phase-stepping sequence and compute the map", group="Measurement")
        self.act_continuous = A("Continuous stepping", self.toggle_continuous, checkable=True,
                                tip="Repeat phase-stepping sequences for a live map", group="Measurement")
        self.act_capture = A("Capture step", self.capture_manual_step, ["Space", "F8"],
                             tip="Store the current frame as the next phase step (manual stepping)",
                             group="Measurement")
        self.act_undo_step = A("Undo last step", self.undo_manual_step, "Ctrl+Z",
                               tip="Forget the last captured manual step", group="Measurement")
        self.act_reset_steps = A("Start over", lambda: self.reset_manual_steps(),
                                 tip="Forget all captured manual steps", group="Measurement")
        self.act_cancel = A("Cancel", self._escape, "Esc", tip="Cancel recording / restore the normal view",
                            group="Measurement")
        self.act_hold = A("Hold live map", self._toggle_hold, "H", checkable=True,
                          tip="Freeze the live height map so you can measure or save it", group="Measurement")
        self.act_lock_carrier = A("Lock fringe frequency", self.lock_carrier, checkable=True,
                                  tip="Keep the fringe frequency found in the last frame (steadier live map)",
                                  group="Measurement")
        self.act_reanalyse = A("Recalculate with current settings", self.reanalyse,
                               tip="Apply the current unwrapping / levelling / unit settings to the last map",
                               group="Measurement")

        # files
        self.act_quick_save = A("Quick save", self.quick_save, "Ctrl+S",
                                tip="Save the frame, map, picture and an info sheet - no questions asked",
                                group="Files", icon=sp.SP_DialogSaveButton)
        self.act_save_as = A("Save measurement as…", self.save_measurement_as, "Ctrl+Shift+S",
                             tip="Choose folder and name, then save everything like Quick save", group="Files")
        self.act_open = A("Open image(s)…", self.open_images, "Ctrl+O",
                          tip="One image: analyse it like a live frame. Several: a phase-step sequence.",
                          group="Files")
        self.act_bypass = A("Bypass all processing", self._toggle_bypass, "Ctrl+B", checkable=True,
                            tip="Temporarily switch every filter and transformation off (compare raw vs. filtered)",
                            group="Processing")

        # view
        self.view_group = QtGui.QActionGroup(self)
        self.view_group.setExclusive(True)
        self.view_actions = {}
        for key, label, keyseq, tip in (("live", "Live only", "Alt+1", "Show only the live camera"),
                                        ("both", "Side by side", "Alt+2", "Live camera and analysis side by side"),
                                        ("analysis", "Analysis only", "Alt+3", "Show only the analysis")):
            act = A(label, lambda _=False, k=key: self.set_view_mode(k), keyseq, checkable=True, tip=tip,
                    group="View")
            self.view_group.addAction(act)
            self.view_actions[key] = act
        self.view_actions["both"].setChecked(True)
        self.act_2d = A("2D view", lambda: self.view_tabs.setCurrentIndex(0), "Ctrl+2", group="View")
        self.act_3d = A("3D view", lambda: self.view_tabs.setCurrentIndex(1), "Ctrl+3", group="View")
        self.act_panels = A("Side panels", self._toggle_panels, "F9", checkable=True,
                            tip="Show / hide the Camera and Phase panels", group="View")
        self.act_panels.setChecked(True)
        self.act_fullscreen = A("Full screen", self._toggle_fullscreen, "F11", checkable=True, group="View")
        self.act_auto_levels = A("Reset colour scale", lambda: setattr(self.analysis_panel, "auto_levels", True),
                                 "A", tip="Fit the colour scale to the data again", group="View")
        self.act_profile = A("Line profile", lambda on: self.profile_check.setChecked(on), "L", checkable=True,
                             tip="Cross-section with step-height tool", group="View")
        self.act_reset_zoom = A("Reset zoom", lambda: (self.live_panel.reset_view(),
                                                        self.analysis_panel.reset_view()), ["Home", "Z"],
                                group="View")
        self.act_help = A("Keyboard shortcuts", self.show_shortcuts, "F1", group="Help")

    def _operation_menu(self, title: str, category: str) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(title, self)
        menu.setToolTipsVisible(True)
        groups: dict[str, QtWidgets.QMenu] = {}
        for op in operations(category):
            parent = menu
            if op.group:
                if op.group not in groups:
                    groups[op.group] = menu.addMenu(op.group)
                    groups[op.group].setToolTipsVisible(True)
                parent = groups[op.group]
            name = op.name if getattr(op, "phase_safe", True) else f"{op.name}  (view only)"
            act = parent.addAction(name)
            if op.description:
                act.setToolTip(op.description)
                act.setStatusTip(op.description)
            act.triggered.connect(lambda _=False, k=op.key: self.add_operation(k))
        menu.addSeparator()
        menu.addAction("Remove all", lambda _=False, c=category: self._clear_category(c))
        return menu

    def _radio_menu(self, title: str, items: dict[str, str], handler) -> tuple[QtWidgets.QMenu, dict]:
        menu = QtWidgets.QMenu(title, self)
        group = QtGui.QActionGroup(menu)
        group.setExclusive(True)
        actions = {}
        for key, label in items.items():
            act = menu.addAction(label)
            act.setCheckable(True)
            group.addAction(act)
            act.triggered.connect(lambda _=False, k=key: handler(k))
            actions[key] = act
        return menu, actions

    def _build_menus(self) -> None:
        # 2.a / 2.b pop-up menus
        self.filters_menu = self._operation_menu("Filters", "filter")
        self.transforms_menu = self._operation_menu("Transformations", "transform")

        # 2.c phase pop-up menu
        m = QtWidgets.QMenu("Phase", self)
        m.setToolTipsVisible(True)
        self.phase_menu = m
        for act in self.mode_actions.values():
            m.addAction(act)
        m.addSeparator()
        sub, self.method_actions = self._radio_menu(
            "Method", PHASE_METHODS, lambda k: self.method_combo.setCurrentIndex(list(PHASE_METHODS).index(k)))
        m.addMenu(sub)
        sub, self.unwrap_actions = self._radio_menu(
            "Unwrapping", UNWRAP_LABELS, lambda k: self.unwrap_combo.setCurrentIndex(list(UNWRAP_METHODS).index(k)))
        m.addMenu(sub)
        sub, self.detrend_actions = self._radio_menu(
            "Background removal", DETREND_LABELS,
            lambda k: self.detrend_combo.setCurrentIndex(list(DETREND_LABELS).index(k)))
        m.addMenu(sub)
        sub, self.output_actions = self._radio_menu(
            "Output quantity", OUTPUT_LABELS, lambda k: self.output_combo.setCurrentIndex(list(OUTPUT_LABELS).index(k)))
        m.addMenu(sub)
        m.addSeparator()
        for a in (self.act_acquire, self.act_continuous, self.act_capture, self.act_undo_step, self.act_reset_steps,
                  self.act_hold, self.act_lock_carrier, self.act_reanalyse):
            m.addAction(a)
        m.addSeparator()
        sub, self.shifter_actions = self._radio_menu(
            "Phase shifter", SHIFTERS, lambda k: self.shifter_combo.setCurrentIndex(list(SHIFTERS).index(k)))
        m.addMenu(sub)
        m.addAction("Piezo controller settings…", self.edit_serial_settings)
        m.addAction("Show the Phase panel", lambda: self._raise_dock(self.phase_dock))

        self.cmap_menu, self.cmap_actions = self._radio_menu(
            "Colour map", {"Auto": "Auto (per display)", **{c: c for c in COLORMAPS}},
            lambda k: self.cmap_combo.setCurrentText(k))

    def _context_menu(self, panel) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        menu.addMenu(self.filters_menu)
        menu.addMenu(self.transforms_menu)
        menu.addMenu(self.phase_menu)
        menu.addSeparator()
        if panel is self.live_panel:
            menu.addAction(self.act_true_brightness)
            menu.addAction(self.act_show_clipped)
            menu.addAction(self.act_live_hist)
            menu.addAction(self.act_auto_exposure)
            menu.addAction("Reset zoom", panel.reset_view)
        elif isinstance(panel, ImagePanel):
            menu.addMenu(self.cmap_menu)
            menu.addAction(self.act_auto_levels)
            menu.addAction(self.act_profile)
            menu.addAction("Use the visible area as flat reference", self._reference_from_view)
            menu.addAction("Reset zoom", panel.reset_view)
        else:
            menu.addMenu(self.cmap_menu)
            menu.addAction("Reset camera", panel.reset_camera)
        menu.addSeparator()
        menu.addAction("Save this view as PNG…", lambda: self.save_view(panel))
        return menu

    # ======================================================================
    # Central area: live camera (1) and analysis (3)
    # ======================================================================
    def _header(self, title: str) -> tuple[QtWidgets.QHBoxLayout, QtWidgets.QLabel]:
        row = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel(title)
        f = lab.font()
        f.setBold(True)
        f.setPointSize(f.pointSize() + 1)
        lab.setFont(f)
        row.addWidget(lab)
        return row, lab

    def _build_central(self, use_gl: bool) -> None:
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        # ---- live camera ----------------------------------------------------
        self.live_box = QtWidgets.QFrame()
        lv = QtWidgets.QVBoxLayout(self.live_box)
        lv.setContentsMargins(4, 4, 4, 4)
        row, self.live_title = self._header("Live camera")
        self.live_info = _muted("")
        row.addStretch(1)
        row.addWidget(self.live_info)
        lv.addLayout(row)
        self.live_panel = ImagePanel(histogram=True, roi=True)
        self.live_panel.set_popup_menu(self._context_menu(self.live_panel))
        self.live_panel.hovered.connect(lambda x, y, v: self.cursor_label.setText(f"live ({x}, {y}) = {v:.5g}"))
        self.live_panel.roi_changed.connect(self._roi_changed)
        self.live_panel.double_clicked.connect(lambda: self._toggle_single_view("live"))
        lv.addWidget(self.live_panel, 1)
        meter = QtWidgets.QHBoxLayout()
        self.exposure_pill = StatusPill()
        self.exposure_pill.set_state("muted", "Exposure: –")
        meter.addWidget(self.exposure_pill)
        self.exposure_hint = _muted("")
        meter.addWidget(self.exposure_hint, 1)
        b = QtWidgets.QToolButton()
        b.setDefaultAction(self.act_auto_exposure)
        meter.addWidget(b)
        lv.addLayout(meter)
        self.splitter.addWidget(self.live_box)

        # ---- analysis --------------------------------------------------------
        self.analysis_box = QtWidgets.QFrame()
        av = QtWidgets.QVBoxLayout(self.analysis_box)
        av.setContentsMargins(4, 4, 4, 4)
        row, self.analysis_title = self._header("Analysis")
        row.addStretch(1)
        self.headline = QtWidgets.QLabel("")
        f = self.headline.font()
        f.setBold(True)
        f.setPointSize(f.pointSize() + 2)
        self.headline.setFont(f)
        self.headline.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.headline)
        av.addLayout(row)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Display"))
        self.channel_combo = QtWidgets.QComboBox()
        for key, label in CHANNELS.items():
            self.channel_combo.addItem(label, key)
        self.channel_combo.setToolTip("What the analysis area shows")
        self.channel_combo.currentIndexChanged.connect(lambda _: self._update_analysis(force=True))
        bar.addWidget(self.channel_combo, 1)
        bar.addWidget(QtWidgets.QLabel("Colours"))
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["Auto"] + COLORMAPS)
        self.cmap_combo.currentTextChanged.connect(self._cmap_changed)
        bar.addWidget(self.cmap_combo, 1)
        self.channel_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.channel_combo.setMinimumContentsLength(12)
        self.cmap_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)  # short list
        self.autolevel_check = QtWidgets.QCheckBox("Auto colour scale")
        self.autolevel_check.setToolTip("Fit the colour scale to every new map (untick after dragging the scale "
                                        "to keep your own)  (A)")
        self.autolevel_check.setChecked(True)
        self.autolevel_check.toggled.connect(lambda v: setattr(self.analysis_panel, "auto_levels", v))
        bar.addWidget(self.autolevel_check)
        self.profile_check = QtWidgets.QCheckBox("Profile")
        self.profile_check.setToolTip("Line profile with step-height tool: drag the magenta line on the map  (L)")
        self.profile_check.toggled.connect(self._profile_toggled)
        bar.addWidget(self.profile_check)
        av.addLayout(bar)

        self.view_tabs = QtWidgets.QTabWidget()
        self.analysis_panel = ImagePanel(histogram=True, profile=True)
        self.analysis_panel.set_popup_menu(self._context_menu(self.analysis_panel))
        self.analysis_panel.hovered.connect(self._analysis_hover)
        self.analysis_panel.auto_levels_changed.connect(self._auto_levels_changed)
        self.analysis_panel.double_clicked.connect(lambda: self._toggle_single_view("analysis"))
        self.view_tabs.addTab(self.analysis_panel, "2D")
        self.surface_panel = Surface3DPanel(use_gl=use_gl)
        self.surface_panel.set_popup_menu(self._context_menu(self.surface_panel))
        self.surface_panel.double_clicked.connect(lambda: self._toggle_single_view("analysis"))
        self.view_tabs.addTab(self.surface_panel, "3D" + ("" if self.surface_panel.use_gl else " (compatible)"))
        self.view_tabs.setTabToolTip(0, "Flat colour map  (Ctrl+2)")
        self.view_tabs.setTabToolTip(1, "Rotatable 3-D surface: drag to rotate, wheel to zoom  (Ctrl+3)")
        self.view_tabs.currentChanged.connect(lambda _: self._update_analysis(force=True))
        av.addWidget(self.view_tabs, 1)

        # flat reference area for levelling (lives on the 2-D map)
        self.ref_roi = pg.RectROI([10, 10], [100, 80], pen=pg.mkPen((60, 220, 120), width=2,
                                                                     style=QtCore.Qt.PenStyle.DashLine))
        self.ref_roi.addScaleHandle([0, 0], [1, 1])
        label = pg.TextItem("flat reference", color=(60, 220, 120), anchor=(0, 1))
        label.setParentItem(self.ref_roi)
        self.ref_roi.hide()
        self.ref_roi.sigRegionChangeFinished.connect(lambda *_: self._phase_widgets_changed())
        self.analysis_panel.vb.addItem(self.ref_roi)
        self.analysis_panel.add_popup_item(self.ref_roi)

        res = QtWidgets.QHBoxLayout()
        self.fringe_pill = StatusPill()
        self.fringe_pill.hide()
        res.addWidget(self.fringe_pill)
        self.stale_pill = StatusPill()
        self.stale_pill.hide()
        res.addWidget(self.stale_pill)
        self.result_label = QtWidgets.QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        res.addWidget(self.result_label, 1)
        self.copy_btn = QtWidgets.QToolButton()
        self.copy_btn.setText("Copy")
        self.copy_btn.setToolTip("Copy the numbers of this result (tab separated)")
        self.copy_btn.clicked.connect(self._copy_result)
        self.copy_btn.hide()
        res.addWidget(self.copy_btn)
        av.addLayout(res)
        self.splitter.addWidget(self.analysis_box)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([10000, 10000])
        self.setCentralWidget(self.splitter)

    # ======================================================================
    # Side panels
    # ======================================================================
    def _build_docks(self) -> None:
        self.setDockOptions(QtWidgets.QMainWindow.DockOption.AnimatedDocks
                            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks)
        self.setTabPosition(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, QtWidgets.QTabWidget.TabPosition.North)
        self.camera_dock = self._dock("Camera", "Camera / processing", self._build_camera_panel())
        self.phase_dock = self._dock("Phase", "Phase / measurement", self._build_phase_panel())
        self.tabifyDockWidget(self.camera_dock, self.phase_dock)
        self.camera_dock.raise_()
        self.resizeDocks([self.camera_dock], [360], QtCore.Qt.Orientation.Horizontal)

    def _dock(self, name: str, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(name)  # stable for saveState()
        for combo in widget.findChildren(QtWidgets.QComboBox):  # let long entries shrink
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # safety net for big fonts
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(widget)
        dock.setWidget(scroll)
        dock.setMinimumWidth(320)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        return dock

    def _raise_dock(self, dock: QtWidgets.QDockWidget) -> None:
        dock.show()
        dock.raise_()

    def _build_camera_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        g = QtWidgets.QGroupBox("1  Camera")
        f = QtWidgets.QFormLayout(g)
        self.camera_form = f
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(list(SOURCES.values()))
        for i, key in enumerate(SOURCES):
            self.source_combo.setItemData(i, SOURCE_TIPS[key], QtCore.Qt.ItemDataRole.ToolTipRole)
        self.source_combo.currentIndexChanged.connect(self._source_ui_changed)
        f.addRow("Source", self.source_combo)
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.setEditable(True)
        self.device_combo.addItem("First camera found")
        self.device_combo.setToolTip("Which camera: a Thorlabs serial number or a USB camera number (0, 1, …)")
        self.detect_btn = QtWidgets.QPushButton("Detect")
        self.detect_btn.setToolTip("Look for connected cameras")
        self.detect_btn.clicked.connect(self.detect_cameras)
        drow = QtWidgets.QHBoxLayout()
        drow.addWidget(self.device_combo, 1)
        drow.addWidget(self.detect_btn)
        self.device_row = _wrap(drow)
        f.addRow("Device", self.device_row)
        row = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(lambda: self.connect_camera())
        self.disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_camera)
        row.addWidget(self.connect_btn)
        row.addWidget(self.disconnect_btn)
        f.addRow(_wrap(row))
        self.exposure_spin = QtWidgets.QDoubleSpinBox()
        self.exposure_spin.setDecimals(3)
        self.exposure_spin.setSuffix(" ms")
        self.exposure_spin.setKeyboardTracking(False)
        self.exposure_spin.setStepType(QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self.exposure_spin.setSpecialValueText("–")
        self.exposure_spin.valueChanged.connect(self._exposure_changed)
        erow = QtWidgets.QHBoxLayout()
        erow.addWidget(self.exposure_spin, 1)
        for text, factor, tip in (("÷2", 0.5, "Halve the exposure"), ("×2", 2.0, "Double the exposure")):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=factor: self.exposure_spin.setValue(self.exposure_spin.value() * k))
            erow.addWidget(b)
        b = QtWidgets.QToolButton()  # short label; mirrors the Auto exposure action
        b.setText("Auto")
        b.setToolTip(self.act_auto_exposure.toolTip())
        b.clicked.connect(self.act_auto_exposure.trigger)
        self.act_auto_exposure.changed.connect(lambda b=b: b.setEnabled(self.act_auto_exposure.isEnabled()))
        erow.addWidget(b)
        self.exposure_row = _wrap(erow)
        f.addRow("Exposure", self.exposure_row)
        self.gain_spin = QtWidgets.QDoubleSpinBox()
        self.gain_spin.setKeyboardTracking(False)
        self.gain_spin.setSpecialValueText("–")
        self.gain_spin.valueChanged.connect(self._gain_changed)
        f.addRow("Gain", self.gain_spin)
        self.pixel_spin = QtWidgets.QDoubleSpinBox()
        self.pixel_spin.setRange(0.0, 1000.0)
        self.pixel_spin.setDecimals(4)
        self.pixel_spin.setSingleStep(0.01)
        self.pixel_spin.setSuffix(" µm/px")
        self.pixel_spin.setSpecialValueText("not calibrated (px)")
        self.pixel_spin.setToolTip("Size of one camera pixel on the sample = pixel pitch ÷ magnification, "
                                   "e.g. 3.45 µm / 20× = 0.1725 µm/px. Used by the line profile.")
        self.pixel_spin.valueChanged.connect(lambda v: self.analysis_panel.set_pixel_size(v))
        f.addRow("Pixel size", self.pixel_spin)
        self.roi_check = QtWidgets.QCheckBox("Analyse only the yellow box")
        self.roi_check.setToolTip("Drag / resize the yellow box on the live image to analyse just that region")
        self.roi_check.toggled.connect(self._roi_toggled)
        f.addRow(self.roi_check)
        self.camera_info = _muted("")
        f.addRow(self.camera_info)
        v.addWidget(g)

        g = QtWidgets.QGroupBox("2  Processing (runs on every frame)")
        self.pipeline_group = g
        pv = QtWidgets.QVBoxLayout(g)
        row = QtWidgets.QHBoxLayout()
        for text, menu in (("Add filter", self.filters_menu), ("Add transformation", self.transforms_menu)):
            b = QtWidgets.QToolButton()
            b.setText(text + " ")
            b.setMenu(menu)
            b.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
            row.addWidget(b)
        row.addStretch(1)
        pv.addLayout(row)
        self.pipeline_empty = _muted("No processing yet. Add a filter or transformation here, or right-click "
                                     "the image.")
        pv.addWidget(self.pipeline_empty)
        self.pipeline_list = QtWidgets.QListWidget()
        self.pipeline_list.setToolTip("Steps run top to bottom on every frame. Untick to bypass, double-click "
                                      "to edit.")
        self.pipeline_list.setMinimumHeight(90)
        self.pipeline_list.itemChanged.connect(self._pipeline_item_changed)
        self.pipeline_list.itemDoubleClicked.connect(lambda _: self.edit_step())
        pv.addWidget(self.pipeline_list, 1)
        row = QtWidgets.QHBoxLayout()
        for text, slot, tip in (("Edit…", self.edit_step, "Edit the selected step"),
                                ("▲", lambda: self.move_step(-1), "Move the selected step up"),
                                ("▼", lambda: self.move_step(1), "Move the selected step down"),
                                ("Remove", self.remove_step, "Remove the selected step")):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        more = QtWidgets.QToolButton()
        more.setText("More ")
        more.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        mm = QtWidgets.QMenu(more)
        mm.addAction("Re-capture background / restart averaging", self.recapture_background)
        mm.addAction(self.act_bypass)
        mm.addSeparator()
        mm.addAction("Remove all steps", self.clear_pipeline)
        more.setMenu(mm)
        row.addWidget(more)
        row.addStretch(1)
        pv.addLayout(row)
        v.addWidget(g, 1)
        return w

    def _build_phase_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        def combo(items: dict[str, str]):
            c = QtWidgets.QComboBox()
            c.addItems(list(items.values()))
            c.currentIndexChanged.connect(self._phase_widgets_changed)
            return c

        def dspin(lo, hi, val, step, dec=3, suffix=""):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSingleStep(step)
            s.setValue(val)
            s.setSuffix(suffix)
            s.setKeyboardTracking(False)
            s.valueChanged.connect(self._phase_widgets_changed)
            return s

        def ispin(lo, hi, val, suffix=""):
            s = QtWidgets.QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSuffix(suffix)
            s.setKeyboardTracking(False)
            s.valueChanged.connect(self._phase_widgets_changed)
            return s

        cfg = self.phase_cfg
        # 1. method
        g = QtWidgets.QGroupBox("Method")
        f = QtWidgets.QFormLayout(g)
        self.method_combo = combo(PHASE_METHODS)
        f.addRow(self.method_combo)
        self.method_hint = _muted("")
        f.addRow(self.method_hint)
        v.addWidget(g)

        # 2. sample & light (what the user changes between experiments)
        g = QtWidgets.QGroupBox("Sample && light")
        self.output_group = g
        f = QtWidgets.QFormLayout(g)
        self.output_form = f
        self.output_combo = QtWidgets.QComboBox()
        self.output_combo.addItems(list(OUTPUT_LABELS.values()))
        self.output_combo.setCurrentIndex(list(OUTPUT_LABELS).index(cfg.output))
        self.output_combo.currentIndexChanged.connect(self._phase_widgets_changed)
        f.addRow("Show as", self.output_combo)
        self.wavelength_spin = dspin(100, 20000, cfg.wavelength_nm, 1, 2, " nm")
        wl_btn = QtWidgets.QToolButton()
        wl_btn.setText("Laser ")
        wl_btn.setToolTip("Pick a common laser wavelength")
        wl_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        wl_menu = QtWidgets.QMenu(wl_btn)
        for label, nm in WAVELENGTHS:
            wl_menu.addAction(label, lambda _=False, x=nm: self.wavelength_spin.setValue(x))
        wl_btn.setMenu(wl_menu)
        wrow = QtWidgets.QHBoxLayout()
        wrow.addWidget(self.wavelength_spin, 1)
        wrow.addWidget(wl_btn)
        self.wavelength_row = _wrap(wrow)
        f.addRow("Wavelength", self.wavelength_row)
        self.nsample_spin = dspin(0.5, 5.0, cfg.n_sample, 0.01, 4)
        self.nsample_spin.setToolTip("Refractive index of the sample material")
        f.addRow("n sample", self.nsample_spin)
        self.nmedium_spin = dspin(0.5, 5.0, cfg.n_medium, 0.01, 4)
        self.nmedium_spin.setToolTip("Refractive index around the sample (air 1.000, water 1.333)")
        f.addRow("n medium", self.nmedium_spin)
        self.thickness_spin = dspin(0.001, 100000, cfg.thickness_um, 0.1, 3, " µm")
        self.thickness_spin.setToolTip("Known sample thickness, needed to turn phase into Δn")
        f.addRow("Thickness", self.thickness_spin)
        self.formula_hint = _muted("")
        f.addRow(self.formula_hint)
        v.addWidget(g)

        # 3a. phase stepping
        g = QtWidgets.QGroupBox("Phase stepping")
        self.stepping_group = g
        f = QtWidgets.QFormLayout(g)
        self.stepping_form = f
        self.nsteps_spin = ispin(3, 16, cfg.n_steps)
        f.addRow("Number of steps", self.nsteps_spin)
        self.shifter_combo = QtWidgets.QComboBox()
        self.shifter_combo.addItems(list(SHIFTERS.values()))
        for i, key in enumerate(SHIFTERS):
            self.shifter_combo.setItemData(i, SHIFTER_TIPS[key], QtCore.Qt.ItemDataRole.ToolTipRole)
        # choosing 'Serial piezo' never connects at once - the user may first need to set the port
        self.shifter_combo.currentIndexChanged.connect(lambda *_: self._shifter_changed(connect=False, user=True))
        f.addRow("Phase shifter", self.shifter_combo)
        srow = QtWidgets.QHBoxLayout()
        self.shifter_connect_btn = QtWidgets.QToolButton()
        self.shifter_connect_btn.setText("Connect")
        self.shifter_connect_btn.setToolTip("Connect the piezo controller")
        self.shifter_connect_btn.clicked.connect(lambda: self._shifter_changed(connect=True))
        srow.addWidget(self.shifter_connect_btn)
        b = QtWidgets.QToolButton()
        b.setText("Settings…")
        b.setToolTip("Piezo controller settings")
        b.clicked.connect(self.edit_serial_settings)
        srow.addWidget(b)
        srow.addStretch(1)
        f.addRow("", _wrap(srow))
        self.shifter_status = _muted("")
        f.addRow(self.shifter_status)
        grid = QtWidgets.QGridLayout()
        self.acquire_btn = QtWidgets.QPushButton("Acquire")
        self.acquire_btn.clicked.connect(self._acquire_or_cancel)
        self.continuous_btn = QtWidgets.QPushButton("Continuous")
        self.continuous_btn.setCheckable(True)
        self.continuous_btn.toggled.connect(self.toggle_continuous)
        self.capture_btn = QtWidgets.QPushButton("Capture step")
        self.capture_btn.clicked.connect(self.capture_manual_step)
        self.undo_btn = QtWidgets.QPushButton("Undo last step")
        self.undo_btn.clicked.connect(self.undo_manual_step)
        self.startover_btn = QtWidgets.QPushButton("Start over")
        self.startover_btn.clicked.connect(lambda: self.reset_manual_steps())
        for i, b in enumerate((self.acquire_btn, self.continuous_btn, self.capture_btn, self.undo_btn,
                               self.startover_btn)):
            grid.addWidget(b, i // 2, i % 2)
        f.addRow(_wrap(grid))
        self.step_progress = QtWidgets.QProgressBar()
        self.step_progress.setFormat("step %v / %m")
        f.addRow(self.step_progress)
        self.step_label = _muted("")
        f.addRow(self.step_label)
        v.addWidget(g)

        # 3b. Fourier method
        g = QtWidgets.QGroupBox("Live phase (Fourier transform)")
        self.ftp_group = g
        f = QtWidgets.QFormLayout(g)
        self.ftp_pill = StatusPill()
        self.ftp_pill.set_state("muted", "Fringes: waiting…")
        f.addRow(self.ftp_pill)
        self.ftp_hint = _muted("Tilt the reference mirror until you see straight fringes 5–20 px apart.")
        f.addRow(self.ftp_hint)
        crow = QtWidgets.QHBoxLayout()
        self.carrier_x_spin = ispin(-4096, 4096, 0)
        self.carrier_y_spin = ispin(-4096, 4096, 0)
        for sp_ in (self.carrier_x_spin, self.carrier_y_spin):
            sp_.setMinimumWidth(56)
            sp_.setToolTip("Fringes across the image in x / y. 0, 0 = find automatically in every frame")
        crow.addWidget(QtWidgets.QLabel("x"))
        crow.addWidget(self.carrier_x_spin)
        crow.addWidget(QtWidgets.QLabel("y"))
        crow.addWidget(self.carrier_y_spin)
        f.addRow("Fringe frequency", _wrap(crow))
        lock = QtWidgets.QToolButton()
        lock.setDefaultAction(self.act_lock_carrier)
        f.addRow(lock)
        v.addWidget(g)

        # 4. unwrapping & levelling
        g = QtWidgets.QGroupBox("Unwrapping && levelling")
        self.unwrap_group = g
        f = QtWidgets.QFormLayout(g)
        self.unwrap_combo = QtWidgets.QComboBox()
        self.unwrap_combo.addItems([UNWRAP_LABELS[k] for k in UNWRAP_METHODS])
        for i, k in enumerate(UNWRAP_METHODS):
            self.unwrap_combo.setItemData(i, UNWRAP_METHODS[k], QtCore.Qt.ItemDataRole.ToolTipRole)
        self.unwrap_combo.setCurrentIndex(list(UNWRAP_METHODS).index(cfg.unwrap))
        self.unwrap_combo.currentIndexChanged.connect(self._phase_widgets_changed)
        if not skimage_available():
            self.unwrap_combo.model().item(list(UNWRAP_METHODS).index("quality_guided")).setEnabled(False)
        f.addRow("Unwrapping", self.unwrap_combo)
        self.detrend_combo = combo(DETREND_LABELS)
        self.detrend_combo.setCurrentIndex(list(DETREND_LABELS).index(cfg.detrend))
        self.detrend_combo.setToolTip("What to subtract so the map shows the sample, not the instrument tilt")
        f.addRow("Remove", self.detrend_combo)
        self.fit_combo = QtWidgets.QComboBox()
        self.fit_combo.addItems(["Whole image", "Flat reference area (green box)"])
        self.fit_combo.setToolTip("Fit the background on the whole map, or only on an area you know is flat "
                                  "(substrate) - gives correct step heights when a film covers much of the image")
        self.fit_combo.currentIndexChanged.connect(self._fit_mode_changed)
        f.addRow("Fit on", self.fit_combo)
        v.addWidget(g)

        # 5. advanced (collapsed)
        self.adv_btn = QtWidgets.QToolButton()
        self.adv_btn.setText("Advanced settings")
        self.adv_btn.setCheckable(True)
        self.adv_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.adv_btn.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.adv_btn.setStyleSheet("QToolButton { border: none; }")
        v.addWidget(self.adv_btn)
        self.adv_box = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(self.adv_box)
        self.adv_form = f
        self.settle_spin = dspin(0, 5000, cfg.settle_ms, 10, 0, " ms")
        self.settle_spin.setToolTip("Wait after each phase step before grabbing (lets vibrations die down)")
        f.addRow("Settle time", self.settle_spin)
        self.discard_spin = ispin(0, 20, cfg.discard_frames)
        self.discard_spin.setToolTip("Frames thrown away after each step (exposed while the reference moved)")
        f.addRow("Discard frames", self.discard_spin)
        self.fps_step_spin = ispin(1, 100, cfg.frames_per_step)
        self.fps_step_spin.setToolTip("Frames averaged at every phase step (less noise)")
        f.addRow("Frames per step", self.fps_step_spin)
        self.ftp_radius_spin = dspin(0.05, 1.0, cfg.ftp_filter_radius, 0.05, 2)
        self.ftp_radius_spin.setToolTip("Fourier method: filter size as a fraction of the fringe frequency "
                                        "(smaller = smoother, larger = more detail)")
        f.addRow("Fourier filter size", self.ftp_radius_spin)
        self.dc_spin = ispin(1, 200, cfg.ftp_dc_exclude, " px")
        self.dc_spin.setToolTip("Fourier method: ignore this many frequency bins around zero when looking for "
                                "the fringes")
        f.addRow("Ignore low frequencies", self.dc_spin)
        self.mod_spin = dspin(0.0, 1.0, cfg.modulation_threshold, 0.01, 3)
        self.mod_spin.setToolTip("Pixels whose fringe contrast is below this fraction of the best pixels are "
                                 "left blank")
        f.addRow("Min. fringe contrast", self.mod_spin)
        self.adv_box.setVisible(False)
        self.adv_btn.toggled.connect(self._advanced_toggled)
        v.addWidget(self.adv_box)
        v.addStretch(1)
        return w

    def _build_toolbars(self) -> None:
        tb = self.addToolBar("Main")
        tb.setObjectName("Main toolbar")
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.main_toolbar = tb
        tb.addAction(self.act_live)
        self.status_pill = StatusPill()
        self.status_pill.setToolTip("Camera status")
        tb.addWidget(self.status_pill)
        tb.addSeparator()
        for key in ("image", "live_phase", "measure"):
            tb.addAction(self.mode_actions[key])
        self.algo_combo = QtWidgets.QComboBox()
        for key, algo in ALGORITHMS.items():
            self.algo_combo.addItem(algo.name, key)
            self.algo_combo.setItemData(self.algo_combo.count() - 1, METHOD_HINTS.get(key, ""),
                                        QtCore.Qt.ItemDataRole.ToolTipRole)
        self.algo_combo.setCurrentIndex(list(ALGORITHMS).index("4step"))
        self.algo_combo.setToolTip("Phase-stepping method used by Measure")
        self.algo_combo.currentIndexChanged.connect(self._algo_changed)
        tb.addWidget(self.algo_combo)
        self.primary_btn = QtWidgets.QToolButton()
        self.primary_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.primary_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.primary_btn.setStyleSheet("QToolButton { font-weight: bold; padding: 3px 10px; }")
        self.primary_btn.clicked.connect(self._primary_clicked)
        self.primary_menu = QtWidgets.QMenu(self.primary_btn)
        self.primary_btn.setMenu(self.primary_menu)
        tb.addWidget(self.primary_btn)
        tb.addSeparator()
        self.menu_buttons = {}
        for menu in (self.filters_menu, self.transforms_menu, self.phase_menu):
            b = QtWidgets.QToolButton()
            b.setText(menu.title() + " ")
            b.setMenu(menu)
            b.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
            b.setToolTip(f"{menu.title()} pop-up menu (also on right-click of the images)")
            tb.addWidget(b)
            self.menu_buttons[menu.title()] = b

        self.addToolBarBreak()
        tb2 = self.addToolBar("Save and view")
        tb2.setObjectName("Save toolbar")
        tb2.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb2.addWidget(QtWidgets.QLabel(" Sample "))
        self.sample_edit = QtWidgets.QLineEdit()
        self.sample_edit.setPlaceholderText("name / note for file names")
        self.sample_edit.setMaximumWidth(200)
        tb2.addWidget(self.sample_edit)
        tb2.addAction(self.act_quick_save)
        tb2.addSeparator()
        for key in ("live", "both", "analysis"):
            tb2.addAction(self.view_actions[key])

    def _build_menubar(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addAction(self.act_open)
        m.addSeparator()
        m.addAction(self.act_quick_save)
        m.addAction(self.act_save_as)
        m.addAction("Choose save folder…", self.choose_save_folder)
        m.addAction("Open save folder", self.open_save_folder)
        m.addSeparator()
        m.addAction("Save raw frame…", self.save_raw)
        m.addAction("Save processed image…", self.save_processed)
        m.addAction("Save phase result…", self.save_result)
        m.addAction("Save 2D view as PNG…", lambda: self.save_view(self.analysis_panel))
        m.addAction("Save 3D view as PNG…", lambda: self.save_view(self.surface_panel))
        m.addSeparator()
        m.addAction(self._action("Exit", self.close, "Ctrl+Q", group="Files"))
        m = mb.addMenu("&Camera")
        for a in (self.act_live, self.act_stop, self.act_connect, self.act_disconnect, self.act_auto_exposure):
            m.addAction(a)
        m.addSeparator()
        m.addAction("Detect cameras", self.detect_cameras)
        m.addAction("Camera settings", lambda: self._raise_dock(self.camera_dock))
        mb.addMenu(self.filters_menu)
        mb.addMenu(self.transforms_menu)
        mb.addMenu(self.phase_menu)
        m = mb.addMenu("&View")
        for key in ("live", "both", "analysis"):
            m.addAction(self.view_actions[key])
        m.addAction(self.act_2d)
        m.addAction(self.act_3d)
        m.addSeparator()
        m.addMenu(self.cmap_menu)
        for a in (self.act_auto_levels, self.act_profile, self.act_reset_zoom, self.act_true_brightness,
                  self.act_show_clipped, self.act_live_hist):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.act_panels)
        m.addAction(self.act_fullscreen)
        m.addAction("Reset layout", self.reset_layout)
        m = mb.addMenu("&Help")
        m.addAction("Quick guide", self.show_help)
        m.addAction(self.act_help)
        m.addAction("Detect cameras", self.detect_cameras)
        m.addAction("Reset all settings…", self.reset_settings)
        m.addAction("About", lambda: QtWidgets.QMessageBox.about(
            self, "About PhaseScope",
            f"PhaseScope {__version__}\nLive imaging, filtering, phase-stepping interferometry "
            "and 2D/3D visualisation."))

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(140)
        self.progress.setFormat("step %v / %m")
        self.progress.hide()
        self.fps_label = QtWidgets.QLabel("")
        self.cursor_label = QtWidgets.QLabel("")
        self.cursor_label.setMinimumWidth(220)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.cursor_label)
        sb.addPermanentWidget(self.fps_label)

    # ======================================================================
    # Status (pill, banners, empty-view messages)
    # ======================================================================
    def _notify(self, message: str, ms: int = 5000) -> None:
        self.statusBar().showMessage(message, ms)

    def _set_state(self, state: str) -> None:
        self._state = state
        running = self.thread is not None and self.thread.isRunning()
        self.act_live.blockSignals(True)
        self.act_live.setChecked(running)
        self.act_live.blockSignals(False)
        sp = QtWidgets.QStyle.StandardPixmap
        self.act_live.setText("Pause live" if running else "Start live")
        self.act_live.setIcon(self.style().standardIcon(sp.SP_MediaPause if running else sp.SP_MediaPlay))
        self.act_stop.setEnabled(running)
        connected = self.source is not None
        self.act_disconnect.setEnabled(connected)
        self.disconnect_btn.setEnabled(connected)
        if not connected:
            self.exposure_row.setEnabled(False)
            self.gain_spin.setEnabled(False)
            self.act_auto_exposure.setEnabled(False)
        self.connect_btn.setText("Reconnect" if connected else "Connect")
        self._refresh_pill(force=True)
        self._refresh_live_overlays()
        self._update_controls()

    def _refresh_pill(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_pill_t < 0.25:
            return
        self._last_pill_t = now
        st, th = self._state, self.thread
        if st == "disconnected":
            self.status_pill.set_state("muted", "● Not connected", "Choose a source in the Camera tab and press "
                                                                   "Connect")
        elif st == "connecting":
            self.status_pill.set_state("muted", "Connecting…")
        elif st == "error":
            self.status_pill.set_state("error", "✖ Camera stopped", self._last_error)
        elif st == "stalled":
            self.status_pill.set_state("warn", "⚠ No frames", "The camera stopped sending images")
        elif st == "paused":
            self.status_pill.set_state("warn", "❚❚ Paused", "Press Start live (F5)")
        elif self._recording and th is not None and th.analysing and not self.act_continuous.isChecked():
            self.status_pill.set_state("rec", "Calculating the map…")
        elif self._recording and self.last_packet is not None and self.last_packet.step:
            self.status_pill.set_state("rec", f"● Recording {self.last_packet.step}")
        elif isinstance(self.source, StaticImageSource):
            self.status_pill.set_state("info", "Image file", self.source.describe())
        elif th is not None and self.last_packet is not None:
            exp = self.source.get_exposure_ms() if self.source is not None and self.source.supports_exposure else None
            text = f"● LIVE {self.last_packet.fps:.0f} fps"
            if exp:
                text += f" · {exp:.3g} ms"
            self.status_pill.set_state("ok", text, self.source.describe() if self.source else "")
        else:
            self.status_pill.set_state("muted", "Starting…")

    def _refresh_live_overlays(self) -> None:
        """Message and banner on the live image, highest priority first."""
        p, st = self.live_panel, self._state
        if self.source is None:
            p.set_message("No camera connected.\nChoose a source in the Camera tab and press Connect - "
                          "or use the simulated interferometer (no hardware).")
        elif st == "connecting":
            p.set_message("Connecting…")
        elif self.last_packet is None and self.thread is not None:
            p.set_message("Waiting for the first camera frame…")
        else:
            p.set_message(None)
        show_hist = self.act_live_hist.isChecked() and (self.source is not None or self.last_packet is not None)
        if p.histogram_visible() != show_hist:
            p.set_histogram_visible(show_hist)
        p.set_dimmed(st == "paused")
        if st == "error":
            p.set_banner(f"Camera stopped: {self._last_error} - press Start live (F5) to retry", "error")
        elif st == "stalled":
            age = time.monotonic() - self._last_frame_t
            p.set_banner(f"No frames for {age:.0f} s - check the camera and its cable", "warn")
        elif self._recording and self.thread is not None and self.thread.analysing \
                and not self.act_continuous.isChecked():
            p.set_banner("All steps recorded - calculating the map…", "rec")
        elif self._recording:
            step = self.last_packet.step if self.last_packet is not None else ""
            p.set_banner(f"Recording phase steps {step} - keep the set-up still", "rec")
        elif self._capture is not None:
            p.set_banner(f"Capturing step {len(self.manual_frames) + 1}… "
                         f"({self._capture['got']}/{self._capture['target']} frames)", "rec")
        elif self._manual_active():
            p.set_banner(self._manual_prompt(), "info")
        elif st == "paused":
            p.set_banner("Paused - press Start live (F5)", "warn")
        elif self.pipeline.bypass:
            p.set_banner("All processing bypassed (Ctrl+B)", "warn")
        elif self._live_notice is not None:
            p.set_banner(*self._live_notice)
        elif isinstance(self.source, SimulatedInterferometer):
            p.set_banner("Demo mode - this is the built-in simulator, not your camera. "
                         "Pick your camera in the Camera tab.", "info")
        else:
            p.set_banner(None)

    # ======================================================================
    # Camera
    # ======================================================================
    def _source_key(self) -> str:
        return list(SOURCES)[self.source_combo.currentIndex()]

    def _device_text(self) -> str:
        text = self.device_combo.currentText().strip()
        return "" if text.lower().startswith("first camera") else text.split(" ")[0]

    def _source_ui_changed(self, *_) -> None:
        _set_row_visible(self.camera_form, self.device_row, self._source_key() != "sim")

    def _open_source(self, key: str, dev: str) -> CameraSource:
        if key == "sim":
            src: CameraSource = SimulatedInterferometer()
        elif key == "opencv":
            if dev and not dev.isdigit():
                raise CameraError(f"'{dev}' is not a USB camera number - use 0, 1, 2 …", kind="not_found")
            src = OpenCVCamera(int(dev) if dev else 0)
        else:
            src = ThorlabsCamera(dev or None)
        src.open()
        return src

    def connect_camera(self, fallback_to_sim: bool = False, quiet: bool = False) -> None:
        self.disconnect_camera()
        self._live_notice = None
        key, dev = self._source_key(), self._device_text()
        fb = self._fallback
        if fb is not None and key != "sim" and fb[1] in SHIFTERS:  # retrying after a fallback
            self._sync = True
            self.shifter_combo.setCurrentIndex(list(SHIFTERS).index(fb[1]))
            self._sync = False
        self._set_state("connecting")
        QtWidgets.QApplication.processEvents()
        self._finish_closing()  # a camera still ending a long exposure must be closed before reopening
        try:
            src = self._open_source(key, dev)
        except Exception as exc:
            self._set_state("disconnected")
            if fallback_to_sim and key != "sim":
                # show the simulator but keep the user's camera selected, so Connect retries it and the
                # remembered settings still name their camera and phase shifter
                wanted_shifter = fb[1] if fb is not None else self._shifter_key()
                self.source_key = "sim"
                self._connected_device = ""
                self._set_source(self._open_source("sim", ""))
                self._fallback = (key, wanted_shifter)  # after _set_source, which picks the demo shifter
                self.start_live()
                self._live_notice = (f"Could not open the {SOURCE_SHORT[key]} ({exc}) - showing the simulator. "
                                     f"Press Connect in the Camera tab to try the {SOURCE_SHORT[key]} again.", "warn")
                self._refresh_live_overlays()
                self._notify(f"Could not open the {SOURCE_SHORT[key]} - showing the simulator", 10000)
                return
            if not quiet:
                self._camera_problem(key, dev, exc)
            return
        self._fallback = None
        self.source_key = key
        self._connected_device = dev if key != "sim" else ""  # the hidden Device box means nothing here
        self._set_source(src)
        self.start_live()

    def _set_source(self, src: CameraSource) -> None:
        self.source = src
        self._source_serial += 1
        self._levels_source = None
        self._meter_history, self._meter_level = [], None
        self.last_packet = None
        self.camera_info.setText(src.describe())
        self.live_title.setText(f"Live camera - {SOURCE_SHORT.get(self.source_key, 'camera')}"
                                if not isinstance(src, StaticImageSource) else "Image file")
        self._sync = True
        saved = self._exposure_by_source.get(self.source_key, {})
        self.exposure_row.setEnabled(src.supports_exposure)
        self.act_auto_exposure.setEnabled(src.supports_exposure)
        self.exposure_spin.setSpecialValueText("" if src.supports_exposure else "–")
        if src.supports_exposure:
            self.exposure_spin.setRange(*src.exposure_range_ms)
            exp = saved.get("exposure_ms")
            if exp is not None and src.exposure_range_ms[0] <= exp <= src.exposure_range_ms[1]:
                src.set_exposure_ms(exp)
            self.exposure_spin.setValue(src.get_exposure_ms() or src.exposure_range_ms[0])
        self.gain_spin.setEnabled(src.supports_gain)
        self.gain_spin.setSpecialValueText("" if src.supports_gain else "–")
        if src.supports_gain:
            self.gain_spin.setDecimals(getattr(src, "gain_decimals", 1))
            unit = getattr(src, "gain_unit", "")
            self.gain_spin.setSuffix(f" {unit}" if unit else "")
            self.gain_spin.setRange(*src.gain_range)
            gain = saved.get("gain")
            if gain is not None and src.gain_range[0] <= gain <= src.gain_range[1]:
                src.set_gain(gain)
            self.gain_spin.setValue(src.get_gain() or 0.0)
        else:
            self.gain_spin.setSuffix("")
        # sensible default phase shifter for this source
        wanted = "sim" if isinstance(src, SimulatedInterferometer) else (
            "manual" if self._shifter_key() == "sim" else self._shifter_key())
        self.shifter_combo.setCurrentIndex(list(SHIFTERS).index(wanted))
        self._sync = False
        if wanted == "serial" and isinstance(self.shifter, SerialPiezoShifter) and self._serial_connected():
            self.reset_manual_steps(quiet=True)  # keep a working piezo connection across camera reconnects
            self._update_controls()
        else:
            self._shifter_changed(connect=False)

    def _serial_connected(self) -> bool:
        sh = self.shifter
        return isinstance(sh, SerialPiezoShifter) and getattr(sh, "ser", None) is not None

    def disconnect_camera(self) -> None:
        running = self.stop_live(state=None)
        self._ae = None
        if self.source is not None:
            self._remember_exposure()
            src, self.source = self.source, None
            if running is not None:  # still inside a grab: close the camera once the thread has really ended
                self._closing.append((running, src))
                running.finished.connect(lambda src=src: self._close_source(src))
            else:
                self._close_source(src)
        self.last_packet = None
        self.camera_info.setText("")
        self.live_info.setText("")
        self.live_title.setText("Live camera")
        self.live_panel.set_image(None)
        self.live_panel.set_mask_overlay(None)
        self.exposure_pill.set_state("muted", "Exposure: –")
        self.exposure_hint.setText("")
        self.exposure_spin.setSpecialValueText("–")
        self.gain_spin.setSpecialValueText("–")
        self.exposure_spin.setValue(self.exposure_spin.minimum())
        self.gain_spin.setValue(self.gain_spin.minimum())
        self._set_state("disconnected")
        self._update_analysis(force=True)

    def _close_source(self, src: CameraSource) -> None:
        self._closing = [(th, s) for th, s in self._closing if s is not src]
        if id(src) in self._closed_ids:
            return
        self._closed_ids.add(id(src))
        try:
            src.close()
        except Exception as exc:
            self._notify(f"Closing the camera: {exc}")

    def _finish_closing(self) -> None:
        """Wait for cameras still finishing a grab and close them (the Thorlabs SDK allows one at a time)."""
        if not self._closing:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            for th, src in list(self._closing):
                exp = src.get_exposure_ms() if src.supports_exposure else None
                th.wait(int(max(3000, 2 * (exp or 0) + 2000)))  # a grab ends within the camera timeout
                self._close_source(src)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _remember_exposure(self) -> None:
        src = self.source
        if src is None or isinstance(src, StaticImageSource):
            return
        entry = {}
        if src.supports_exposure and src.get_exposure_ms() is not None:
            entry["exposure_ms"] = float(src.get_exposure_ms())
        if src.supports_gain and src.get_gain() is not None:
            entry["gain"] = float(src.get_gain())
        self._exposure_by_source[self.source_key] = entry

    def start_live(self) -> None:
        if self.source is None:
            self.connect_camera()
            return
        if self.thread is not None and self.thread.isRunning():
            return
        th = AcquisitionThread(self.source, self.pipeline, self)
        th.config = self.phase_cfg
        th.shifter = self.shifter
        th.roi = self.live_panel.roi_rect() if self.roi_check.isChecked() else None
        th.hold = self._hold
        th.step_frames = self._manual_active()
        th.continuous_stepping = self.act_continuous.isChecked()
        th.error.connect(lambda msg: self._notify(msg, 8000))
        th.stopped_on_error.connect(self._stopped_on_error)
        th.sequence_progress.connect(self._sequence_progress)
        self._dropped_seq = set()  # sequence ids restart with every camera thread
        self._seq_id = -1
        th.sequence_finished.connect(lambda seq, ok, reason, th=th: self._sequence_finished(th, seq, ok, reason))
        th.finished.connect(self._thread_finished)
        self._pause_after_seq = False
        if self.act_continuous.isChecked():
            self._seq_id = th.sequence_id
            self._seq_context = self._context()
        self.thread = th
        self._last_frame_t = time.monotonic()
        th.start()
        self._set_state("live")

    def stop_live(self, state: str | None = "paused") -> AcquisitionThread | None:
        """Stop acquisition. Returns the thread if it is still finishing a blocking grab."""
        running = None
        self._pause_after_seq = False
        if self.thread is not None:
            if self._recording and not self.act_continuous.isChecked():
                self._notify("Measurement discarded - the camera was stopped before it finished", 8000)
                self._progress_note = "cancelled"
                self.step_progress.setFormat(self._progress_note)
            th, self.thread = self.thread, None
            th.stop(1500)
            if th.isRunning():  # still inside a blocking grab - keep it referenced until it ends
                self._stale_threads.append(th)
                th.finished.connect(lambda th=th: self._forget_thread(th))
                running = th
            else:
                th.deleteLater()
        self._set_continuous(False)
        self._recording = False
        self._capture = None
        self.progress.hide()
        self.fps_label.setText("")
        if state is not None and self.source is not None:
            self._set_state(state)
        return running

    def _forget_thread(self, th: AcquisitionThread) -> None:
        if th in self._stale_threads:
            self._stale_threads.remove(th)
            th.deleteLater()

    def _toggle_live(self, on: bool) -> None:
        if on:
            self.start_live()
        else:
            self.pause_live()
        self._set_state(self._state if self.thread is None else "live")

    def pause_live(self) -> None:
        """Pause the camera - after the map of a just-recorded sequence has been calculated."""
        th = self.thread
        if th is not None and self._recording and not self.act_continuous.isChecked() and th.analysing:
            self._pause_after_seq = True
            self._notify("The live view pauses as soon as the map is calculated…", 6000)
            self._set_state("live")
            return
        self.stop_live()

    def _thread_finished(self) -> None:
        th = self.thread
        if th is not None and not th.isRunning():
            self.thread = None
            if self._recording and th.analysing and self._seq_id >= 0:
                # the camera gave up, but the recorded sequence is still being calculated: keep its map
                self._orphan = (th, self._seq_id, self._seq_context)
                self._notify("Camera stopped - the recorded sequence is still being calculated…", 15000)
            self._recording = False
            self._pause_after_seq = False
            self._seq_id = -1
            self.progress.hide()
            self._set_state("error" if self._state == "error" else
                            ("paused" if self.source is not None else "disconnected"))
            self._update_controls()

    def _stopped_on_error(self, message: str) -> None:
        self._last_error = message
        self._pause_after_seq = False
        self._set_state("error")
        self._notify(f"Camera stopped: {message}", 15000)

    def _orphan_finished(self, seq_id: int, ok: bool, reason: str) -> None:
        """The map of a sequence recorded before the camera stopped is ready (or failed)."""
        th, wanted, context = self._orphan
        if seq_id != wanted:
            return
        self._orphan = None
        res = th.take_phase() if ok else None
        if res is not None and res.config is not None and not self._hold and \
                (res.config.method, res.config.n_steps) == (self.phase_cfg.method, self.phase_cfg.n_steps):
            self._show_result(res, context=self._sequence_context(res, context))
            self._notify(f"{self._seq_method or 'Phase'} measurement finished (the camera stopped afterwards)", 10000)
        elif not ok and reason != "cancelled":
            self._notify(f"Measurement aborted: {reason}", 12000)
        else:
            self._notify("Measurement discarded - the camera stopped while the map was calculated", 10000)

    def _run_on_camera(self, fn) -> None:
        if self.thread is not None and self.thread.isRunning():
            self.thread.submit(fn)
        else:
            try:
                fn()
            except Exception as exc:
                self._notify(str(exc))

    def _exposure_changed(self, value: float) -> None:
        if not self._sync and self.source is not None and self.source.supports_exposure:
            src = self.source
            self._run_on_camera(lambda: src.set_exposure_ms(value))
            self._meter_history.clear()
            self._meter_level = None
            self.reset_manual_steps(quiet=True)  # steps at different exposures cannot be combined

    def _gain_changed(self, value: float) -> None:
        if not self._sync and self.source is not None and self.source.supports_gain:
            src = self.source
            self._run_on_camera(lambda: src.set_gain(value))
            self.reset_manual_steps(quiet=True)

    def _roi_toggled(self, on: bool) -> None:
        self.live_panel.show_roi(on)
        if not on:
            self._roi_changed(None)

    def _roi_changed(self, rect) -> None:
        if not self.roi_check.isChecked():
            rect = None
        if self.thread is not None:
            self.thread.roi = rect
        self.reset_manual_steps(quiet=True)
        self._mark_result_stale()

    def detect_cameras(self) -> None:
        key = self._source_key()
        if key == "sim":
            self._notify("The simulator needs no detection - choose a real camera as Source first")
            return
        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("Searching…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        sdk = self.source.sdk if isinstance(self.source, ThorlabsCamera) else None
        busy_index = self.source.index if isinstance(self.source, OpenCVCamera) else None

        def work():
            try:
                if key == "thorlabs":
                    found = [(s, s) for s in ThorlabsCamera.list_cameras(sdk=sdk)]
                else:
                    found = []
                    for i in range(5):
                        if i == busy_index:
                            found.append((str(i), f"{i}  (connected)"))
                            continue
                        cap = cv2.VideoCapture(i)
                        if cap.isOpened():
                            found.append((str(i), f"{i}  (USB camera)"))
                        cap.release()
                self._detector.done.emit((key, found, ""))
            except Exception as exc:
                self._detector.done.emit((key, [], str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _detect_done(self, payload) -> None:
        key, found, error = payload
        QtWidgets.QApplication.restoreOverrideCursor()
        self.detect_btn.setEnabled(True)
        self.detect_btn.setText("Detect")
        if error:
            self._problem("No cameras found", f"Looking for {SOURCE_SHORT[key]}s failed: {error}",
                          self._camera_fixes(key), error)
            return
        current = self.device_combo.currentText()
        self.device_combo.clear()
        self.device_combo.addItem("First camera found")
        for value, label in found:
            self.device_combo.addItem(label, value)
        self.device_combo.setEditText(current)
        if found:
            self.device_combo.showPopup()
        self._notify(f"Found {len(found)} {SOURCE_SHORT[key]}(s)" if found else
                     f"No {SOURCE_SHORT[key]} found - is it plugged in and closed in other programs?", 8000)

    # ---- friendly problem dialogs ---------------------------------------------
    def _camera_fixes(self, key: str) -> list[str]:
        if key == "thorlabs":
            return ["Close ThorCam - only one program can use the camera at a time.",
                    "Check the USB cable and that the camera's LED is on, then wait a few seconds.",
                    "Make sure the Thorlabs Python SDK is installed (see README)."]
        return ["Close other programs that may use the camera (Zoom, Teams, the Camera app).",
                "Try camera number 0 or 1 in the Device box, or press Detect."]

    def _problem(self, title: str, what: str, fixes: list[str], details: str = "",
                 actions: list[tuple[str, object]] = ()) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("PhaseScope")
        box.setText(f"<b>{title}</b><br>{what}")
        if fixes:
            box.setInformativeText("What to try:\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(fixes, 1)))
        if details:
            box.setDetailedText(details)
        buttons = {}
        for label, fn in actions:
            buttons[box.addButton(label, QtWidgets.QMessageBox.ButtonRole.ActionRole)] = fn
        box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
        box.exec()
        fn = buttons.get(box.clickedButton())
        if fn is not None:
            fn()

    def _use_simulator(self) -> None:
        self.source_combo.setCurrentIndex(list(SOURCES).index("sim"))
        self.connect_camera()

    def _camera_problem(self, key: str, dev: str, exc: Exception) -> None:
        kind = getattr(exc, "kind", "")
        sim = ("Use the simulated camera", self._use_simulator)
        retry = ("Retry", lambda: self.connect_camera())
        copy_pip = ("Copy install command", lambda: QtWidgets.QApplication.clipboard().setText(
            f'pip install "{SDK_ZIP}"'))
        if key == "thorlabs" and kind == "sdk_missing":
            self._problem("The Thorlabs camera software is not installed for Python.",
                          "PhaseScope needs the Python part of the Thorlabs camera SDK.",
                          ["Install ThorCam from thorlabs.com (it contains the SDK).",
                           f'Run: pip install "{SDK_ZIP}"',
                           "Set THORLABS_TSI_DLL_DIR to the SDK's Native_64_lib folder and restart PhaseScope."],
                          str(exc), [copy_pip, sim])
        elif key == "thorlabs" and kind == "dll":
            self._problem("The Thorlabs camera driver files could not be loaded.",
                          "The SDK is installed but its DLL files were not found.",
                          [r"Set THORLABS_TSI_DLL_DIR to …\Scientific Camera Interfaces\SDK\Native Toolkit\dlls"
                           r"\Native_64_lib", "Restart PhaseScope."], str(exc), [retry, sim])
        elif kind == "not_found" and getattr(exc, "available", ()):
            acts = [(f"Use {s}", lambda s=s: (self.device_combo.setEditText(s), self.connect_camera()))
                    for s in exc.available[:4]]
            self._problem(f"Camera {dev} was not found.", "These cameras are connected:",
                          [", ".join(exc.available)], str(exc), acts + [sim])
        elif key == "thorlabs":
            self._problem("PhaseScope cannot see a Thorlabs camera.", str(exc), self._camera_fixes(key),
                          str(exc), [retry, sim])
        elif key == "opencv":
            self._problem(f"No USB camera at number {dev or 0}.", str(exc), self._camera_fixes(key), str(exc),
                          [("Retry with camera 0", lambda: (self.device_combo.setEditText("0"),
                                                            self.connect_camera())), sim])
        else:
            self._problem("The camera could not be opened.", str(exc), [], str(exc), [retry])

    # ======================================================================
    # Live loop (GUI thread)
    # ======================================================================
    def _poll(self) -> None:
        th = self.thread
        if th is None:
            return
        pkt = th.take_frame()
        now = time.monotonic()
        if pkt is not None:
            first = self.last_packet is None
            self.last_packet = pkt
            self._packet_thread = th
            self._last_frame_t = now
            if self._state in ("stalled", "connecting") or (first and self._state != "error"):
                self._set_state("live")
            self._configure_live_levels(pkt.raw)
            self.live_panel.set_image(pkt.raw)
            self._update_live_info(pkt)
            if self._channel() in IMAGE_CHANNELS:
                self._update_analysis()
            if self._capture is not None:
                self._feed_manual_capture(pkt)
            if self._ae is not None:
                self._auto_exposure_step(pkt)
            if first:
                self._refresh_live_overlays()
        res = th.take_phase()
        if res is not None:
            self._accept_result(res)
        # watchdog: frames stopped arriving
        allowance = 2.0
        if self.source is not None and self.source.supports_exposure:
            allowance = max(allowance, 3 * (self.source.get_exposure_ms() or 0) / 1000.0 + 1.0)
        if self._recording:
            allowance += self.phase_cfg.settle_ms / 1000.0 * 2
        if self._state == "live" and not th.analysing and now - self._last_frame_t > allowance:
            self._set_state("stalled")
        elif self._state == "stalled":
            self._refresh_live_overlays()

    def _accept_result(self, res: PhaseResult) -> None:
        """Show a map from the camera thread unless it belongs to a cancelled run or another method."""
        if res.config is None or self._hold:
            return
        seq = res.info.get("sequence_id")
        if seq is not None and seq in self._dropped_seq:
            return
        if (res.config.method, res.config.n_steps) != (self.phase_cfg.method, self.phase_cfg.n_steps):
            return
        # a stepping map belongs to the settings it was recorded with, not those of its arrival
        self._show_result(res, context=self._sequence_context(res, self._seq_context) if seq is not None else None)

    def _slow_update(self) -> None:
        """Twice a second: pipeline status, staleness, live-phase errors."""
        self._refresh_pipeline_status()
        self._refresh_stale_badge()
        th = self.thread
        if th is not None and self.phase_cfg.method == "ftp" and not self._hold:
            err = th.phase_error
            if err:
                self.fringe_pill.set_state("error", "Fringes: none found", err)
                self.fringe_pill.show()
                self.ftp_pill.set_state("error", "No fringes found", err)
                if self.last_result is None:
                    self._update_analysis()
            rate = th.phase_rate
            if self.last_packet is not None:
                self.fps_label.setText(f"camera {self.last_packet.fps:.0f} fps · phase {rate:.1f} maps/s")

    def _configure_live_levels(self, raw: np.ndarray) -> None:
        """True brightness: pin the live view to the camera's full scale."""
        key = (self._source_serial, self.act_true_brightness.isChecked())
        if self._levels_source == key:
            return
        self._levels_source = key
        src = self.source
        if key[1] and src is not None and src.supports_exposure and np.issubdtype(raw.dtype, np.integer):
            self.live_panel.set_fixed_levels((0.0, float((1 << src.bit_depth) - 1)))
        else:
            self.live_panel.set_fixed_levels(None)
            self.live_panel.auto_levels = True

    def _true_brightness_toggled(self, on: bool) -> None:
        self._levels_source = None
        if self.last_packet is not None:
            self._configure_live_levels(self.last_packet.raw)
            self.live_panel.set_image(self.last_packet.raw)

    def _update_live_info(self, pkt: FramePacket) -> None:
        raw = pkt.raw
        depth = self.source.bit_depth if self.source is not None else 8
        self.live_info.setText(f"{raw.shape[1]}×{raw.shape[0]} px · {depth}-bit")
        self.live_info.setToolTip(f"dtype {raw.dtype}, frame {pkt.index}")
        if self._recording or self._capture is not None or (pkt.step and self._state == "live"):
            self._refresh_live_overlays()
        self._refresh_pill()
        if self.thread is not None and self.phase_cfg.method != "ftp":
            self.fps_label.setText(f"{pkt.fps:.0f} fps")
        # exposure meter
        if not np.issubdtype(raw.dtype, np.integer) or isinstance(self.source, StaticImageSource):
            self.exposure_pill.set_state("muted", "Exposure: n/a", "Not a live camera image")
            self.exposure_hint.setText("")
            return
        full = float((1 << depth) - 1)
        sub = raw[::4, ::4]
        sat = float(np.mean(sub >= full)) * 100.0
        fill = float(np.percentile(sub, 99.9)) / full
        level = "sat" if sat > 0.05 else ("dark" if fill < 0.25 else ("bright" if fill > 0.85 else "ok"))
        hist = self._meter_history = (self._meter_history + [level])[-3:]
        if level == "sat" or self._meter_level is None or (len(hist) == 3 and len(set(hist)) == 1):
            self._meter_level = level  # change only when 3 frames agree (no flicker), clipping at once
        level = self._meter_level
        exp = self.source.get_exposure_ms() if self.source.supports_exposure else None
        suggestion = ""
        if exp and level in ("sat", "dark"):
            lo, hi = self.source.exposure_range_ms
            new = exp * 0.6 if level == "sat" else exp * 0.75 / max(fill, 0.01)
            suggestion = f" - try {float(np.clip(new, lo, hi)):.3g} ms or press Auto"
        if level == "sat":
            self.exposure_pill.set_state("error", f"Too bright: {sat:.1f} % clipped")
            self.exposure_hint.setText("Reduce exposure" + suggestion)
        elif level == "dark":
            self.exposure_pill.set_state("warn", f"Too dark: peak at {fill:.0%}")
            self.exposure_hint.setText("Increase exposure" + suggestion)
        elif level == "bright":
            self.exposure_pill.set_state("warn", f"Bright: peak at {fill:.0%}")
            self.exposure_hint.setText("Close to clipping")
        else:
            self.exposure_pill.set_state("ok", f"Exposure OK: peak at {fill:.0%}")
            self.exposure_hint.setText("")
        if self.act_show_clipped.isChecked() and sat > 0:
            self.live_panel.set_mask_overlay(raw[::2, ::2] >= full)
        else:
            self.live_panel.set_mask_overlay(None)

    # ---- auto exposure -------------------------------------------------------------
    def start_auto_exposure(self) -> None:
        src = self.source
        if src is None or not src.supports_exposure:
            self._notify("This camera has no adjustable exposure")
            return
        if self._recording:
            self._notify("Wait until the phase-stepping sequence has finished")
            return
        if self.thread is None:
            self.start_live()
        self._ae = {"iter": 0, "wait_index": self._current_index() + 2}
        self._notify("Auto exposure…", 3000)

    def _current_index(self) -> int:
        """Index of the newest frame of the running thread (0 right after a (re)start)."""
        if self.last_packet is not None and self._packet_thread is self.thread:
            return self.last_packet.index
        return 0

    def _auto_exposure_step(self, pkt: FramePacket) -> None:
        ae, src = self._ae, self.source
        if src is None or self._recording:
            self._ae = None
            return
        if pkt.index < ae["wait_index"] or not np.issubdtype(pkt.raw.dtype, np.integer):
            return
        raw = pkt.raw
        roi = self.thread.roi if self.thread is not None else None
        if roi is not None:
            x, y, w, h = roi
            raw = raw[y:y + h, x:x + w]
        full = float((1 << src.bit_depth) - 1)
        sub = raw[::4, ::4]
        p = float(np.percentile(sub, 99.9))
        sat = float(np.mean(sub >= full))
        exp = src.get_exposure_ms() or self.exposure_spin.value()
        target = 0.75 * full
        if sat <= 0.0005 and abs(p / target - 1.0) < 0.12:
            self._ae = None
            self._notify(f"Auto exposure: {exp:.3g} ms (peak at {p / full:.0%} of full scale)", 6000)
            return
        new = exp * 0.5 if sat > 0.005 else exp * target / max(p, 1.0)
        lo, hi = src.exposure_range_ms
        new = float(np.clip(new, lo, hi))
        ae["iter"] += 1
        if abs(new - exp) < 1e-9 or ae["iter"] > 8:
            self._ae = None
            limit = "longest" if new >= hi else ("shortest" if new <= lo else "")
            self._notify(f"Auto exposure stopped at {exp:.3g} ms" +
                         (f" ({limit} possible) - adjust the gain or the light" if limit else ""), 8000)
            return
        self.exposure_spin.setValue(new)
        ae["wait_index"] = pkt.index + 3

    # ======================================================================
    # Analysis display
    # ======================================================================
    def _channel(self) -> str:
        return self.channel_combo.currentData() or "processed"

    def _set_channel(self, key: str) -> None:
        self.channel_combo.setCurrentIndex(list(CHANNELS).index(key))

    def _channel_data(self):
        ch = self._channel()
        if ch in IMAGE_CHANNELS:
            if self.last_packet is None:
                return None, CHANNELS[ch], "intensity"
            return (self.last_packet.processed if ch == "processed" else self.last_packet.raw,
                    CHANNELS[ch], "Intensity (counts)")
        r = self.last_result
        if r is None:
            return None, self.channel_combo.currentText(), ""
        if ch == "wrapped":
            return r.wrapped, CHANNELS[ch], "Phase (rad)"
        if ch == "unwrapped":
            return np.where(r.mask, r.unwrapped, np.nan), CHANNELS[ch], "Phase (rad)"
        if ch == "modulation":
            return r.modulation, CHANNELS[ch], "Modulation (counts)"
        label = f"{r.quantity} ({r.unit})" if r.unit else r.quantity
        return r.physical, self.channel_combo.currentText(), label

    def _empty_hint(self) -> str:
        ch, cfg = self._channel(), self.phase_cfg
        if self.source is None and self.last_packet is None:
            return "No camera connected.\nChoose a source in the Camera tab and press Connect (F5)."
        if ch in IMAGE_CHANNELS:
            return "Waiting for the first camera frame…"
        if cfg.method == "off":
            return ("Phase is off.\nPress Live phase (F3) for a live height map,\n"
                    "or Measure (F4) for a precise phase-stepping measurement.")
        if cfg.method == "ftp":
            err = self.thread.phase_error if self.thread is not None else ""
            text = "Waiting for fringes…\nTilt the reference mirror until straight fringes 5–20 px apart appear."
            return text + (f"\n\n({err})" if err else "")
        n = len(cfg.shifts())
        if self._recording:
            k, total = self._seq_progress
            if total and k >= total:
                return "Calculating the map…"
            return f"Recording {n} phase-shifted frames… (step {min(k + 1, n)} of {n})"
        if self._calc_busy and self._calc_kind == "manual":
            return "All steps captured - calculating the map…"
        if isinstance(self.shifter, ManualShifter):
            return "Manual stepping.\n" + self._manual_prompt().replace(": ", ":\n", 1)
        if self.shifter is None:
            return "No measurement yet.\n" + self._acquire_reason()
        return f"No measurement yet.\nPress Acquire (Ctrl+P) to record {n} phase-shifted frames."

    def _refresh_empty_hint(self) -> None:
        """While there is no map yet, keep the hint in the analysis area up to date."""
        if hasattr(self, "surface_panel") and self.last_result is None and self._channel() not in IMAGE_CHANNELS:
            self._update_analysis()

    def _update_analysis(self, force: bool = False) -> None:
        data, title, units = self._channel_data()
        ch = self._channel()
        cmap = self.cmap_combo.currentText()
        if cmap == "Auto":
            cmap = AUTO_CMAP.get(ch, "gray")
        if self.analysis_panel.colormap_name != cmap:
            self.analysis_panel.set_colormap(cmap)
        if self.surface_panel.colormap_name != cmap:
            self.surface_panel.set_colormap(cmap)
            force = True
        self.analysis_title.setText(title)
        self.analysis_panel.set_units(units)
        self.surface_panel.set_units(units.split("(")[-1].rstrip(")") if "(" in units else "")
        three_d = self.view_tabs.currentIndex() == 1
        if data is None:
            hint = self._empty_hint()
            self.analysis_panel.set_image(None)
            self.analysis_panel.set_message(hint)
            if self.analysis_panel.histogram_visible():
                self.analysis_panel.set_histogram_visible(False)
            self.surface_panel.clear()
            self.surface_panel.set_message(hint)
            self.headline.setText("")
            return
        self.analysis_panel.set_message(None)
        self.surface_panel.set_message(None)
        if not self.analysis_panel.histogram_visible():
            self.analysis_panel.set_histogram_visible(True)
        if three_d:
            if force:
                self.surface_panel.show_data(data)
            else:
                self.surface_panel.set_data(data)
        else:
            self.analysis_panel.set_image(data)
        self.headline.setText(self._headline_text if ch == "physical" and self.last_result is not None else "")

    def _cmap_changed(self, name: str) -> None:
        act = self.cmap_actions.get(name)
        if act is not None:
            act.setChecked(True)
        self._update_analysis(force=True)

    def _auto_levels_changed(self, on: bool) -> None:
        self.autolevel_check.blockSignals(True)
        self.autolevel_check.setChecked(on)
        self.autolevel_check.blockSignals(False)

    def _profile_toggled(self, on: bool) -> None:
        self.act_profile.blockSignals(True)
        self.act_profile.setChecked(on)
        self.act_profile.blockSignals(False)
        if on and self.view_tabs.currentIndex() != 0:
            self.view_tabs.setCurrentIndex(0)
        self.analysis_panel.show_profile(on)

    def _analysis_hover(self, x: int, y: int, v: float) -> None:
        ch, r = self._channel(), self.last_result
        if ch in IMAGE_CHANNELS:
            unit = "counts"
        elif ch in ("wrapped", "unwrapped"):
            unit = "rad"
        elif ch == "physical" and r is not None:
            unit = r.unit
        else:
            unit = ""
        self.cursor_label.setText(f"({x}, {y}) = {v:.5g} {unit}".rstrip())

    def _context(self) -> tuple:
        """Where a map comes from: camera, analysed region and processing steps."""
        roi = self.thread.roi if self.thread is not None else (
            self.live_panel.roi_rect() if self.roi_check.isChecked() else None)
        return (self.source_key, self._connected_device), (roi, self.pipeline.signature(), self.pipeline.bypass)

    def _sequence_context(self, res: PhaseResult, context: tuple | None) -> tuple | None:
        """Context of a stepping map: the region and processing its sequence was recorded with."""
        settings = res.info.get("sequence_settings")
        if settings is None:
            return context
        camera = context[0] if context is not None else self._context()[0]
        return camera, settings

    def _show_result(self, res: PhaseResult, context: tuple | None = None,
                     files: list[str] | None = None) -> None:
        self.last_result = res
        self._recalc_error = ""
        self._result_files = files
        self._result_source = self._source_serial
        self._result_context = context if context is not None else self._context()
        self._last_result_t = time.monotonic()
        fields = res.fields()
        head = [f"{label} {value}" for label, value, _ in fields if label in ("Peak-to-valley", "RMS")]
        prefix = f"{res.quantity}  " if res.quantity and res.quantity not in ("Height",) else ""
        self._headline_text = prefix + "   ".join(head).replace("Peak-to-valley", "PV")
        nbsp = "\u00a0"  # keep each label with its value when the line wraps
        self.result_label.setText("   ·   ".join(f"{label} {value}".replace(" ", nbsp) for label, value, _ in fields
                                               if label not in ("Peak-to-valley", "RMS")))
        self.result_label.setToolTip("\n".join(f"{label}: {tip}" for label, _, tip in fields))
        self.copy_btn.show()
        level, message = fringe_quality(res.info)
        kind = {"good": "ok", "fair": "warn", "poor": "error"}.get(level)
        if kind:
            short = {"good": "Fringes OK", "fair": "Fringes fair", "poor": "Fringes poor"}[level]
            self.fringe_pill.set_state(kind, short, message)
            self.fringe_pill.show()
            if res.config is not None and res.config.method == "ftp":
                self.ftp_pill.set_state(kind, short, message)
                self.ftp_hint.setText(message)
        else:
            self.fringe_pill.hide()
        note = res.info.get("level_note")
        if level == "poor":
            self.analysis_panel.set_banner(f"Result unreliable - {message}", "warn")
        elif note:
            self.analysis_panel.set_banner(note.capitalize(), "warn")
        elif self._hold:
            self.analysis_panel.set_banner("Held - live map frozen (press H or Resume)", "info")
        else:
            self.analysis_panel.set_banner(None)
        self._refresh_stale_badge()
        if self._channel() not in IMAGE_CHANNELS:
            self._update_analysis()

    def _result_is_stale(self) -> str:
        r = self.last_result
        if r is None or r.config is None:
            return ""
        if self._recalc_error:
            return f"not recalculated: {self._recalc_error}"
        if (r.config.method, r.config.n_steps) != (self.phase_cfg.method, self.phase_cfg.n_steps):
            return "different method"
        if self._result_context is not None:
            camera, settings = self._context()
            if camera != self._result_context[0]:
                return "different camera"
            if settings != self._result_context[1]:
                return "settings changed"
        if r.config.method == "ftp" and not self._hold:
            rate = self.thread.phase_rate if self.thread is not None else 0.0
            limit = max(2.0, 3.0 / rate) if rate > 0 else 2.0
            if self.thread is None or time.monotonic() - self._last_result_t > limit:
                return "not live"
        return ""

    def _refresh_stale_badge(self) -> None:
        reason = self._result_is_stale()
        r = self.last_result
        if reason.startswith("not recalculated") and r is not None:
            self.stale_pill.set_state("warn", "Not recalculated",
                                      f"The map still uses the earlier settings: {self._recalc_error}")
            self.stale_pill.show()
        elif reason and r is not None:
            when = time.strftime("%H:%M:%S", time.localtime(r.timestamp))
            self.stale_pill.set_state("muted", f"Earlier result · {when}",
                                      f"This map is from {when} ({reason}). It is kept so you can still save it.")
            self.stale_pill.show()
        elif r is not None and self._calc_busy and self._calc_kind == "reanalyse":
            self.stale_pill.set_state("info", "Recalculating…", "The map is being recalculated with the new settings.")
            self.stale_pill.show()
        else:
            self.stale_pill.hide()
        dim = reason.startswith("not recalculated") or reason in ("different method", "different camera",
                                                                   "settings changed")
        self.analysis_panel.set_dimmed(dim and self._channel() not in IMAGE_CHANNELS)

    def _mark_result_stale(self) -> None:
        self._refresh_stale_badge()

    def _copy_result(self) -> None:
        r = self.last_result
        if r is None:
            return
        fields = r.fields()
        QtWidgets.QApplication.clipboard().setText(
            "\t".join(label for label, _, _ in fields) + "\n" + "\t".join(value for _, value, _ in fields))
        self._notify("Result copied to the clipboard")

    # ======================================================================
    # Processing pipeline (2.a filters / 2.b transformations)
    # ======================================================================
    def add_operation(self, key: str) -> None:
        op = REGISTRY[key]
        self.pipeline.add(key)
        index = len(self.pipeline) - 1
        self._refresh_pipeline_list(select=index)
        self._raise_dock(self.camera_dock)
        if op.params and not self._edit_step(index, new=True):
            return
        if not getattr(op, "phase_safe", True) and self.phase_cfg.method != "off":
            self._notify(f"{op.name} changes the fringe pattern - phase results will be wrong while it is on", 8000)
        else:
            self._notify(f"Added {op.name}", 3000)

    def _edit_step(self, index: int, new: bool = False) -> bool:
        step = self.pipeline.steps[index]
        old = dict(step.params)
        kind = "Filter" if step.op.category == "filter" else "Transformation"
        subtitle = f"{kind} · {step.op.group}" if step.op.group else kind
        dlg = ParamDialog(step.op.name, step.op.params, old, step.op.description, self, subtitle=subtitle, new=new)
        dlg.changed.connect(lambda p: self._step_params_live(index, p))
        dlg.adjustSize()
        top_right = self.mapToGlobal(QtCore.QPoint(self.width() - dlg.width() - 24, 90))
        dlg.move(top_right)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.pipeline.set_params(index, dlg.values())
            self._refresh_pipeline_list(select=index)
            return True
        if new:
            self.pipeline.remove(index)
        else:
            self.pipeline.set_params(index, old)
        self._refresh_pipeline_list(select=None if new else index)
        return False

    def _step_params_live(self, index: int, params: dict) -> None:
        if not 0 <= index < len(self.pipeline):
            return
        self.pipeline.set_params(index, params)
        item = self.pipeline_list.item(index)
        if item is not None:
            self.pipeline_list.blockSignals(True)  # setText would otherwise fire itemChanged and rebuild the list
            self._decorate_item(item, self.pipeline.steps[index])
            self.pipeline_list.blockSignals(False)
        self.reset_manual_steps(quiet=True)  # steps processed differently cannot be combined

    def _selected_step(self) -> int:
        return self.pipeline_list.currentRow()

    def edit_step(self) -> None:
        i = self._selected_step()
        if 0 <= i < len(self.pipeline) and self.pipeline.steps[i].op.params:
            self._edit_step(i)

    def move_step(self, delta: int) -> None:
        i = self._selected_step()
        if 0 <= i < len(self.pipeline):
            self._refresh_pipeline_list(select=self.pipeline.move(i, delta))

    def remove_step(self) -> None:
        i = self._selected_step()
        if 0 <= i < len(self.pipeline):
            self.pipeline.remove(i)
            self._refresh_pipeline_list(select=min(i, len(self.pipeline) - 1))

    def clear_pipeline(self) -> None:
        self.pipeline.clear()
        self._refresh_pipeline_list()

    def _clear_category(self, category: str) -> None:
        for i in reversed(range(len(self.pipeline))):
            if self.pipeline.steps[i].op.category == category:
                self.pipeline.remove(i)
        self._refresh_pipeline_list()

    def recapture_background(self) -> None:
        steps = self.pipeline.steps
        idx = [i for i, s in enumerate(steps) if s.op.key in ("background", "temporal_average")]
        sel = self._selected_step()
        if sel in idx:
            idx = [sel]
        if not idx:
            self._notify("There is no background or averaging step to restart")
            return
        for i in idx:
            self.pipeline.reset_step(i)
        self._notify("Learning the background again - remove the sample (or move to a blank area) now", 8000)

    def _toggle_bypass(self, on: bool) -> None:
        self.pipeline.bypass = on
        self._refresh_live_overlays()
        self._refresh_pipeline_list()

    def _refresh_pipeline_list(self, select: int | None = None) -> None:
        self.pipeline_list.blockSignals(True)
        self.pipeline_list.clear()
        for step in self.pipeline.steps:
            item = QtWidgets.QListWidgetItem()
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if step.enabled else QtCore.Qt.CheckState.Unchecked)
            self._decorate_item(item, step)
            self.pipeline_list.addItem(item)
        if select is not None and 0 <= select < self.pipeline_list.count():
            self.pipeline_list.setCurrentRow(select)
        self.pipeline_list.blockSignals(False)
        n = len(self.pipeline)
        self.pipeline_empty.setVisible(n == 0)
        nf = sum(1 for s in self.pipeline.steps if s.op.category == "filter" and s.enabled)
        nt = sum(1 for s in self.pipeline.steps if s.op.category == "transform" and s.enabled)
        self.menu_buttons["Filters"].setText(f"Filters ({nf}) " if nf else "Filters ")
        self.menu_buttons["Transformations"].setText(f"Transformations ({nt}) " if nt else "Transformations ")
        title = "2  Processing (runs on every frame)"
        if self.pipeline.bypass:
            title += " - BYPASSED"
        self.pipeline_group.setTitle(title)
        self.reset_manual_steps(quiet=True)

    def _decorate_item(self, item: QtWidgets.QListWidgetItem, step) -> None:
        status = getattr(step, "status", "")
        text = step.label + (f"   [{status}]" if status else "")
        unsafe = not getattr(step.op, "phase_safe", True) and step.enabled and self.phase_cfg.method != "off"
        if step.error:
            item.setText("⚠ " + text)
            item.setForeground(QtGui.QBrush(QtGui.QColor(STATUS_COLORS["error"])))
            item.setToolTip(f"Bypassed because of an error: {step.error}")
        elif unsafe:
            item.setText("⚠ " + text)
            item.setForeground(QtGui.QBrush(QtGui.QColor(STATUS_COLORS["warn"])))
            item.setToolTip("Changes the fringe pattern - phase / height results will be wrong. "
                            "Untick it while measuring.")
        else:
            item.setText(text)
            item.setForeground(self.pipeline_list.palette().brush(QtGui.QPalette.ColorRole.Text))
            kind = "Filter" if step.op.category == "filter" else "Transformation"
            item.setToolTip(f"{kind}: {step.op.description or step.op.name}")

    def _refresh_pipeline_status(self) -> None:
        steps = self.pipeline.steps
        if len(steps) != self.pipeline_list.count():
            return
        self.pipeline_list.blockSignals(True)
        for i, step in enumerate(steps):
            self._decorate_item(self.pipeline_list.item(i), step)
        self.pipeline_list.blockSignals(False)

    def _pipeline_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        i = self.pipeline_list.row(item)
        if 0 <= i < len(self.pipeline):
            self.pipeline.set_enabled(i, item.checkState() == QtCore.Qt.CheckState.Checked)
            self._refresh_pipeline_list(select=i)

    # ======================================================================
    # Phase (2.c)
    # ======================================================================
    def _shifter_key(self) -> str:
        return list(SHIFTERS)[self.shifter_combo.currentIndex()]

    def _mode_for(self, method: str) -> str:
        return "image" if method == "off" else ("live_phase" if method == "ftp" else "measure")

    def set_mode(self, mode: str) -> None:
        """One-click modes: Live image / Live phase / Measure."""
        if mode == "image":
            method = "off"
        elif mode == "live_phase":
            method = "ftp"
        else:
            method = self.algo_combo.currentData() or "4step"
        self.method_combo.setCurrentIndex(list(PHASE_METHODS).index(method))
        if mode == "image" and self._channel() not in IMAGE_CHANNELS:
            self._set_channel("processed")
        elif mode != "image" and self._channel() in IMAGE_CHANNELS:
            self._set_channel("physical")
        if mode == "live_phase" and self.thread is None and self.source is not None:
            self.start_live()
        if mode == "measure" and isinstance(self.shifter, ManualShifter):
            self._raise_dock(self.phase_dock)

    def _algo_changed(self, *_) -> None:
        if self._sync:
            return
        if self.phase_cfg.is_stepping:
            self.method_combo.setCurrentIndex(list(PHASE_METHODS).index(self.algo_combo.currentData()))
        self._update_controls()  # the main button names the method also in the live modes

    def _reference_region(self):
        if self.fit_combo.currentIndex() != 1:
            return None
        pos, size = self.ref_roi.pos(), self.ref_roi.size()
        return (int(round(pos.x())), int(round(pos.y())), max(2, int(round(size.x()))), max(2, int(round(size.y()))))

    def _fit_mode_changed(self, index: int) -> None:
        on = index == 1
        if on:
            data = self.analysis_panel.data()
            if data is not None:
                h, w = data.shape[:2]
                pos, size = self.ref_roi.pos(), self.ref_roi.size()
                if pos.x() + size.x() > w or pos.y() + size.y() > h or size.x() < 4:
                    self.ref_roi.setPos([w * 0.03, h * 0.03], finish=False)
                    self.ref_roi.setSize([w * 0.2, h * 0.2], finish=False)
            if self._channel() in IMAGE_CHANNELS:
                self._set_channel("physical")
            self.view_tabs.setCurrentIndex(0)
            self._notify("Drag the green box onto an area that is flat (bare substrate)", 8000)
        self.ref_roi.setVisible(on)
        self._phase_widgets_changed()

    def _reference_from_view(self) -> None:
        data = self.analysis_panel.data()
        if data is None:
            return
        (x0, x1), (y0, y1) = self.analysis_panel.vb.viewRange()
        h, w = data.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        self.ref_roi.setPos([x0, y0], finish=False)
        self.ref_roi.setSize([x1 - x0, y1 - y0], finish=False)
        if self.fit_combo.currentIndex() != 1:
            self.fit_combo.setCurrentIndex(1)
        else:
            self._phase_widgets_changed()

    def _phase_widgets_changed(self, *_) -> None:
        if self._sync:
            return
        old = self.phase_cfg
        cfg = PhaseConfig(
            method=list(PHASE_METHODS)[self.method_combo.currentIndex()],
            n_steps=self.nsteps_spin.value(),
            unwrap=list(UNWRAP_METHODS)[self.unwrap_combo.currentIndex()],
            detrend=list(DETREND_LABELS)[self.detrend_combo.currentIndex()],
            modulation_threshold=self.mod_spin.value(),
            output=list(OUTPUT_LABELS)[self.output_combo.currentIndex()],
            wavelength_nm=self.wavelength_spin.value(),
            n_sample=self.nsample_spin.value(),
            n_medium=self.nmedium_spin.value(),
            thickness_um=self.thickness_spin.value(),
            ftp_filter_radius=self.ftp_radius_spin.value(),
            ftp_carrier=(self.carrier_x_spin.value(), self.carrier_y_spin.value()),
            ftp_dc_exclude=self.dc_spin.value(),
            settle_ms=self.settle_spin.value(),
            discard_frames=self.discard_spin.value(),
            frames_per_step=self.fps_step_spin.value(),
            level_region=self._reference_region(),
        )
        self.phase_cfg = cfg
        if self.thread is not None:
            self.thread.config = cfg
        # keep menus, mode bar and labels in sync with the panel
        self.method_actions[cfg.method].setChecked(True)
        self.unwrap_actions[cfg.unwrap].setChecked(True)
        self.detrend_actions[cfg.detrend].setChecked(True)
        self.output_actions[cfg.output].setChecked(True)
        self.mode_actions[self._mode_for(cfg.method)].setChecked(True)
        if cfg.is_stepping and self.algo_combo.currentData() != cfg.method:
            self._sync = True
            self.algo_combo.setCurrentIndex(list(ALGORITHMS).index(cfg.method))
            self._sync = False
        self.act_lock_carrier.blockSignals(True)
        self.act_lock_carrier.setChecked(cfg.ftp_carrier != (0, 0))
        self.act_lock_carrier.blockSignals(False)
        self.channel_combo.setItemText(list(CHANNELS).index("physical"), PHYSICAL_LABELS[cfg.output])
        self.method_hint.setText(METHOD_HINTS.get(cfg.method, ""))
        self.formula_hint.setText(OUTPUT_FORMULAS.get(cfg.output, ""))
        self._update_phase_visibility(cfg)
        if cfg.method != old.method or cfg.n_steps != old.n_steps:
            self._progress_note = ""
            if self._recording and not (self.act_continuous.isChecked() and cfg.is_stepping):
                self.cancel_sequence("Measurement cancelled - the method was changed")
            self.reset_manual_steps(quiet=True)
            if cfg.method == "off" or not cfg.is_stepping:
                self._set_continuous(False)
            if cfg.method != "ftp" and self._hold:
                self._toggle_hold(False)
            if cfg.method != "off" and self._channel() in IMAGE_CHANNELS:
                self._set_channel("physical")
            elif cfg.method == "off" and self._channel() not in IMAGE_CHANNELS:
                self._set_channel("processed")
            self._refresh_pipeline_status()
            self._update_analysis(force=True)
        self._update_controls()
        post_only = cfg.method == old.method and cfg.n_steps == old.n_steps and cfg != old
        if post_only and self.last_result is not None and (cfg.is_stepping or self.thread is None or self._hold
                                                           or self._result_is_stale()):
            self._schedule_reanalyse()

    def _update_phase_visibility(self, cfg: PhaseConfig) -> None:
        stepping, ftp, off = cfg.is_stepping, cfg.method == "ftp", cfg.method == "off"
        self.stepping_group.setVisible(stepping)
        self.ftp_group.setVisible(ftp)
        self.unwrap_group.setVisible(not off)
        self.output_group.setVisible(not off)
        _set_row_visible(self.stepping_form, self.nsteps_spin, cfg.method == "nstep")
        _set_row_visible(self.output_form, self.wavelength_row, cfg.output != "phase")
        _set_row_visible(self.output_form, self.nsample_spin, cfg.output == "transmission")
        _set_row_visible(self.output_form, self.nmedium_spin, cfg.output == "transmission")
        _set_row_visible(self.output_form, self.thickness_spin, cfg.output == "index")
        for w in (self.settle_spin, self.discard_spin, self.fps_step_spin):
            _set_row_visible(self.adv_form, w, stepping)
        for w in (self.ftp_radius_spin, self.dc_spin):
            _set_row_visible(self.adv_form, w, ftp)
        _set_row_visible(self.adv_form, self.mod_spin, not off)
        self.adv_btn.setVisible(not off)
        self.adv_box.setVisible(self.adv_btn.isChecked() and not off)
        self.algo_combo.setEnabled(not self._recording)

    def _advanced_toggled(self, on: bool) -> None:
        self.adv_btn.setArrowType(QtCore.Qt.ArrowType.DownArrow if on else QtCore.Qt.ArrowType.RightArrow)
        self.adv_box.setVisible(on and self.phase_cfg.method != "off")

    def _acquire_reason(self) -> str:
        """Why automatic acquisition is unavailable ('' when it is)."""
        if self.source is None:
            return "Connect a camera first (Camera tab)."
        if self.shifter is None:
            key = self._shifter_key()
            if key == "sim":
                return ("The simulated shifter only works with the simulated camera - choose Manual or "
                        "Serial piezo as phase shifter in the Phase tab.")
            if key == "serial":
                return "The piezo controller is not connected - press Connect next to it in the Phase tab."
            return "Choose a phase shifter in the Phase tab."
        if not self.shifter.automatic:
            return "Manual stepping: press Capture step (Space) for each step."
        return ""

    def _manual_active(self) -> bool:
        return self.phase_cfg.is_stepping and isinstance(self.shifter, ManualShifter) and self.source is not None

    def _manual_prompt(self) -> str:
        cfg = self.phase_cfg
        shifts = cfg.shifts()
        k = len(self.manual_frames)
        if k >= len(shifts):
            return "Sequence complete."
        deg = np.degrees(shifts[k])
        mirror = shifts[k] / (4 * np.pi) * cfg.wavelength_nm
        return (f"Step {k + 1} of {len(shifts)}: set the reference to {deg:.0f}° "
                f"(mirror +{mirror:.1f} nm), then press Space")

    def _update_controls(self) -> None:
        """Enable/label every measurement control from one place."""
        if not hasattr(self, "primary_btn"):
            return
        cfg = self.phase_cfg
        reason = self._acquire_reason()
        auto = cfg.is_stepping and reason == ""
        manual = cfg.is_stepping and isinstance(self.shifter, ManualShifter)
        n = len(cfg.shifts()) if cfg.is_stepping else 0
        k = len(self.manual_frames)
        rec = self._recording
        if self.thread is not None:  # manual captures must not include temporal averaging
            self.thread.step_frames = self._capture is not None

        # phase panel buttons: hide what does not apply
        self.acquire_btn.setVisible(not manual)
        self.continuous_btn.setVisible(not manual)
        for b in (self.capture_btn, self.undo_btn, self.startover_btn):
            b.setVisible(manual)
        self.acquire_btn.setText("Cancel (Esc)" if rec and not self.act_continuous.isChecked()
                                 else f"Acquire {n} steps")
        self.acquire_btn.setEnabled(auto or rec)
        self.continuous_btn.setEnabled(auto)
        self.capture_btn.setText(f"Capture step {min(k + 1, n)} of {n}" if n else "Capture step")
        self.capture_btn.setEnabled(manual and self._capture is None)
        self.undo_btn.setEnabled(manual and k > 0)
        self.startover_btn.setEnabled(manual and k > 0)
        for w, ok, why in ((self.acquire_btn, auto or rec, reason),
                           (self.continuous_btn, auto, reason),
                           (self.capture_btn, manual, "Only for the Manual phase shifter.")):
            w.setToolTip("" if ok else why)
        self.act_acquire.setEnabled(not rec)
        self._set_tip(self.act_acquire, "Record one phase-stepping sequence and compute the map" if not reason
                      else reason)
        self.act_continuous.setEnabled(auto)
        self.act_capture.setEnabled(manual)
        self.act_undo_step.setEnabled(manual and k > 0)
        self.act_reset_steps.setEnabled(manual and k > 0)
        self.act_hold.setEnabled(cfg.method == "ftp")
        self.act_lock_carrier.setEnabled(cfg.method == "ftp")
        self.act_reanalyse.setEnabled(self.last_result is not None)
        if manual:
            self.step_progress.setFormat("captured %v of %m")
            self.step_progress.setRange(0, max(n, 1))
            self.step_progress.setValue(k)
            self.step_label.setText(self._manual_prompt())
        elif cfg.is_stepping:
            if not rec and not self._progress_note:
                self.step_progress.setFormat("step %v / %m")
                self.step_progress.setRange(0, max(n, 1))
                self.step_progress.setValue(0)
            self.step_label.setText(reason or (f"{n} frames per sequence, shifts "
                                               + ", ".join(f"{np.degrees(s):.0f}°" for s in cfg.shifts())))

        # the one primary button on the toolbar
        menu = self.primary_menu
        menu.clear()
        mode = self._mode_for(cfg.method)
        self.primary_btn.setEnabled(True)
        if rec:
            text, tip = ("Stop continuous" if self.act_continuous.isChecked() else "Cancel (Esc)"), \
                "Stop the phase-stepping sequence"
        elif mode == "image":
            algo = ALGORITHMS[self.algo_combo.currentData() or "4step"]
            text, tip = f"Measure ({algo.name.split(' ')[0]})", "Switch to Measure and record a sequence (F4)"
        elif mode == "live_phase":
            text = "Resume live" if self._hold else "Hold"
            tip = "Freeze / resume the live height map  (H)"
            menu.addAction(self.act_lock_carrier)
        elif manual:
            text = f"Capture step {min(k + 1, n)}/{n}"
            tip = "Store the current frame as the next step  (Space)"
            menu.addAction(self.act_undo_step)
            menu.addAction(self.act_reset_steps)
        else:
            text = f"Acquire {n} steps"
            tip = reason or "Record one phase-stepping sequence  (Ctrl+P)"
            self.primary_btn.setEnabled(not reason)
            menu.addAction(self.act_continuous)
        menu.addSeparator()
        menu.addAction(self.act_reanalyse)
        self.primary_btn.setText(text)
        self.primary_btn.setToolTip(tip)
        self._refresh_live_overlays()
        self._refresh_empty_hint()

    def _primary_clicked(self) -> None:
        cfg = self.phase_cfg
        mode = self._mode_for(cfg.method)
        if self._recording:
            self._stop_continuous() if self.act_continuous.isChecked() else self.cancel_sequence()
        elif mode == "image":
            self.set_mode("measure")
            self.acquire_sequence()
        elif mode == "live_phase":
            self._toggle_hold(not self._hold)
        elif isinstance(self.shifter, ManualShifter):
            self.capture_manual_step()
        else:
            self.acquire_sequence()

    def _acquire_or_cancel(self) -> None:
        if self._recording and not self.act_continuous.isChecked():
            self.cancel_sequence()
        else:
            self.acquire_sequence()

    def _toggle_hold(self, on: bool) -> None:
        self._hold = bool(on) and self.phase_cfg.method == "ftp"
        self.act_hold.blockSignals(True)
        self.act_hold.setChecked(self._hold)
        self.act_hold.blockSignals(False)
        if self.thread is not None:
            self.thread.hold = self._hold  # no live maps are computed while held
        if self._hold:
            self.analysis_panel.set_banner("Held - live map frozen (press H or Resume)", "info")
        else:
            self.analysis_panel.set_banner(None)
        self._update_controls()

    def _shifter_changed(self, connect: bool = True, user: bool = False) -> None:
        key = self._shifter_key()
        self.shifter_actions[key].setChecked(True)
        self.shifter_connect_btn.setVisible(key == "serial")
        if self._sync:
            return
        if self._recording:
            self.cancel_sequence("Measurement cancelled - the phase shifter was changed")
        if self._fallback is not None:  # the user's own choice wins over the remembered one
            self._fallback = (self._fallback[0], key)
        self._progress_note = ""
        self.step_progress.setValue(0)
        if self.shifter is not None:
            try:
                self.shifter.disconnect()
            except Exception:
                pass
        shifter: PhaseShifter | None = None
        status = ""
        if key == "sim":
            if isinstance(self.source, SimulatedInterferometer):
                shifter = SimulatedShifter(self.source)
            elif self.source is not None:
                status = "Only works with the simulated camera."
        elif key == "manual":
            shifter = ManualShifter()
            status = "You move the reference yourself; PhaseScope tells you where."
        elif connect:
            shifter = SerialPiezoShifter(**self.serial_settings)
            try:
                shifter.connect()
                status = f"✔ Connected ({self.serial_settings['port']})"
            except Exception as exc:
                shifter = None
                status = f"Not connected - {exc}"
                self._serial_problem(exc)
        else:
            status = f"Not connected ({self.serial_settings['port']}) - press Connect"
            if user:  # chosen from a menu maybe: show where to connect it
                self._raise_dock(self.phase_dock)
                self._notify(f"Serial piezo selected - press Connect in the Phase tab (port "
                             f"{self.serial_settings['port']}), or Settings… to choose the port", 10000)
        self.shifter = shifter
        self.shifter_status.setText(status)
        if self.thread is not None:
            self.thread.shifter = shifter
        self.reset_manual_steps(quiet=True)
        self._update_controls()

    def _serial_problem(self, exc: Exception) -> None:
        kind = getattr(exc, "kind", "")
        port = self.serial_settings["port"]
        ports = list_serial_ports()
        acts = [(f"Use {dev}", lambda d=dev: self._set_serial_port(d)) for dev, _ in ports[:3] if dev != port]
        acts += [("Piezo settings…", self.edit_serial_settings),
                 ("Use manual stepping", lambda: self.shifter_combo.setCurrentIndex(list(SHIFTERS).index("manual")))]
        listing = [f"{dev} - {desc}" for dev, desc in ports] or ["(no serial ports found)"]
        if kind == "port_busy":
            what = f"{port} is in use by another program."
            fixes = ["Close Kinesis / APT or any terminal program using the port.", "Then press Connect again."]
        else:
            what = str(exc)
            fixes = ["Check that the controller is switched on and its USB cable is plugged in.",
                     "Choose the right port. Ports found: " + "; ".join(listing)]
        self._problem("The piezo controller could not be connected.", what, fixes, str(exc), acts)

    def _set_serial_port(self, port: str) -> None:
        self.serial_settings["port"] = port
        self._shifter_changed(connect=True)

    def edit_serial_settings(self) -> None:
        ports = list_serial_ports()
        found = ", ".join(f"{d} ({desc})" for d, desc in ports) or "none found"
        dlg = ParamDialog("Piezo controller", SERIAL_PARAMS, self.serial_settings,
                          "ASCII serial piezo controller. Defaults match the Thorlabs MDT693B / MDT694B - check the "
                          f"command syntax in your controller's manual. Serial ports: {found}.", self,
                          subtitle="Phase shifter")
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.serial_settings = dlg.values()
            if self._shifter_key() == "serial":
                self._shifter_changed(connect=True)

    def lock_carrier(self, on: bool | None = None) -> None:
        if on is None:
            on = self.phase_cfg.ftp_carrier == (0, 0)
        if not on:
            kx = ky = 0
        else:
            r = self.last_result
            if r is None or "carrier_px" not in r.info:
                self._notify("No live phase map yet - start Live phase first")
                self.act_lock_carrier.setChecked(False)
                return
            kx, ky = r.info["carrier_px"]
        self._sync = True
        self.carrier_x_spin.setValue(kx)
        self.carrier_y_spin.setValue(ky)
        self._sync = False
        self._phase_widgets_changed()

    def acquire_sequence(self) -> None:
        if self._recording:
            return
        if not self.phase_cfg.is_stepping:
            self.set_mode("measure")
            self._notify(f"Switched to {ALGORITHMS[self.phase_cfg.method].name} phase stepping")
        if isinstance(self.shifter, ManualShifter):
            self._raise_dock(self.phase_dock)
            self._notify("Manual stepping: " + self._manual_prompt(), 8000)
            self._update_controls()
            return
        reason = self._acquire_reason()
        if reason:
            self._problem("Cannot record a sequence yet.", reason, [])
            return
        if self.thread is None:
            self.start_live()
        if self.thread is not None:
            self._recording = True
            self._progress_note = ""
            self._seq_started = time.monotonic()
            self.step_progress.setRange(0, len(self.phase_cfg.shifts()))
            self.step_progress.setValue(0)
            self.step_progress.setFormat("step %v / %m")
            self.progress.setRange(0, len(self.phase_cfg.shifts()))
            self.progress.setValue(0)
            self.progress.show()
            self._seq_method = ALGORITHMS[self.phase_cfg.method].name
            self._seq_context = self._context()
            self._seq_progress = (0, len(self.phase_cfg.shifts()))
            self._seq_id = self.thread.request_sequence()
            self._update_controls()

    def cancel_sequence(self, message: str = "Measurement cancelled") -> None:
        """Stop the measurement the window waits for; its late reports and its map are dropped."""
        was = self._recording
        if self._seq_id >= 0:
            self._dropped_seq.add(self._seq_id)
        self._seq_id = -1
        self._pause_after_seq = False
        if self.thread is not None:
            self.thread.cancel_sequence()
        self._set_continuous(False)
        self._recording = False
        self.progress.hide()
        if was:
            self._progress_note = "cancelled"
            self.step_progress.setFormat(self._progress_note)
            self._notify(message, 6000)
        self._update_controls()

    def toggle_continuous(self, on: bool) -> None:
        if not on:
            self._stop_continuous()
            return
        if self._acquire_reason():
            self._set_continuous(False)
            self._notify(self._acquire_reason())
            return
        if not self.phase_cfg.is_stepping:
            self.set_mode("measure")
        self._set_continuous(True)
        self._seq_method = ALGORITHMS[self.phase_cfg.method].name
        self._seq_started = time.monotonic()
        self._seq_context = self._context()
        self._seq_progress = (0, len(self.phase_cfg.shifts()))
        if self.thread is None:
            self.start_live()  # picks up the continuous flag and the sequence id
        else:
            self._seq_id = self.thread.sequence_id
        self._recording = True
        self._progress_note = ""
        self.progress.show()
        self._update_controls()

    def _stop_continuous(self) -> None:
        """Maps of sequences already recorded are still shown; the partly recorded one is dropped."""
        was = self._recording
        if self.thread is not None:
            self.thread.cancel_sequence()  # abort the sequence being recorded (and any wait for an analysis)
        self._seq_id = -1  # no 'finished' / 'cancelled' notes from the stopped run
        self._set_continuous(False)
        self._recording = False
        self.progress.hide()
        if was:
            self._notify("Continuous stepping stopped", 4000)
        self._update_controls()

    def _set_continuous(self, on: bool) -> None:
        """Tick / untick Continuous and tell the camera thread (unticking aborts a partial sequence)."""
        for w in (self.continuous_btn, self.act_continuous):
            w.blockSignals(True)
            w.setChecked(on)
            w.blockSignals(False)
        if self.thread is not None:
            self.thread.continuous_stepping = on
            if not on and self.thread.sequence_id == self._seq_id:
                self.thread.cancel_sequence()

    def _sequence_progress(self, seq_id: int, k: int, n: int) -> None:
        if seq_id != self._seq_id:
            return  # a cancelled sequence still finishing its current step
        for bar in (self.step_progress, self.progress):
            bar.setRange(0, n)
            bar.setValue(k)
        self._seq_progress = (k, n)
        self._refresh_pill(force=True)
        self._refresh_empty_hint()

    def _sequence_finished(self, th, seq_id: int, ok: bool, reason: str = "") -> None:
        if self._orphan is not None and th is self._orphan[0]:
            self._orphan_finished(seq_id, ok, reason)
            return
        if th is not self.thread or seq_id != self._seq_id:
            return  # an older sequence (cancelled, or from a previous camera thread) - not what we wait for
        continuous = self.act_continuous.isChecked()
        when = time.strftime("%H:%M:%S")
        if ok and continuous:
            pass  # the next sequence starts by itself
        elif ok:
            took = time.monotonic() - self._seq_started
            self._notify(f"{self._seq_method or 'Phase'} measurement finished at {when} ({took:.1f} s)", 8000)
            self._progress_note = f"Done ✓ {when}"
            self.step_progress.setFormat(self._progress_note)
            self._seq_id = -1
        else:  # failed, or cancelled by something other than the user (e.g. the camera)
            self._seq_id = -1
            self._set_continuous(False)
            if reason == "cancelled":
                self._notify("Measurement cancelled", 5000)
            else:
                self._notify(f"Measurement aborted: {reason}", 12000)
            self._progress_note = "cancelled" if reason == "cancelled" else "failed"
            self.step_progress.setFormat(self._progress_note)
        if not (ok and continuous):
            self._recording = False
            self.progress.hide()
        if self._pause_after_seq and not self._recording:
            self._pause_after_seq = False
            res = th.take_phase()  # the map is published just before this report
            if res is not None:
                self._accept_result(res)
            self.stop_live()
        self._refresh_pill(force=True)
        self._update_controls()

    # ---- manual phase stepping -----------------------------------------------------
    def capture_manual_step(self) -> None:
        cfg = self.phase_cfg
        if not cfg.is_stepping or not isinstance(self.shifter, ManualShifter) or self._capture is not None:
            return
        if self.thread is None or self.last_packet is None:
            self._notify("Start the live camera first (F5)")
            return
        self._capture = {"target": max(1, cfg.frames_per_step), "got": 0, "acc": None,
                         "not_before": time.monotonic() + cfg.settle_ms / 1000.0,
                         "skip": cfg.discard_frames, "after": self._current_index()}
        self._update_controls()

    def _feed_manual_capture(self, pkt: FramePacket) -> None:
        cap = self._capture
        if pkt.index <= cap["after"] or time.monotonic() < cap["not_before"] or not pkt.step_exact:
            return  # (not step_exact: processed before the temporal steps could be split off)
        if cap["skip"] > 0:
            cap["skip"] -= 1
            return
        step = pkt.processed_step if pkt.processed_step is not None else pkt.processed
        frame = step.astype(np.float64)
        if cap["acc"] is not None and cap["acc"].shape != frame.shape:
            cap["acc"], cap["got"] = None, 0
        cap["acc"] = frame if cap["acc"] is None else cap["acc"] + frame
        cap["got"] += 1
        if cap["got"] < cap["target"]:
            self._refresh_live_overlays()
            return
        frame = (cap["acc"] / cap["got"]).astype(np.float32)
        self._capture = None
        if self.manual_frames and frame.shape != self.manual_frames[0].shape:
            self.manual_frames.clear()
        self.manual_frames.append(frame)
        cfg = self.phase_cfg
        n = len(cfg.shifts())
        if len(self.manual_frames) < n:
            self._update_controls()
            return
        frames, self.manual_frames = self.manual_frames, []
        self._update_controls()
        self.step_label.setText("All steps captured - calculating the map…")
        self._run_calculation("manual", lambda: analyse_stack(frames, cfg), context=self._context(),
                              name=ALGORITHMS[cfg.method].name, cfg=cfg)

    def undo_manual_step(self) -> None:
        if self.manual_frames:
            self.manual_frames.pop()
            self._update_controls()

    def reset_manual_steps(self, quiet: bool = False) -> None:
        had = bool(self.manual_frames) or self._capture is not None
        self.manual_frames = []
        self._capture = None
        if had and quiet:
            self._notify("Manual phase steps discarded (settings changed)", 4000)
        self._update_controls()

    # ---- re-analysis (coalesced) ---------------------------------------------------
    def _schedule_reanalyse(self) -> None:
        if not hasattr(self, "_reanalyse_timer"):
            self._reanalyse_timer = QtCore.QTimer(self)
            self._reanalyse_timer.setSingleShot(True)
            self._reanalyse_timer.timeout.connect(self.reanalyse)
        self._reanalyse_timer.start(200)

    def reanalyse(self, only_if_needed: bool = False) -> None:
        """Apply the current unwrapping / levelling / unit settings to the last map (in the background)."""
        if self.last_result is None:
            if not only_if_needed:
                self._notify("No map to recalculate yet")
            return
        r = self.last_result
        if only_if_needed and r.config is not None and \
                replace(self.phase_cfg, method=r.config.method, n_steps=r.config.n_steps) == r.config:
            return  # the map already uses the current settings
        if self._calc_busy:
            self._calc_reanalyse = True  # done after the running job, on the newest map and settings
            return
        r, cfg = self.last_result, self.phase_cfg
        keep = ("method", "carrier_px", "fringe_period_px", "estimated_step_deg", "carrier_snr", "contrast")
        info = {k: v for k, v in r.info.items() if k in keep}
        # unwrapping is the slow part: reuse it when only levelling / units changed
        same_unwrap = r.config is not None and (r.config.unwrap, r.config.modulation_threshold) == \
            (cfg.unwrap, cfg.modulation_threshold)
        unwrapped = r.unwrapped if same_unwrap else None
        self._start_calculation("reanalyse", lambda: analyse(r.wrapped, r.modulation, cfg, info, unwrapped=unwrapped),
                                original=r, context=self._result_context, t=self._last_result_t, cfg=cfg,
                                files=self._result_files)

    def _run_calculation(self, kind: str, fn, **meta) -> None:
        """Queue a manual-stepping analysis: it is never dropped, and runs after the current job."""
        if self._calc_busy:
            self._calc_queue.append((kind, fn, meta))
        else:
            self._start_calculation(kind, fn, **meta)

    def _start_calculation(self, kind: str, fn, **meta) -> None:
        """Run one phase calculation in a background thread (only one at a time - they are big)."""
        self._calc_busy = True
        self._calc_kind = kind
        self._job_serial += 1
        job = self._job_serial
        self._notify("Calculating…", 0)  # until the calculation is done
        self._refresh_stale_badge()

        def work():
            try:
                out, err = fn(), ""
            except Exception as exc:
                out, err = None, str(exc) or type(exc).__name__
            try:
                self._calc.done.emit((job, kind, out, err, meta))
            except RuntimeError:  # window already closed
                pass

        threading.Thread(target=work, daemon=True, name="PhaseScope calculation").start()

    def _next_calculation(self) -> None:
        self._calc_busy = False
        self._calc_kind = ""
        if self._calc_queue:
            kind, fn, meta = self._calc_queue.pop(0)
            self._start_calculation(kind, fn, **meta)
        elif self._calc_reanalyse:
            self._calc_reanalyse = False
            self.reanalyse(only_if_needed=True)
        if not self._calc_busy:
            if self.statusBar().currentMessage() == "Calculating…":
                self.statusBar().clearMessage()
            self._refresh_stale_badge()
            self._refresh_empty_hint()

    def _calculation_done(self, payload) -> None:
        job, kind, res, err, meta = payload
        if job != self._job_serial:
            return  # the window was closed meanwhile
        try:
            if kind == "manual":
                if err:
                    self._problem("Phase stepping failed.", err, ["Press Start over and capture the steps again."])
                else:
                    self._show_result(res, context=meta["context"])
                    self._notify(f"Manual {meta['name']} measurement finished at {time.strftime('%H:%M:%S')}",
                                 8000)
                    if self._channel() in IMAGE_CHANNELS:
                        self._set_channel("physical")
                    if self.phase_cfg != meta["cfg"]:
                        self._calc_reanalyse = True  # settings changed while it was calculated
                self._update_controls()
                return
            r = meta["original"]
            if self.last_result is not r:
                self._calc_reanalyse = True  # a newer map arrived meanwhile: recalculate that one
                return
            if err:
                self._notify(f"Recalculation failed: {err}", 8000)
                self._recalc_error = err  # the map on screen does not match the panel settings
                self._refresh_stale_badge()
                self.analysis_panel.set_banner(f"Not recalculated - {err}", "warn")
                return
            res.timestamp = r.timestamp
            if r.config is not None:  # the settings the job used, with the method the map was recorded with
                res.config = replace(meta["cfg"], method=r.config.method, n_steps=r.config.n_steps)
            self._show_result(res, context=meta["context"], files=meta["files"])
            self._last_result_t = meta["t"]
            self._refresh_stale_badge()
        finally:
            self._next_calculation()

    # ======================================================================
    # Files
    # ======================================================================
    def open_images(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open image(s)", self._save_folder(), IMAGE_FILTER)
        if paths:  # natural order, so 'step 2' comes before 'step 10'
            self.load_images(sorted(paths, key=lambda p: _natural_key(os.path.basename(p))))

    def load_images(self, paths: list[str]) -> None:
        try:  # a multi-page TIFF counts as a sequence of its pages
            images = [img for p in paths for img in load_pages(p)]
        except Exception as exc:
            self._problem("The image could not be opened.", str(exc), [])
            return
        if len(images) == 1:
            self.disconnect_camera()
            self.source_key = "file"
            self._set_source(StaticImageSource(images[0], os.path.basename(paths[0])))
            self.start_live()
            return
        n = len(images)  # several images: a phase-stepping sequence
        if not 3 <= n <= 16:
            self._problem("Cannot use these images as a sequence.",
                          f"A phase-stepping sequence needs 3 to 16 images; "
                          f"{'you selected' if len(paths) > 1 else 'the file contains'} {n}.", [])
            return
        cfg = self.phase_cfg
        if not cfg.is_stepping or len(cfg.shifts()) != n:
            key = {3: "3step", 4: "4step", 5: "5step"}.get(n, "nstep")
            self._sync = True
            self.nsteps_spin.setValue(n)
            self.method_combo.setCurrentIndex(list(PHASE_METHODS).index(key))
            self._sync = False
            self._phase_widgets_changed()  # rebuild the settings from the widgets explicitly
            cfg = self.phase_cfg
        try:
            frames = [self.pipeline.apply(img, include_temporal=False) for img in images]
            res = analyse_stack(frames, cfg)
        except Exception as exc:
            self._problem("Phase stepping failed.", str(exc), ["Check that the images are the same size and in "
                                                                "phase-step order."])
            return
        self.stop_live()
        self.live_panel.set_image(images[0])
        self.live_panel.set_message(None)
        self.last_packet = FramePacket(images[0], frames[0], 0, 0.0, 0.0)
        names = [os.path.basename(p) for p in paths]
        self.live_info.setText(f"{n}-frame sequence: {names[0]}{' …' if len(names) > 1 else ''}")
        self._set_channel("physical")
        self._show_result(res, files=names)

    def _save_folder(self) -> str:
        if self.settings is not None:
            folder = self.settings.value("save_folder", "", type=str)
            if folder:
                return folder
        docs = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.DocumentsLocation)
        return os.path.join(docs or os.path.expanduser("~"), "PhaseScope data", time.strftime("%Y-%m-%d"))

    def choose_save_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Save folder", self._save_folder())
        if folder and self.settings is not None:
            self.settings.setValue("save_folder", folder)
            self._notify(f"Measurements will be saved in {folder}")

    def open_save_folder(self) -> None:
        folder = self._save_folder()
        os.makedirs(folder, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))

    def _base_name(self) -> str:
        r = self.last_result
        method = r.config.method if r is not None and r.config is not None else self.phase_cfg.method
        mode = method if method != "off" else "image"
        parts = [time.strftime("%Y%m%d_%H%M%S"), _safe_name(self.sample_edit.text()), mode]
        return "_".join(p for p in parts if p)

    def quick_save(self) -> None:
        folder = self._save_folder()
        base = self._base_name()
        path = os.path.join(folder, base)
        i = 2
        while os.path.exists(path + "_info.txt"):
            path = os.path.join(folder, f"{base}_{i}")
            i += 1
        self.save_measurement(path)

    def save_measurement_as(self) -> None:
        default = os.path.join(self._save_folder(), self._base_name())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save measurement as", default,
                                                        "Measurement (*_info.txt);;All files (*)")
        if not path:
            return
        base = next((path[:-len(suf)] for suf in MEASUREMENT_FILES if path.lower().endswith(suf)), None)
        if base is None:  # e.g. 'wafer7' or 'wafer7.txt' (a dot inside the name is kept: 'Si_0.5um')
            root, ext = os.path.splitext(path)
            base = root if ext.lower() in (".tif", ".tiff", ".png", ".npz", ".npy", ".txt", ".csv") else path
        existing = os.path.exists(base + "_info.txt")
        if existing and QtWidgets.QMessageBox.question(
                self, "Replace measurement?", f"A measurement named '{os.path.basename(base)}' already exists "
                "in this folder. Replace it?") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.save_measurement(base, overwrite=existing)

    def _metadata(self) -> dict:
        src = self.source
        meta = {"saved": time.strftime("%Y-%m-%d %H:%M:%S"), "sample": self.sample_edit.text(),
                "source": src.describe() if src is not None else "",
                "exposure_ms": src.get_exposure_ms() if src is not None and src.supports_exposure else None,
                "gain": src.get_gain() if src is not None and src.supports_gain else None,
                "pixel_size_um": self.pixel_spin.value() or None,
                "roi": self.thread.roi if self.thread is not None else None,
                "pipeline": [{"step": s.label, "key": s.op.key, "params": s.params, "enabled": s.enabled}
                             for s in self.pipeline.steps],
                "phase_settings": asdict(self.phase_cfg)}
        r = self.last_result
        if r is not None and self._result_files:  # computed from files, not from the camera
            meta.update(source="files: " + ", ".join(self._result_files), exposure_ms=None, gain=None)
        if r is not None:
            if r.config is not None:
                meta["result_settings"] = asdict(r.config)  # the settings this map was computed with
            if self._result_context is not None:
                roi, steps, bypass = self._result_context[1]
                meta["result_roi"] = roi
                meta["result_pipeline"] = [{"key": k, "params": p, "enabled": e} for k, p, e in steps]
                meta["result_processing_bypassed"] = bypass
            meta["result"] = {label: value for label, value, _ in r.fields()}
            meta["quantity"], meta["unit"] = r.quantity, r.unit
            meta["result_info"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.info.items()
                                   if k != "sequence_settings"}  # already in result_roi / result_pipeline
            step = self.analysis_panel.step_height() if self.profile_check.isChecked() else None
            if step is not None:
                meta["profile_step_height"] = f"{step[0]:.4g} ± {step[1]:.2g} {r.unit}"
        return meta

    def save_measurement(self, base_path: str, overwrite: bool = False) -> None:
        """Write raw frame, map, picture and a readable info sheet with one base name.

        ``overwrite``: also delete the files of an earlier measurement with that name that this
        one does not write (e.g. its map), so the set never mixes two measurements."""
        folder = os.path.dirname(base_path)
        written = []
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            os.makedirs(folder, exist_ok=True)
            if self.last_packet is not None:
                save_array(base_path + "_raw.tif", self.last_packet.raw)
                written.append("_raw.tif")
            r = self.last_result
            meta = self._metadata()
            if r is not None:
                save_array(base_path + "_map.tif", r.physical)
                np.savez_compressed(base_path + "_maps.npz", wrapped=r.wrapped, unwrapped=r.unwrapped,
                                    physical=r.physical, modulation=r.modulation, mask=r.mask,
                                    metadata=json.dumps(meta, default=str))
                written += ["_map.tif", "_maps.npz"]
            panel = self.surface_panel if self.view_tabs.currentIndex() == 1 else self.analysis_panel
            if panel.grab_image().save(base_path + "_view.png"):
                written.append("_view.png")
            with open(base_path + "_info.txt", "w", encoding="utf-8") as fh:
                for key, value in meta.items():
                    if isinstance(value, dict):
                        fh.write(f"{key}:\n")
                        for k, v in value.items():
                            fh.write(f"    {k}: {v}\n")
                    elif isinstance(value, list):
                        fh.write(f"{key}:\n")
                        for v in value:
                            fh.write(f"    - {v}\n")
                    else:
                        fh.write(f"{key}: {value}\n")
            written.append("_info.txt")
            if overwrite:
                for suffix in MEASUREMENT_FILES:
                    if suffix not in written and os.path.exists(base_path + suffix):
                        os.remove(base_path + suffix)
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._problem("Saving failed.", str(exc), ["Choose another folder with File → Choose save folder…"])
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self._notify(f"Saved {len(written)} files as {os.path.basename(base_path)} in {folder}", 8000)

    def _save_dialog(self, title: str, name: str, filters: str) -> str:
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(self, title, os.path.join(self._save_folder(), name),
                                                             filters)
        exts = {e.lower() for e in re.findall(r"\*(\.\w+)", filters)}
        if ".tif" in exts:
            exts.add(".tiff")
        if path and os.path.splitext(path)[1].lower() not in exts:  # e.g. 'Si_0.5um': add the selected type
            m = re.search(r"\*(\.\w+)", chosen or filters)
            path += m.group(1) if m else ""
        return path

    def save_raw(self) -> None:
        if self.last_packet is None:
            self._notify("No camera frame yet")
            return
        frame = self.last_packet.raw.copy()
        path = self._save_dialog("Save raw frame", self._base_name() + "_raw.tif",
                                 "TIFF (*.tif);;PNG (*.png);;NumPy (*.npy);;CSV (*.csv)")
        if path:
            self._try(lambda: save_array(path, frame), f"Saved {path}")

    def save_processed(self) -> None:
        if self.last_packet is None:
            self._notify("No camera frame yet")
            return
        frame = self.last_packet.processed.copy()
        path = self._save_dialog("Save processed image", self._base_name() + "_processed.tif",
                                 "32-bit TIFF (*.tif);;PNG 8-bit (*.png);;NumPy (*.npy);;CSV (*.csv)")
        if path:
            self._try(lambda: save_array(path, frame), f"Saved {path}")

    def save_result(self) -> None:
        r = self.last_result
        if r is None:
            self._notify("No phase result yet - press Acquire or Live phase")
            return
        path = self._save_dialog("Save phase result", self._base_name() + "_maps.npz",
                                 "All maps + metadata (*.npz);;Map as TIFF (*.tif);;Map as CSV (*.csv);;"
                                 "Map as PNG (*.png)")
        if not path:
            return
        if path.lower().endswith(".npz"):
            meta = self._metadata()
            self._try(lambda: np.savez_compressed(
                path, wrapped=r.wrapped, unwrapped=r.unwrapped, physical=r.physical,
                modulation=r.modulation, mask=r.mask, metadata=json.dumps(meta, default=str)), f"Saved {path}")
        else:
            self._try(lambda: save_array(path, r.physical), f"Saved {path}")

    def save_view(self, panel) -> None:
        path = self._save_dialog("Save view", self._base_name() + "_view.png", "PNG (*.png)")
        if path:
            self._try(lambda: _check_written(panel.grab_image().save(path), path), f"Saved {path}")

    def _try(self, fn, ok_message: str) -> None:
        try:
            fn()
            self._notify(ok_message, 5000)
        except Exception as exc:
            self._problem("Saving failed.", str(exc), [])

    # ======================================================================
    # View
    # ======================================================================
    def set_view_mode(self, mode: str) -> None:
        self._view_mode = mode
        self.view_actions[mode].setChecked(True)
        self.live_box.setVisible(mode != "analysis")
        self.analysis_box.setVisible(mode != "live")

    def _toggle_single_view(self, panel: str) -> None:
        self.set_view_mode("both" if self._view_mode == panel else panel)

    def _toggle_panels(self, on: bool) -> None:
        for dock in (self.camera_dock, self.phase_dock):
            dock.setVisible(on)

    def _toggle_fullscreen(self, on: bool) -> None:
        self.showFullScreen() if on else self.showNormal()

    def _escape(self) -> None:
        if self._recording:
            self._stop_continuous() if self.act_continuous.isChecked() else self.cancel_sequence()
        elif self._capture is not None:
            self._capture = None
            self._update_controls()
        elif self._ae is not None:
            self._ae = None
            self._notify("Auto exposure cancelled")
        elif self._view_mode != "both":
            self.set_view_mode("both")
        elif self.isFullScreen():
            self.act_fullscreen.setChecked(False)
            self.showNormal()

    def reset_layout(self) -> None:
        self.restoreState(self._default_state)
        self.set_view_mode("both")
        self.splitter.setSizes([10000, 10000])
        self.act_panels.setChecked(True)

    # ======================================================================
    # Settings
    # ======================================================================
    def _restore_settings(self) -> str | None:
        """Restore hardware / physical settings and layout. Returns the saved source key."""
        s = self.settings
        if s is None:
            self._default_geometry()
            return None
        try:
            state = json.loads(s.value("state/v2", "", type=str) or "{}")
        except Exception:
            state = {}
        geometry = s.value("geometry")
        if geometry is not None and self.restoreGeometry(geometry):
            pass
        else:
            self._default_geometry()
        if s.value("windowState") is not None:
            self.restoreState(s.value("windowState"))
        if s.value("splitter") is not None:
            self.splitter.restoreState(s.value("splitter"))
        p = state.get("phase", {})
        try:
            if "wavelength_nm" in p:
                self.wavelength_spin.setValue(float(p["wavelength_nm"]))
            for spin, key in ((self.nsample_spin, "n_sample"), (self.nmedium_spin, "n_medium"),
                              (self.thickness_spin, "thickness_um"), (self.mod_spin, "modulation_threshold"),
                              (self.ftp_radius_spin, "ftp_filter_radius"), (self.settle_spin, "settle_ms")):
                if key in p:
                    spin.setValue(float(p[key]))
            for spin, key in ((self.dc_spin, "ftp_dc_exclude"), (self.discard_spin, "discard_frames"),
                              (self.fps_step_spin, "frames_per_step"), (self.nsteps_spin, "n_steps")):
                if key in p:
                    spin.setValue(int(p[key]))
            for combo, items, key in ((self.output_combo, OUTPUT_LABELS, "output"),
                                      (self.unwrap_combo, UNWRAP_METHODS, "unwrap"),
                                      (self.detrend_combo, DETREND_LABELS, "detrend")):
                if p.get(key) in items:
                    combo.setCurrentIndex(list(items).index(p[key]))
            if state.get("algo") in ALGORITHMS:
                self.algo_combo.setCurrentIndex(list(ALGORITHMS).index(state["algo"]))
            self.serial_settings.update({k: v for k, v in state.get("serial", {}).items()
                                         if k in self.serial_settings})
            if state.get("shifter") in SHIFTERS:
                self.shifter_combo.setCurrentIndex(list(SHIFTERS).index(state["shifter"]))
            self._exposure_by_source = dict(state.get("exposure", {}))
            self.pixel_spin.setValue(float(state.get("pixel_size_um", 0.0)))
            if state.get("cmap") in ["Auto"] + COLORMAPS:
                self.cmap_combo.setCurrentText(state["cmap"])
            self.sample_edit.setText(state.get("sample", ""))
            self.adv_btn.setChecked(bool(state.get("advanced", False)))
            self.act_true_brightness.setChecked(bool(state.get("true_brightness", True)))
            self.act_show_clipped.setChecked(bool(state.get("show_clipped", True)))
            if state.get("device"):
                self.device_combo.setEditText(str(state["device"]))
        except Exception:
            pass  # a damaged settings file must never stop the program
        src = state.get("source")
        return src if src in SOURCES else None

    def _default_geometry(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            self.resize(1400, 860)
            return
        geo = screen.availableGeometry()
        self.resize(int(geo.width() * 0.9), int(geo.height() * 0.9))

    def _save_settings(self) -> None:
        s = self.settings
        if s is None:
            return
        self._remember_exposure()
        cfg = self.phase_cfg
        state = {
            # the camera / shifter the user chose - also while the simulator stands in after a failed start
            "source": self._source_key(),
            "device": self._device_text(),
            "exposure": self._exposure_by_source,
            "shifter": self._fallback[1] if self._fallback is not None and self.source_key == "sim"
            else self._shifter_key(),
            "serial": self.serial_settings,
            "algo": self.algo_combo.currentData(),
            "phase": {k: getattr(cfg, k) for k in ("wavelength_nm", "n_sample", "n_medium", "thickness_um",
                                                   "output", "unwrap", "detrend", "modulation_threshold",
                                                   "ftp_filter_radius", "ftp_dc_exclude", "settle_ms",
                                                   "discard_frames", "frames_per_step", "n_steps")},
            "pixel_size_um": self.pixel_spin.value(),
            "cmap": self.cmap_combo.currentText(),
            "sample": self.sample_edit.text(),
            "advanced": self.adv_btn.isChecked(),
            "true_brightness": self.act_true_brightness.isChecked(),
            "show_clipped": self.act_show_clipped.isChecked(),
        }
        s.setValue("state/v2", json.dumps(state, default=str))
        s.setValue("geometry", self.saveGeometry())
        s.setValue("windowState", self.saveState())
        s.setValue("splitter", self.splitter.saveState())

    def reset_settings(self) -> None:
        if self.settings is None:
            return
        if QtWidgets.QMessageBox.question(self, "Reset all settings",
                                          "Forget all remembered settings (camera, wavelength, layout, save "
                                          "folder…)? They return to the defaults when PhaseScope restarts.") \
                == QtWidgets.QMessageBox.StandardButton.Yes:
            self.settings.clear()
            self._persist = False  # do not write the current state back on exit
            self._notify("Settings cleared - restart PhaseScope to use the defaults", 10000)

    # ======================================================================
    # Help
    # ======================================================================
    def show_shortcuts(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Keyboard shortcuts")
        lay = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Group", "Action", "Keys"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        seen = set()
        for group, act in sorted(self._shortcut_actions, key=lambda ga: ga[0]):
            if id(act) in seen:
                continue
            seen.add(id(act))
            row = table.rowCount()
            table.insertRow(row)
            for col, text in enumerate((group, act.text().replace("&", ""), self._shortcut_text(act))):
                table.setItem(row, col, QtWidgets.QTableWidgetItem(text))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(table)
        lay.addWidget(_muted("Right-click any image for the Filters, Transformations and Phase pop-up menus. "
                             "Double-click an image to enlarge it; Esc returns."))
        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(dlg.reject)
        lay.addWidget(box)
        dlg.resize(520, 560)
        dlg.exec()

    def show_help(self) -> None:
        QtWidgets.QMessageBox.information(self, "Quick guide", (
            "1. Camera tab: choose your camera and press Connect. The live image appears on the left. Use Auto "
            "exposure (Ctrl+E) until the meter under the image says 'Exposure OK'.\n\n"
            "2. Processing: add filters / transformations from the toolbar, the Camera tab, or by right-clicking "
            "the image. They run on every frame; untick a step to compare.\n\n"
            "3. Choose what you want:\n"
            "   • Live image (F2) - just the camera.\n"
            "   • Live phase (F3) - a height map from every frame. Tilt the reference mirror until you see straight "
            "fringes 5–20 px apart; the 'Fringes' badge tells you if they are good.\n"
            "   • Measure (F4) - a precise phase-stepping measurement: press Acquire (piezo / simulator) or "
            "capture each step with Space (manual).\n"
            "   The Phase tab sets wavelength and what the map shows: height, thickness or refractive-index change.\n\n"
            "4. Look at the result: Display chooses the map, 2D / 3D tabs switch views, Profile draws a "
            "cross-section with a step-height readout. Double-click an image to enlarge it.\n\n"
            "5. Ctrl+S saves the frame, the map, a picture and an info sheet in your save folder.\n\n"
            "F1 lists all keyboard shortcuts."))

    def closeEvent(self, ev) -> None:
        if self._persist:
            self._save_settings()
        self.hide()  # the window disappears at once even if the camera needs a moment to stop
        self.timer.stop()
        self._slow_timer.stop()
        self._job_serial += 1  # drop results of background calculations still running
        self.disconnect_camera()
        for th in list(self._stale_threads):
            th.wait()  # never let Qt destroy a running thread; a grab ends within the camera timeout
        if self.shifter is not None:
            try:
                self.shifter.disconnect()
            except Exception:
                pass
        super().closeEvent(ev)
