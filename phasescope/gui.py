"""PhaseScope main window."""

from __future__ import annotations

import codecs
import json
import os
from dataclasses import asdict

import cv2
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import __version__
from .acquisition import AcquisitionThread, FramePacket
from .camera import (CameraSource, ManualShifter, OpenCVCamera, PhaseShifter, SerialPiezoShifter,
                     SimulatedInterferometer, SimulatedShifter, StaticImageSource, ThorlabsCamera)
from .phase import (ALGORITHMS, DETREND_MODES, OUTPUT_MODES, UNWRAP_METHODS, PhaseConfig, PhaseResult,
                    analyse, analyse_stack, skimage_available)
from .processing import REGISTRY, Param, Pipeline, operations
from .widgets import COLORMAPS, ImagePanel, ParamDialog, Surface3DPanel

SOURCES = {
    "sim": "Simulated interferometer (no hardware)",
    "opencv": "OpenCV / USB / webcam",
    "thorlabs": "Thorlabs scientific camera (TSI SDK)",
}
SHIFTERS = {
    "sim": "Simulated (simulated camera only)",
    "manual": "Manual (capture each step)",
    "serial": "Serial piezo controller (MDT693B/694B…)",
}
PHASE_METHODS = {"off": "Off", "ftp": "Fourier transform (single frame, live)",
                 **{k: a.name for k, a in ALGORITHMS.items()}}
CHANNELS = ["Processed image", "Raw camera", "Wrapped phase", "Unwrapped phase",
            "Height / physical", "Modulation (fringe contrast)"]
PHASE_CHANNELS = CHANNELS[2:]
AUTO_CMAP = {"Processed image": "gray", "Raw camera": "gray", "Wrapped phase": "CET-C2 (cyclic)",
             "Unwrapped phase": "viridis", "Height / physical": "viridis",
             "Modulation (fringe contrast)": "inferno"}

SERIAL_PARAMS = (
    Param("port", "Serial port", "str", "COM3", tooltip="e.g. COM3 on Windows, /dev/ttyUSB0 on Linux"),
    Param("baudrate", "Baud rate", "int", 115200, 300, 1000000),
    Param("command", "Command template", "str", "xvoltage={volts:.3f}\\r",
          tooltip="{volts} is replaced by the voltage; \\r and \\n are allowed"),
    Param("volts_per_2pi", "Volts per 2π phase", "float", 3.2, 0.001, 150.0, 0.1, decimals=3,
          tooltip="Reflection: 2π = λ/2 mirror travel. Calibrate with the Carré algorithm."),
    Param("offset_v", "Offset voltage (V)", "float", 20.0, 0.0, 150.0, 1.0),
    Param("max_v", "Maximum voltage (V)", "float", 75.0, 1.0, 150.0, 1.0),
)

IMAGE_FILTER = "Images (*.png *.tif *.tiff *.bmp *.jpg *.jpeg *.npy);;All files (*)"


def load_image(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        img = np.load(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"cannot read image {path}")
    if img.ndim == 3:
        img = img[:, :, :3].mean(axis=2).astype(np.float32)
    return img


def save_array(path: str, data: np.ndarray) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, data)
    elif ext in (".csv", ".txt"):
        np.savetxt(path, data, delimiter=",", fmt="%.6g")
    elif ext in (".tif", ".tiff"):
        arr = data if data.dtype in (np.uint8, np.uint16) else data.astype(np.float32)
        if not cv2.imwrite(path, arr):
            raise IOError(f"could not write {path}")
    else:  # 8/16-bit display image for png / jpg / bmp
        if data.dtype in (np.uint8, np.uint16) and ext == ".png":
            arr = data
        else:
            finite = data[np.isfinite(data)]
            lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
            arr = np.clip((np.nan_to_num(data, nan=lo) - lo) / max(hi - lo, 1e-12) * 255, 0, 255).astype(np.uint8)
        if not cv2.imwrite(path, arr):
            raise IOError(f"could not write {path}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, source: str = "sim", device: str = "", use_gl: bool = True,
                 autostart: bool = True) -> None:
        super().__init__()
        self.setWindowTitle(f"PhaseScope {__version__} - live imaging & phase-stepping interferometry")
        self.resize(1500, 900)

        self.pipeline = Pipeline()
        self.source: CameraSource | None = None
        self.thread: AcquisitionThread | None = None
        self.shifter: PhaseShifter | None = None
        self.phase_cfg = PhaseConfig()
        self.serial_settings = {p.name: p.default for p in SERIAL_PARAMS}
        self.last_packet: FramePacket | None = None
        self.last_result: PhaseResult | None = None
        self.manual_frames: list[np.ndarray] = []
        self._sync = True  # True while widgets are updated programmatically
        self._last_error = ""
        self._stale_threads: list[AcquisitionThread] = []

        self._build_menus()
        self._build_central(use_gl)
        self._build_docks()
        self._build_toolbar()
        self._build_menubar()
        self._build_statusbar()

        self.source_combo.setCurrentIndex(list(SOURCES).index(source if source in SOURCES else "sim"))
        self.device_edit.setText(device)
        self._sync = False
        self._phase_widgets_changed()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(25)
        self._pipeline_timer = QtCore.QTimer(self)
        self._pipeline_timer.timeout.connect(self._refresh_pipeline_errors)
        self._pipeline_timer.start(500)
        if autostart:
            QtCore.QTimer.singleShot(0, self.connect_camera)

    # ======================================================================
    # UI construction
    # ======================================================================
    def _action(self, text, slot=None, shortcut=None, checkable=False, tip=None, parent=None):
        act = QtGui.QAction(text, parent or self)
        if shortcut:
            act.setShortcut(QtGui.QKeySequence(shortcut))
        act.setCheckable(checkable)
        if tip:
            act.setStatusTip(tip)
            act.setToolTip(tip)
        if slot is not None:
            act.triggered.connect(slot)
        return act

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
            act = parent.addAction(op.name)
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
        # ---- 2.a / 2.b pop-up menus -----------------------------------------
        self.filters_menu = self._operation_menu("Filters", "filter")
        self.transforms_menu = self._operation_menu("Transformations", "transform")

        # ---- 2.c phase pop-up menu ------------------------------------------
        m = QtWidgets.QMenu("Phase", self)
        self.phase_menu = m
        sub, self.method_actions = self._radio_menu(
            "Phase-stepping / retrieval method", PHASE_METHODS,
            lambda k: self.method_combo.setCurrentIndex(list(PHASE_METHODS).index(k)))
        m.addMenu(sub)
        sub, self.unwrap_actions = self._radio_menu(
            "Phase unwrapping", UNWRAP_METHODS,
            lambda k: self.unwrap_combo.setCurrentIndex(list(UNWRAP_METHODS).index(k)))
        m.addMenu(sub)
        sub, self.detrend_actions = self._radio_menu(
            "Background removal", DETREND_MODES,
            lambda k: self.detrend_combo.setCurrentIndex(list(DETREND_MODES).index(k)))
        m.addMenu(sub)
        sub, self.output_actions = self._radio_menu(
            "Output quantity", OUTPUT_MODES,
            lambda k: self.output_combo.setCurrentIndex(list(OUTPUT_MODES).index(k)))
        m.addMenu(sub)
        m.addSeparator()
        self.act_acquire = self._action("Acquire phase-stepping sequence", self.acquire_sequence,
                                        "Ctrl+P", tip="Record one phase-stepped sequence and compute the phase")
        self.act_continuous = self._action("Continuous phase stepping", self.toggle_continuous,
                                           checkable=True, tip="Repeat phase-stepping sequences for a live phase map")
        self.act_capture = self._action("Capture manual step", self.capture_manual_step, "Space",
                                        tip="Store the current frame as the next phase step (manual shifter)")
        self.act_reset_steps = self._action("Reset manual steps", self.reset_manual_steps)
        self.act_reanalyse = self._action("Re-analyse last result", self.reanalyse,
                                          tip="Apply the current unwrapping / conversion settings to the last result")
        for a in (self.act_acquire, self.act_continuous, self.act_capture, self.act_reset_steps,
                  self.act_reanalyse):
            m.addAction(a)
        m.addSeparator()
        sub, self.shifter_actions = self._radio_menu(
            "Phase shifter", SHIFTERS,
            lambda k: self.shifter_combo.setCurrentIndex(list(SHIFTERS).index(k)))
        m.addMenu(sub)
        m.addAction("Piezo controller settings…", self.edit_serial_settings)
        m.addAction("Show phase settings panel", lambda: self._raise_dock(self.phase_dock))

        # ---- colour maps -----------------------------------------------------
        self.cmap_menu, self.cmap_actions = self._radio_menu(
            "Colour map", {"Auto": "Auto (per channel)", **{c: c for c in COLORMAPS}},
            lambda k: self.cmap_combo.setCurrentText(k))

    def _context_menu(self, panel) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        menu.addMenu(self.filters_menu)
        menu.addMenu(self.transforms_menu)
        menu.addMenu(self.phase_menu)
        menu.addSeparator()
        menu.addMenu(self.cmap_menu)
        if isinstance(panel, ImagePanel):
            menu.addAction("Auto levels", lambda: setattr(panel, "auto_levels", True))
            menu.addAction("Reset zoom", panel.reset_view)
        else:
            menu.addAction("Reset camera", panel.reset_camera)
        menu.addSeparator()
        menu.addAction("Save this view as PNG…", lambda: self.save_view(panel))
        return menu

    def _build_central(self, use_gl: bool) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # ---- 1. live camera area --------------------------------------------
        live_box = QtWidgets.QGroupBox("Live camera")
        lv = QtWidgets.QVBoxLayout(live_box)
        self.live_panel = ImagePanel(histogram=True, roi=True)
        self.live_panel.set_popup_menu(self._context_menu(self.live_panel))
        self.live_panel.hovered.connect(lambda x, y, v: self.cursor_label.setText(f"live ({x}, {y}) = {v:.5g}"))
        self.live_panel.roi_changed.connect(self._roi_changed)
        lv.addWidget(self.live_panel, 1)
        self.live_info = QtWidgets.QLabel("No camera")
        lv.addWidget(self.live_info)
        splitter.addWidget(live_box)

        # ---- 3. analysis area: 2-D / 3-D ------------------------------------
        ana_box = QtWidgets.QGroupBox("Analysis")
        self.analysis_box = ana_box
        av = QtWidgets.QVBoxLayout(ana_box)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Display:"))
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(CHANNELS)
        self.channel_combo.currentIndexChanged.connect(lambda _: self._update_analysis(force=True))
        bar.addWidget(self.channel_combo)
        bar.addWidget(QtWidgets.QLabel("Colour map:"))
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["Auto"] + COLORMAPS)
        self.cmap_combo.currentTextChanged.connect(self._cmap_changed)
        bar.addWidget(self.cmap_combo)
        self.autolevel_check = QtWidgets.QCheckBox("Auto levels")
        self.autolevel_check.setChecked(True)
        self.autolevel_check.toggled.connect(lambda v: setattr(self.analysis_panel, "auto_levels", v))
        bar.addWidget(self.autolevel_check)
        self.profile_check = QtWidgets.QCheckBox("Line profile")
        self.profile_check.toggled.connect(lambda v: self.analysis_panel.show_profile(v))
        bar.addWidget(self.profile_check)
        bar.addStretch(1)
        av.addLayout(bar)

        self.view_tabs = QtWidgets.QTabWidget()
        self.analysis_panel = ImagePanel(histogram=True, profile=True)
        self.analysis_panel.set_popup_menu(self._context_menu(self.analysis_panel))
        self.analysis_panel.hovered.connect(self._analysis_hover)
        self.analysis_panel.auto_levels_changed.connect(self._auto_levels_changed)
        self.view_tabs.addTab(self.analysis_panel, "2D view")
        self.surface_panel = Surface3DPanel(use_gl=use_gl)
        self.surface_panel.set_popup_menu(self._context_menu(self.surface_panel))
        self.view_tabs.addTab(self.surface_panel, "3D view" + ("" if self.surface_panel.use_gl else " (matplotlib)"))
        self.view_tabs.currentChanged.connect(lambda _: self._update_analysis(force=True))
        av.addWidget(self.view_tabs, 1)
        self.result_label = QtWidgets.QLabel("")
        self.result_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        av.addWidget(self.result_label)
        splitter.addWidget(ana_box)
        splitter.setSizes([560, 640])
        self.setCentralWidget(splitter)

    def _build_docks(self) -> None:
        self.setDockOptions(QtWidgets.QMainWindow.DockOption.AnimatedDocks
                            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks)
        # ---- camera ---------------------------------------------------------
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(list(SOURCES.values()))
        f.addRow("Source", self.source_combo)
        self.device_edit = QtWidgets.QLineEdit()
        self.device_edit.setPlaceholderText("index (OpenCV) / serial no. (Thorlabs); blank = first")
        f.addRow("Device", self.device_edit)
        row = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("Connect && start")
        self.connect_btn.clicked.connect(self.connect_camera)
        self.disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_camera)
        row.addWidget(self.connect_btn)
        row.addWidget(self.disconnect_btn)
        f.addRow(row)
        self.exposure_spin = QtWidgets.QDoubleSpinBox()
        self.exposure_spin.setDecimals(3)
        self.exposure_spin.setSuffix(" ms")
        self.exposure_spin.setKeyboardTracking(False)
        self.exposure_spin.valueChanged.connect(self._exposure_changed)
        f.addRow("Exposure", self.exposure_spin)
        self.gain_spin = QtWidgets.QDoubleSpinBox()
        self.gain_spin.setDecimals(1)
        self.gain_spin.setKeyboardTracking(False)
        self.gain_spin.valueChanged.connect(self._gain_changed)
        f.addRow("Gain", self.gain_spin)
        self.roi_check = QtWidgets.QCheckBox("Analyse only the yellow ROI of the live image")
        self.roi_check.toggled.connect(self._roi_toggled)
        f.addRow(self.roi_check)
        self.camera_info = QtWidgets.QLabel("")
        self.camera_info.setWordWrap(True)
        f.addRow(self.camera_info)
        self.camera_dock = self._dock("Camera", w)

        # ---- processing pipeline -------------------------------------------
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        for text, menu in (("Add filter", self.filters_menu), ("Add transformation", self.transforms_menu)):
            b = QtWidgets.QToolButton()
            b.setText(text + " ")
            b.setMenu(menu)
            b.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
            b.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.pipeline_list = QtWidgets.QListWidget()
        self.pipeline_list.setToolTip("Steps run top to bottom on every frame. Untick to bypass, "
                                      "double-click to edit.")
        self.pipeline_list.itemChanged.connect(self._pipeline_item_changed)
        self.pipeline_list.itemDoubleClicked.connect(lambda _: self.edit_step())
        v.addWidget(self.pipeline_list, 1)
        grid = QtWidgets.QGridLayout()
        for i, (text, slot, tip) in enumerate((
                ("Edit…", self.edit_step, "Edit parameters"),
                ("Move up", lambda: self.move_step(-1), "Move the selected step up"),
                ("Move down", lambda: self.move_step(1), "Move the selected step down"),
                ("Remove", self.remove_step, "Remove the selected step"),
                ("Clear all", self.clear_pipeline, "Remove all steps"),
                ("Reset state", self.reset_pipeline_state, "Forget stored backgrounds / averages"))):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            grid.addWidget(b, i // 3, i % 3)
        v.addLayout(grid)
        self.pipeline_dock = self._dock("Processing pipeline", w)

        # ---- phase ---------------------------------------------------------
        self.phase_dock = self._dock("Phase", self._build_phase_panel())
        self.tabifyDockWidget(self.camera_dock, self.pipeline_dock)
        self.tabifyDockWidget(self.pipeline_dock, self.phase_dock)
        self.camera_dock.raise_()
        self.resizeDocks([self.camera_dock], [370], QtCore.Qt.Orientation.Horizontal)

    def _dock(self, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(title)
        for combo in widget.findChildren(QtWidgets.QComboBox):  # let long entries shrink
            combo.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(14)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(widget)
        dock.setWidget(scroll)
        dock.setMinimumWidth(340)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        return dock

    def _raise_dock(self, dock: QtWidgets.QDockWidget) -> None:
        dock.show()
        dock.raise_()

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
        # method & acquisition
        g = QtWidgets.QGroupBox("Phase retrieval")
        f = QtWidgets.QFormLayout(g)
        self.method_combo = combo(PHASE_METHODS)
        f.addRow("Method", self.method_combo)
        self.nsteps_spin = ispin(3, 16, cfg.n_steps)
        f.addRow("N (N-step)", self.nsteps_spin)
        self.shifter_combo = QtWidgets.QComboBox()
        self.shifter_combo.addItems(list(SHIFTERS.values()))
        self.shifter_combo.currentIndexChanged.connect(self._shifter_changed)
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(self.shifter_combo, 1)
        b = QtWidgets.QPushButton("…")
        b.setToolTip("Piezo controller settings")
        b.setMaximumWidth(32)
        b.clicked.connect(self.edit_serial_settings)
        srow.addWidget(b)
        f.addRow("Phase shifter", srow)
        self.settle_spin = dspin(0, 5000, cfg.settle_ms, 10, 0, " ms")
        self.settle_spin.setToolTip("Wait after each piezo move before grabbing")
        f.addRow("Settle time", self.settle_spin)
        self.discard_spin = ispin(0, 20, cfg.discard_frames)
        self.discard_spin.setToolTip("Frames thrown away after each move (exposed while moving)")
        f.addRow("Discard frames", self.discard_spin)
        self.fps_step_spin = ispin(1, 100, cfg.frames_per_step)
        self.fps_step_spin.setToolTip("Frames averaged at every phase step")
        f.addRow("Frames per step", self.fps_step_spin)
        grid = QtWidgets.QGridLayout()
        self.acquire_btn = QtWidgets.QPushButton("Acquire")
        self.acquire_btn.clicked.connect(self.acquire_sequence)
        self.continuous_btn = QtWidgets.QPushButton("Continuous")
        self.continuous_btn.setCheckable(True)
        self.continuous_btn.toggled.connect(self.toggle_continuous)
        self.capture_btn = QtWidgets.QPushButton("Capture step")
        self.capture_btn.clicked.connect(self.capture_manual_step)
        reset = QtWidgets.QPushButton("Reset steps")
        reset.clicked.connect(lambda: self.reset_manual_steps())
        for i, b in enumerate((self.acquire_btn, self.continuous_btn, self.capture_btn, reset)):
            grid.addWidget(b, i // 2, i % 2)
        f.addRow(grid)
        self.step_progress = QtWidgets.QProgressBar()
        self.step_progress.setFormat("step %v / %m")
        f.addRow(self.step_progress)
        self.step_label = QtWidgets.QLabel("")
        self.step_label.setWordWrap(True)
        f.addRow(self.step_label)
        v.addWidget(g)

        # Fourier-transform method
        g = QtWidgets.QGroupBox("Fourier-transform method")
        f = QtWidgets.QFormLayout(g)
        self.ftp_radius_spin = dspin(0.05, 1.0, cfg.ftp_filter_radius, 0.05, 2)
        self.ftp_radius_spin.setToolTip("Side-lobe filter radius as a fraction of the carrier frequency")
        f.addRow("Filter radius", self.ftp_radius_spin)
        crow = QtWidgets.QHBoxLayout()
        self.carrier_x_spin = ispin(-4096, 4096, 0)
        self.carrier_y_spin = ispin(-4096, 4096, 0)
        for sp_ in (self.carrier_x_spin, self.carrier_y_spin):
            sp_.setMinimumWidth(60)
        crow.addWidget(QtWidgets.QLabel("x"))
        crow.addWidget(self.carrier_x_spin)
        crow.addWidget(QtWidgets.QLabel("y"))
        crow.addWidget(self.carrier_y_spin)
        f.addRow("Carrier", crow)
        hint = QtWidgets.QLabel("0, 0 = find automatically in every frame")
        hint.setStyleSheet("color: gray")
        f.addRow(hint)
        lock = QtWidgets.QPushButton("Lock current carrier")
        lock.setToolTip("Fix the carrier found in the last frame (steadier live phase)")
        lock.clicked.connect(self.lock_carrier)
        f.addRow(lock)
        self.dc_spin = ispin(1, 200, cfg.ftp_dc_exclude, " px")
        f.addRow("Ignore around DC", self.dc_spin)
        v.addWidget(g)

        # unwrapping
        g = QtWidgets.QGroupBox("Unwrapping && background")
        f = QtWidgets.QFormLayout(g)
        self.unwrap_combo = combo(UNWRAP_METHODS)
        self.unwrap_combo.setCurrentIndex(list(UNWRAP_METHODS).index(cfg.unwrap))
        if not skimage_available():
            idx = list(UNWRAP_METHODS).index("quality_guided")
            self.unwrap_combo.model().item(idx).setEnabled(False)
        f.addRow("Unwrapping", self.unwrap_combo)
        self.detrend_combo = combo(DETREND_MODES)
        self.detrend_combo.setCurrentIndex(list(DETREND_MODES).index(cfg.detrend))
        f.addRow("Remove", self.detrend_combo)
        self.mod_spin = dspin(0.0, 1.0, cfg.modulation_threshold, 0.01, 3)
        self.mod_spin.setToolTip("Pixels whose fringe contrast is below this fraction are masked")
        f.addRow("Modulation threshold", self.mod_spin)
        v.addWidget(g)

        # output
        g = QtWidgets.QGroupBox("Output (phase → physical units)")
        f = QtWidgets.QFormLayout(g)
        self.output_combo = combo(OUTPUT_MODES)
        self.output_combo.setCurrentIndex(list(OUTPUT_MODES).index(cfg.output))
        f.addRow("Quantity", self.output_combo)
        self.wavelength_spin = dspin(100, 20000, cfg.wavelength_nm, 1, 2, " nm")
        f.addRow("Wavelength", self.wavelength_spin)
        self.nsample_spin = dspin(0.5, 5.0, cfg.n_sample, 0.01, 4)
        f.addRow("n sample", self.nsample_spin)
        self.nmedium_spin = dspin(0.5, 5.0, cfg.n_medium, 0.01, 4)
        f.addRow("n medium", self.nmedium_spin)
        self.thickness_spin = dspin(0.001, 100000, cfg.thickness_um, 0.1, 3, " µm")
        f.addRow("Sample thickness", self.thickness_spin)
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setObjectName("Main toolbar")
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        style = self.style()
        sp = QtWidgets.QStyle.StandardPixmap
        self.act_start = self._action("Start", self.start_live, "F5", tip="Start continuous acquisition")
        self.act_start.setIcon(style.standardIcon(sp.SP_MediaPlay))
        self.act_stop = self._action("Stop", self.stop_live, "F6", tip="Stop acquisition")
        self.act_stop.setIcon(style.standardIcon(sp.SP_MediaStop))
        self.act_snapshot = self._action("Snapshot", self.save_raw, "Ctrl+Shift+S",
                                         tip="Save the current raw camera frame")
        self.act_snapshot.setIcon(style.standardIcon(sp.SP_DialogSaveButton))
        tb.addAction(self.act_start)
        tb.addAction(self.act_stop)
        tb.addAction(self.act_snapshot)
        tb.addSeparator()
        for menu in (self.filters_menu, self.transforms_menu, self.phase_menu):
            b = QtWidgets.QToolButton()
            b.setText(menu.title() + " ")
            b.setMenu(menu)
            b.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
            tb.addWidget(b)
        tb.addSeparator()
        tb.addAction(self.act_acquire)
        tb.addSeparator()
        self.act_2d = self._action("2D", lambda: self.view_tabs.setCurrentIndex(0), "Ctrl+2", tip="2-D view")
        self.act_3d = self._action("3D", lambda: self.view_tabs.setCurrentIndex(1), "Ctrl+3", tip="3-D view")
        tb.addAction(self.act_2d)
        tb.addAction(self.act_3d)

    def _build_menubar(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addAction(self._action("Open image(s)…", self.open_images, "Ctrl+O",
                                 tip="One image: analyse it like a live frame. Several: phase-step sequence."))
        m.addSeparator()
        m.addAction(self._action("Save raw frame…", self.save_raw))
        m.addAction(self._action("Save processed image…", self.save_processed))
        m.addAction(self._action("Save phase result…", self.save_result, "Ctrl+S"))
        m.addAction(self._action("Save 2D view as PNG…", lambda: self.save_view(self.analysis_panel)))
        m.addAction(self._action("Save 3D view as PNG…", lambda: self.save_view(self.surface_panel)))
        m.addSeparator()
        m.addAction(self._action("Exit", self.close, "Ctrl+Q"))
        m = mb.addMenu("&Camera")
        m.addAction(self.act_start)
        m.addAction(self.act_stop)
        m.addAction(self.act_snapshot)
        m.addAction("Connect selected source", self.connect_camera)
        m.addAction("Disconnect", self.disconnect_camera)
        m.addAction("Camera settings", lambda: self._raise_dock(self.camera_dock))
        mb.addMenu(self.filters_menu)
        mb.addMenu(self.transforms_menu)
        mb.addMenu(self.phase_menu)
        m = mb.addMenu("&View")
        m.addAction(self.act_2d)
        m.addAction(self.act_3d)
        m.addMenu(self.cmap_menu)
        m.addAction("Auto levels", lambda: setattr(self.analysis_panel, "auto_levels", True))
        m.addAction("Reset zoom", lambda: (self.live_panel.reset_view(), self.analysis_panel.reset_view()))
        m.addSeparator()
        for d in (self.camera_dock, self.pipeline_dock, self.phase_dock):
            m.addAction(d.toggleViewAction())
        m = mb.addMenu("&Help")
        m.addAction("Quick guide", self.show_help)
        m.addAction("About", lambda: QtWidgets.QMessageBox.about(
            self, "About PhaseScope",
            f"PhaseScope {__version__}\nLive imaging, filtering, phase-stepping interferometry "
            "and 2D/3D visualisation."))

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self.fps_label = QtWidgets.QLabel("")
        self.cursor_label = QtWidgets.QLabel("")
        self.cursor_label.setMinimumWidth(260)
        sb.addPermanentWidget(self.cursor_label)
        sb.addPermanentWidget(self.fps_label)

    # ======================================================================
    # Camera
    # ======================================================================
    def _source_key(self) -> str:
        return list(SOURCES)[self.source_combo.currentIndex()]

    def connect_camera(self) -> None:
        self.disconnect_camera()
        key, dev = self._source_key(), self.device_edit.text().strip()
        try:
            if key == "sim":
                src: CameraSource = SimulatedInterferometer()
            elif key == "opencv":
                src = OpenCVCamera(int(dev) if dev else 0)
            else:
                src = ThorlabsCamera(dev or None)
            src.open()
        except Exception as exc:
            self._error(f"Could not open {SOURCES[key]}:\n{exc}", dialog=True)
            return
        self._set_source(src)
        self.start_live()

    def _set_source(self, src: CameraSource) -> None:
        self.source = src
        self.camera_info.setText(src.describe())
        self._sync = True
        self.exposure_spin.setEnabled(src.supports_exposure)
        if src.supports_exposure:
            self.exposure_spin.setRange(*src.exposure_range_ms)
            self.exposure_spin.setValue(src.get_exposure_ms() or src.exposure_range_ms[0])
        self.gain_spin.setEnabled(src.supports_gain)
        if src.supports_gain:
            self.gain_spin.setRange(*src.gain_range)
            self.gain_spin.setValue(src.get_gain() or 0.0)
        # sensible default shifter for this source
        wanted = "sim" if isinstance(src, SimulatedInterferometer) else (
            "manual" if self._shifter_key() == "sim" else self._shifter_key())
        self.shifter_combo.setCurrentIndex(list(SHIFTERS).index(wanted))
        self._sync = False
        self.pipeline.reset_state()
        self._shifter_changed()

    def disconnect_camera(self) -> None:
        self.stop_live()
        if self.source is not None:
            try:
                self.source.close()
            except Exception as exc:
                self._error(f"closing camera: {exc}")
            self.source = None
        self.camera_info.setText("")
        self.live_info.setText("No camera")

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
        th.continuous_stepping = self.continuous_btn.isChecked()
        th.error.connect(self._error)
        th.sequence_progress.connect(self._sequence_progress)
        th.sequence_finished.connect(self._sequence_finished)
        th.finished.connect(self._thread_finished)
        self.thread = th
        th.start()
        self.statusBar().showMessage(f"Live: {self.source.describe()}", 3000)

    def stop_live(self) -> None:
        if self.thread is not None:
            th, self.thread = self.thread, None
            th.stop()
            if th.isRunning():  # still inside a blocking grab - let it finish
                self._stale_threads.append(th)
            else:
                th.deleteLater()
        self.fps_label.setText("stopped")
        self._set_continuous(False)

    def _thread_finished(self) -> None:
        if self.thread is not None and not self.thread.isRunning():
            self.thread = None
            self.fps_label.setText("stopped")

    def _run_on_camera(self, fn) -> None:
        if self.thread is not None and self.thread.isRunning():
            self.thread.submit(fn)
        else:
            try:
                fn()
            except Exception as exc:
                self._error(str(exc))

    def _exposure_changed(self, value: float) -> None:
        if not self._sync and self.source is not None and self.source.supports_exposure:
            src = self.source
            self._run_on_camera(lambda: src.set_exposure_ms(value))

    def _gain_changed(self, value: float) -> None:
        if not self._sync and self.source is not None and self.source.supports_gain:
            src = self.source
            self._run_on_camera(lambda: src.set_gain(value))

    def _roi_toggled(self, on: bool) -> None:
        self.live_panel.show_roi(on)
        if not on:
            self._roi_changed(None)

    def _roi_changed(self, rect) -> None:
        if not self.roi_check.isChecked():
            rect = None
        if self.thread is not None:
            self.thread.roi = rect
        self.pipeline.reset_state()
        self.reset_manual_steps(quiet=True)

    # ======================================================================
    # Live update loop (GUI thread)
    # ======================================================================
    def _poll(self) -> None:
        th = self.thread
        if th is None:
            return
        pkt = th.take_frame()
        if pkt is not None:
            self.last_packet = pkt
            self.live_panel.set_image(pkt.raw)
            self._update_live_info(pkt)
            if self.channel_combo.currentText() in ("Processed image", "Raw camera"):
                self._update_analysis()
        res = th.take_phase()
        if res is not None and res.config is not None and \
                (res.config.method, res.config.n_steps) == (self.phase_cfg.method, self.phase_cfg.n_steps):
            self._show_result(res)  # (results of a method that was just switched off are dropped)

    def _update_live_info(self, pkt: FramePacket) -> None:
        raw = pkt.raw
        sub = raw[::4, ::4]
        depth = self.source.bit_depth if self.source is not None else 8
        maxval = (1 << depth) - 1 if np.issubdtype(raw.dtype, np.integer) else np.inf
        sat = float(np.mean(sub >= maxval)) * 100 if np.isfinite(maxval) else 0.0
        text = f"{raw.shape[1]}×{raw.shape[0]}  {raw.dtype}  min {sub.min():.0f}  max {sub.max():.0f}"
        if sat > 0.05:
            text += f"   <b style='color:#e53935'>saturated {sat:.1f}% - reduce exposure</b>"
        if pkt.step:
            text += f"   <b>recording step {pkt.step}</b>"
        if self.manual_frames:
            text += f"   <b>manual steps captured: {len(self.manual_frames)}</b>"
        self.live_info.setText(text)
        self.fps_label.setText(f"{pkt.fps:5.1f} fps  frame {pkt.index}")
        if pkt.phase_error and pkt.phase_error != self._last_error:
            self._last_error = pkt.phase_error
            self.statusBar().showMessage(f"Phase: {pkt.phase_error}", 4000)

    def _channel_data(self):
        ch = self.channel_combo.currentText()
        if ch in ("Processed image", "Raw camera"):
            if self.last_packet is None:
                return None, ch, "intensity"
            return (self.last_packet.processed if ch == "Processed image" else self.last_packet.raw,
                    ch, "intensity (counts)")
        r = self.last_result
        if r is None:
            hint = ("press Acquire" if self.phase_cfg.is_stepping else
                    "choose a phase method in the Phase menu")
            return None, f"{ch}: no phase result yet - {hint}", ""
        if ch == "Wrapped phase":
            return r.wrapped, ch, "phase (rad)"
        if ch == "Unwrapped phase":
            return np.where(r.mask, r.unwrapped, np.nan), ch, "phase (rad)"
        if ch == "Modulation (fringe contrast)":
            return r.modulation, ch, "modulation (counts)"
        label = f"{r.quantity} ({r.unit})" if r.unit else r.quantity
        return r.physical, label, label

    def _update_analysis(self, force: bool = False) -> None:
        data, title, units = self._channel_data()
        cmap = self.cmap_combo.currentText()
        if cmap == "Auto":
            cmap = AUTO_CMAP.get(self.channel_combo.currentText(), "gray")
        three_d = self.view_tabs.currentIndex() == 1
        if self.analysis_panel.colormap_name != cmap:
            self.analysis_panel.set_colormap(cmap)
        if self.surface_panel._cmap_name != cmap:
            self.surface_panel._cmap_name = cmap
            force = True
        self.analysis_box.setTitle(f"Analysis - {title}")
        self.analysis_panel.set_units(units)
        self.surface_panel.set_units(units.split("(")[-1].rstrip(")") if "(" in units else "")
        if data is None:
            self.analysis_panel.set_image(None)
            return
        if three_d:
            if force:
                self.surface_panel._data = data
                self.surface_panel._redraw(force=True)
            else:
                self.surface_panel.set_data(data)
        else:
            self.analysis_panel.set_image(data)

    def _cmap_changed(self, name: str) -> None:
        act = self.cmap_actions.get(name)
        if act is not None:
            act.setChecked(True)
        self._update_analysis(force=True)

    def _auto_levels_changed(self, on: bool) -> None:
        self.autolevel_check.blockSignals(True)
        self.autolevel_check.setChecked(on)
        self.autolevel_check.blockSignals(False)

    def _analysis_hover(self, x: int, y: int, v: float) -> None:
        _, _, units = self._channel_data()
        self.cursor_label.setText(f"({x}, {y}) = {v:.5g} {units}")

    def _show_result(self, res: PhaseResult) -> None:
        self.last_result = res
        self.result_label.setText(res.summary())
        if self.channel_combo.currentText() in PHASE_CHANNELS:
            self._update_analysis()

    # ======================================================================
    # Processing pipeline (2.a filters / 2.b transformations)
    # ======================================================================
    def add_operation(self, key: str) -> None:
        op = REGISTRY[key]
        self.pipeline.add(key)
        index = len(self.pipeline) - 1
        self._refresh_pipeline_list(select=index)
        if op.params and not self._edit_step(index, new=True):
            return
        self.statusBar().showMessage(f"Added {op.name}", 3000)

    def _edit_step(self, index: int, new: bool = False) -> bool:
        step = self.pipeline.steps[index]
        old = dict(step.params)
        dlg = ParamDialog(step.op.name, step.op.params, old, step.op.description, self)
        dlg.changed.connect(lambda p: (self.pipeline.set_params(index, p),
                                       self._refresh_pipeline_list(select=index)))
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.pipeline.set_params(index, dlg.values())
            self._refresh_pipeline_list(select=index)
            return True
        if new:
            self.pipeline.remove(index)
        else:
            self.pipeline.set_params(index, old)
        self._refresh_pipeline_list()
        return False

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

    def reset_pipeline_state(self) -> None:
        self.pipeline.reset_state()
        self.statusBar().showMessage("Pipeline state reset (backgrounds / averages re-learned)", 3000)

    def _refresh_pipeline_list(self, select: int | None = None) -> None:
        self.pipeline_list.blockSignals(True)
        self.pipeline_list.clear()
        for step in self.pipeline.steps:
            tag = "Filter" if step.op.category == "filter" else "Transform"
            item = QtWidgets.QListWidgetItem(f"[{tag}] {step.label}")
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if step.enabled else QtCore.Qt.CheckState.Unchecked)
            self._decorate_item(item, step)
            self.pipeline_list.addItem(item)
        if select is not None and 0 <= select < self.pipeline_list.count():
            self.pipeline_list.setCurrentRow(select)
        self.pipeline_list.blockSignals(False)
        self.reset_manual_steps(quiet=True)

    def _decorate_item(self, item: QtWidgets.QListWidgetItem, step) -> None:
        if step.error:
            item.setForeground(QtGui.QBrush(QtGui.QColor("#e53935")))
            item.setToolTip(f"Bypassed - {step.error}")
        else:
            item.setForeground(self.pipeline_list.palette().brush(QtGui.QPalette.ColorRole.Text))
            item.setToolTip(step.op.description or step.op.name)

    def _refresh_pipeline_errors(self) -> None:
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

    # ======================================================================
    # Phase (2.c)
    # ======================================================================
    def _shifter_key(self) -> str:
        return list(SHIFTERS)[self.shifter_combo.currentIndex()]

    def _phase_widgets_changed(self, *_) -> None:
        if self._sync:
            return
        old = self.phase_cfg
        cfg = PhaseConfig(
            method=list(PHASE_METHODS)[self.method_combo.currentIndex()],
            n_steps=self.nsteps_spin.value(),
            unwrap=list(UNWRAP_METHODS)[self.unwrap_combo.currentIndex()],
            detrend=list(DETREND_MODES)[self.detrend_combo.currentIndex()],
            modulation_threshold=self.mod_spin.value(),
            output=list(OUTPUT_MODES)[self.output_combo.currentIndex()],
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
        )
        self.phase_cfg = cfg
        if self.thread is not None:
            self.thread.config = cfg
        # keep the pop-up menus in sync with the panel
        self.method_actions[cfg.method].setChecked(True)
        self.unwrap_actions[cfg.unwrap].setChecked(True)
        self.detrend_actions[cfg.detrend].setChecked(True)
        self.output_actions[cfg.output].setChecked(True)
        self.nsteps_spin.setEnabled(cfg.method == "nstep")
        if cfg.method != old.method or cfg.n_steps != old.n_steps:
            self.reset_manual_steps(quiet=True)
            if cfg.method == "off":
                self._set_continuous(False)
            elif self.channel_combo.currentText() in ("Processed image", "Raw camera"):
                self.channel_combo.setCurrentText("Height / physical")
            if not cfg.is_stepping:
                self._set_continuous(False)
        self._update_phase_buttons()
        post_only = cfg.method == old.method and cfg.n_steps == old.n_steps
        if post_only and self.last_result is not None and (cfg.is_stepping or self.thread is None):
            self.reanalyse()

    def _update_phase_buttons(self) -> None:
        cfg = self.phase_cfg
        auto = self.shifter is not None and self.shifter.automatic
        manual = isinstance(self.shifter, ManualShifter)
        for w in (self.acquire_btn, self.act_acquire):
            w.setEnabled(cfg.is_stepping and auto)
        for w in (self.continuous_btn, self.act_continuous):
            w.setEnabled(cfg.is_stepping and auto)
        for w in (self.capture_btn, self.act_capture):
            w.setEnabled(cfg.is_stepping and manual)
        n = len(cfg.shifts()) if cfg.is_stepping else 0
        self.step_progress.setRange(0, max(n, 1))
        if cfg.is_stepping and manual:
            self._manual_hint()
        elif cfg.is_stepping and not auto:
            self.step_label.setText("Select a working phase shifter to record sequences.")
        elif cfg.method == "ftp":
            self.step_label.setText("Live phase from every frame - needs tilt (carrier) fringes.")
        elif cfg.is_stepping:
            self.step_label.setText(f"{n} frames per sequence, shifts "
                                    + ", ".join(f"{np.degrees(s):.0f}°" for s in cfg.shifts()))
        else:
            self.step_label.setText("")

    def _shifter_changed(self, *_) -> None:
        key = self._shifter_key()
        self.shifter_actions[key].setChecked(True)
        if self._sync:
            return
        if self.shifter is not None:
            try:
                self.shifter.disconnect()
            except Exception:
                pass
        shifter: PhaseShifter | None = None
        if key == "sim":
            if isinstance(self.source, SimulatedInterferometer):
                shifter = SimulatedShifter(self.source)
            elif self.source is not None:
                self.statusBar().showMessage("The simulated shifter only works with the simulated camera", 5000)
        elif key == "manual":
            shifter = ManualShifter()
        else:
            s = dict(self.serial_settings)
            s["command"] = codecs.decode(str(s["command"]), "unicode_escape")
            shifter = SerialPiezoShifter(**s)
            try:
                shifter.connect()
            except Exception as exc:
                self._error(f"Piezo controller: {exc}", dialog=True)
                shifter = None
        self.shifter = shifter
        if self.thread is not None:
            self.thread.shifter = shifter
        self.reset_manual_steps(quiet=True)
        self._update_phase_buttons()

    def edit_serial_settings(self) -> None:
        dlg = ParamDialog("Piezo controller (serial)", SERIAL_PARAMS, self.serial_settings,
                          "ASCII serial piezo controller. Defaults match the Thorlabs MDT693B/MDT694B; "
                          "check the command syntax in your controller's manual.", self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.serial_settings = dlg.values()
            if self._shifter_key() == "serial":
                self._shifter_changed()

    def lock_carrier(self) -> None:
        r = self.last_result
        if r is None or "carrier_px" not in r.info:
            self.statusBar().showMessage("No Fourier-method result yet", 3000)
            return
        kx, ky = r.info["carrier_px"]
        self._sync = True
        self.carrier_x_spin.setValue(kx)
        self.carrier_y_spin.setValue(ky)
        self._sync = False
        self._phase_widgets_changed()

    def acquire_sequence(self) -> None:
        if not self.phase_cfg.is_stepping:
            self.method_combo.setCurrentIndex(list(PHASE_METHODS).index("4step"))
        if self.shifter is None or not self.shifter.automatic:
            self._error("Recording a sequence needs an automatic phase shifter. With the manual "
                        "shifter use 'Capture step' (Space) for every step.", dialog=True)
            return
        if self.thread is None:
            self.start_live()
        if self.thread is not None:
            self.step_progress.setValue(0)
            self.thread.request_sequence()

    def toggle_continuous(self, on: bool) -> None:
        if on and not (self.phase_cfg.is_stepping and self.shifter is not None and self.shifter.automatic):
            self._set_continuous(False)
            return
        self._set_continuous(on)
        if on and self.thread is None:
            self.start_live()

    def _set_continuous(self, on: bool) -> None:
        for w in (self.continuous_btn, self.act_continuous):
            w.blockSignals(True)
            w.setChecked(on)
            w.blockSignals(False)
        if self.thread is not None:
            self.thread.continuous_stepping = on
            if not on:
                self.thread.cancel_sequence()

    def _sequence_progress(self, k: int, n: int) -> None:
        self.step_progress.setRange(0, n)
        self.step_progress.setValue(k)

    def _sequence_finished(self, ok: bool) -> None:
        if not ok:
            self.statusBar().showMessage("Phase-stepping sequence aborted", 4000)
            self._set_continuous(False)

    def _manual_hint(self) -> None:
        cfg = self.phase_cfg
        shifts = cfg.shifts()
        k = len(self.manual_frames)
        if k >= len(shifts):
            return
        deg = np.degrees(shifts[k])
        mirror = shifts[k] / (4 * np.pi) * cfg.wavelength_nm
        self.step_label.setText(
            f"Manual: set the reference to {deg:.0f}° (mirror +{mirror:.1f} nm in reflection) "
            f"and press 'Capture step' (Space) - step {k + 1} of {len(shifts)}.")
        self.step_progress.setRange(0, len(shifts))
        self.step_progress.setValue(k)

    def capture_manual_step(self) -> None:
        cfg = self.phase_cfg
        if not cfg.is_stepping or not isinstance(self.shifter, ManualShifter):
            return
        if self.last_packet is None:
            self._error("No camera frame to capture yet")
            return
        frame = self.last_packet.processed.copy()
        if self.manual_frames and frame.shape != self.manual_frames[0].shape:
            self.manual_frames.clear()
        self.manual_frames.append(frame)
        n = len(cfg.shifts())
        if len(self.manual_frames) < n:
            self._manual_hint()
            return
        try:
            res = analyse_stack(self.manual_frames, cfg)
        except Exception as exc:
            self._error(f"Phase stepping failed: {exc}", dialog=True)
            res = None
        self.manual_frames = []
        self.step_progress.setValue(n)
        if res is not None:
            self._show_result(res)
            self.step_label.setText("Sequence complete - capture again for a new measurement.")

    def reset_manual_steps(self, quiet: bool = False) -> None:
        had = bool(self.manual_frames)
        self.manual_frames = []
        if hasattr(self, "step_progress"):
            self.step_progress.setValue(0)
            if isinstance(self.shifter, ManualShifter) and self.phase_cfg.is_stepping:
                self._manual_hint()
        if had and quiet:
            self.statusBar().showMessage("Manual phase steps discarded (settings changed)", 3000)

    def reanalyse(self) -> None:
        r = self.last_result
        if r is None:
            return
        try:
            info = {k: v for k, v in r.info.items() if k in ("method", "carrier_px", "fringe_period_px",
                                                              "estimated_step_deg")}
            self._show_result(analyse(r.wrapped, r.modulation, self.phase_cfg, info))
        except Exception as exc:
            self._error(f"Re-analysis failed: {exc}")

    # ======================================================================
    # Files
    # ======================================================================
    def open_images(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open image(s)", "", IMAGE_FILTER)
        if paths:
            self.load_images(sorted(paths))

    def load_images(self, paths: list[str]) -> None:
        try:
            images = [load_image(p) for p in paths]
        except Exception as exc:
            self._error(str(exc), dialog=True)
            return
        if len(images) == 1:
            self.disconnect_camera()
            self._set_source(StaticImageSource(images[0], os.path.basename(paths[0])))
            self.start_live()
            return
        # several images: a phase-stepping sequence
        n = len(images)
        cfg = self.phase_cfg
        if not cfg.is_stepping or len(cfg.shifts()) != n:
            key = {3: "3step", 4: "4step", 5: "5step"}.get(n, "nstep")
            self._sync = True
            self.nsteps_spin.setValue(min(max(n, 3), 16))
            self._sync = False
            self.method_combo.setCurrentIndex(list(PHASE_METHODS).index(key))
            cfg = self.phase_cfg
        try:
            frames = [self.pipeline.apply(img, include_temporal=False) for img in images]
            res = analyse_stack(frames, cfg)
        except Exception as exc:
            self._error(f"Phase stepping failed: {exc}", dialog=True)
            return
        self.stop_live()
        self.live_panel.set_image(images[0])
        self.last_packet = FramePacket(images[0], frames[0], 0, 0.0, 0.0)
        self.live_info.setText(f"Loaded {n}-frame sequence: {os.path.basename(paths[0])} …")
        self.channel_combo.setCurrentText("Height / physical")
        self._show_result(res)

    def _save_dialog(self, title: str, default: str, filters: str) -> str:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, default, filters)
        return path

    def save_raw(self) -> None:
        if self.last_packet is None:
            return
        frame = self.last_packet.raw.copy()
        path = self._save_dialog("Save raw frame", "frame.tif",
                                 "TIFF (*.tif);;PNG (*.png);;NumPy (*.npy);;CSV (*.csv)")
        if path:
            self._try(lambda: save_array(path, frame), f"Saved {path}")

    def save_processed(self) -> None:
        if self.last_packet is None:
            return
        frame = self.last_packet.processed.copy()
        path = self._save_dialog("Save processed image", "processed.tif",
                                 "32-bit TIFF (*.tif);;PNG 8-bit (*.png);;NumPy (*.npy);;CSV (*.csv)")
        if path:
            self._try(lambda: save_array(path, frame), f"Saved {path}")

    def save_result(self) -> None:
        r = self.last_result
        if r is None:
            self._error("No phase result to save yet", dialog=True)
            return
        path = self._save_dialog("Save phase result", "phase_result.npz",
                                 "All maps + metadata (*.npz);;Height map TIFF (*.tif);;"
                                 "Height map CSV (*.csv);;Height map PNG (*.png)")
        if not path:
            return
        if path.lower().endswith(".npz"):
            meta = {"quantity": r.quantity, "unit": r.unit, "config": asdict(self.phase_cfg),
                    "info": {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.info.items()}}
            self._try(lambda: np.savez_compressed(
                path, wrapped=r.wrapped, unwrapped=r.unwrapped, physical=r.physical,
                modulation=r.modulation, mask=r.mask, metadata=json.dumps(meta, default=str)),
                f"Saved {path}")
        else:
            self._try(lambda: save_array(path, r.physical), f"Saved {path}")

    def save_view(self, panel) -> None:
        path = self._save_dialog("Save view", "view.png", "PNG (*.png)")
        if path:
            self._try(lambda: panel.grab_image().save(path), f"Saved {path}")

    def _try(self, fn, ok_message: str) -> None:
        try:
            fn()
            self.statusBar().showMessage(ok_message, 4000)
        except Exception as exc:
            self._error(str(exc), dialog=True)

    # ======================================================================
    # Misc
    # ======================================================================
    def _error(self, message: str, dialog: bool = False) -> None:
        self.statusBar().showMessage(message, 8000)
        if dialog:
            QtWidgets.QMessageBox.warning(self, "PhaseScope", message)

    def show_help(self) -> None:
        QtWidgets.QMessageBox.information(self, "Quick guide", (
            "1. Camera panel: choose the source and press 'Connect & start'. The live image "
            "appears on the left. Tick the ROI box to analyse only part of the frame.\n\n"
            "2. Right-click the image (or use the toolbar / menu bar) for the pop-up menus:\n"
            "   • Filters - smoothing, edges, contrast, Fourier, background, averaging\n"
            "   • Transformations - flip, rotate, crop, bin, scale, spectrum\n"
            "   • Phase - stepping algorithm, unwrapping, background removal, output\n"
            "   Steps are listed in the 'Processing pipeline' panel and run on every frame.\n\n"
            "3. Phase:\n"
            "   • Fourier transform: live phase from every frame (tilt the reference so you see "
            "straight carrier fringes).\n"
            "   • 3/4/5-step, Carré, N-step: press Acquire (piezo / simulator) or capture each "
            "step with Space (manual).\n"
            "   • Output converts phase to height (reflection), thickness (transmission, needs "
            "the refractive indices) or refractive-index change (needs the thickness).\n\n"
            "4. Analysis: choose what to display, switch between the 2D and 3D tabs. In 2D, "
            "'Line profile' shows a cross-section; in 3D drag to rotate, wheel to zoom."))

    def closeEvent(self, ev) -> None:
        self.timer.stop()
        self.disconnect_camera()
        for th in self._stale_threads:
            th.wait(5000)
        if self.shifter is not None:
            try:
                self.shifter.disconnect()
            except Exception:
                pass
        super().closeEvent(ev)
