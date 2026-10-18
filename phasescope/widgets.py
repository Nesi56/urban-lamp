"""Reusable Qt widgets: status pill, 2-D image panel, 3-D surface panel, parameter dialog."""

from __future__ import annotations

import csv
import math
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, isQObjectAlive
from scipy import ndimage

from .processing import Param

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)


# --------------------------------------------------------------------------
# Colour maps
# --------------------------------------------------------------------------

COLORMAPS = ["gray", "viridis", "inferno", "magma", "plasma", "cividis", "turbo",
             "CET-C2 (cyclic)", "CET-D1 (diverging)", "CET-R4 (rainbow)"]


def get_colormap(name: str) -> pg.ColorMap:
    if name == "gray":
        return pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])
    try:
        return pg.colormap.get(name.split(" ")[0])
    except Exception:
        return pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])


def robust_levels(data: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> tuple[float, float]:
    sample = data[:: max(1, data.shape[0] // 256), :: max(1, data.shape[1] // 256)]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# Status pill and floating overlays
# --------------------------------------------------------------------------

STATUS_COLORS = {"ok": "#2e7d32", "warn": "#b26a00", "error": "#c62828", "info": "#1565c0",
                 "rec": "#6a1b9a", "muted": "#616161"}


def pill_stylesheet(kind: str) -> str:
    """Style sheet of a rounded status badge: white bold text on ``STATUS_COLORS[kind]``."""
    color = STATUS_COLORS.get(kind, STATUS_COLORS["muted"])
    return (f"color: white; font-weight: bold; background-color: {color}; "
            "border-radius: 9px; padding: 2px 10px;")


class StatusPill(QtWidgets.QLabel):
    """Small coloured badge ("Live", "Saturated", "Recording" ...)."""

    def __init__(self, text: str = "", kind: str = "muted", parent=None) -> None:
        super().__init__(parent)
        self._kind = ""
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.set_state(kind, text)

    @property
    def kind(self) -> str:
        return self._kind

    def set_state(self, kind: str, text: str, tooltip: str = "") -> None:
        kind = kind if kind in STATUS_COLORS else "muted"
        if kind != self._kind:  # re-polishing a style sheet is costly: only on change
            self._kind = kind
            self.setStyleSheet(pill_stylesheet(kind))
        if text != self.text():
            self.setText(text)
        if tooltip != self.toolTip():
            self.setToolTip(tooltip)


_MUTED_STYLE = f"color: {STATUS_COLORS['muted']};"
_MESSAGE_STYLE = ("QLabel { color: white; background-color: rgba(0, 0, 0, 170); border-radius: 10px; "
                  "border: 1px solid rgba(255, 255, 255, 40); padding: 12px 18px; font-size: 13pt; }")


def _banner_style(kind: str) -> str:
    r, g, b, _ = QtGui.QColor(STATUS_COLORS.get(kind, STATUS_COLORS["info"])).getRgb()
    return (f"QLabel {{ color: white; font-weight: bold; background-color: rgba({r}, {g}, {b}, 230); "
            "padding: 6px 12px; }")


def _make_overlay(parent: QtWidgets.QWidget, style: str, centered: bool) -> QtWidgets.QLabel:
    """Hidden, word-wrapped label floating over ``parent`` that lets mouse events through."""
    label = QtWidgets.QLabel(parent)
    label.setWordWrap(True)
    label.setStyleSheet(style)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter if centered
                       else QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
    label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    label.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
    label.hide()
    return label


def _place_message(label: QtWidgets.QLabel, area: QtCore.QRect) -> None:
    """Centre ``label`` in ``area`` with a comfortable reading width."""
    avail = max(60, min(area.width() - 24, 620))
    label.setWordWrap(False)  # natural width of the longest line ...
    natural = label.sizeHint().width() + 2
    label.setWordWrap(True)   # ... then wrap only what does not fit
    width = min(avail, natural)
    height = min(label.heightForWidth(width), max(30, area.height() - 8))
    label.setGeometry(area.left() + (area.width() - width) // 2,
                      area.top() + (area.height() - height) // 2, width, height)
    label.raise_()


def _place_banner(label: QtWidgets.QLabel, area: QtCore.QRect) -> None:
    """Stretch ``label`` across the top edge of ``area``."""
    width = max(20, area.width())
    label.setGeometry(area.left(), area.top(), width, label.heightForWidth(width))
    label.raise_()


def _set_overlay_text(label: QtWidgets.QLabel, text: str | None) -> bool:
    """Show ``text`` in ``label`` (None/empty hides it); True when it needs placing."""
    if not text:
        label.hide()
        return False
    text = str(text)
    if text == label.text() and not label.isHidden():
        return False  # unchanged: cheap enough to call on every refresh
    label.setText(text)
    label.show()
    return True


def _unit_of(label: str) -> str:
    """Unit in the last parentheses of an axis label: "Height (nm)" -> "nm"."""
    i = label.rfind("(")
    j = label.find(")", i + 1)
    return label[i + 1:j].strip() if i >= 0 and j > i else ""


def _decimals(ref: float) -> int:
    """Number of decimals that shows a quantity of magnitude ``ref`` with ~4 significant digits."""
    ref = abs(float(ref))
    if not math.isfinite(ref) or ref == 0:
        return 2
    if ref >= 1000:
        return 0
    if ref >= 10:
        return 1
    if ref >= 1:
        return 2
    return min(9, int(math.ceil(-math.log10(ref))) + 2)


def _fmt(value: float, decimals: int) -> str:
    if not math.isfinite(value):
        return "–"
    text = f"{value:.{decimals}f}"
    return text.lstrip("-") if float(text) == 0 else text  # no "-0.0"


# --------------------------------------------------------------------------
# 2-D image panel
# --------------------------------------------------------------------------

class _MenuImageItem(pg.ImageItem):
    """ImageItem that pops up the application's context menu on right click."""

    sigDoubleClicked = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.popup_menu: QtWidgets.QMenu | None = None

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton and self.popup_menu is not None:
            ev.accept()
            self.popup_menu.popup(ev.screenPos().toPoint())
            return
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and ev.double():
            ev.accept()
            self.sigDoubleClicked.emit()
            return
        super().mouseClickEvent(ev)


_PROFILE_A = (66, 165, 245)   # reference band (blue)
_PROFILE_B = (255, 152, 0)    # feature band (orange)


class ImagePanel(QtWidgets.QWidget):
    """Zoomable image display with colour bar, optional ROI and line-profile tool.

    * mouse wheel / drag: zoom & pan, right click anywhere in the view: pop-up menu
    * ``hovered(x, y, value)`` is emitted while the mouse moves over the image
    * ``double_clicked()`` is emitted when the image is double-clicked
    """

    hovered = QtCore.Signal(int, int, float)
    roi_changed = QtCore.Signal(object)  # (x, y, w, h) or None
    auto_levels_changed = QtCore.Signal(bool)
    double_clicked = QtCore.Signal()

    def __init__(self, title: str = "", histogram: bool = True, profile: bool = False,
                 roi: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._data: np.ndarray | None = None
        self._auto_levels = True
        self._programmatic = False
        self._cmap_name = "gray"
        self._shape = None
        self._fixed_levels: tuple[float, float] | None = None
        self._levels_before_fixed = None
        self._mask: np.ndarray | None = None
        self._units_label = ""
        self._um_per_px = 0.0
        self._profile: tuple[np.ndarray, np.ndarray] | None = None
        self._profile_length = 0.0
        self._profile_placed = False
        self._region_fracs = [(0.10, 0.25), (0.75, 0.90)]
        self._syncing_regions = False

        self.view_widget = pg.GraphicsLayoutWidget()
        self.view_widget.setBackground("k")
        if title:
            self.title = self.view_widget.addLabel(title, row=0, col=0)
        self.vb = self.view_widget.addViewBox(row=1, col=0, lockAspect=True, invertY=True)
        self.vb.setMenuEnabled(True)
        self.image = _MenuImageItem()
        self.image.sigDoubleClicked.connect(self.double_clicked)
        self.vb.addItem(self.image)
        # boolean mask (e.g. saturated pixels) drawn in red above the image
        self.mask_item = pg.ImageItem()
        self.mask_item.setZValue(5)
        self.mask_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.mask_item.hide()
        self.vb.addItem(self.mask_item)

        self.hist = None
        self._hist_unit_item = None
        if histogram:
            # compact horizontal colour bar under the image
            self.hist = pg.HistogramLUTItem(self.image, orientation="horizontal")
            self.hist.gradient.showTicks(False)
            self.hist.gradient.setMaxDim(self.hist.gradient.rectSize + 3)
            self.hist.vb.setMinimumHeight(20)
            self.hist.axis.showLabel(False)
            self.hist.sigLevelChangeFinished.connect(self._user_levels)
            self.view_widget.addItem(self.hist, row=2, col=0)
            self.view_widget.ci.layout.setRowMaximumHeight(2, 70)
            self._hist_unit_item = QtWidgets.QGraphicsSimpleTextItem("", self.hist.vb)
            font = self._hist_unit_item.font()
            font.setPointSizeF(max(6.0, font.pointSizeF() - 1))
            self._hist_unit_item.setFont(font)
            self._hist_unit_item.setBrush(pg.mkBrush(225, 225, 225))
            self._hist_unit_item.setPos(4, 1)
            self._hist_unit_item.setZValue(2000)
        self.view_widget.scene().sigMouseMoved.connect(self._mouse_moved)

        # floating message (centre) and banner (top) over the image area
        self._viewport = self.view_widget.viewport()
        self.message_label = _make_overlay(self._viewport, _MESSAGE_STYLE, centered=True)
        self.banner_label = _make_overlay(self._viewport, _banner_style("info"), centered=False)
        self._viewport.installEventFilter(self)
        self.vb.sigResized.connect(self._place_overlays)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # optional analysis ROI (rectangle)
        self.roi_item = None
        if roi:
            self.roi_item = pg.RectROI([20, 20], [200, 150], pen=pg.mkPen("y", width=2),
                                       removable=False, rotatable=False)
            self.roi_item.addScaleHandle([0, 0], [1, 1])
            self.roi_item.setZValue(10)
            self.roi_item.hide()
            self.roi_item.sigRegionChangeFinished.connect(self._roi_moved)
            self.vb.addItem(self.roi_item)

        # optional line profile
        self.profile_line = None
        self.profile_plot = None
        self.profile_box = None
        self.profile_readout = None
        self.profile_hover_label = None
        self.profile_width_spin = None
        self.region_a = self.region_b = None
        if profile:
            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            splitter.addWidget(self.view_widget)
            splitter.addWidget(self._build_profile_box())
            splitter.setSizes([700, 260])
            layout.addWidget(splitter)
            self.profile_line = pg.LineSegmentROI([[40, 40], [240, 160]], pen=pg.mkPen("m", width=2))
            self.profile_line.setZValue(10)
            self.profile_line.hide()
            self.profile_line.sigRegionChanged.connect(self._update_profile)
            self.vb.addItem(self.profile_line)
            self._update_readout()
        else:
            layout.addWidget(self.view_widget)
        self.set_colormap("gray")

    def _build_profile_box(self) -> QtWidgets.QWidget:
        """Profile plot with its compact control row and step-height readout."""
        self.profile_box = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(self.profile_box)
        box.setContentsMargins(4, 2, 4, 2)
        box.setSpacing(2)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QtWidgets.QLabel("Width"))
        self.profile_width_spin = QtWidgets.QSpinBox()
        self.profile_width_spin.setRange(1, 25)
        self.profile_width_spin.setValue(1)
        self.profile_width_spin.setSuffix(" px")
        self.profile_width_spin.setToolTip("Average this many parallel lines (reduces noise)")
        self.profile_width_spin.valueChanged.connect(self._update_profile)
        row.addWidget(self.profile_width_spin)
        self.profile_copy_btn = QtWidgets.QPushButton("Copy")
        self.profile_copy_btn.setToolTip("Copy the profile as two tab-separated columns (paste into Excel/Origin)")
        self.profile_copy_btn.clicked.connect(self.copy_profile)
        row.addWidget(self.profile_copy_btn)
        self.profile_save_btn = QtWidgets.QPushButton("Save CSV…")
        self.profile_save_btn.setToolTip("Save the profile and the step-height result as a CSV file")
        self.profile_save_btn.clicked.connect(self._save_profile_dialog)
        row.addWidget(self.profile_save_btn)
        row.addStretch(1)
        self.profile_hover_label = QtWidgets.QLabel("")
        self.profile_hover_label.setStyleSheet(_MUTED_STYLE)
        row.addWidget(self.profile_hover_label)
        box.addLayout(row)

        self.profile_plot = pg.PlotWidget()
        self.profile_plot.setLabel("bottom", "distance along line", units="px")
        self.profile_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.profile_plot.showGrid(x=True, y=True, alpha=0.3)
        self.profile_plot.setMinimumHeight(90)
        self.profile_curve = self.profile_plot.plot(pen=pg.mkPen("c", width=2))
        self.region_a = self._make_region(_PROFILE_A, "A (reference)")
        self.region_b = self._make_region(_PROFILE_B, "B (feature)")
        self.profile_cursor = pg.InfiniteLine(angle=90, movable=False,
                                              pen=pg.mkPen((255, 255, 255, 140), width=1,
                                                           style=QtCore.Qt.PenStyle.DashLine))
        self.profile_cursor.hide()
        self.profile_plot.addItem(self.profile_cursor, ignoreBounds=True)
        self.profile_plot.scene().sigMouseMoved.connect(self._profile_hover)
        self.profile_plot.installEventFilter(self)
        box.addWidget(self.profile_plot, 1)

        self.profile_readout = QtWidgets.QLabel("")
        self.profile_readout.setWordWrap(True)
        self.profile_readout.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.profile_readout.setToolTip("Step = mean of band B (orange) − mean of band A (blue); ± combines the "
                                        "standard deviations inside both bands. Drag the bands to move them.")
        box.addWidget(self.profile_readout)
        self.profile_box.hide()
        return self.profile_box

    def _make_region(self, rgb: tuple[int, int, int], text: str) -> pg.LinearRegionItem:
        region = pg.LinearRegionItem(values=(0, 1), orientation="vertical", brush=pg.mkBrush(*rgb, 45),
                                     hoverBrush=pg.mkBrush(*rgb, 80), pen=pg.mkPen(*rgb, width=1.5),
                                     swapMode="block")
        region.setZValue(10)
        self.profile_plot.addItem(region, ignoreBounds=True)
        label = pg.InfLineLabel(region.lines[0], text, position=0.97, anchors=[(0, 0), (0, 0)], color=rgb)
        font = label.textItem.font()
        font.setPointSizeF(max(6.0, font.pointSizeF() - 1))
        label.setFont(font)
        region.label = label
        region.sigRegionChanged.connect(self._region_moved)
        return region

    # -- configuration ----------------------------------------------------
    def set_popup_menu(self, menu: QtWidgets.QMenu) -> None:
        """Right click on the image, its margins, the colour bar, the boxes and the profile line
        opens ``menu`` (in place of pyqtgraph's View All / Axis / Export and gradient menus)."""
        self.image.popup_menu = menu
        self._popup = menu
        self.vb.raiseContextMenu = self._raise_popup
        if self.hist is not None:
            self.hist.vb.raiseContextMenu = self._raise_popup
            self.hist.gradient.showMenu = self._raise_popup  # its presets would bypass the Colours box
        for item in (self.roi_item, self.profile_line):
            if item is not None:
                self.add_popup_item(item)

    def add_popup_item(self, roi: pg.ROI) -> None:
        """Let a box or line drawn on the image open the panel's menu on a right click."""
        roi.contextMenuEnabled = lambda: getattr(self, "_popup", None) is not None
        roi.raiseContextMenu = self._raise_popup

    def _raise_popup(self, ev) -> None:
        menu = getattr(self, "_popup", None)
        if menu is not None:
            pos = ev.screenPos()
            menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))

    def set_colormap(self, name: str) -> None:
        self._cmap_name = name
        cmap = get_colormap(name)
        if self.hist is not None:
            self.hist.gradient.setColorMap(cmap)
            self.hist.gradient.showTicks(False)
        else:
            self.image.setLookupTable(cmap.getLookupTable(nPts=256))

    @property
    def colormap_name(self) -> str:
        return self._cmap_name

    def set_title(self, text: str) -> None:
        if hasattr(self, "title"):
            self.title.setText(text)

    def set_units(self, label: str) -> None:
        self._units_label = label
        if self.hist is not None:
            self.hist.axis.setLabel(label)
            self.hist.axis.showLabel(False)  # no room in the compact bar: shown in its corner
            self._hist_unit_item.setText(label)
        if self.profile_plot is not None:
            self.profile_plot.setLabel("left", label)
            self._update_readout()

    def set_histogram_visible(self, visible: bool) -> None:
        if self.hist is None or visible == self.histogram_visible():
            return
        layout = self.view_widget.ci.layout
        if visible:  # a hidden item still takes room in the grid: take it out of the layout instead
            layout.addItem(self.hist, 2, 0)
            self.hist.show()
        else:
            self.hist.hide()
            layout.removeItem(self.hist)

    def histogram_visible(self) -> bool:
        return self.hist is not None and self.hist.isVisible()

    def set_pixel_size(self, um_per_px: float) -> None:
        """Calibrate profile distances: 0 = pixels, >0 = micrometres per pixel."""
        self._um_per_px = max(0.0, float(um_per_px or 0.0))
        if self.profile_plot is not None:
            self.profile_plot.setLabel("bottom", "distance along line", units=self._distance_unit())
            self._update_profile()

    # -- overlays ---------------------------------------------------------
    def set_message(self, text: str | None) -> None:
        """Centred multi-line hint over the image (None hides it)."""
        if _set_overlay_text(self.message_label, text):
            self._place_overlays()

    def set_banner(self, text: str | None, kind: str = "info") -> None:
        """Coloured strip across the top of the image (None hides it)."""
        style = _banner_style(kind)
        if style != self.banner_label.styleSheet():
            self.banner_label.setStyleSheet(style)
        if _set_overlay_text(self.banner_label, text):
            self._place_overlays()

    def set_dimmed(self, dimmed: bool) -> None:
        self.image.setOpacity(0.4 if dimmed else 1.0)

    def set_mask_overlay(self, mask: np.ndarray | None) -> None:
        """Paint True pixels of ``mask`` red over the image; the mask may be subsampled."""
        if mask is None:
            self._mask = None
            self.mask_item.hide()
            return
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2 or mask.size == 0:
            raise ValueError("mask must be a non-empty 2-D array")
        self._mask = mask
        if not mask.any():  # the usual case for a per-frame saturation mask: nothing to paint
            self.mask_item.hide()
            return
        rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
        rgba[mask] = (255, 40, 40, 190)
        self.mask_item.setImage(rgba, autoLevels=False, levels=None)
        self._place_mask()
        self.mask_item.show()

    def _place_mask(self) -> None:
        if self.mask_item.image is None:  # nothing drawn yet
            return
        h, w = (self._data.shape[:2] if self._data is not None else self._mask.shape)
        self.mask_item.setRect(QtCore.QRectF(0, 0, w, h))

    def _image_area(self) -> QtCore.QRect:
        full = self._viewport.rect()
        area = self.view_widget.mapFromScene(self.vb.sceneBoundingRect()).boundingRect().intersected(full)
        return area if area.width() >= 40 and area.height() >= 30 else full

    def _place_overlays(self, *_) -> None:
        if self.message_label.isHidden() and self.banner_label.isHidden():
            return
        area = self._image_area()
        if not self.message_label.isHidden():
            _place_message(self.message_label, area)
        if not self.banner_label.isHidden():
            _place_banner(self.banner_label, area)

    def eventFilter(self, obj, ev) -> bool:
        etype = ev.type()
        if etype == QtCore.QEvent.Type.Resize and obj is getattr(self, "_viewport", None):
            self._place_overlays()
        elif etype == QtCore.QEvent.Type.Leave and obj is getattr(self, "profile_plot", None):
            self._hide_cursor()
        return super().eventFilter(obj, ev)

    # -- data -------------------------------------------------------------
    def set_image(self, data: np.ndarray | None) -> None:
        if data is None:
            self.image.clear()
            self._data = None
            self._update_profile()
            return
        data = np.asarray(data)
        first = self._shape != data.shape
        self._data = data
        self._programmatic = True  # level changes below are not user edits
        try:
            if self._fixed_levels is not None:
                self.image.setImage(data, autoLevels=False, levels=self._fixed_levels)
                self._pin_histogram()
            elif self._auto_levels or first:
                levels = robust_levels(data)
                self.image.setImage(data, autoLevels=False, levels=levels)
                if self.hist is not None:
                    self.hist.setLevels(*levels)
            else:
                self.image.setImage(data, autoLevels=False)
        finally:
            self._programmatic = False
        if first:
            self._shape = data.shape
            self.vb.autoRange(padding=0.01)
            if self._mask is not None:
                self._place_mask()
        if (self.profile_line is not None and self.profile_line.isVisible()
                and not (self._profile_placed and self._line_inside())):
            self._place_default_line()
        self._update_profile()

    def data(self) -> np.ndarray | None:
        return self._data

    def reset_view(self) -> None:
        self.vb.autoRange(padding=0.01)

    @property
    def auto_levels(self) -> bool:
        return self._auto_levels

    @auto_levels.setter
    def auto_levels(self, value: bool) -> None:
        value = bool(value)
        if value != self._auto_levels:
            self._auto_levels = value
            self.auto_levels_changed.emit(value)
        if value and self._data is not None:
            self.set_image(self._data)

    def _user_levels(self) -> None:
        # the user dragged the colour bar: keep their levels from now on
        if not self._programmatic:
            self.auto_levels = False

    @property
    def fixed_levels(self) -> tuple[float, float] | None:
        return self._fixed_levels

    def set_fixed_levels(self, levels: tuple[float, float] | None) -> None:
        """Pin the display range (e.g. to the full sensor range); None restores normal levels."""
        if levels is None:
            if self._fixed_levels is None:
                return
            self._fixed_levels = None
            if self.hist is not None:
                self.hist.region.setMovable(True)
                self.hist.vb.setMouseEnabled(x=True, y=False)
                self.hist.autoHistogramRange()
                self.hist.setToolTip("")
            previous, self._levels_before_fixed = self._levels_before_fixed, None
            if not self._auto_levels and previous is not None:
                self._apply_levels(previous)
            if self._data is not None:
                self.set_image(self._data)
            return
        lo, hi = float(levels[0]), float(levels[1])
        if hi <= lo:
            hi = lo + 1.0
        if self._fixed_levels == (lo, hi):
            return
        if self._fixed_levels is None:
            current = self.image.getLevels()
            self._levels_before_fixed = None if current is None else tuple(np.ravel(current)[:2])
        self._fixed_levels = (lo, hi)
        self._apply_levels(self._fixed_levels)
        if self.hist is not None:
            self.hist.region.setMovable(False)
            self.hist.vb.setMouseEnabled(x=False, y=False)
            self.hist.setToolTip(f"Display range fixed to {lo:g} … {hi:g}")
        self._pin_histogram()

    def _apply_levels(self, levels) -> None:
        self._programmatic = True
        try:
            self.image.setLevels(levels)
            if self.hist is not None:
                self.hist.setLevels(*levels)
        finally:
            self._programmatic = False

    def _pin_histogram(self) -> None:
        if self.hist is None or self._fixed_levels is None:
            return
        self._programmatic = True
        try:
            self.hist.setLevels(*self._fixed_levels)
            self.hist.setHistogramRange(*self._fixed_levels, padding=0.02)
        finally:
            self._programmatic = False

    # -- ROI ----------------------------------------------------------------
    def show_roi(self, visible: bool) -> None:
        if self.roi_item is None:
            return
        self.roi_item.setVisible(visible)
        if visible and self._data is not None:
            h, w = self._data.shape[:2]
            pos, size = self.roi_item.pos(), self.roi_item.size()
            if pos.x() + size.x() > w or pos.y() + size.y() > h:
                self.roi_item.setPos([w * 0.25, h * 0.25], finish=False)
                self.roi_item.setSize([w * 0.5, h * 0.5], finish=False)
        self._roi_moved()

    def roi_rect(self) -> tuple[int, int, int, int] | None:
        if self.roi_item is None or not self.roi_item.isVisible():
            return None
        pos, size = self.roi_item.pos(), self.roi_item.size()
        x, y = int(round(pos.x())), int(round(pos.y()))
        w, h = int(round(size.x())), int(round(size.y()))
        if self._data is not None:
            H, W = self._data.shape[:2]
            x, y = max(0, min(x, W - 2)), max(0, min(y, H - 2))
            w, h = max(2, min(w, W - x)), max(2, min(h, H - y))
        return x, y, w, h

    def _roi_moved(self) -> None:
        self.roi_changed.emit(self.roi_rect())

    # -- line profile -----------------------------------------------------
    def show_profile(self, visible: bool) -> None:
        if self.profile_line is None:
            return
        self.profile_line.setVisible(visible)
        self.profile_box.setVisible(visible)
        if not visible:
            self._hide_cursor()
        elif self._data is not None and not (self._profile_placed and self._line_inside()):
            self._place_default_line()
        self._update_profile()

    def _line_points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        p0, p1 = (self.profile_line.mapToParent(h.pos()) for h in self.profile_line.getHandles()[:2])
        return (p0.x(), p0.y()), (p1.x(), p1.y())

    def _line_inside(self) -> bool:
        if self._data is None:
            return False
        h, w = self._data.shape[:2]
        return all(0 <= x <= w and 0 <= y <= h for x, y in self._line_points())

    def _place_default_line(self) -> None:
        h, w = self._data.shape[:2]
        handles = self.profile_line.getHandles()
        self.profile_line.movePoint(handles[0], [0.1 * w, 0.5 * h], finish=False)
        self.profile_line.movePoint(handles[1], [0.9 * w, 0.5 * h])
        self._profile_placed = True

    def _distance_unit(self) -> str:
        return "µm" if self._um_per_px > 0 else "px"

    def _sample_profile(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Values along the segment (1 sample/px), averaged over ``width`` parallel lines."""
        img = self._data
        if img.ndim == 3:
            img = img.mean(axis=2)
        if img.dtype.kind == "c":
            img = img.real
        if img.dtype.kind not in "iu" and img.dtype not in (np.float32, np.float64):
            img = img.astype(np.float32)
        h, w = img.shape[:2]
        (x0, y0), (x1, y1) = self._line_points()
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6 or h == 0 or w == 0:
            return None
        t = np.linspace(0.0, 1.0, max(2, int(math.ceil(length)) + 1))
        width = self.profile_width_spin.value()
        offsets = np.arange(width) - (width - 1) / 2.0
        xs = x0 + t[None, :] * dx - offsets[:, None] * dy / length
        ys = y0 + t[None, :] * dy + offsets[:, None] * dx / length
        inside = (xs >= 0) & (xs <= w) & (ys >= 0) & (ys <= h)
        # pixel (r, c) covers [c, c+1) x [r, r+1): its centre is at array coordinate (c+0.5, r+0.5)
        vals = ndimage.map_coordinates(img, [ys.ravel() - 0.5, xs.ravel() - 0.5], output=np.float64,
                                       order=1, mode="nearest", prefilter=False).reshape(xs.shape)
        vals = np.where(inside & np.isfinite(vals), vals, np.nan)
        count = np.isfinite(vals).sum(axis=0)
        values = np.where(count > 0, np.nansum(vals, axis=0) / np.maximum(count, 1), np.nan)
        scale = self._um_per_px if self._um_per_px > 0 else 1.0
        return t * length * scale, values

    def _update_profile(self, *_) -> None:
        if self.profile_line is None:
            return
        profile = None
        if self.profile_line.isVisible() and self._data is not None:
            try:
                profile = self._sample_profile()
            except Exception:
                profile = None
        if profile is None and self._profile is None:
            return  # nothing shown before or now (e.g. live frames while the profile is off)
        self._profile = profile
        if profile is None:
            self.profile_curve.setData([], [])
            self._profile_length = 0.0
        else:
            dist, values = profile
            self.profile_curve.setData(dist, values, connect="finite")
            self._profile_length = float(dist[-1])
            self._sync_regions()
        self._update_readout()

    def _sync_regions(self) -> None:
        length = self._profile_length
        self._syncing_regions = True
        try:
            for region, (f0, f1) in zip((self.region_a, self.region_b), self._region_fracs):
                region.setBounds((0.0, length))
                region.setRegion((f0 * length, f1 * length))
        finally:
            self._syncing_regions = False

    def _region_moved(self, *_) -> None:
        if self._syncing_regions or self._profile_length <= 0:
            return
        length = self._profile_length
        self._region_fracs = [tuple(min(1.0, max(0.0, v / length)) for v in sorted(r.getRegion()))
                              for r in (self.region_a, self.region_b)]
        self._update_readout()

    def _band_stats(self) -> dict | None:
        if self._profile is None:
            return None
        dist, values = self._profile
        out = {}
        for key, region in (("a", self.region_a), ("b", self.region_b)):
            lo, hi = sorted(region.getRegion())
            sel = values[(dist >= lo) & (dist <= hi) & np.isfinite(values)]
            if sel.size == 0:
                return None
            out[key], out["s" + key] = float(sel.mean()), float(sel.std())
        out["step"] = out["b"] - out["a"]
        out["err"] = math.hypot(out["sa"], out["sb"])
        return out

    def _value_suffix(self) -> str:
        unit = _unit_of(self._units_label)
        return f" {unit}" if unit else ""

    def _fmt_distance(self, value: float) -> str:
        return f"{value:.0f}" if self._um_per_px <= 0 else _fmt(value, _decimals(self._profile_length))

    def _result_text(self) -> str:
        stats = self._band_stats()
        if stats is None:
            return ""
        u = self._value_suffix()
        dec = _decimals(max(abs(stats["a"]), abs(stats["b"]), abs(stats["step"])))
        return (f"Step B − A = {_fmt(stats['step'], dec)}{u} (± {_fmt(stats['err'], dec)})    "
                f"A {_fmt(stats['a'], dec)}{u}    B {_fmt(stats['b'], dec)}{u}    "
                f"line {self._fmt_distance(self._profile_length)} {self._distance_unit()}")

    def _update_readout(self) -> None:
        if self.profile_readout is None:
            return
        text = self._result_text()
        if not text:
            text = ("Drag the ends of the magenta line on the image. Place band A (blue) on the reference "
                    "and band B (orange) on the feature to measure a step height.")
        self.profile_readout.setText(text)

    def _profile_hover(self, pos) -> None:
        vb = self.profile_plot.getPlotItem().vb
        if self._profile is None or not vb.sceneBoundingRect().contains(pos):
            self._hide_cursor()
            return
        x = vb.mapSceneToView(pos).x()
        dist, values = self._profile
        if not dist[0] <= x <= dist[-1]:
            self._hide_cursor()
            return
        value = float(np.interp(x, dist, values))
        finite = values[np.isfinite(values)]
        dec = _decimals(np.abs(finite).max() if finite.size else value)
        self.profile_cursor.setPos(x)
        self.profile_cursor.show()
        self.profile_hover_label.setText(f"x {self._fmt_distance(x)} {self._distance_unit()} → "
                                         f"{_fmt(value, dec)}{self._value_suffix()}")

    def _hide_cursor(self) -> None:
        if self.profile_plot is not None:
            self.profile_cursor.hide()
            self.profile_hover_label.setText("")

    def profile_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(distance in px or µm, values) along the profile line, or None when no profile is shown."""
        if self._profile is None:
            return None
        return self._profile[0].copy(), self._profile[1].copy()

    def step_height(self) -> tuple[float, float] | None:
        """(mean(B) − mean(A), sqrt(std_A² + std_B²)) of the two profile bands."""
        stats = self._band_stats()
        return None if stats is None else (stats["step"], stats["err"])

    def _profile_header(self, sep: str) -> str:
        return f"distance ({self._distance_unit()}){sep}{self._units_label or 'value'}"

    def profile_text(self, sep: str = "\t") -> str | None:
        """Header line plus one ``distance<sep>value`` row per sample (blank value = no data)."""
        if self._profile is None:
            return None
        rows = [self._profile_header(sep)]
        rows += [f"{d:.6g}{sep}{v:.6g}" if np.isfinite(v) else f"{d:.6g}{sep}" for d, v in zip(*self._profile)]
        return "\n".join(rows) + "\n"

    def copy_profile(self) -> bool:
        text = self.profile_text("\t")
        if text is None:
            return False
        QtWidgets.QApplication.clipboard().setText(text)
        return True

    def save_profile_csv(self, path: str) -> None:
        """Write the profile with a commented header (units, bands and step-height result)."""
        if self._profile is None:
            raise ValueError("no line profile to save")
        (x0, y0), (x1, y1) = self._line_points()
        du = self._distance_unit()
        bands = "; ".join(f"{name} {lo:.6g} – {hi:.6g} {du}" for name, (lo, hi) in
                          (("A (reference)", sorted(self.region_a.getRegion())),
                           ("B (feature)", sorted(self.region_b.getRegion()))))
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            fh.write("# PhaseScope line profile\n")
            fh.write(f"# {self._result_text() or 'step height: not available'}\n")
            fh.write(f"# bands: {bands}\n")
            fh.write(f"# line from ({x0:.1f}, {y0:.1f}) to ({x1:.1f}, {y1:.1f}) px, averaged over "
                     f"{self.profile_width_spin.value()} px"
                     + (f", {self._um_per_px:g} µm/px" if self._um_per_px > 0 else "") + "\n")
            writer = csv.writer(fh)
            writer.writerow([f"distance ({du})", self._units_label or "value"])
            for d, v in zip(*self._profile):
                writer.writerow([f"{d:.6g}", f"{v:.6g}" if np.isfinite(v) else ""])

    def _save_profile_dialog(self) -> None:
        if self._profile is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save line profile", "line_profile.csv",
                                                        "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            self.save_profile_csv(path)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Save line profile", f"Could not save the profile:\n{exc}")

    # -- mouse ------------------------------------------------------------
    def _mouse_moved(self, pos) -> None:
        if self._data is None or not self.vb.sceneBoundingRect().contains(pos):
            return
        p = self.vb.mapSceneToView(pos)
        x, y = int(np.floor(p.x())), int(np.floor(p.y()))
        h, w = self._data.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            self.hovered.emit(x, y, float(self._data[y, x]))

    def grab_image(self) -> QtGui.QImage:
        shown = [lab for lab in (self.message_label, self.banner_label) if not lab.isHidden()]
        for lab in shown:  # keep hints out of saved pictures
            lab.hide()
        try:
            return self.view_widget.grab().toImage()
        finally:
            for lab in shown:
                lab.show()


# --------------------------------------------------------------------------
# 3-D surface panel
# --------------------------------------------------------------------------

def opengl_available() -> bool:
    try:
        import OpenGL.GL  # noqa: F401
        import pyqtgraph.opengl  # noqa: F401
        return True
    except Exception:
        return False


REDRAW_IDLE_FACTOR = 4.0  # live 3-D updates leave the GUI idle >= 4x the last redraw's duration
_WIRE_COLOR = (0.0, 0.0, 0.0, 0.6)


class Surface3DPanel(QtWidgets.QWidget):
    """Interactive 3-D surface of a 2-D map (OpenGL, matplotlib fallback).

    Left-drag rotates, right/middle-drag pans, the wheel zooms; ``double_clicked()`` on double click.
    """

    double_clicked = QtCore.Signal()

    def __init__(self, use_gl: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._data = None
        self._cmap_name = "viridis"
        self._units = ""
        self._last_draw = -math.inf  # perf_counter() at the start of the last redraw
        self._draw_cost = 0.0  # duration of the last redraw (s)
        self._stale = False  # set_data() data not drawn yet
        self._max_rate_hz = 12.0
        self._legacy_gl = False  # pyqtgraph < 0.14: GLSurfacePlotItem.setData has no grid options
        self._pending = False
        self._extent = None
        self.max_points = 200
        self.z_scale = 0.35
        self.use_gl = use_gl and opengl_available()

        self.controls_widget = QtWidgets.QWidget()
        self.controls_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                           QtWidgets.QSizePolicy.Policy.Fixed)
        controls = QtWidgets.QHBoxLayout(self.controls_widget)
        controls.setContentsMargins(4, 2, 4, 2)
        z_label = QtWidgets.QLabel("Z scale")
        z_label.setToolTip("Vertical exaggeration of the surface")
        controls.addWidget(z_label)
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.z_slider.setRange(1, 200)
        self.z_slider.setValue(int(round(self.z_scale * 100)))
        self.z_slider.setMinimumWidth(60)
        self.z_slider.setToolTip(z_label.toolTip())
        self.z_slider.valueChanged.connect(self._z_changed)
        controls.addWidget(self.z_slider, 2)
        self.z_value_label = QtWidgets.QLabel(self._z_text(self.z_slider.value()))
        self.z_value_label.setMinimumWidth(self.z_value_label.fontMetrics().horizontalAdvance("×0.00") + 4)
        controls.addWidget(self.z_value_label)
        detail_label = QtWidgets.QLabel("Detail")
        controls.addWidget(detail_label)
        self.res_spin = QtWidgets.QSpinBox()
        self.res_spin.setRange(32, 800)
        self.res_spin.setSingleStep(16)
        self.res_spin.setValue(self.max_points)
        self.res_spin.setSuffix(" pts")
        self.res_spin.setToolTip("Maximum number of mesh points per side (higher = finer, slower)")
        detail_label.setToolTip(self.res_spin.toolTip())
        self.res_spin.valueChanged.connect(self._res_changed)
        controls.addWidget(self.res_spin)
        self.wire_check = QtWidgets.QCheckBox("Wireframe")
        self.wire_check.toggled.connect(lambda _: self._redraw(force=True))
        controls.addWidget(self.wire_check)
        self.reset_btn = QtWidgets.QPushButton("Reset view")
        self.reset_btn.setToolTip("Back to the default viewing angle and zoom")
        self.reset_btn.clicked.connect(self.reset_camera)
        controls.addWidget(self.reset_btn)
        self.range_label = QtWidgets.QLabel("")
        self.range_label.setWordWrap(True)
        self.range_label.setContentsMargins(4, 0, 4, 0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.controls_widget)
        layout.addWidget(self.range_label)

        if self.use_gl:
            import pyqtgraph.opengl as gl
            self._gl = gl
            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor("k")
            self.view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            self.grid = gl.GLGridItem()
            self.grid.setColor((255, 255, 255, 40))
            self.view.addItem(self.grid)
            self.surface = gl.GLSurfacePlotItem(computeNormals=False, smooth=False, glOptions="opaque")
            self.view.addItem(self.surface)
            layout.addWidget(self.view, 1)
        else:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.figure = Figure(facecolor="black")
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.ax = self.figure.add_subplot(111, projection="3d")
            self.view = self.canvas
            self.view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            self.max_points = 90
            self.res_spin.setValue(90)
            layout.addWidget(self.canvas, 1)
        self.message_label = _make_overlay(self.view, _MESSAGE_STYLE, centered=True)
        self.view.installEventFilter(self)
        self.reset_camera()

    def set_popup_menu(self, menu: QtWidgets.QMenu) -> None:
        self.view.customContextMenuRequested.connect(
            lambda pos: menu.popup(self.view.mapToGlobal(pos)))

    def set_colormap(self, name: str) -> None:
        self._cmap_name = name
        self._redraw(force=True)

    @property
    def colormap_name(self) -> str:
        return self._cmap_name

    def set_units(self, units: str) -> None:
        self._units = units

    def set_message(self, text: str | None) -> None:
        """Centred multi-line hint over the 3-D view (None hides it)."""
        if _set_overlay_text(self.message_label, text):
            _place_message(self.message_label, self.view.rect())

    def eventFilter(self, obj, ev) -> bool:
        if obj is getattr(self, "view", None):
            etype = ev.type()
            if etype == QtCore.QEvent.Type.MouseButtonDblClick:
                self.double_clicked.emit()
            elif etype == QtCore.QEvent.Type.Resize and not self.message_label.isHidden():
                _place_message(self.message_label, self.view.rect())
        return super().eventFilter(obj, ev)

    @staticmethod
    def _z_text(v: int) -> str:
        return f"×{v / 100.0:.2f}"

    def _z_changed(self, v: int) -> None:
        self.z_scale = v / 100.0
        self.z_value_label.setText(self._z_text(v))
        self._redraw(force=True)

    def _res_changed(self, v: int) -> None:
        self.max_points = v
        self._redraw(force=True)

    def reset_camera(self) -> None:
        if self.use_gl:
            self.view.setCameraPosition(distance=1.6 * (self._extent or self.max_points),
                                        elevation=35, azimuth=-60)
            self.view.opts["center"] = pg.Vector(0, 0, 0)
            self.view.update()
        else:
            self.ax.view_init(elev=35, azim=-60)
            self.canvas.draw_idle()

    def set_data(self, data: np.ndarray | None, max_rate_hz: float = 12.0) -> None:
        """Live update, throttled adaptively: at most ``max_rate_hz`` redraws per second, and the
        next one only after :data:`REDRAW_IDLE_FACTOR` times the last redraw's duration (so the
        slow matplotlib fallback cannot starve the GUI).  The latest data is always drawn."""
        self._data = data
        self._stale = True
        self._max_rate_hz = max_rate_hz
        self._throttled_redraw()

    def _throttled_redraw(self) -> None:
        interval = max(1.0 / self._max_rate_hz, (1.0 + REDRAW_IDLE_FACTOR) * self._draw_cost)
        wait = self._last_draw + interval - time.perf_counter()
        if wait <= 0:
            self._redraw()
        elif not self._pending:  # make sure the last frame is drawn
            self._pending = True
            QtCore.QTimer.singleShot(max(1, math.ceil(wait * 1000)), self._deferred)

    def show_data(self, data: np.ndarray | None) -> None:
        """Draw ``data`` immediately (no throttling), even while the panel is hidden."""
        if data is None:
            self.clear()
            return
        self._data = data
        self._redraw(force=True)

    def clear(self) -> None:
        """Remove the surface; the next data shown brings it back."""
        self._data = None
        self.range_label.setText("")
        if self.use_gl:
            self.surface.hide()
            self.grid.hide()
        else:
            self.ax.clear()
            self.ax.set_facecolor("black")
            self.ax.set_axis_off()
            self.canvas.draw_idle()

    def _deferred(self) -> None:
        self._pending = False
        if self._stale:
            self._throttled_redraw()

    def _prepare(self):
        data = np.asarray(self._data, dtype=np.float32)
        step = int(np.ceil(max(data.shape) / float(self.max_points)))
        z = data[::step, ::step]
        finite = np.isfinite(z)
        if not finite.any():
            return None
        lo, hi = robust_levels(z, 0.2, 99.8)
        z = np.where(finite, np.clip(z, lo, hi), lo)
        norm = (z - lo) / (hi - lo)
        return z, norm, lo, hi, step

    def _redraw(self, force: bool = False) -> None:
        """Draw the current data now (timed, for :meth:`set_data`'s throttle)."""
        if self._data is None or (not self.isVisible() and not force):
            return
        start = time.perf_counter()
        self._stale = False
        try:
            self._draw()
        finally:
            self._last_draw = start
            self._draw_cost = time.perf_counter() - start

    def _draw(self) -> None:
        prepared = self._prepare()
        if prepared is None:
            return
        z, norm, lo, hi, step = prepared
        unit = f" {self._units}" if self._units else ""
        self.range_label.setText(f"z range {lo:.4g} … {hi:.4g}{unit}   (mesh {z.shape[1]}×{z.shape[0]}, 1:{step})")
        rows, cols = z.shape
        extent = max(rows, cols)
        height = norm * self.z_scale * extent
        cmap = get_colormap(self._cmap_name)
        if self.use_gl:
            lut = cmap.getLookupTable(0.0, 1.0, 256, alpha=True, mode=pg.ColorMap.FLOAT)
            idx = np.clip((norm * 255).astype(np.int32), 0, 255)
            # surface x = image column, y = image row (flipped so row 0 is at the back)
            zz = height[::-1, :].T
            colors = lut[idx[::-1, :].T]
            x = np.arange(cols, dtype=np.float32) - cols / 2.0
            y = np.arange(rows, dtype=np.float32) - rows / 2.0
            self._set_surface(x, y, zz.astype(np.float32), colors.reshape(-1, 4).astype(np.float32))
            self.surface.show()
            self.grid.show()
            self.grid.resetTransform()
            self.grid.setSize(cols, rows)
            self.grid.setSpacing(max(1, cols // 10), max(1, rows // 10))
            if self._extent != extent:  # new image size: frame the whole surface
                self._extent = extent
                self.view.setCameraPosition(distance=1.6 * extent)
        else:
            ax = self.ax
            elev, azim = ax.elev, ax.azim
            ax.clear()
            ax.set_facecolor("black")
            yy, xx = np.mgrid[:rows, :cols]
            ax.plot_surface(xx, yy, z, facecolors=cmap.getLookupTable(0, 1, 256, alpha=True,
                                                                     mode=pg.ColorMap.FLOAT)[
                np.clip((norm * 255).astype(int), 0, 255)], linewidth=0.2 if self.wire_check.isChecked() else 0,
                edgecolor="k" if self.wire_check.isChecked() else None, antialiased=False, shade=False)
            ax.set_zlim(lo, lo + (hi - lo) / max(self.z_scale, 1e-3))
            ax.invert_yaxis()
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            self.canvas.draw()  # now, not idle: the draw time feeds set_data's throttle

    def _set_surface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray) -> None:
        """Update the GL mesh; the wireframe uses ``showGrid`` (pyqtgraph >= 0.14) or mesh edges."""
        wire = self.wire_check.isChecked()
        if not self._legacy_gl:
            try:
                self.surface.setData(x=x, y=y, z=z, colors=colors, showGrid=wire, lineColor=_WIRE_COLOR)
                return
            except TypeError:  # pyqtgraph < 0.14: setData(x, y, z, colors) only
                self._legacy_gl = True
        self.surface.setData(x=x, y=y, z=z, colors=colors)
        self.surface.opts.update(drawEdges=wire, edgeColor=_WIRE_COLOR)
        self.surface.update()

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        self._redraw(force=True)

    def grab_image(self) -> QtGui.QImage:
        if self.use_gl:
            return self.view.grabFramebuffer()
        shown = not self.message_label.isHidden()
        self.message_label.hide()
        try:
            return self.canvas.grab().toImage()
        finally:
            self.message_label.setVisible(shown)


# --------------------------------------------------------------------------
# Parameter dialog
# --------------------------------------------------------------------------

SLIDER_STEPS = 1000
EMIT_INTERVAL_MS = 100  # live updates from the dialog are coalesced to at most 10 per second


def _has_slider(p: Param) -> bool:
    return (p.kind in ("int", "float") and p.minimum is not None and p.maximum is not None
            and float(p.maximum) > float(p.minimum))


def _slider_is_log(p: Param) -> bool:
    return float(p.minimum) > 0 and float(p.maximum) / float(p.minimum) > 100


def _value_to_slider(p: Param, value: float) -> int:
    lo, hi = float(p.minimum), float(p.maximum)
    value = min(hi, max(lo, float(value)))
    if _slider_is_log(p):
        frac = math.log(value / lo) / math.log(hi / lo)
    else:
        frac = (value - lo) / (hi - lo)
    return int(round(frac * SLIDER_STEPS))


def _slider_to_value(p: Param, pos: int) -> int | float:
    lo, hi = float(p.minimum), float(p.maximum)
    frac = min(1.0, max(0.0, pos / float(SLIDER_STEPS)))
    value = lo * (hi / lo) ** frac if _slider_is_log(p) else lo + frac * (hi - lo)
    if p.kind == "float":
        return min(hi, max(lo, round(value, p.decimals)))
    lo_i, hi_i = int(math.ceil(lo)), int(math.floor(hi))
    value = min(hi_i, max(lo_i, int(round(value))))
    if p.odd and value % 2 == 0:
        value = value + 1 if value + 1 <= hi_i else value - 1
    return value


class ParamDialog(QtWidgets.QDialog):
    """Dialog generated from a list of :class:`Param`; emits ``changed`` live (at most 10 Hz)."""

    changed = QtCore.Signal(dict)

    def __init__(self, title: str, params: tuple[Param, ...] | list[Param],
                 values: dict | None = None, description: str = "", parent=None,
                 subtitle: str = "", new: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(380)
        self.params = list(params)
        self._widgets: dict[str, QtWidgets.QWidget] = {}
        self._sliders: dict[str, QtWidgets.QSlider] = {}
        self._last_emit = -math.inf
        self._last_values: dict | None = None
        self._closed = False
        self._emit_timer = QtCore.QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.timeout.connect(self._emit_now)
        values = dict(values or {})

        layout = QtWidgets.QVBoxLayout(self)
        self.header_label = QtWidgets.QLabel(title)
        font = self.header_label.font()
        font.setBold(True)
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() + 3)
        self.header_label.setFont(font)
        layout.addWidget(self.header_label)
        self.subtitle_label = None
        if subtitle:
            self.subtitle_label = QtWidgets.QLabel(subtitle)
            self.subtitle_label.setStyleSheet(_MUTED_STYLE)
            layout.addWidget(self.subtitle_label)
        self.description_label = QtWidgets.QLabel(description or "Adjust the values – the image updates live.")
        self.description_label.setWordWrap(True)
        layout.addWidget(self.description_label)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addLayout(form)
        for p in self.params:
            w = self._make_widget(p, values.get(p.name, p.default))
            if p.tooltip:
                w.setToolTip(p.tooltip)
            self._widgets[p.name] = w
            field = w
            if _has_slider(p):
                field = self._with_slider(p, w)
            form.addRow(p.label, field)

        self.footer_label = QtWidgets.QLabel("Add step keeps it · Cancel removes it" if new
                                             else "Cancel restores the previous values")
        self.footer_label.setStyleSheet(_MUTED_STYLE)
        self.footer_label.setWordWrap(True)
        layout.addWidget(self.footer_label)

        self.button_box = buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Add step" if new else "Apply")
        for button in buttons.buttons():
            button.setIcon(QtGui.QIcon())
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self.restore_defaults)
        layout.addWidget(buttons)

    def _make_widget(self, p: Param, value) -> QtWidgets.QWidget:
        if p.kind == "int":
            w = QtWidgets.QSpinBox()
            w.setRange(int(p.minimum if p.minimum is not None else -10 ** 9),
                       int(p.maximum if p.maximum is not None else 10 ** 9))
            w.setSingleStep(int(p.step or (2 if p.odd else 1)))
            w.setValue(int(value))
            w.valueChanged.connect(lambda _v, p=p: self._spin_changed(p))
            w.editingFinished.connect(lambda p=p: self._spin_finished(p))
        elif p.kind == "float":
            w = QtWidgets.QDoubleSpinBox()
            w.setDecimals(p.decimals)
            w.setRange(float(p.minimum if p.minimum is not None else -1e12),
                       float(p.maximum if p.maximum is not None else 1e12))
            w.setSingleStep(float(p.step or 0.1))
            w.setValue(float(value))
            w.valueChanged.connect(lambda _v, p=p: self._spin_changed(p))
            w.editingFinished.connect(lambda p=p: self._spin_finished(p))
        elif p.kind == "choice":
            w = QtWidgets.QComboBox()
            w.addItems(list(p.choices))
            w.setCurrentText(str(value))
            w.currentIndexChanged.connect(self._emit)
        elif p.kind == "bool":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(self._emit)
        else:
            w = QtWidgets.QLineEdit(str(value))
            w.editingFinished.connect(self._emit)
        return w

    def _with_slider(self, p: Param, spin: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Put a two-way linked slider (log scale for wide positive ranges) beside ``spin``."""
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, SLIDER_STEPS)
        slider.setPageStep(SLIDER_STEPS // 20)
        slider.setValue(_value_to_slider(p, spin.value()))
        slider.setMinimumWidth(120)
        if p.tooltip:
            slider.setToolTip(p.tooltip)
        slider.valueChanged.connect(lambda pos, p=p: self._slider_changed(p, pos))
        slider.sliderReleased.connect(self._flush)
        self._sliders[p.name] = slider
        spin.setMinimumWidth(90)
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(spin)
        row.addWidget(slider, 1)
        return box

    def _sync_slider(self, p: Param) -> None:
        slider = self._sliders.get(p.name)
        if slider is not None:
            blocked = slider.blockSignals(True)
            slider.setValue(_value_to_slider(p, self._widgets[p.name].value()))
            slider.blockSignals(blocked)

    def _spin_changed(self, p: Param) -> None:
        self._sync_slider(p)
        self._emit()

    def _spin_finished(self, p: Param) -> None:
        w = self._widgets[p.name]
        if p.kind == "int" and p.odd and w.value() % 2 == 0:  # typed an even kernel size
            w.setValue(w.value() + 1 if w.value() + 1 <= w.maximum() else w.value() - 1)
        self._flush()

    def _slider_changed(self, p: Param, pos: int) -> None:
        w = self._widgets[p.name]
        blocked = w.blockSignals(True)
        w.setValue(_slider_to_value(p, pos))
        w.blockSignals(blocked)
        self._emit()

    def values(self) -> dict:
        out = {}
        for p in self.params:
            w = self._widgets[p.name]
            if p.kind in ("int", "float"):
                out[p.name] = p.coerce(w.value())
            elif p.kind == "choice":
                out[p.name] = w.currentText()
            elif p.kind == "bool":
                out[p.name] = w.isChecked()
            else:
                out[p.name] = w.text()
        return out

    def restore_defaults(self) -> None:
        for p in self.params:
            w = self._widgets[p.name]
            w.blockSignals(True)
            if p.kind in ("int", "float"):
                w.setValue(p.default)
            elif p.kind == "choice":
                w.setCurrentText(p.default)
            elif p.kind == "bool":
                w.setChecked(p.default)
            else:
                w.setText(str(p.default))
            w.blockSignals(False)
            self._sync_slider(p)
        self._emit_now()

    # -- live updates (coalesced) ----------------------------------------
    def _live(self) -> bool:
        # editingFinished fires when focus leaves on close/destruction: no updates after that
        return not self._closed and isQObjectAlive(self)

    def _emit(self, *_) -> None:
        """Emit now if the last emission is old enough, otherwise once the interval has passed."""
        if not self._live() or self._emit_timer.isActive():
            return
        wait_ms = EMIT_INTERVAL_MS - (time.monotonic() - self._last_emit) * 1000.0
        if wait_ms <= 0:
            self._emit_now()
        else:
            self._emit_timer.start(int(math.ceil(wait_ms)))

    def _emit_now(self) -> None:
        if not isQObjectAlive(self):
            return
        self._emit_timer.stop()
        self._last_values = self.values()
        self._last_emit = time.monotonic()
        self.changed.emit(dict(self._last_values))

    def _flush(self) -> None:
        """Emit the current values unless they were the last ones emitted."""
        if self._live() and (self._emit_timer.isActive() or self.values() != self._last_values):
            self._emit_now()

    def accept(self) -> None:
        self._flush()
        super().accept()

    def showEvent(self, ev) -> None:
        self._closed = False
        super().showEvent(ev)

    def done(self, result: int) -> None:
        self._closed = True
        accepted = QtWidgets.QDialog.DialogCode.Accepted
        if int(getattr(result, "value", result)) != int(getattr(accepted, "value", accepted)):
            self._emit_timer.stop()  # a late update must not undo the caller's restore on Cancel
        super().done(result)
