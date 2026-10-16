"""Reusable Qt widgets: 2-D image panel, 3-D surface panel, parameter dialog."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

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
# 2-D image panel
# --------------------------------------------------------------------------

class _MenuImageItem(pg.ImageItem):
    """ImageItem that pops up the application's context menu on right click."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.popup_menu: QtWidgets.QMenu | None = None

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton and self.popup_menu is not None:
            ev.accept()
            self.popup_menu.popup(ev.screenPos().toPoint())
            return
        super().mouseClickEvent(ev)


class ImagePanel(QtWidgets.QWidget):
    """Zoomable image display with colour bar, optional ROI and line profile.

    * mouse wheel / drag: zoom & pan, right click on the image: pop-up menu
    * ``hovered(x, y, value)`` is emitted while the mouse moves over the image
    """

    hovered = QtCore.Signal(int, int, float)
    roi_changed = QtCore.Signal(object)  # (x, y, w, h) or None
    auto_levels_changed = QtCore.Signal(bool)

    def __init__(self, title: str = "", histogram: bool = True, profile: bool = False,
                 roi: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._data: np.ndarray | None = None
        self._auto_levels = True
        self._programmatic = False
        self._cmap_name = "gray"
        self._shape = None

        self.view_widget = pg.GraphicsLayoutWidget()
        self.view_widget.setBackground("k")
        if title:
            self.title = self.view_widget.addLabel(title, row=0, col=0, colspan=2)
        self.vb = self.view_widget.addViewBox(row=1, col=0, lockAspect=True, invertY=True)
        self.vb.setMenuEnabled(True)
        self.image = _MenuImageItem()
        self.vb.addItem(self.image)
        self.hist = None
        if histogram:
            self.hist = pg.HistogramLUTItem(self.image)
            self.hist.sigLevelChangeFinished.connect(self._user_levels)
            self.view_widget.addItem(self.hist, row=1, col=1)
        self.view_widget.scene().sigMouseMoved.connect(self._mouse_moved)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # optional analysis ROI (rectangle)
        self.roi_item = None
        if roi:
            self.roi_item = pg.RectROI([20, 20], [200, 150], pen=pg.mkPen("y", width=2),
                                       removable=False, rotatable=False)
            self.roi_item.addScaleHandle([0, 0], [1, 1])
            self.roi_item.hide()
            self.roi_item.sigRegionChangeFinished.connect(self._roi_moved)
            self.vb.addItem(self.roi_item)

        # optional line profile
        self.profile_line = None
        self.profile_plot = None
        if profile:
            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            splitter.addWidget(self.view_widget)
            self.profile_plot = pg.PlotWidget()
            self.profile_plot.setLabel("bottom", "distance along line", units="px")
            self.profile_plot.showGrid(x=True, y=True, alpha=0.3)
            self.profile_curve = self.profile_plot.plot(pen=pg.mkPen("c", width=2))
            splitter.addWidget(self.profile_plot)
            splitter.setSizes([700, 200])
            self.profile_plot.hide()
            layout.addWidget(splitter)
            self.profile_line = pg.LineSegmentROI([[40, 40], [240, 160]], pen=pg.mkPen("m", width=2))
            self.profile_line.hide()
            self.profile_line.sigRegionChanged.connect(self._update_profile)
            self.vb.addItem(self.profile_line)
        else:
            layout.addWidget(self.view_widget)
        self.set_colormap("gray")

    # -- configuration ----------------------------------------------------
    def set_popup_menu(self, menu: QtWidgets.QMenu) -> None:
        self.image.popup_menu = menu

    def set_colormap(self, name: str) -> None:
        self._cmap_name = name
        cmap = get_colormap(name)
        if self.hist is not None:
            self.hist.gradient.setColorMap(cmap)
        else:
            self.image.setLookupTable(cmap.getLookupTable(nPts=256))

    @property
    def colormap_name(self) -> str:
        return self._cmap_name

    def set_title(self, text: str) -> None:
        if hasattr(self, "title"):
            self.title.setText(text)

    def set_units(self, label: str) -> None:
        if self.hist is not None:
            self.hist.axis.setLabel(label)
        if self.profile_plot is not None:
            self.profile_plot.setLabel("left", label)

    # -- data -------------------------------------------------------------
    def set_image(self, data: np.ndarray | None) -> None:
        if data is None:
            self.image.clear()
            self._data = None
            return
        data = np.asarray(data)
        first = self._shape != data.shape
        self._data = data
        self._programmatic = True  # level changes below are not user edits
        try:
            if self._auto_levels or first:
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

    # -- ROI / profile ------------------------------------------------------
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

    def show_profile(self, visible: bool) -> None:
        if self.profile_line is None:
            return
        self.profile_line.setVisible(visible)
        self.profile_plot.setVisible(visible)
        if visible and self._data is not None:
            h, w = self._data.shape[:2]
            self.profile_line.movePoint(self.profile_line.getHandles()[0], [0.1 * w, 0.5 * h])
            self.profile_line.movePoint(self.profile_line.getHandles()[1], [0.9 * w, 0.5 * h])
        self._update_profile()

    def _update_profile(self) -> None:
        if self.profile_line is None or not self.profile_line.isVisible() or self._data is None:
            return
        try:
            values = self.profile_line.getArrayRegion(self._data.astype(np.float32), self.image)
        except Exception:
            return
        values = np.asarray(values, dtype=float).ravel()
        if values.size:
            self.profile_curve.setData(np.arange(values.size), values, connect="finite")

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
        return self.view_widget.grab().toImage()


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


class Surface3DPanel(QtWidgets.QWidget):
    """Interactive 3-D surface of a 2-D map (OpenGL, matplotlib fallback).

    Left-drag rotates, right/middle-drag pans, the wheel zooms.
    """

    def __init__(self, use_gl: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._data = None
        self._cmap_name = "viridis"
        self._units = ""
        self._last_draw = 0.0
        self._pending = False
        self._extent = None
        self.max_points = 200
        self.z_scale = 0.35
        self.use_gl = use_gl and opengl_available()

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Height exaggeration"))
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.z_slider.setRange(1, 200)
        self.z_slider.setValue(int(self.z_scale * 100))
        self.z_slider.valueChanged.connect(self._z_changed)
        controls.addWidget(self.z_slider, 2)
        controls.addWidget(QtWidgets.QLabel("Mesh"))
        self.res_spin = QtWidgets.QSpinBox()
        self.res_spin.setRange(32, 800)
        self.res_spin.setSingleStep(16)
        self.res_spin.setValue(self.max_points)
        self.res_spin.setSuffix(" pts")
        self.res_spin.setToolTip("Maximum number of mesh points per side (higher = finer, slower)")
        self.res_spin.valueChanged.connect(self._res_changed)
        controls.addWidget(self.res_spin)
        self.wire_check = QtWidgets.QCheckBox("Wireframe")
        self.wire_check.toggled.connect(lambda _: self._redraw(force=True))
        controls.addWidget(self.wire_check)
        self.reset_btn = QtWidgets.QPushButton("Reset camera")
        self.reset_btn.clicked.connect(self.reset_camera)
        controls.addWidget(self.reset_btn)
        self.range_label = QtWidgets.QLabel("")
        self.z_slider.setMinimumWidth(120)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
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
        self.reset_camera()

    def set_popup_menu(self, menu: QtWidgets.QMenu) -> None:
        self.view.customContextMenuRequested.connect(
            lambda pos: menu.popup(self.view.mapToGlobal(pos)))

    def set_colormap(self, name: str) -> None:
        self._cmap_name = name
        self._redraw(force=True)

    def set_units(self, units: str) -> None:
        self._units = units

    def _z_changed(self, v: int) -> None:
        self.z_scale = v / 100.0
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
        self._data = data
        now = QtCore.QTime.currentTime().msecsSinceStartOfDay() / 1000.0
        if now - self._last_draw >= 1.0 / max_rate_hz:
            self._last_draw = now
            self._redraw()
        elif not self._pending:  # make sure the last frame is drawn
            self._pending = True
            QtCore.QTimer.singleShot(int(1000 / max_rate_hz), self._deferred)

    def _deferred(self) -> None:
        self._pending = False
        self._redraw()

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
        if self._data is None or (not self.isVisible() and not force):
            return
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
            self.surface.setData(x=x, y=y, z=zz.astype(np.float32),
                                 colors=colors.reshape(-1, 4).astype(np.float32),
                                 showGrid=self.wire_check.isChecked(), lineColor=(0, 0, 0, 0.6))
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
            self.canvas.draw_idle()

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        self._redraw(force=True)

    def grab_image(self) -> QtGui.QImage:
        if self.use_gl:
            return self.view.grabFramebuffer()
        return self.canvas.grab().toImage()


# --------------------------------------------------------------------------
# Parameter dialog
# --------------------------------------------------------------------------

class ParamDialog(QtWidgets.QDialog):
    """Dialog generated from a list of :class:`Param`; emits ``changed`` live."""

    changed = QtCore.Signal(dict)

    def __init__(self, title: str, params: tuple[Param, ...] | list[Param],
                 values: dict | None = None, description: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.params = list(params)
        self._widgets: dict[str, QtWidgets.QWidget] = {}
        values = dict(values or {})
        layout = QtWidgets.QVBoxLayout(self)
        if description:
            lab = QtWidgets.QLabel(description)
            lab.setWordWrap(True)
            layout.addWidget(lab)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        for p in self.params:
            w = self._make_widget(p, values.get(p.name, p.default))
            if p.tooltip:
                w.setToolTip(p.tooltip)
            self._widgets[p.name] = w
            form.addRow(p.label, w)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults)
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
            w.valueChanged.connect(self._emit)
        elif p.kind == "float":
            w = QtWidgets.QDoubleSpinBox()
            w.setDecimals(p.decimals)
            w.setRange(float(p.minimum if p.minimum is not None else -1e12),
                       float(p.maximum if p.maximum is not None else 1e12))
            w.setSingleStep(float(p.step or 0.1))
            w.setValue(float(value))
            w.valueChanged.connect(self._emit)
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
        self._emit()

    def _emit(self, *_) -> None:
        self.changed.emit(self.values())
