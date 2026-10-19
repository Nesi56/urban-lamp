"""Offscreen tests of the reusable widgets (image panel, 3-D panel, parameter dialog)."""

import math
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pg = pytest.importorskip("pyqtgraph")
from pyqtgraph.Qt import QtCore, QtGui, QtTest, QtWidgets  # noqa: E402

from phasescope.processing import Param  # noqa: E402
from phasescope.widgets import (STATUS_COLORS, ImagePanel, ParamDialog, StatusPill,  # noqa: E402
                                Surface3DPanel, pill_stylesheet)


@pytest.fixture(scope="module")
def app():
    return pg.mkQApp()


def _pump(app, seconds=0.0):
    end = time.time() + seconds
    app.processEvents()
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


def _step_image(h=100, w=200, low=0.0, high=100.0):
    img = np.full((h, w), low, dtype=np.float32)
    img[:, w // 2:] = high
    return img


def _levels(item):
    return tuple(float(v) for v in np.ravel(item.getLevels()))


# -- status pill ------------------------------------------------------------

def test_status_pill(app):
    css = pill_stylesheet("warn")
    assert STATUS_COLORS["warn"] in css and "border-radius: 9px" in css and "bold" in css
    pill = StatusPill("Idle")
    pill.set_state("rec", "Recording", "Saving frames to disk")
    assert pill.text() == "Recording" and pill.toolTip() == "Saving frames to disk"
    assert STATUS_COLORS["rec"] in pill.styleSheet() and pill.kind == "rec"


# -- image panel: levels, histogram ------------------------------------------

def test_fixed_levels_pin_image_and_histogram(app):
    panel = ImagePanel(histogram=True)
    bright = np.random.default_rng(1).uniform(1000, 4000, (60, 80))
    panel.set_image(bright)
    panel.set_fixed_levels((0, 4095))
    assert panel.fixed_levels == (0.0, 4095.0)
    panel.set_image(bright * 0.01)  # a much darker frame must still use the fixed range
    assert _levels(panel.image) == pytest.approx((0.0, 4095.0))
    assert tuple(panel.hist.getLevels()) == pytest.approx((0.0, 4095.0))
    lo, hi = panel.hist.getHistogramRange()
    assert lo <= 0.0 and hi >= 4095.0 and hi - lo < 4095.0 * 1.1
    assert not panel.hist.region.movable
    assert panel.auto_levels  # pinning is not a user edit

    panel.set_fixed_levels(None)
    assert panel.fixed_levels is None and panel.hist.region.movable
    assert _levels(panel.image)[1] < 100.0  # back to robust auto levels of the dark frame


def test_user_level_drag_still_disables_auto_levels(app):
    panel = ImagePanel(histogram=True)
    assert panel.hist.orientation == "horizontal"
    panel.set_image(np.arange(100.0).reshape(10, 10))
    assert panel.auto_levels
    panel.hist.region.setRegion((10.0, 20.0))  # same path as dragging the level lines
    assert not panel.auto_levels
    panel.set_image(np.arange(100.0).reshape(10, 10) * 2)
    assert _levels(panel.image) == pytest.approx((10.0, 20.0))
    panel.auto_levels = True
    assert _levels(panel.image) != pytest.approx((10.0, 20.0))


def test_histogram_visibility_toggle(app):
    panel = ImagePanel(histogram=True)
    panel.resize(400, 400)
    panel.show()
    panel.set_image(np.random.default_rng(0).random((50, 50)))
    _pump(app, 0.05)
    assert panel.histogram_visible()
    assert panel.hist.geometry().height() <= 72  # compact bar under the image
    height_with = panel.vb.geometry().height()
    panel.set_histogram_visible(False)
    _pump(app, 0.05)
    assert not panel.histogram_visible()
    assert panel.vb.geometry().height() > height_with  # the image takes the freed room
    panel.set_histogram_visible(True)
    _pump(app, 0.05)
    assert panel.histogram_visible()
    assert ImagePanel(histogram=False).histogram_visible() is False
    panel.close()


# -- image panel: overlays ----------------------------------------------------

def test_message_and_banner_toggle(app):
    panel = ImagePanel("Live camera")
    panel.resize(500, 400)
    panel.show()
    _pump(app, 0.05)
    for label in (panel.message_label, panel.banner_label):
        assert not label.isVisible()
        assert label.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    panel.set_message("No camera connected.\nPress Connect to start.")
    assert panel.message_label.isVisible() and "Press Connect" in panel.message_label.text()
    geo, area = panel.message_label.geometry(), panel.view_widget.viewport().rect()
    assert area.contains(geo) and abs(geo.center().x() - area.center().x()) <= 30

    panel.set_banner("Saturated pixels – reduce exposure", "warn")
    assert panel.banner_label.isVisible()
    r, g, b, _ = QtGui.QColor(STATUS_COLORS["warn"]).getRgb()
    assert f"rgba({r}, {g}, {b}, 230)" in panel.banner_label.styleSheet()
    assert panel.banner_label.geometry().width() > 300

    panel.resize(700, 500)  # overlays follow the image area
    _pump(app, 0.05)
    assert abs(panel.message_label.geometry().center().x()
               - panel.view_widget.viewport().rect().center().x()) <= 30

    panel.set_message(None)
    panel.set_banner(None)
    assert not panel.message_label.isVisible() and not panel.banner_label.isVisible()
    panel.close()


def test_mask_overlay_subsampled_and_dimmed(app):
    panel = ImagePanel()
    panel.set_image(np.zeros((100, 200)))
    mask = np.zeros((25, 50), bool)
    mask[5:10, 10:20] = True
    panel.set_mask_overlay(mask)
    assert panel.mask_item.isVisible()
    rect = panel.mask_item.mapRectToParent(panel.mask_item.boundingRect())
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == pytest.approx((0, 0, 200, 100))
    assert panel.mask_item.zValue() > panel.image.zValue()
    rgba = panel.mask_item.image
    assert tuple(rgba[7, 15]) == (255, 40, 40, 190) and rgba[0, 0, 3] == 0
    panel.set_mask_overlay(np.zeros((25, 50), bool))  # nothing saturated: nothing drawn
    assert not panel.mask_item.isVisible()
    panel.set_mask_overlay(mask)
    panel.set_mask_overlay(None)
    assert not panel.mask_item.isVisible()
    panel.set_dimmed(True)
    assert panel.image.opacity() == pytest.approx(0.4)
    panel.set_dimmed(False)
    assert panel.image.opacity() == pytest.approx(1.0)


def test_double_click_signal(app):
    panel = ImagePanel()
    panel.set_image(np.zeros((40, 40)))
    hits = []
    panel.double_clicked.connect(lambda: hits.append(True))

    class _Click:
        def __init__(self, double):
            self._double, self.accepted = double, False

        def button(self):
            return QtCore.Qt.MouseButton.LeftButton

        def double(self):
            return self._double

        def accept(self):
            self.accepted = True

    ev = _Click(True)
    panel.image.mouseClickEvent(ev)
    assert hits == [True] and ev.accepted
    panel.image.mouseClickEvent(_Click(False))
    assert hits == [True]


class _RightClick:
    """Minimal stand-in for pyqtgraph's MouseClickEvent (right button)."""

    def __init__(self, x=10, y=20):
        self._pos, self.accepted = QtCore.QPointF(x, y), False

    def button(self):
        return QtCore.Qt.MouseButton.RightButton

    def double(self):
        return False

    def screenPos(self):
        return self._pos

    def accept(self):
        self.accepted = True


def test_right_click_anywhere_opens_the_app_menu(app):
    panel = ImagePanel("Live camera")
    menu = QtWidgets.QMenu()
    shown = []
    menu.popup = lambda pos, *a: shown.append((pos.x(), pos.y()))
    panel.set_popup_menu(menu)

    ev = _RightClick(10, 20)  # empty panel (before the first frame): the ViewBox gets the click
    panel.vb.mouseClickEvent(ev)
    assert ev.accepted and shown == [(10, 20)]
    panel.set_image(np.zeros((40, 60)))
    panel.image.mouseClickEvent(_RightClick(30, 40))  # on the image pixels
    panel.vb.mouseClickEvent(_RightClick(5, 6))  # in the black margin around the image
    assert shown == [(10, 20), (30, 40), (5, 6)]

    panel.resize(500, 400)  # a real right click in the margin of a wide panel
    panel.set_image(np.zeros((40, 10)))
    panel.show()
    _pump(app, 0.05)
    viewport = panel.view_widget.viewport()
    rect = panel.vb.sceneBoundingRect()
    margin = panel.view_widget.mapFromScene(QtCore.QPointF(rect.left() + 5, rect.center().y()))
    assert not panel.image.sceneBoundingRect().contains(panel.view_widget.mapToScene(margin))
    QtTest.QTest.mouseClick(viewport, QtCore.Qt.MouseButton.RightButton, QtCore.Qt.KeyboardModifier.NoModifier,
                            margin)
    _pump(app, 0.05)
    assert len(shown) == 4
    panel.close()


# -- image panel: line profile ------------------------------------------------

def test_profile_step_height(app):
    panel = ImagePanel(profile=True)
    assert panel.step_height() is None and panel.profile_data() is None
    panel.set_image(_step_image())
    panel.set_units("Height (nm)")
    panel.show_profile(True)
    (x0, y0), (x1, y1) = panel._line_points()
    assert (x0, y0, x1, y1) == pytest.approx((20, 50, 180, 50))  # default: mid-height, 10 % → 90 %
    step, err = panel.step_height()
    assert step == pytest.approx(100.0, abs=1e-3) and err == pytest.approx(0.0, abs=1e-3)
    dist, values = panel.profile_data()
    assert dist[0] == 0 and dist[-1] == pytest.approx(160.0) and len(dist) == len(values)
    text = panel.profile_readout.text()
    assert text.startswith("Step B − A = 100.0 nm (± 0.0)") and "line 160 px" in text
    assert panel.region_a.getRegion() == pytest.approx((16.0, 40.0))
    assert panel.region_b.getRegion() == pytest.approx((120.0, 144.0))

    panel.region_b.setRegion((5.0, 10.0))  # both bands on the low side: no step
    assert panel.step_height()[0] == pytest.approx(0.0, abs=1e-3)

    panel.show_profile(False)
    assert panel.profile_data() is None and panel.step_height() is None


def test_profile_pixel_size_changes_distance_unit(app):
    panel = ImagePanel(profile=True)
    panel.set_image(_step_image())
    panel.show_profile(True)
    assert panel.profile_data()[0][-1] == pytest.approx(160.0)
    panel.set_pixel_size(0.5)
    dist, _ = panel.profile_data()
    assert dist[-1] == pytest.approx(80.0)
    assert "µm" in panel.profile_readout.text()
    assert "µm" in panel.profile_plot.getAxis("bottom").labelString()
    assert panel.step_height()[0] == pytest.approx(100.0, abs=1e-3)
    assert panel.region_a.getRegion() == pytest.approx((8.0, 20.0))  # bands keep their place
    panel.set_pixel_size(0)
    assert panel.profile_data()[0][-1] == pytest.approx(160.0)
    assert panel.profile_readout.text().endswith("px")


def test_profile_integer_and_nan_images(app):
    panel = ImagePanel(profile=True)
    panel.set_image(_step_image(low=10, high=4000).astype(np.uint16))
    panel.show_profile(True)
    assert panel.step_height()[0] == pytest.approx(3990.0)
    img = _step_image()
    img[:, 30:45] = np.nan  # masked pixels inside band A are ignored
    panel.set_image(img)
    assert panel.step_height()[0] == pytest.approx(100.0)
    assert np.isnan(panel.profile_data()[1]).any()


def test_profile_width_averages_parallel_lines(app):
    panel = ImagePanel(profile=True)
    noise = np.random.default_rng(3).normal(0.0, 1.0, (120, 200))
    panel.set_image(noise)
    panel.show_profile(True)
    std1 = np.nanstd(panel.profile_data()[1])
    panel.profile_width_spin.setValue(9)
    std9 = np.nanstd(panel.profile_data()[1])
    assert std9 < 0.6 * std1


def test_show_profile_keeps_or_resets_handles(app):
    panel = ImagePanel(profile=True)
    panel.set_image(np.zeros((100, 200)))
    panel.show_profile(True)
    handles = panel.profile_line.getHandles()
    panel.profile_line.movePoint(handles[0], [30, 20])
    panel.profile_line.movePoint(handles[1], [150, 80])
    panel.show_profile(False)
    panel.show_profile(True)
    assert np.ravel(panel._line_points()) == pytest.approx([30, 20, 150, 80])
    panel.show_profile(False)
    panel.set_image(np.zeros((50, 60)))  # smaller image: old handles fall outside
    panel.show_profile(True)
    assert np.ravel(panel._line_points()) == pytest.approx([6, 25, 54, 25])


def test_profile_copy_and_csv(app, tmp_path):
    panel = ImagePanel(profile=True)
    panel.set_image(_step_image())
    panel.set_units("Height (nm)")
    panel.show_profile(True)
    assert panel.copy_profile()
    lines = QtWidgets.QApplication.clipboard().text().splitlines()
    assert lines[0] == "distance (px)\tHeight (nm)"
    assert len(lines) == 1 + len(panel.profile_data()[0])
    assert lines[1].split("\t") == ["0", "0"]

    path = tmp_path / "profile.csv"
    panel.save_profile_csv(str(path))
    text = path.read_text(encoding="utf-8-sig")
    header = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert any("Step B − A = 100.0 nm" in ln for ln in header)
    assert "distance (px),Height (nm)" in text
    data = np.loadtxt(str(path), delimiter=",", comments="#", skiprows=len(header) + 1, encoding="utf-8-sig")
    assert data.shape == (161, 2) and data[-1, 1] == pytest.approx(100.0)


# -- 3-D panel ----------------------------------------------------------------

def test_surface_panel_clear_show_data_message(app):
    pytest.importorskip("matplotlib")
    panel = Surface3DPanel(use_gl=False)
    assert panel.colormap_name == "viridis"
    panel.set_colormap("magma")
    assert panel.colormap_name == "magma"
    assert panel.controls_widget.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Ignored
    assert panel.range_label.wordWrap()
    panel.z_slider.setValue(120)
    assert panel.z_value_label.text() == "×1.20"

    data = np.outer(np.linspace(0, 1, 40), np.linspace(0, 2, 50))
    panel.show_data(data)  # immediate even though the panel was never shown
    assert panel._data is data and panel.range_label.text().startswith("z range")
    panel.clear()
    assert panel._data is None and panel.range_label.text() == ""
    assert len(panel.ax.collections) == 0
    panel.show_data(data)
    assert len(panel.ax.collections) > 0

    panel.resize(500, 400)
    panel.show()
    _pump(app, 0.05)
    panel.set_message("Run a phase measurement\nto see the surface.")
    assert panel.message_label.isVisible()
    assert panel.message_label.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    panel.set_message(None)
    assert not panel.message_label.isVisible()

    hits = []
    panel.double_clicked.connect(lambda: hits.append(True))
    ev = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseButtonDblClick, QtCore.QPointF(50, 50),
                           QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.LeftButton,
                           QtCore.Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(panel.view, ev)
    assert hits == [True]
    panel.close()


def test_surface_set_data_throttles_adaptively(app):
    pytest.importorskip("matplotlib")
    panel = Surface3DPanel(use_gl=False)
    panel.resize(400, 300)
    panel.show()
    _pump(app, 0.05)
    drawn = []
    delay = [0.04]

    def slow_draw():
        time.sleep(delay[0])
        drawn.append(panel._data)

    panel._draw = slow_draw
    frames = [np.full((8, 8), float(i)) for i in range(500)]

    def stream(seconds, rate_hz):
        drawn.clear()
        end, i = time.perf_counter() + seconds, 0
        while time.perf_counter() < end:
            panel.set_data(frames[i % len(frames)], max_rate_hz=rate_hz)
            i += 1
            _pump(app)
            time.sleep(0.005)
        return len(drawn)

    # 40 ms per draw: about one redraw per 5 x 40 ms, far below the 50 Hz cap
    assert 2 <= stream(1.0, 50.0) <= 8
    assert panel._draw_cost >= 0.04
    last = frames[-1]
    panel.set_data(last, max_rate_hz=50.0)  # the latest data is drawn once the wait is over
    _pump(app, 0.4)
    assert drawn[-1] is last
    n = len(drawn)
    _pump(app, 0.3)
    assert len(drawn) == n  # and not drawn again

    delay[0] = 0.0  # fast redraws (OpenGL) still run at the requested rate
    assert stream(1.0, 20.0) >= 12

    delay[0] = 0.04  # show_data and control changes are never throttled
    drawn.clear()
    panel.show_data(frames[1])
    panel.show_data(frames[2])
    panel.z_slider.setValue(panel.z_slider.value() + 1)
    assert len(drawn) == 3
    panel.close()


def test_surface_gl_wireframe_falls_back_on_old_pyqtgraph(app):
    from phasescope.widgets import opengl_available
    if not opengl_available():
        pytest.skip("pyqtgraph.opengl / PyOpenGL not available")
    panel = Surface3DPanel(use_gl=True)
    calls = []

    def old_set_data(x=None, y=None, z=None, colors=None):  # pyqtgraph 0.13: no showGrid / lineColor
        calls.append((x.shape, y.shape, z.shape, colors.shape))

    panel.surface.setData = old_set_data
    data = np.outer(np.linspace(0, 1, 30), np.linspace(0, 2, 40))
    panel.wire_check.setChecked(True)
    panel.show_data(data)
    assert calls == [((40,), (30,), (40, 30), (1200, 4))]
    assert panel.surface.opts["drawEdges"] is True and panel.surface.opts["edgeColor"] == (0, 0, 0, 0.6)
    assert panel.surface.visible() and panel.range_label.text().startswith("z range")
    panel.wire_check.setChecked(False)
    assert len(calls) == 2 and panel.surface.opts["drawEdges"] is False

    new = Surface3DPanel(use_gl=True)  # pyqtgraph >= 0.14: grid options passed to setData
    kwargs = []
    new.surface.setData = lambda **kw: kwargs.append(kw)
    new.wire_check.setChecked(True)
    new.show_data(data)
    assert kwargs[-1]["showGrid"] is True and kwargs[-1]["lineColor"] == (0, 0, 0, 0.6)
    assert new.surface.opts.get("drawEdges", False) is False
    panel.close()
    new.close()


# -- parameter dialog -----------------------------------------------------------

PARAMS = (
    Param("amount", "Amount", "float", 1.0, 0.0, 10.0, 0.1),
    Param("sigma", "Sigma (px)", "float", 2.0, 0.1, 50.0, 0.5),             # 500x range: log slider
    Param("ksize", "Kernel size (px)", "int", 5, 1, 99, 2, odd=True),
    Param("mode", "Mode", "choice", "a", choices=("a", "b")),
    Param("shift", "Shift", "float", 0.0),                                   # no range: no slider
)


def test_param_dialog_layout(app):
    dlg = ParamDialog("Gaussian blur", PARAMS, subtitle="Filter · Smoothing", new=True)
    assert dlg.minimumWidth() >= 380
    assert dlg.header_label.text() == "Gaussian blur" and dlg.header_label.font().bold()
    assert dlg.subtitle_label.text() == "Filter · Smoothing"
    assert "updates live" in dlg.description_label.text()
    assert dlg.footer_label.text() == "Add step keeps it · Cancel removes it"
    ok = dlg.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert ok.text() == "Add step"
    assert all(b.icon().isNull() for b in dlg.button_box.buttons())
    assert set(dlg._sliders) == {"amount", "sigma", "ksize"}

    edit = ParamDialog("Gaussian blur", PARAMS, description="Blurs the image.")
    assert edit.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).text() == "Apply"
    assert edit.footer_label.text() == "Cancel restores the previous values"
    assert edit.description_label.text() == "Blurs the image." and edit.subtitle_label is None


def test_param_dialog_slider_spin_sync(app):
    dlg = ParamDialog("Test", PARAMS, values={"amount": 2.5})
    amount, s_amount = dlg._widgets["amount"], dlg._sliders["amount"]
    assert s_amount.maximum() - s_amount.minimum() == 1000
    assert s_amount.value() == 250
    s_amount.setValue(700)
    assert amount.value() == pytest.approx(7.0)
    amount.setValue(4.0)
    assert s_amount.value() == 400

    sigma, s_sigma = dlg._widgets["sigma"], dlg._sliders["sigma"]
    s_sigma.setValue(500)  # log mapping: middle of the slider is the geometric mean
    assert sigma.value() == pytest.approx(math.sqrt(0.1 * 50.0), abs=0.01)
    s_sigma.setValue(0)
    assert sigma.value() == pytest.approx(0.1)
    s_sigma.setValue(1000)
    assert sigma.value() == pytest.approx(50.0)
    sigma.setValue(5.0)
    assert s_sigma.value() == round(1000 * math.log(50) / math.log(500))

    ksize, s_ksize = dlg._widgets["ksize"], dlg._sliders["ksize"]
    for pos in range(0, 1001, 7):
        s_ksize.setValue(pos)
        assert ksize.value() % 2 == 1 and 1 <= ksize.value() <= 99
    s_ksize.setValue(1000)
    assert ksize.value() == 99
    assert dlg.values()["ksize"] % 2 == 1

    dlg.restore_defaults()
    assert dlg.values()["amount"] == 1.0 and s_amount.value() == 100


def test_param_dialog_coalesces_changed(app):
    dlg = ParamDialog("Test", PARAMS)
    emitted = []
    dlg.changed.connect(emitted.append)
    slider = dlg._sliders["amount"]
    for pos in range(100, 1001, 100):  # ten rapid drag steps
        slider.setValue(pos)
    _pump(app, 0.3)
    assert 1 <= len(emitted) < 10
    assert emitted[-1]["amount"] == pytest.approx(10.0)

    # the final value is emitted at once on slider release and before accept()
    _pump(app, 0.15)
    emitted.clear()
    dlg._widgets["amount"].setValue(3.0)
    dlg._widgets["amount"].setValue(4.0)
    slider.sliderReleased.emit()
    assert emitted and emitted[-1]["amount"] == pytest.approx(4.0)
    emitted.clear()
    dlg._widgets["amount"].setValue(5.0)
    dlg._widgets["amount"].setValue(6.0)
    dlg.accept()
    assert emitted and emitted[-1]["amount"] == pytest.approx(6.0)


def test_param_dialog_reject_drops_pending_update(app):
    dlg = ParamDialog("Test", PARAMS)
    emitted = []
    dlg.changed.connect(emitted.append)
    dlg._widgets["amount"].setValue(2.0)  # emitted immediately
    dlg._widgets["amount"].setValue(3.0)  # pending
    dlg.reject()
    _pump(app, 0.25)
    assert [e["amount"] for e in emitted] == [pytest.approx(2.0)]


def test_colour_bar_and_boxes_open_the_panel_menu(app):
    from pyqtgraph.Qt import QtWidgets

    from phasescope.widgets import ImagePanel

    import gc
    import weakref

    class Click:
        def screenPos(self):
            return QtCore.QPointF(10, 20)

    panel = ImagePanel(histogram=True, profile=True, roi=True)
    menu = QtWidgets.QMenu()
    opened = []
    menu.popup = lambda pos: opened.append((pos.x(), pos.y()))
    panel.set_popup_menu(menu)
    hooks = [panel.vb.raiseContextMenu, panel.hist.vb.raiseContextMenu,
             panel.hist.gradient.showMenu]  # no pyqtgraph gradient presets
    for item in (panel.roi_item, panel.profile_line):
        assert item.contextMenuEnabled()
        hooks.append(item.raiseContextMenu)
    for hook in hooks:
        hook(Click())
    assert opened == [(10, 20)] * len(hooks)
    # the handlers keep no reference to the panel (a cycle the PySide6 GC can crash on)
    del hooks
    ref = weakref.ref(panel)
    gc.disable()
    try:
        del panel
        assert ref() is None
    finally:
        gc.enable()
