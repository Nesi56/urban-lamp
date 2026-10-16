"""Headless smoke test of the main window with the simulated camera."""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pg = pytest.importorskip("pyqtgraph")
pytest.importorskip("matplotlib")


@pytest.fixture(scope="module")
def app():
    return pg.mkQApp()


def _pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def _wait_for(app, condition, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end and not condition():
        app.processEvents()
        time.sleep(0.01)
    return condition()


def test_main_window_live_filters_and_phase(app, tmp_path):
    from phasescope.gui import PHASE_METHODS, MainWindow

    win = MainWindow(source="sim", use_gl=False, autostart=True)
    win.show()
    try:
        assert _wait_for(app, lambda: win.last_packet is not None), "no live frames"

        # filters / transformations from the pop-up menus
        win.pipeline.add("gaussian", {"sigma": 1.5})
        win.pipeline.add("bin", {"factor": 2})
        win._refresh_pipeline_list()
        assert _wait_for(app, lambda: win.last_packet.processed.shape == (240, 320))
        win.clear_pipeline()

        # live Fourier-transform phase
        win.method_combo.setCurrentIndex(list(PHASE_METHODS).index("ftp"))
        assert _wait_for(app, lambda: win.last_result is not None)
        assert win.channel_combo.currentText() == "Height / physical"

        # automatic 4-step sequence with the simulated phase shifter
        win.method_combo.setCurrentIndex(list(PHASE_METHODS).index("4step"))
        win.last_result = None
        win.acquire_sequence()
        assert _wait_for(app, lambda: win.last_result is not None)
        assert "4-step" in win.last_result.info["method"]
        assert np.isfinite(win.last_result.physical).mean() > 0.9

        # 3-D view and saving
        win.view_tabs.setCurrentIndex(1)
        _pump(app, 0.3)
        path = str(tmp_path / "result.npz")
        np.savez_compressed(path, physical=win.last_result.physical)
        assert os.path.getsize(path) > 0
    finally:
        win.close()
