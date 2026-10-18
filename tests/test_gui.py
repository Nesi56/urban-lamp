"""Headless smoke tests of the main window with the simulated camera."""

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


@pytest.fixture
def win(app):
    from phasescope.gui import MainWindow

    w = MainWindow(source="sim", use_gl=False, autostart=True, restore_settings=False)
    w.show()
    assert _wait_for(app, lambda: w.last_packet is not None), "no live frames"
    yield w
    w.close()


def test_live_view_status_and_exposure_meter(app, win):
    assert win._state == "live"
    assert "LIVE" in win.status_pill.text()
    assert win.act_live.isChecked()
    # true brightness: the live view is pinned to the 12-bit full scale
    assert win.live_panel.fixed_levels == (0.0, 4095.0)
    assert _wait_for(app, lambda: "Exposure" in win.exposure_pill.text() or "Bright" in win.exposure_pill.text())
    # pause / resume with the single toggle
    win.act_live.trigger()
    assert win._state == "paused" and win.thread is None
    win.act_live.trigger()
    assert _wait_for(app, lambda: win._state == "live" and win.thread is not None)


def test_auto_exposure_reaches_target(app, win):
    win.exposure_spin.setValue(1.0)  # far too dark
    _pump(app, 0.4)
    win.start_auto_exposure()
    assert _wait_for(app, lambda: win._ae is None, timeout=15)
    raw = win.last_packet.raw
    peak = np.percentile(raw[::4, ::4], 99.9) / 4095
    assert 0.55 < peak < 0.95


def test_filters_modes_and_measurement(app, win):
    from phasescope.gui import PHASE_METHODS

    # filters / transformations from the pop-up menus
    win.pipeline.add("gaussian", {"sigma": 1.5})
    win.pipeline.add("bin", {"factor": 2})
    win._refresh_pipeline_list()
    assert win.menu_buttons["Filters"].text().startswith("Filters (1)")
    assert _wait_for(app, lambda: win.last_packet.processed.shape == (240, 320))
    win.clear_pipeline()

    # Live phase mode (F3): Fourier method, height channel, live results
    win.mode_actions["live_phase"].trigger()
    assert win.phase_cfg.method == "ftp"
    assert win.channel_combo.currentData() == "physical"
    assert _wait_for(app, lambda: win.last_result is not None)
    assert win.fringe_pill.isVisible()
    assert win.ftp_group.isVisibleTo(win.phase_dock.widget().widget())
    assert not win.stepping_group.isVisibleTo(win.phase_dock.widget().widget())

    # Hold freezes the live map
    win._primary_clicked()
    assert win._hold
    held = win.last_result
    _pump(app, 0.5)
    assert win.last_result is held
    win._primary_clicked()
    assert not win._hold

    # Measure mode (F4) with the simulated phase shifter: one primary click records a sequence
    win.mode_actions["measure"].trigger()
    assert win.phase_cfg.method == "4step"
    assert win.primary_btn.text() == "Acquire 4 steps"
    win.last_result = None
    win._primary_clicked()
    assert win._recording
    assert _wait_for(app, lambda: win.last_result is not None and not win._recording)
    assert "4-step" in win.last_result.info["method"]
    assert np.isfinite(win.last_result.physical).mean() > 0.9
    assert "PV" in win.headline.text()

    # switching method keeps the old result but marks it as earlier
    win.method_combo.setCurrentIndex(list(PHASE_METHODS).index("carre"))
    win._refresh_stale_badge()
    assert win.stale_pill.isVisibleTo(win)

    # 3-D view
    win.mode_actions["measure"].trigger()
    win.view_tabs.setCurrentIndex(1)
    _pump(app, 0.3)


def test_cancel_and_manual_stepping(app, win):
    from phasescope.gui import SHIFTERS

    win.set_mode("measure")
    win.acquire_sequence()
    win.cancel_sequence()
    assert not win._recording
    _pump(app, 0.5)
    assert not win._recording

    # manual phase stepping: the prompt is on the live image, Space captures
    win.shifter_combo.setCurrentIndex(list(SHIFTERS).index("manual"))
    assert win._manual_active()
    assert "Step 1 of 4" in win._manual_prompt()
    win.last_result = None
    for k in range(4):
        win.source.set_phase_shift(k * np.pi / 2)
        _pump(app, 0.15)
        win.capture_manual_step()
        assert _wait_for(app, lambda: win._capture is None)
        if k == 1:
            win.undo_manual_step()
            assert len(win.manual_frames) == 1
            win.source.set_phase_shift(np.pi / 2)
            _pump(app, 0.15)
            win.capture_manual_step()
            assert _wait_for(app, lambda: win._capture is None)
    assert _wait_for(app, lambda: win.last_result is not None)
    assert win.manual_frames == []


def test_reference_area_levelling_and_quick_save(app, win, tmp_path):
    win.set_mode("measure")
    win.acquire_sequence()
    assert _wait_for(app, lambda: win.last_result is not None and not win._recording)
    win.fit_combo.setCurrentIndex(1)
    assert win.phase_cfg.level_region is not None
    assert win.ref_roi.isVisible()

    win.sample_edit.setText("wafer A")
    win.save_measurement(str(tmp_path / "meas"))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert "meas_raw.tif" in names and "meas_maps.npz" in names and "meas_info.txt" in names
    info = (tmp_path / "meas_info.txt").read_text(encoding="utf-8")
    assert "wafer A" in info and "phase_settings" in info


def test_view_modes(app, win):
    win.set_view_mode("live")
    assert not win.analysis_box.isVisible()
    win._escape()
    assert win.analysis_box.isVisible() and win.live_box.isVisible()
    win._toggle_single_view("analysis")
    assert not win.live_box.isVisible()
    win._toggle_single_view("analysis")
    assert win.live_box.isVisible()


# --- regressions found in review -------------------------------------------------------------------

def test_exposure_change_discards_manual_steps(app, win):
    from phasescope.gui import SHIFTERS

    win.set_mode("measure")
    win.shifter_combo.setCurrentIndex(list(SHIFTERS).index("manual"))
    win.capture_manual_step()
    assert _wait_for(app, lambda: win._capture is None)
    assert len(win.manual_frames) == 1
    win.exposure_spin.setValue(win.exposure_spin.value() * 2)
    assert win.manual_frames == []


def test_cancel_then_acquire_runs_exactly_one_sequence(app, win):
    win.settle_spin.setValue(300)
    win.set_mode("measure")
    finished = []
    win.thread.sequence_finished.connect(lambda seq, ok, reason: finished.append((seq, ok, reason)))
    win.acquire_sequence()
    _pump(app, 0.2)
    win.cancel_sequence()
    win.acquire_sequence()
    assert win._recording
    assert _wait_for(app, lambda: any(ok for _, ok, _ in finished), timeout=20)
    _pump(app, 0.3)
    assert not win._recording
    assert [ok for _, ok, _ in finished].count(True) == 1
    assert win.last_result is not None


def test_method_change_cancels_a_recording(app, win):
    from phasescope.gui import PHASE_METHODS

    win.settle_spin.setValue(300)
    win.set_mode("measure")
    win.acquire_sequence()
    _pump(app, 0.2)
    win.method_combo.setCurrentIndex(list(PHASE_METHODS).index("carre"))
    assert not win._recording


def test_method_off_shows_the_camera_image(app, win):
    from phasescope.gui import PHASE_METHODS

    win.set_mode("measure")
    win.acquire_sequence()
    assert _wait_for(app, lambda: win.last_result is not None and not win._recording)
    win.method_combo.setCurrentIndex(list(PHASE_METHODS).index("off"))
    assert win.channel_combo.currentData() == "processed"
    assert win.headline.text() == ""


def test_failed_camera_falls_back_but_keeps_the_choice(app, monkeypatch):
    from phasescope import gui as G
    from phasescope.camera import CameraError

    def fail(self):
        raise CameraError("no Thorlabs camera detected", kind="no_camera")

    monkeypatch.setattr(G.ThorlabsCamera, "open", fail)
    w = G.MainWindow(source="thorlabs", use_gl=False, autostart=True, restore_settings=False)
    w.show()
    try:
        assert _wait_for(app, lambda: w.last_packet is not None)
        assert isinstance(w.source, G.SimulatedInterferometer)
        assert w._source_key() == "thorlabs"  # Connect retries the user's camera
        assert w._fallback == ("thorlabs", "sim")
    finally:
        w.close()


def test_loading_bad_files_and_natural_order(app, win, tmp_path):
    from phasescope.gui import _natural_key, _safe_name, load_image

    names = [f"step {i}.tif" for i in (1, 10, 2)]
    assert sorted(names, key=_natural_key) == ["step 1.tif", "step 2.tif", "step 10.tif"]
    assert _safe_name("Glas ölbeschichtet / 0.5 µm") == "Glas-ölbeschichtet-0-5-µm"
    np.save(tmp_path / "bad.npy", np.arange(10))
    with pytest.raises(IOError):
        load_image(str(tmp_path / "bad.npy"))
    (tmp_path / "empty.png").write_bytes(b"")
    with pytest.raises(IOError):
        load_image(str(tmp_path / "empty.png"))
    np.save(tmp_path / "ok.npy", np.zeros((20, 30), bool))
    assert load_image(str(tmp_path / "ok.npy")).dtype == np.uint8


def test_disconnect_clears_the_analysis(app, win):
    win.disconnect_camera()
    assert win.analysis_panel.data() is None
    assert win._state == "disconnected"
    assert win.exposure_spin.specialValueText() == "–"


# --- sequence lifecycle while a map is being calculated ----------------------------------------------

@pytest.fixture
def slow_analysis(monkeypatch):
    import phasescope.acquisition as acq

    real = acq.analyse_stack

    def slow(frames, cfg):
        time.sleep(1.0)
        return real(frames, cfg)

    monkeypatch.setattr(acq, "analyse_stack", slow)


def _status(win):
    return win.statusBar().currentMessage()


def test_cancel_while_calculating_drops_the_map(app, win, slow_analysis):
    win.set_mode("measure")
    win.last_result = None
    win.acquire_sequence()
    assert _wait_for(app, lambda: win.thread.analysing)
    win._escape()
    assert not win._recording and "cancelled" in _status(win).lower()
    _pump(app, 1.6)
    assert win.last_result is None
    assert "finished" not in _status(win)


def test_pause_while_calculating_keeps_the_map(app, win, slow_analysis):
    win.set_mode("measure")
    win.last_result = None
    win.acquire_sequence()
    assert _wait_for(app, lambda: win.thread is not None and win.thread.analysing)
    win.act_live.trigger()  # pause
    assert win.thread is not None  # waits for the map
    assert _wait_for(app, lambda: win.thread is None, timeout=10)
    assert win.last_result is not None
    assert win._state == "paused"


def test_stop_continuous_and_continuous_errors(app, win):
    win.set_mode("measure")
    win.continuous_btn.setChecked(True)
    assert win._recording
    assert _wait_for(app, lambda: win.last_result is not None, timeout=15)
    win.continuous_btn.setChecked(False)
    assert not win._recording
    _pump(app, 1.0)
    assert not win._recording and "finished" not in _status(win)

    # a failing analysis keeps its reason visible (was replaced by 'Measurement cancelled')
    win.output_combo.setCurrentIndex(list(__import__("phasescope.gui", fromlist=["OUTPUT_LABELS"])
                                          .OUTPUT_LABELS).index("transmission"))
    win.nsample_spin.setValue(1.5)
    win.nmedium_spin.setValue(1.5)
    win.continuous_btn.setChecked(True)
    assert _wait_for(app, lambda: not win._recording, timeout=15)
    _pump(app, 0.5)
    assert "aborted" in _status(win).lower()


def test_manual_result_survives_a_settings_change(app, win, monkeypatch):
    import phasescope.gui as G
    from phasescope.gui import SHIFTERS

    real = G.analyse_stack

    def slow(frames, cfg):
        time.sleep(0.8)
        return real(frames, cfg)

    monkeypatch.setattr(G, "analyse_stack", slow)
    win.set_mode("measure")
    win.shifter_combo.setCurrentIndex(list(SHIFTERS).index("manual"))
    win.last_result = None
    for k in range(4):
        win.source.set_phase_shift(k * np.pi / 2)
        _pump(app, 0.12)
        win.capture_manual_step()
        assert _wait_for(app, lambda: win._capture is None)
    win.wavelength_spin.setValue(532.0)  # changed while the final map is calculated
    assert _wait_for(app, lambda: win.last_result is not None, timeout=10)
    assert _wait_for(app, lambda: win.last_result.config.wavelength_nm == 532.0, timeout=10)
