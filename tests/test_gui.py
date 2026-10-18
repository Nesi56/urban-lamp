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
    assert _safe_name("Glas ölbeschichtet / 0.5 µm") == "Glas-ölbeschichtet-0.5-µm"
    assert _safe_name("नमूना 1") == "नमूना-1"  # vowel signs (marks) are kept
    assert _safe_name("  film_1.2 / ") == "film_1.2"
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


def _slow(monkeypatch, module, name, seconds):
    real = getattr(module, name)

    def slow(*args, **kwargs):
        time.sleep(seconds)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, slow)


def _measure_once(app, win):
    win.set_mode("measure")
    win.last_result = None
    win.acquire_sequence()
    assert _wait_for(app, lambda: win.last_result is not None and not win._recording, timeout=15)


def test_a_setting_changed_during_a_recalculation_is_applied(app, win, monkeypatch):
    import phasescope.gui as G

    _measure_once(app, win)
    win.pause_live()
    std0 = float(np.nanstd(win.last_result.physical))
    lam0 = win.last_result.config.wavelength_nm
    _slow(monkeypatch, G, "analyse", 0.6)
    win.wavelength_spin.setValue(700.0)
    assert _wait_for(app, lambda: win._calc_busy)
    assert win.stale_pill.isVisibleTo(win) and "Recalculating" in win.stale_pill.text()
    win.wavelength_spin.setValue(800.0)  # while the 700 nm recalculation runs
    assert _wait_for(app, lambda: not win._calc_busy and not win._calc_reanalyse
                     and win.last_result.config.wavelength_nm == 800.0, timeout=10)
    _wait_for(app, lambda: not win._calc_busy, timeout=3)
    assert np.isclose(np.nanstd(win.last_result.physical), std0 * 800.0 / lam0, rtol=0.01)
    assert _status(win) != "Calculating…"
    assert not win.stale_pill.isVisibleTo(win)


def test_a_failed_recalculation_marks_the_map(app, win):
    from phasescope.gui import OUTPUT_LABELS

    _measure_once(app, win)
    win.pause_live()
    win.output_combo.setCurrentIndex(list(OUTPUT_LABELS).index("transmission"))
    win.nsample_spin.setValue(1.5)
    win.nmedium_spin.setValue(1.0)
    assert _wait_for(app, lambda: win.last_result.config.output == "transmission" and not win._calc_busy)
    win.nsample_spin.setValue(1.0)  # equal indices: the thickness cannot be computed
    assert _wait_for(app, lambda: win._recalc_error != "", timeout=5)
    assert "Not recalculated" in win.stale_pill.text() and win.stale_pill.isVisibleTo(win)
    assert win._result_is_stale().startswith("not recalculated")
    win.nsample_spin.setValue(1.6)
    assert _wait_for(app, lambda: win._recalc_error == "" and not win._calc_busy, timeout=5)
    assert win._result_is_stale() == ""


def test_stop_continuous_stops_the_piezo(app, win):
    win.set_mode("measure")
    win.settle_spin.setValue(300)
    calls = []
    shifter = win.shifter
    real = shifter.set_phase
    shifter.set_phase = lambda d: (calls.append((time.monotonic(), d)), real(d))
    win.continuous_btn.setChecked(True)
    assert _wait_for(app, lambda: win._seq_progress[0] >= 1, timeout=10)
    win._primary_clicked()  # 'Stop continuous'
    t_stop = time.monotonic()
    assert not win._recording
    _pump(app, 1.5)
    later = [d for t, d in calls if t > t_stop + 0.05]
    assert all(d == 0.0 for d in later), later  # no further phase steps, only the reset to 0
    assert _wait_for(app, lambda: win.last_packet is not None and win.last_packet.step == "", timeout=3)


def test_camera_error_while_calculating_keeps_the_map(app, win, monkeypatch):
    import phasescope.acquisition as acq
    from phasescope.camera import CameraError

    _slow(monkeypatch, acq, "analyse_stack", 2.5)
    win.set_mode("measure")
    win.last_result = None
    win.acquire_sequence()
    assert _wait_for(app, lambda: win.thread is not None and win.thread.analysing, timeout=10)
    win.pause_live()  # wants to pause after the map
    assert win._pause_after_seq

    def broken():
        raise CameraError("USB transfer failed")

    win.source.grab = broken
    assert _wait_for(app, lambda: win.thread is None, timeout=10)
    assert not win._pause_after_seq
    assert _wait_for(app, lambda: win.last_result is not None, timeout=10)
    assert "finished" in _status(win)


def test_empty_hint_follows_shifter_and_recording(app, win):
    from phasescope.gui import SHIFTERS

    win.set_mode("measure")
    win.last_result = None
    win._update_analysis(force=True)
    hint = win.analysis_panel.message_label.text
    win.shifter_combo.setCurrentIndex(list(SHIFTERS).index("manual"))
    assert "Manual stepping" in hint() and "Step 1 of 4" in hint()
    win.shifter_combo.setCurrentIndex(list(SHIFTERS).index("serial"))
    assert "not connected" in hint()
    assert "Phase" in _status(win) and "Connect" in _status(win)  # tells where to connect it
    win.shifter_combo.setCurrentIndex(list(SHIFTERS).index("sim"))
    win.acquire_sequence()
    assert "Recording" in hint() or "Calculating" in hint()


def test_simulator_ignores_the_hidden_device_box(app, win):
    from phasescope.gui import SOURCES

    _measure_once(app, win)
    win.source_combo.setCurrentIndex(list(SOURCES).index("opencv"))
    win.device_combo.setEditText("1")
    win.source_combo.setCurrentIndex(list(SOURCES).index("sim"))
    win.connect_camera()
    assert win._connected_device == ""
    assert win._result_is_stale() != "different camera"


def test_live_image_main_button_names_the_chosen_method(app, win):
    win.set_mode("image")
    win.algo_combo.setCurrentIndex(win.algo_combo.findData("carre"))
    assert "Carré" in win.primary_btn.text()


def test_multipage_tiff_is_a_sequence_and_named_in_the_metadata(app, win, tmp_path):
    import cv2

    if not hasattr(cv2, "imwritemulti"):
        pytest.skip("OpenCV without multi-page TIFF support")
    frames = []
    for k in range(4):
        win.source.set_phase_shift(k * np.pi / 2)
        frames.append(win.source.grab())
    path = str(tmp_path / "stack.tif")
    assert cv2.imwritemulti(path, frames)
    win.load_images([path])
    assert win.last_result is not None and win.last_result.config.method == "4step"
    meta = win._metadata()
    assert meta["source"] == "files: stack.tif" and meta["exposure_ms"] is None


def test_replacing_a_measurement_removes_its_old_files(app, win, tmp_path, monkeypatch):
    from pyqtgraph.Qt import QtWidgets

    _measure_once(app, win)
    win.settings = None
    base = str(tmp_path / "wafer7")
    win.save_measurement(base)
    assert os.path.exists(base + "_maps.npz")
    win.last_result = None  # the replacement has no map
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (base + "_info.txt", "")))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))
    win.save_measurement_as()
    assert os.path.exists(base + "_info.txt") and not os.path.exists(base + "_info.txt_info.txt")
    assert not os.path.exists(base + "_maps.npz") and not os.path.exists(base + "_map.tif")


def test_save_dialog_adds_the_type_to_names_with_a_dot(app, win, monkeypatch):
    from pyqtgraph.Qt import QtWidgets

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("/data/Si_0.5um", "TIFF (*.tif)")))
    assert win._save_dialog("t", "x", "TIFF (*.tif);;PNG (*.png)") == "/data/Si_0.5um.tif"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("/data/frame.tiff", "TIFF (*.tif)")))
    assert win._save_dialog("t", "x", "TIFF (*.tif);;PNG (*.png)") == "/data/frame.tiff"


def test_manual_capture_skips_frames_not_split_from_temporal_steps(app, win):
    from phasescope.acquisition import FramePacket

    win._capture = {"target": 1, "got": 0, "acc": None, "not_before": 0.0, "skip": 0, "after": -1}
    img = np.zeros((8, 8), np.float32)
    win._feed_manual_capture(FramePacket(img, img, 5, 0.0, 0.0, step_exact=False))
    assert win._capture is not None and win._capture["got"] == 0
    win._capture = None


def test_continuous_maps_carry_the_processing_of_their_sequence(app, win):
    win.set_mode("measure")
    win.continuous_btn.setChecked(True)
    assert _wait_for(app, lambda: win.last_result is not None, timeout=15)
    win.pipeline.add("gaussian", {"sigma": 1.0})
    win._refresh_pipeline_list()
    assert _wait_for(app, lambda: any(k == "gaussian" for k, _, _ in
                                      win.last_result.info.get("sequence_settings", (None, (), False))[1]),
                     timeout=20)
    assert win._result_is_stale() == ""
    assert win._metadata()["result_pipeline"][0]["key"] == "gaussian"
    win.continuous_btn.setChecked(False)
