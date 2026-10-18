"""AcquisitionThread with the simulated interferometer (headless)."""

import os
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pg = pytest.importorskip("pyqtgraph")

from phasescope import acquisition as A  # noqa: E402
from phasescope.camera import (CameraError, CameraSource, SimulatedInterferometer,  # noqa: E402
                               SimulatedShifter, StaticImageSource)
from phasescope.phase import PhaseConfig  # noqa: E402
from phasescope.processing import Pipeline, as_float  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return pg.mkQApp()


def _wait_for(app, condition, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end and not condition():
        app.processEvents()
        time.sleep(0.005)
    return condition()


@pytest.fixture
def make_thread(app):
    threads = []

    def make(source, cfg=None, shifter=None):
        th = A.AcquisitionThread(source, Pipeline())
        th.config = cfg or PhaseConfig()
        th.shifter = shifter
        threads.append(th)
        return th

    yield make
    for th in threads:
        th.stop()
        app.processEvents()


def _sim(**kw):
    kw.setdefault("fps", 100)
    return SimulatedInterferometer(width=192, height=144, seed=0, **kw)


def test_live_fourier_phase_in_worker(app, make_thread):
    th = make_thread(_sim(), PhaseConfig(method="ftp"))
    assert th.last_frame_time == 0.0 and th.phase_rate == 0.0 and th.phase_error == ""
    th.start()
    frames, results = [], []

    def collect():
        pkt, res = th.take_frame(), th.take_phase()
        if pkt is not None:
            frames.append(pkt)
        if res is not None:
            results.append(res)
        return len(frames) >= 10 and len(results) >= 4

    assert _wait_for(app, collect), (len(frames), len(results))
    assert th.phase_rate > 0 and th.phase_error == ""
    assert th.last_frame_time > 0 and frames[-1].phase_error == ""
    res = results[-1]
    assert res.physical.shape == (144, 192) and "carrier_snr" in res.info and "contrast" in res.info
    worker = th._worker
    assert worker is not None and worker.daemon and worker.is_alive()
    th.stop()
    assert not th.isRunning()
    worker.join(2.0)  # stop() only signals the worker
    assert not worker.is_alive()


def test_slow_phase_does_not_slow_the_camera(app, make_thread, monkeypatch):
    real = A.analyse_frame

    def slow(*args, **kwargs):
        time.sleep(0.25)
        return real(*args, **kwargs)

    monkeypatch.setattr(A, "analyse_frame", slow)
    th = make_thread(_sim(fps=100), PhaseConfig(method="ftp"))
    th.start()
    results, last = [], [None]

    def collect():
        pkt, res = th.take_frame(), th.take_phase()
        if pkt is not None:
            last[0] = pkt
        if res is not None:
            results.append(res)
        return len(results) >= 3

    assert _wait_for(app, collect)
    # ~4 maps/s while the camera keeps running at (nearly) its own frame rate
    assert last[0].index > 5 * len(results)
    assert 0 < th.phase_rate < 6


def test_phase_error_reported_without_fringes(app, make_thread):
    flat = StaticImageSource(np.full((64, 80), 1000, np.uint16), fps=200)
    th = make_thread(flat, PhaseConfig(method="ftp"))
    th.start()
    assert _wait_for(app, lambda: th.phase_error != "")
    assert "fringes" in th.phase_error
    errors = []

    def packet_error():
        pkt = th.take_frame()
        if pkt is not None and pkt.phase_error:
            errors.append(pkt.phase_error)
        return bool(errors)

    assert _wait_for(app, packet_error) and errors[0] == th.phase_error
    assert th.take_phase() is None
    th.config = PhaseConfig(method="off")  # live phase off -> error cleared
    assert _wait_for(app, lambda: th.phase_error == "")


def _sequence_events(th):
    events = []
    th.sequence_finished.connect(lambda sid, ok, text: events.append((sid, ok, text)))
    return events


def _pump(app, seconds):
    _wait_for(app, lambda: False, seconds)


def _next(app, take, timeout=10.0):
    """The next non-None value of ``take()`` (th.take_frame / th.take_phase)."""
    end = time.time() + timeout
    while time.time() < end:
        value = take()
        if value is not None:
            return value
        app.processEvents()
        time.sleep(0.005)
    raise AssertionError(f"nothing from {take.__name__} within {timeout} s")


def _fresh_frame(app, th):
    """The next packet published after this call."""
    th.take_frame()
    return _next(app, th.take_frame)


def test_sequence_finished_ok(app, make_thread):
    cam = _sim(animate=False)
    th = make_thread(cam, PhaseConfig(method="4step", settle_ms=5), SimulatedShifter(cam))
    events = _sequence_events(th)
    progress = []
    th.sequence_progress.connect(lambda seq, k, n: progress.append((k, n)))
    th.start()
    sid = th.request_sequence()
    assert sid == th.sequence_id
    assert _wait_for(app, lambda: events)
    assert events == [(sid, True, "")] and not th.analysing
    assert progress[-1] == (4, 4)
    res = th.take_phase()
    assert res is not None and "4-step" in res.info["method"] and "contrast" in res.info
    assert np.isfinite(res.physical).mean() > 0.9
    assert th.request_sequence() == sid + 1


def test_sequence_cancelled(app, make_thread):
    cam = _sim(animate=False)
    th = make_thread(cam, PhaseConfig(method="4step", settle_ms=300), SimulatedShifter(cam))
    events = _sequence_events(th)
    progress = []
    th.sequence_progress.connect(lambda seq, k, n: progress.append(k))
    th.start()
    sid = th.request_sequence()
    assert _wait_for(app, lambda: progress)
    th.cancel_sequence()
    assert _wait_for(app, lambda: events)
    assert events == [(sid, False, "cancelled")]
    assert th.take_phase() is None and not th.continuous_stepping

    # continuous stepping after a cancel starts cleanly, with a new id
    th.config = th.config.with_(settle_ms=5)
    th.continuous_stepping = True
    cid = th.sequence_id
    assert cid == sid + 1
    assert _wait_for(app, lambda: (cid, True, "") in events)
    th.continuous_stepping = True  # already on: same id
    assert th.sequence_id == cid
    th.continuous_stepping = False


@pytest.mark.parametrize("gap_s", [0.0, 0.05])
def test_cancel_then_acquire_runs_one_new_sequence(app, make_thread, gap_s):
    cam = _sim(animate=False)
    th = make_thread(cam, PhaseConfig(method="4step", settle_ms=150), SimulatedShifter(cam))
    events = _sequence_events(th)
    progress = []
    th.sequence_progress.connect(lambda seq, k, n: progress.append(k))
    th.start()
    old = th.request_sequence()
    assert _wait_for(app, lambda: progress)
    th.cancel_sequence()
    if gap_s:
        _pump(app, gap_s)
    new = th.request_sequence()
    assert new == old + 1
    assert _wait_for(app, lambda: len(events) >= 2)
    del progress[:]
    _pump(app, 1.0)  # no hidden second sequence afterwards
    assert events == [(old, False, "cancelled"), (new, True, "")]
    assert progress == [] and not th.analysing
    pkt = th.take_frame()
    assert pkt is not None and pkt.step == ""


def test_cancel_before_the_sequence_starts(app, make_thread):
    cam = _sim(animate=False, fps=4)  # the thread sits in grab() for 250 ms
    th = make_thread(cam, PhaseConfig(method="4step", settle_ms=5), SimulatedShifter(cam))
    events = _sequence_events(th)
    progress = []
    th.sequence_progress.connect(lambda seq, k, n: progress.append(k))
    th.start()
    _next(app, th.take_frame)
    sid = th.request_sequence()
    th.cancel_sequence()
    assert _wait_for(app, lambda: events)
    _pump(app, 0.6)
    assert events == [(sid, False, "cancelled")] and progress == []


def _slow_stack(monkeypatch, seconds):
    """Make analyse_stack slow; returns [running, max running] counters."""
    real = A.analyse_stack
    counts = [0, 0]
    lock = threading.Lock()

    def slow(*args, **kwargs):
        with lock:
            counts[0] += 1
            counts[1] = max(counts)
        try:
            time.sleep(seconds)
            return real(*args, **kwargs)
        finally:
            with lock:
                counts[0] -= 1

    monkeypatch.setattr(A, "analyse_stack", slow)
    return counts


def test_frames_keep_coming_while_a_sequence_is_analysed(app, make_thread, monkeypatch):
    _slow_stack(monkeypatch, 1.0)
    cam = _sim(animate=False)
    th = make_thread(cam, PhaseConfig(method="4step", settle_ms=5), SimulatedShifter(cam))
    events = _sequence_events(th)
    th.start()
    sid = th.request_sequence()
    assert _wait_for(app, lambda: th.analysing)
    th.take_frame()
    live = []

    def collect():
        pkt = th.take_frame()
        if pkt is not None and th.analysing:
            live.append(pkt)
        return not th.analysing

    assert _wait_for(app, collect)
    # ~100 fps camera: the live view keeps running during the 1 s analysis
    assert len(live) > 20 and all(p.step == "" for p in live)
    assert live[-1].index - live[0].index > 20 and th.last_frame_time > 0
    assert _wait_for(app, lambda: events)
    assert events == [(sid, True, "")]
    assert th.take_phase() is not None


def test_continuous_waits_for_the_previous_analysis(app, make_thread, monkeypatch):
    counts = _slow_stack(monkeypatch, 0.3)
    cam = _sim(animate=False)
    th = make_thread(cam, PhaseConfig(method="4step", settle_ms=5), SimulatedShifter(cam))
    events = _sequence_events(th)
    th.start()
    th.continuous_stepping = True
    cid = th.sequence_id
    assert _wait_for(app, lambda: len(events) >= 3)
    th.cancel_sequence()
    assert _wait_for(app, lambda: events[-1][1:] == (False, "cancelled"))
    n = len(events)
    _pump(app, 0.8)
    assert len(events) == n and counts[1] == 1  # one analysis at a time
    assert all(e == (cid, True, "") for e in events[:-1])  # in order: results, then the cancel
    assert events[-1] == (cid, False, "cancelled") and not th.analysing


def test_analysis_error_reported(app, make_thread, monkeypatch):
    def broken(frames, cfg):
        raise ValueError("phase-step frames have different shapes")

    monkeypatch.setattr(A, "analyse_stack", broken)
    cam = _sim(animate=False)
    th = make_thread(cam, PhaseConfig(method="4step", settle_ms=5), SimulatedShifter(cam))
    events = _sequence_events(th)
    th.start()
    th.continuous_stepping = True
    cid = th.sequence_id
    assert _wait_for(app, lambda: events)
    assert events[0] == (cid, False, "ValueError: phase-step frames have different shapes")
    assert not th.continuous_stepping and th.last_error.startswith("ValueError")
    _pump(app, 0.3)
    assert all(not ok for _, ok, _ in events) and th.take_phase() is None


def test_stop_during_analysis_is_fast_and_silent(app, make_thread, monkeypatch):
    _slow_stack(monkeypatch, 1.0)
    cam = _sim(animate=False)
    th = make_thread(cam, PhaseConfig(method="4step", settle_ms=5), SimulatedShifter(cam))
    events = _sequence_events(th)
    th.start()
    th.request_sequence()
    assert _wait_for(app, lambda: th.analysing)
    t0 = time.perf_counter()
    th.stop()
    assert time.perf_counter() - t0 < 0.5 and not th.isRunning() and not th.analysing
    _pump(app, 1.3)  # the discarded analysis has ended by now
    assert events == [] and th.take_phase() is None
    assert not any(t.name == "PhaseScope stepping analysis" for t in threading.enumerate())


def test_stop_is_fast_during_live_phase(app, make_thread, monkeypatch):
    real = A.analyse_frame
    busy = threading.Event()

    def slow(*args, **kwargs):
        busy.set()
        time.sleep(0.6)
        return real(*args, **kwargs)

    monkeypatch.setattr(A, "analyse_frame", slow)
    th = make_thread(_sim(), PhaseConfig(method="ftp"))
    th.start()
    assert busy.wait(5.0)
    worker = th._worker
    t0 = time.perf_counter()
    th.stop()
    assert time.perf_counter() - t0 < 0.3 and not th.isRunning()
    worker.join(2.0)
    assert not worker.is_alive() and th.take_phase() is None  # the map was discarded


def test_hold_stops_live_maps(app, make_thread, monkeypatch):
    real = A.analyse_frame
    calls = [0]

    def counted(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(A, "analyse_frame", counted)
    th = make_thread(_sim(), PhaseConfig(method="ftp"))
    assert th.hold is False
    th.start()
    _next(app, th.take_phase)
    th.hold = True
    _pump(app, 0.3)  # a map already being computed is dropped
    th.take_phase()
    n, index = calls[0], _fresh_frame(app, th).index
    _pump(app, 0.8)
    assert calls[0] == n and th.take_phase() is None
    assert _fresh_frame(app, th).index > index + 20  # the camera view keeps running
    th.hold = False
    _next(app, th.take_phase)


class _Counter(CameraSource):
    """Frame k is filled with k: easy to tell an average from the current frame."""

    def __init__(self):
        self.k = 0

    def grab(self):
        time.sleep(0.005)
        self.k += 1
        return np.full((12, 16), self.k, np.uint16)


def test_processed_step_excludes_temporal_steps(app, make_thread):
    th = make_thread(_Counter())
    th.start()
    pkt = _fresh_frame(app, th)
    assert pkt.processed_step is pkt.processed  # no temporal step

    th.step_frames = True
    _fresh_frame(app, th)
    pkt = _fresh_frame(app, th)
    assert pkt.processed_step is pkt.processed  # still none: nothing extra to compute

    th.pipeline.add("temporal_average", {"mode": "sliding window", "frames": 10})
    _pump(app, 0.3)
    pkt = _fresh_frame(app, th)
    assert np.array_equal(pkt.processed_step, as_float(pkt.raw))  # this frame only
    assert pkt.processed.mean() < pkt.raw.mean() - 3  # the average lags behind

    th.step_frames = False
    _fresh_frame(app, th)
    pkt = _fresh_frame(app, th)
    assert pkt.processed_step is pkt.processed


class _BrokenShifter(SimulatedShifter):
    def set_phase(self, delta_rad):
        if delta_rad:
            raise CameraError("piezo controller is not connected")


def test_sequence_error_text(app, make_thread):
    cam = _sim(animate=False)
    th = make_thread(cam, PhaseConfig(method="4step", settle_ms=0), _BrokenShifter(cam))
    events = _sequence_events(th)
    th.start()
    sid = th.request_sequence()
    assert _wait_for(app, lambda: events)
    assert events == [(sid, False, "piezo controller is not connected")]
    assert th.last_error == "piezo controller is not connected"


class _DeadCamera(CameraSource):
    def grab(self):
        raise CameraError("camera unplugged")


def test_stopped_on_error(app, make_thread):
    th = make_thread(_DeadCamera())
    th.MAX_CONSECUTIVE_ERRORS = 2
    stopped, errors = [], []
    th.stopped_on_error.connect(stopped.append)
    th.error.connect(errors.append)
    th.start()
    assert _wait_for(app, lambda: stopped)
    assert stopped == ["camera unplugged"] and th.last_error == "camera unplugged"
    assert errors.count("camera unplugged") == 2
    assert th.wait(2000) and th.last_frame_time == 0.0


def test_stop_before_run_starts(app):
    th = A.AcquisitionThread(_sim(), Pipeline())
    th.start()
    th.stop()
    assert not th.isRunning()
