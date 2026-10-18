"""Background acquisition thread.

The thread grabs frames continuously, crops them to the optional region of
interest, runs the processing pipeline and - depending on the phase settings -
hands frames to a live Fourier-transform phase worker or drives a
phase-stepping sequence, whose analysis runs in a separate worker thread so
the camera keeps streaming meanwhile.  The GUI *pulls* the most recent results with
:meth:`take_frame` and :meth:`take_phase` from a timer, so a slow display never
makes frames pile up in memory.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from pyqtgraph.Qt import QtCore

from .camera import CameraError, CameraSource, PhaseShifter
from .phase import PhaseConfig, PhaseResult, analyse_frame, analyse_stack
from .processing import Pipeline


@dataclass
class FramePacket:
    raw: np.ndarray
    processed: np.ndarray
    index: int
    timestamp: float
    fps: float
    step: str = ""  # "2/4" while a phase-stepping sequence is recorded
    phase_error: str = ""
    # processed without temporal steps (for manual phase steps, see AcquisitionThread.step_frames);
    # the same array as ``processed`` when no temporal step ran
    processed_step: np.ndarray | None = None


def _error_text(exc: BaseException) -> str:
    """Readable text for an exception (camera errors are already plain language)."""
    text = str(exc).strip()
    if isinstance(exc, CameraError) and text:
        return text
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


class PhaseWorker(threading.Thread):
    """Computes live Fourier-transform phase maps next to the acquisition thread.

    The mailbox holds one job: :meth:`offer` replaces a job that has not started
    yet, so the worker always analyses the newest frame and never builds a backlog,
    while the camera view keeps its own frame rate.
    """

    def __init__(self, publish: Callable[[PhaseResult], None],
                 wanted: Callable[[], bool] = lambda: True) -> None:
        super().__init__(name="PhaseScope phase worker", daemon=True)
        self._publish = publish
        self._wanted = wanted  # checked before publishing (drops results nobody wants)
        self._cond = threading.Condition()
        self._job: tuple[np.ndarray, np.ndarray | None, PhaseConfig] | None = None
        self._stopping = False
        self.error = ""  # text of the last failed analysis, '' after a success
        self._rate = 0.0
        self._t_last: float | None = None

    def offer(self, processed: np.ndarray, raw: np.ndarray | None, cfg: PhaseConfig) -> None:
        with self._cond:
            self._job = (processed, raw, cfg)
            self._cond.notify()

    def discard(self) -> None:
        """Drop a pending job and forget the last error (live phase switched off)."""
        with self._cond:
            self._job = None
            self.error = ""

    def stop(self, timeout: float = 0.0) -> None:
        """Ask the worker to end; it never publishes afterwards.  ``timeout`` > 0 also joins it."""
        with self._cond:
            self._stopping = True
            self._job = None
            self._cond.notify()
        if timeout > 0 and self.is_alive() and threading.current_thread() is not self:
            self.join(timeout)

    @staticmethod
    def _stale_after(rate: float) -> float:
        return max(2.0, 3.0 / rate) if rate > 0 else float("inf")

    @property
    def rate(self) -> float:
        """Phase maps per second (moving average); 0 once maps stop coming."""
        rate, t_last = self._rate, self._t_last
        if t_last is None or time.monotonic() - t_last > self._stale_after(rate):
            return 0.0
        return rate

    def _tick(self) -> None:
        now = time.monotonic()
        if self._t_last is not None:
            dt = now - self._t_last
            if dt > self._stale_after(self._rate):  # first map after a pause
                self._rate = 0.0
            elif dt > 0:
                self._rate = 1.0 / dt if self._rate == 0 else 0.8 * self._rate + 0.2 / dt
        self._t_last = now

    def run(self) -> None:
        while True:
            with self._cond:
                while self._job is None and not self._stopping:
                    self._cond.wait()
                if self._stopping:
                    return
                (processed, raw, cfg), self._job = self._job, None
            try:
                result = analyse_frame(processed, cfg, raw=raw)
            except Exception as exc:  # e.g. no carrier fringes in this frame
                self.error = str(exc) or type(exc).__name__
                continue
            self.error = ""
            with self._cond:  # atomic with stop(): no result is published once it returned
                if self._stopping:
                    return
                self._tick()
                if self._wanted():
                    self._publish(result)


class StackWorker(threading.Thread):
    """Analyses one recorded phase-stepping sequence next to the acquisition thread."""

    def __init__(self, frames: list[np.ndarray], cfg: PhaseConfig, seq_id: int,
                 done: Callable[["StackWorker", PhaseResult | None, str], None]) -> None:
        super().__init__(name="PhaseScope stepping analysis", daemon=True)
        self.frames, self.cfg, self.seq_id = frames, cfg, seq_id
        self._done = done  # (worker, result or None, error text)

    def run(self) -> None:
        try:
            result, error = analyse_stack(self.frames, self.cfg), ""
        except Exception as exc:
            result, error = None, _error_text(exc)
        self.frames = []
        self._done(self, result, error)


class AcquisitionThread(QtCore.QThread):
    error = QtCore.Signal(str)
    sequence_progress = QtCore.Signal(int, int, int)  # (seq_id, steps recorded, steps in the sequence)
    # (seq_id, ok, reason): (id, True, "") analysed and published; (id, False, "cancelled") after
    # cancel_sequence() or stop(); (id, False, error text).  Emitted in recording order.
    sequence_finished = QtCore.Signal(int, bool, str)
    stopped_on_error = QtCore.Signal(str)  # last error text; the thread has given up

    MAX_CONSECUTIVE_ERRORS = 5
    WORKER_JOIN_S = 0.0  # stop() only signals the live phase worker; it drops its last map
    ANALYSIS_POLL_S = 0.05

    def __init__(self, source: CameraSource, pipeline: Pipeline, parent=None) -> None:
        super().__init__(parent)
        self.source = source
        self.pipeline = pipeline
        self.shifter: PhaseShifter | None = None
        self.config = PhaseConfig()  # replaced atomically by the GUI
        self.roi: tuple[int, int, int, int] | None = None  # x, y, w, h on the raw frame
        self.last_error = ""
        self.last_frame_time = 0.0  # time.monotonic() of the last successful grab
        self.hold = False  # the GUI holds the map: no live phase maps are computed
        self.step_frames = False  # also publish frames processed without temporal steps
        self._continuous = False
        self._running = False
        self._stop_requested = False
        self._sequence_requested = False
        self._cancel_sequence = False
        self._seq_lock = threading.Lock()  # sequence requests, cancels and starts
        self._seq_id = 0
        self._active_id: int | None = None  # sequence being recorded
        self._unstarted_cancel: int | None = None  # cancelled before it started: report it
        self._finish_lock = threading.Lock()  # analysis hand-off and order of sequence_finished
        self._analysis: StackWorker | None = None
        self._deferred: list[tuple[int, bool, str]] = []  # reports waiting for the analysis
        self._commands: queue.Queue[Callable[[], None]] = queue.Queue()
        self._lock = threading.Lock()
        self._frame: FramePacket | None = None
        self._phase: PhaseResult | None = None
        self._worker: PhaseWorker | None = None
        self._index = 0
        self._fps = 0.0
        self._t_last = None

    # -- called from the GUI thread ---------------------------------------
    @property
    def continuous_stepping(self) -> bool:
        return self._continuous

    @continuous_stepping.setter
    def continuous_stepping(self, on: bool) -> None:
        """Switching on starts a new sequence id, carried by all continuous sequences."""
        with self._seq_lock:
            if on and not self._continuous:
                self._seq_id += 1
            self._continuous = bool(on)

    @property
    def sequence_id(self) -> int:
        """Id the next (or the currently recorded) sequence reports in sequence_finished."""
        return self._seq_id

    @property
    def analysing(self) -> bool:
        """True while a recorded phase-stepping sequence is being analysed."""
        return self._analysis is not None

    @property
    def phase_error(self) -> str:
        """Why the last live phase map failed ('' when it succeeded)."""
        w = self._worker
        return w.error if w is not None else ""

    @property
    def phase_rate(self) -> float:
        """Live phase maps per second (exponential moving average)."""
        w = self._worker
        return w.rate if w is not None else 0.0

    def submit(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` in the acquisition thread between two frames."""
        self._commands.put(fn)

    def request_sequence(self) -> int:
        """Ask for one phase-stepping sequence; returns the id its sequence_finished carries."""
        with self._seq_lock:
            self._seq_id += 1
            self._sequence_requested = True
            return self._seq_id

    def cancel_sequence(self) -> None:
        """Abort the sequence being recorded, drop a pending request, stop continuous stepping.

        The aborted sequence - or the requested one that had not started yet - reports
        ``(id, False, "cancelled")``; an analysis already running still completes."""
        with self._seq_lock:
            if self._active_id is None and (self._sequence_requested or self._continuous):
                self._unstarted_cancel = self._seq_id
            self._cancel_sequence = True
            self._sequence_requested = False
            self._continuous = False

    def start(self, *args, **kwargs) -> None:
        self._stop_requested = False
        super().start(*args, **kwargs)

    def stop(self, timeout_ms: int = 3000) -> None:
        with self._finish_lock:
            self._stop_requested = True  # also when run() has not started yet
            self._analysis = None  # a running analysis discards its result
        self._running = False
        self._cancel_sequence = True
        self.wait(timeout_ms)
        self._stop_worker()

    def take_frame(self) -> FramePacket | None:
        with self._lock:
            pkt, self._frame = self._frame, None
        return pkt

    def take_phase(self) -> PhaseResult | None:
        with self._lock:
            res, self._phase = self._phase, None
        return res

    # -- acquisition thread -----------------------------------------------
    def run(self) -> None:
        self._running = not self._stop_requested
        errors = 0
        try:
            while self._running:
                self._run_commands()
                seq_id = None
                try:
                    seq_id = self._next_sequence()
                    if seq_id is not None:
                        self._run_sequence(seq_id)
                    else:
                        self._live_frame()
                    errors = 0
                except Exception as exc:
                    errors += 1
                    message = _error_text(exc)
                    self._report(message)
                    if seq_id is not None:
                        self._end_sequences(seq_id)
                        self._finish(seq_id, False, message)
                    if errors >= self.MAX_CONSECUTIVE_ERRORS:
                        self.error.emit("acquisition stopped after repeated errors")
                        self.stopped_on_error.emit(self.last_error)
                        break
                    time.sleep(0.2)
        finally:
            self._running = False
            self._stop_worker()
            with self._finish_lock:  # reports that waited for an analysis stop() discarded
                if self._analysis is None:
                    self._flush_deferred()

    def _report(self, message: str) -> None:
        self.last_error = message
        self._emit("error", message)

    def _emit(self, signal: str, *args) -> None:
        try:
            getattr(self, signal).emit(*args)
        except RuntimeError:  # the QObject has been deleted (window closed)
            pass

    # -- phase-stepping bookkeeping ------------------------------------------
    def _next_sequence(self) -> int | None:
        """Id of the sequence to record now, None for a live frame.

        Consuming the request and clearing the cancel flag happen under the lock that
        cancel_sequence() takes, so a cancel is never lost or applied to a later request."""
        shifter = self.shifter
        stepping = self.config.is_stepping and shifter is not None and shifter.automatic
        seq_id = None
        with self._seq_lock:
            unstarted, self._unstarted_cancel = self._unstarted_cancel, None
            if stepping and (self._sequence_requested or self._continuous):
                self._sequence_requested = False
                self._cancel_sequence = False
                self._active_id = seq_id = self._seq_id
        if unstarted is not None and unstarted != seq_id:
            self._finish(unstarted, False, "cancelled")
        return seq_id

    def _end_sequences(self, seq_id: int) -> None:
        """After a failure of ``seq_id``: like cancel_sequence(), unless a newer one was requested."""
        with self._seq_lock:
            if self._seq_id == seq_id:
                self._cancel_sequence = True
                self._sequence_requested = False
                self._continuous = False

    def _emit_finished(self, seq_id: int, ok: bool, reason: str) -> None:
        self._emit("sequence_finished", seq_id, ok, reason)

    def _flush_deferred(self) -> None:
        deferred, self._deferred = self._deferred, []
        for args in deferred:
            self._emit_finished(*args)

    def _finish(self, seq_id: int, ok: bool, reason: str) -> None:
        """Report a sequence from the acquisition thread - after a running analysis reported."""
        with self._finish_lock:
            if self._analysis is not None:
                self._deferred.append((seq_id, ok, reason))
            else:
                self._emit_finished(seq_id, ok, reason)

    def _hand_off(self, frames: list[np.ndarray], cfg: PhaseConfig, seq_id: int) -> None:
        """Analyse a recorded sequence in a StackWorker, after the previous analysis finished."""
        prev = self._analysis
        while prev is not None and prev.is_alive() and self._running and not self._cancel_sequence:
            prev.join(self.ANALYSIS_POLL_S)
        if self._sequence_cancelled(seq_id):
            return
        with self._finish_lock:
            if self._stop_requested:
                return
            job = self._analysis = StackWorker(frames, cfg, seq_id, self._analysis_done)
        job.start()

    def _analysis_done(self, job: StackWorker, result: PhaseResult | None, error: str) -> None:
        """Runs in the StackWorker: publish and report, unless stop() came first."""
        with self._finish_lock:
            if self._analysis is not job:
                return
            try:
                if result is not None:
                    result.info["sequence_id"] = job.seq_id  # lets the GUI drop maps of cancelled runs
                    self._publish_phase(result)
                    self._emit_finished(job.seq_id, True, "")
                else:
                    self._end_sequences(job.seq_id)
                    self._report(error)
                    self._emit_finished(job.seq_id, False, error)
                self._flush_deferred()
            finally:
                self._analysis = None

    def _run_commands(self) -> None:
        while True:
            try:
                fn = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception as exc:
                self._report(_error_text(exc))

    _abandoned: list[PhaseWorker] = []  # workers told to stop that may still finish a big frame

    def _ensure_worker(self) -> PhaseWorker:
        w = self._worker
        if w is None or not w.is_alive():
            for old in list(AcquisitionThread._abandoned):  # never run several heavy analyses at once
                old.join(5.0)
                if not old.is_alive():
                    AcquisitionThread._abandoned.remove(old)
            w = PhaseWorker(self._publish_phase, lambda: self.config.method == "ftp" and not self.hold)
            w.start()
            self._worker = w
        return w

    def _stop_worker(self) -> None:
        w, self._worker = self._worker, None
        if w is not None:
            w.stop(self.WORKER_JOIN_S)
            if w.is_alive():
                AcquisitionThread._abandoned.append(w)

    def _grab(self) -> np.ndarray | None:
        raw = self.source.grab()
        if raw is None:
            return None
        self.last_frame_time = time.monotonic()
        now = time.perf_counter()
        if self._t_last is not None:
            dt = now - self._t_last
            if dt > 0:
                self._fps = 1.0 / dt if self._fps == 0 else 0.9 * self._fps + 0.1 / dt
        self._t_last = now
        self._index += 1
        return raw

    def _crop(self, raw: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return raw
        x, y, w, h = self.roi
        sub = raw[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w]
        return sub if sub.size else raw

    def _publish(self, raw, processed, step: str = "", phase_error: str = "",
                 processed_step: np.ndarray | None = None) -> None:
        pkt = FramePacket(raw, processed, self._index, time.time(), self._fps, step, phase_error,
                          processed if processed_step is None else processed_step)
        with self._lock:
            self._frame = pkt

    def _publish_phase(self, result: PhaseResult) -> None:
        with self._lock:
            self._phase = result

    def _live_frame(self) -> None:
        raw = self._grab()
        if raw is None:
            return
        cropped = self._crop(raw)
        if self.step_frames and self._temporal_active():
            processed, step = self.pipeline.apply_split(cropped)
        else:
            processed = step = self.pipeline.apply(cropped)
        cfg = self.config
        live_phase = cfg.method == "ftp" and not self.hold
        self._publish(raw, processed, phase_error=self.phase_error if live_phase else "",
                      processed_step=step)
        worker = self._worker
        if live_phase:
            if self._running:
                self._ensure_worker().offer(processed, cropped, cfg)
        elif worker is not None:
            worker.discard()

    def _temporal_active(self) -> bool:
        p = self.pipeline
        return not p.bypass and any(s.enabled and s.op.temporal for s in p.steps)

    def _sequence_cancelled(self, seq_id: int) -> bool:
        if self._cancel_sequence or not self._running:
            self._finish(seq_id, False, "cancelled")
            return True
        return False

    def _run_sequence(self, seq_id: int) -> None:
        try:
            cfg = self.config
            frames = self._record_sequence(cfg, seq_id)
            if frames is not None:
                self._hand_off(frames, cfg, seq_id)
        finally:
            with self._seq_lock:
                self._active_id = None

    def _record_sequence(self, cfg: PhaseConfig, seq_id: int) -> list[np.ndarray] | None:
        """Step the phase and average the frames of each step; None when cancelled."""
        shifter = self.shifter
        if shifter is None:
            raise CameraError("No phase shifter is connected.")
        shifts = cfg.shifts()
        frames: list[np.ndarray] = []
        try:
            for k, delta in enumerate(shifts):
                if self._sequence_cancelled(seq_id):
                    return None
                shifter.set_phase(float(delta))
                if cfg.settle_ms > 0:
                    time.sleep(cfg.settle_ms / 1000.0)
                for _ in range(cfg.discard_frames):  # frames exposed while the piezo moved
                    if self.source.grab() is not None:
                        self.last_frame_time = time.monotonic()
                acc, got, misses = None, 0, 0
                while got < max(1, cfg.frames_per_step):
                    raw = self._grab()
                    if raw is None:
                        misses += 1
                        if misses > 10:
                            raise CameraError("The camera stopped delivering images during the "
                                              "phase-stepping sequence (timeout).")
                        continue
                    p = self.pipeline.apply(self._crop(raw), include_temporal=False)
                    acc = p.astype(np.float64) if acc is None else acc + p
                    got += 1
                    self._publish(raw, p, step=f"{k + 1}/{len(shifts)}")
                frames.append((acc / got).astype(np.float32))
                self._emit("sequence_progress", seq_id, k + 1, len(shifts))
        finally:
            try:
                shifter.set_phase(0.0)
            except Exception:
                pass
        return None if self._sequence_cancelled(seq_id) else frames
