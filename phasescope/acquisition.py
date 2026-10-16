"""Background acquisition thread.

The thread grabs frames continuously, crops them to the optional region of
interest, runs the processing pipeline and - depending on the phase settings -
computes a live Fourier-transform phase map or drives a phase-stepping
sequence.  The GUI *pulls* the most recent results with :meth:`take_frame`
and :meth:`take_phase` from a timer, so a slow display never makes frames
pile up in memory.
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


class AcquisitionThread(QtCore.QThread):
    error = QtCore.Signal(str)
    sequence_progress = QtCore.Signal(int, int)
    sequence_finished = QtCore.Signal(bool)  # True = completed, False = aborted

    MAX_CONSECUTIVE_ERRORS = 5

    def __init__(self, source: CameraSource, pipeline: Pipeline, parent=None) -> None:
        super().__init__(parent)
        self.source = source
        self.pipeline = pipeline
        self.shifter: PhaseShifter | None = None
        self.config = PhaseConfig()  # replaced atomically by the GUI
        self.roi: tuple[int, int, int, int] | None = None  # x, y, w, h on the raw frame
        self.continuous_stepping = False
        self._running = False
        self._sequence_requested = False
        self._cancel_sequence = False
        self._commands: queue.Queue[Callable[[], None]] = queue.Queue()
        self._lock = threading.Lock()
        self._frame: FramePacket | None = None
        self._phase: PhaseResult | None = None
        self._index = 0
        self._fps = 0.0
        self._t_last = None

    # -- called from the GUI thread ---------------------------------------
    def submit(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` in the acquisition thread between two frames."""
        self._commands.put(fn)

    def request_sequence(self) -> None:
        self._cancel_sequence = False
        self._sequence_requested = True

    def cancel_sequence(self) -> None:
        self._cancel_sequence = True
        self._sequence_requested = False
        self.continuous_stepping = False

    def stop(self, timeout_ms: int = 3000) -> None:
        self._running = False
        self._cancel_sequence = True
        self.wait(timeout_ms)

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
        self._running = True
        errors = 0
        while self._running:
            self._run_commands()
            in_sequence = False
            try:
                stepping = self.config.is_stepping and self.shifter is not None and self.shifter.automatic
                if stepping and (self._sequence_requested or self.continuous_stepping):
                    self._sequence_requested = False
                    in_sequence = True
                    self._run_sequence()
                else:
                    self._live_frame()
                errors = 0
            except Exception as exc:
                errors += 1
                self.error.emit(f"{type(exc).__name__}: {exc}")
                if in_sequence:
                    self.cancel_sequence()
                    self.sequence_finished.emit(False)
                if errors >= self.MAX_CONSECUTIVE_ERRORS:
                    self.error.emit("acquisition stopped after repeated errors")
                    break
                time.sleep(0.2)
        self._running = False

    def _run_commands(self) -> None:
        while True:
            try:
                fn = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception as exc:
                self.error.emit(f"{type(exc).__name__}: {exc}")

    def _grab(self) -> np.ndarray | None:
        raw = self.source.grab()
        if raw is None:
            return None
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

    def _publish(self, raw, processed, step: str = "", phase_error: str = "") -> None:
        pkt = FramePacket(raw, processed, self._index, time.time(), self._fps, step, phase_error)
        with self._lock:
            self._frame = pkt

    def _publish_phase(self, result: PhaseResult) -> None:
        with self._lock:
            self._phase = result

    def _live_frame(self) -> None:
        raw = self._grab()
        if raw is None:
            return
        processed = self.pipeline.apply(self._crop(raw))
        cfg = self.config
        phase_error = ""
        if cfg.method == "ftp":
            try:
                self._publish_phase(analyse_frame(processed, cfg))
            except Exception as exc:  # e.g. no carrier fringes in this frame
                phase_error = str(exc)
        self._publish(raw, processed, phase_error=phase_error)

    def _run_sequence(self) -> None:
        cfg = self.config
        shifts = cfg.shifts()
        frames: list[np.ndarray] = []
        try:
            for k, delta in enumerate(shifts):
                if self._cancel_sequence or not self._running:
                    self.sequence_finished.emit(False)
                    return
                self.shifter.set_phase(float(delta))
                if cfg.settle_ms > 0:
                    time.sleep(cfg.settle_ms / 1000.0)
                for _ in range(cfg.discard_frames):  # frames exposed while the piezo moved
                    self.source.grab()
                acc, got, misses = None, 0, 0
                while got < max(1, cfg.frames_per_step):
                    raw = self._grab()
                    if raw is None:
                        misses += 1
                        if misses > 10:
                            raise CameraError("camera timed out during phase stepping")
                        continue
                    p = self.pipeline.apply(self._crop(raw), include_temporal=False)
                    acc = p.astype(np.float64) if acc is None else acc + p
                    got += 1
                    self._publish(raw, p, step=f"{k + 1}/{len(shifts)}")
                frames.append((acc / got).astype(np.float32))
                self.sequence_progress.emit(k + 1, len(shifts))
        finally:
            try:
                self.shifter.set_phase(0.0)
            except Exception:
                pass
        self._publish_phase(analyse_stack(frames, cfg))
        self.sequence_finished.emit(True)
