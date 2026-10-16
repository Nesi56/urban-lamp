"""Phase retrieval (phase stepping and Fourier-transform method), phase
unwrapping, background removal and conversion of phase to physical units.

Conventions
-----------
A phase-stepped interferogram is modelled as::

    I_k(x, y) = A(x, y) + B(x, y) * cos(phi(x, y) + delta_k)

``delta_k`` are the phase shifts listed by each algorithm.  Wrapped phases are
returned in the interval [-pi, pi].
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from scipy import fft as sfft

TWO_PI = 2.0 * np.pi


def wrap(phi: np.ndarray) -> np.ndarray:
    """Wrap a phase into [-pi, pi]."""
    return phi - TWO_PI * np.round(phi / TWO_PI)


# --------------------------------------------------------------------------
# Phase-stepping algorithms
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StepAlgorithm:
    key: str
    name: str
    n_frames: int | None  # None: user chooses N (N-step least squares)
    shifts_fn: Callable[[int], np.ndarray]
    func: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray, dict]]
    description: str = ""

    def shifts(self, n: int | None = None) -> np.ndarray:
        """Phase shifts (radians) the phase shifter must apply, one per frame."""
        return self.shifts_fn(self.n_frames or int(n or 4))

    def frames(self, n: int | None = None) -> int:
        return len(self.shifts(n))


def _equal_shifts(n: int) -> np.ndarray:
    return TWO_PI * np.arange(n) / n


def _stack(frames) -> np.ndarray:
    st = np.asarray(frames, dtype=np.float64)
    if st.ndim != 3:
        raise ValueError("phase stepping needs a stack of 2-D frames with equal shape")
    return st


def _three_step(st: np.ndarray, n: int):
    i1, i2, i3 = st
    num, den = np.sqrt(3.0) * (i3 - i2), 2.0 * i1 - i2 - i3
    return np.arctan2(num, den), np.hypot(num, den) / 3.0, {}


def _four_step(st: np.ndarray, n: int):
    i1, i2, i3, i4 = st
    num, den = i4 - i2, i1 - i3
    return np.arctan2(num, den), np.hypot(num, den) / 2.0, {}


def _hariharan(st: np.ndarray, n: int):
    i1, i2, i3, i4, i5 = st
    num, den = 2.0 * (i4 - i2), i1 - 2.0 * i3 + i5
    return np.arctan2(num, den), np.hypot(num, den) / 4.0, {}


def _carre(st: np.ndarray, n: int):
    """Carré: four frames with equal but unknown step (nominally 90 deg).

    The phase is referred to the middle of the sequence.  The actual step is
    estimated per pixel, which makes this a good tool for calibrating a
    piezo phase shifter (``info['estimated_step_deg']``).
    """
    i1, i2, i3, i4 = st
    a, b = i1 - i4, i2 - i3
    p, q = 3.0 * b - a, a + b
    num = np.sqrt(np.abs(p * q)) * np.sign(b)
    den = (i2 + i3) - (i1 + i4)
    phi = np.arctan2(num, den)
    with np.errstate(divide="ignore", invalid="ignore"):
        tan2 = p / q
        alpha = np.arctan(np.sqrt(np.where(tan2 > 0, tan2, np.nan)))  # half step
        amp = np.hypot(num, den)
        good = np.isfinite(alpha) & (amp > np.nanpercentile(amp, 50))
        step = 2.0 * float(np.nanmedian(alpha[good])) if np.any(good) else np.pi / 2
    s = np.sin(step / 2.0)
    norm = 8.0 * s * s * np.cos(step / 2.0)
    mod = amp / norm if abs(norm) > 1e-6 else amp
    return phi, mod, {"estimated_step_deg": float(np.degrees(step))}


def _n_step(st: np.ndarray, n: int):
    shifts = _equal_shifts(st.shape[0])
    s = np.tensordot(np.sin(shifts), st, axes=1)
    c = np.tensordot(np.cos(shifts), st, axes=1)
    return np.arctan2(-s, c), (2.0 / st.shape[0]) * np.hypot(s, c), {}


ALGORITHMS: dict[str, StepAlgorithm] = {a.key: a for a in (
    StepAlgorithm("3step", "3-step (120°)", 3, _equal_shifts, _three_step,
                  "Fastest algorithm; sensitive to phase-shifter mis-calibration."),
    StepAlgorithm("4step", "4-step (90°)", 4, _equal_shifts, _four_step,
                  "Classic algorithm; good default."),
    StepAlgorithm("5step", "5-step Hariharan (90°)", 5,
                  lambda n: np.arange(5) * np.pi / 2, _hariharan,
                  "Insensitive to linear phase-shift calibration errors."),
    StepAlgorithm("carre", "Carré (4 frames, unknown step)", 4,
                  lambda n: np.arange(4) * np.pi / 2, _carre,
                  "Works with any constant step; reports the measured step."),
    StepAlgorithm("nstep", "N-step least squares", None, _equal_shifts, _n_step,
                  "N equally spaced steps over 2π (N = 3…16); averages noise."),
)}


def phase_from_steps(frames, algorithm: str = "4step") -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(wrapped_phase, modulation, info)`` from a stack of frames."""
    st = _stack(frames)
    algo = ALGORITHMS[algorithm]
    if algo.n_frames is not None and st.shape[0] != algo.n_frames:
        raise ValueError(f"{algo.name} needs {algo.n_frames} frames, got {st.shape[0]}")
    if st.shape[0] < 3:
        raise ValueError("at least 3 frames are required")
    phi, mod, info = algo.func(st, st.shape[0])
    return wrap(phi).astype(np.float32), mod.astype(np.float32), info


# --------------------------------------------------------------------------
# Fourier-transform (Takeda) method - single frame, suited to live imaging
# --------------------------------------------------------------------------

def _freq_index(n: int) -> np.ndarray:
    """Integer FFT frequency index (0, 1, ..., -1) for an axis of length n."""
    return np.fft.fftfreq(n, 1.0 / n).round().astype(np.int64)


def _carrier_from_spectrum(spec: np.ndarray, dc_exclude: int) -> tuple[int, int]:
    h, w = spec.shape
    ky = _freq_index(h)[:, None]
    kx = _freq_index(w)[None, :]
    mag = np.abs(spec)
    blocked = (kx * kx + ky * ky) <= dc_exclude * dc_exclude
    blocked = blocked | (kx < 0) | ((kx == 0) & (ky <= 0))  # keep one half-plane
    mag[blocked] = 0
    iy, ix = np.unravel_index(int(np.argmax(mag)), mag.shape)
    return int(kx[0, ix]), int(ky[iy, 0])


def find_carrier(img: np.ndarray, dc_exclude: int = 4) -> tuple[int, int]:
    """Locate the +1 order of a carrier-fringe pattern.

    Returns ``(kx, ky)``: position of the side-lobe in FFT bins, i.e. the number
    of fringes across the field of view along x and y.
    """
    img = np.asarray(img, dtype=np.float32)
    return _carrier_from_spectrum(sfft.fft2(img - img.mean(), workers=-1), dc_exclude)


def phase_from_fourier(img: np.ndarray, carrier: tuple[int, int] | None = None,
                       filter_radius: float = 0.5, dc_exclude: int = 4
                       ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Single-frame phase retrieval (Takeda) for interferograms with tilt (carrier) fringes.

    ``filter_radius`` is the radius of the side-lobe window as a fraction of the
    carrier distance (0.5 = half way to the zero order).
    """
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape
    spec = sfft.fft2(img - img.mean(), workers=-1)
    if carrier is None or tuple(carrier) == (0, 0):
        carrier = _carrier_from_spectrum(spec, dc_exclude)
    kx, ky = int(carrier[0]), int(carrier[1])
    kr = float(np.hypot(kx, ky))
    if kr < 1:
        raise ValueError("no carrier fringes found - tilt the reference mirror or use phase stepping")
    # distance to the side-lobe on the periodic frequency grid
    dy = (_freq_index(h)[:, None] - ky + h // 2) % h - h // 2
    dx = (_freq_index(w)[None, :] - kx + w // 2) % w - w // 2
    r = max(1.0, filter_radius * kr)
    d2 = (dx * dx + dy * dy).astype(np.float32) / np.float32(r * r)
    window = np.exp(-np.log(2.0) * d2 * d2)  # super-Gaussian, 0.5 at radius r
    side = np.roll(spec * window, shift=(-ky, -kx), axis=(0, 1))  # remove the carrier
    field_ = sfft.ifft2(side, workers=-1)
    info = {"carrier_px": (kx, ky), "fringe_period_px": float(1.0 / np.hypot(kx / w, ky / h))}
    return np.angle(field_).astype(np.float32), (2.0 * np.abs(field_)).astype(np.float32), info


# --------------------------------------------------------------------------
# Phase unwrapping
# --------------------------------------------------------------------------

UNWRAP_METHODS: dict[str, str] = {
    "none": "None (show wrapped phase)",
    "itoh_rows": "Itoh path-following (rows, then columns)",
    "itoh_cols": "Itoh path-following (columns, then rows)",
    "least_squares": "Least-squares, DCT (Ghiglia–Romero)",
    "weighted_ls": "Weighted least-squares, PCG (modulation weighted, slower)",
    "quality_guided": "Quality-guided, reliability sorting (Herráez)",
}


def _wrapped_divergence(psi: np.ndarray, wx=None, wy=None) -> np.ndarray:
    dx = wrap(np.diff(psi, axis=1))
    dy = wrap(np.diff(psi, axis=0))
    if wx is not None:
        dx, dy = dx * wx, dy * wy
    return _div(dx, dy, psi.shape)


def _div(dx: np.ndarray, dy: np.ndarray, shape) -> np.ndarray:
    rho = np.zeros(shape, dtype=np.float64)
    rho[:, :-1] += dx
    rho[:, 1:] -= dx
    rho[:-1, :] += dy
    rho[1:, :] -= dy
    return rho


def _poisson_neumann(rho: np.ndarray) -> np.ndarray:
    """Solve the discrete Poisson equation (Neumann boundaries) with a DCT."""
    m, n = rho.shape
    r_hat = sfft.dctn(rho, type=2, norm="ortho", workers=-1)
    i = np.arange(m)[:, None]
    j = np.arange(n)[None, :]
    denom = 2.0 * (np.cos(np.pi * i / m) + np.cos(np.pi * j / n) - 2.0)
    denom[0, 0] = 1.0
    phi_hat = r_hat / denom
    phi_hat[0, 0] = 0.0
    return sfft.idctn(phi_hat, type=2, norm="ortho", workers=-1)


def unwrap_least_squares(psi: np.ndarray) -> np.ndarray:
    """Unweighted least-squares unwrapping (Ghiglia & Romero, JOSA A 1994)."""
    return _poisson_neumann(_wrapped_divergence(np.asarray(psi, dtype=np.float64)))


def unwrap_weighted_least_squares(psi: np.ndarray, weights: np.ndarray,
                                  max_iter: int = 25, tol: float = 1e-3) -> np.ndarray:
    """Weighted least-squares unwrapping solved by DCT-preconditioned conjugate gradients."""
    psi = np.asarray(psi, dtype=np.float64)
    w = np.clip(np.nan_to_num(np.asarray(weights, dtype=np.float64)), 0.0, None) ** 2
    if w.max() <= 0:
        return unwrap_least_squares(psi)
    w /= w.max()
    wx = np.minimum(w[:, 1:], w[:, :-1])
    wy = np.minimum(w[1:, :], w[:-1, :])

    def apply_a(p):  # A = -(weighted Laplacian), positive semi-definite
        return -_div(np.diff(p, axis=1) * wx, np.diff(p, axis=0) * wy, p.shape)

    b = -_wrapped_divergence(psi, wx, wy)
    phi = np.zeros_like(psi)
    r = b.copy()
    b_norm = np.linalg.norm(b) or 1.0
    z = -_poisson_neumann(r)
    p = z.copy()
    rz = float(np.vdot(r, z))
    for _ in range(max_iter):
        ap = apply_a(p)
        pap = float(np.vdot(p, ap))
        if pap <= 0:
            break
        alpha = rz / pap
        phi += alpha * p
        r -= alpha * ap
        if np.linalg.norm(r) < tol * b_norm:
            break
        z = -_poisson_neumann(r)
        rz_new = float(np.vdot(r, z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    return phi


def skimage_available() -> bool:
    try:
        import skimage.restoration  # noqa: F401
        return True
    except Exception:
        return False


def unwrap(psi: np.ndarray, method: str = "least_squares", mask: np.ndarray | None = None,
           weights: np.ndarray | None = None) -> np.ndarray:
    """Unwrap a wrapped phase map.

    ``mask`` marks *valid* pixels (True) - used by the quality-guided method.
    ``weights`` (e.g. fringe modulation) are used by the weighted least-squares method.
    """
    psi = np.asarray(psi, dtype=np.float64)
    if method == "none":
        return psi
    if method == "itoh_rows":
        return np.unwrap(np.unwrap(psi, axis=1), axis=0)
    if method == "itoh_cols":
        return np.unwrap(np.unwrap(psi, axis=0), axis=1)
    if method == "least_squares":
        return unwrap_least_squares(psi)
    if method == "weighted_ls":
        w = weights if weights is not None else np.ones_like(psi)
        if mask is not None:
            w = np.where(mask, w, 0.0)
        return unwrap_weighted_least_squares(psi, w)
    if method == "quality_guided":
        try:
            from skimage.restoration import unwrap_phase
        except ImportError as exc:
            raise RuntimeError("quality-guided unwrapping needs scikit-image "
                               "(pip install scikit-image)") from exc
        if mask is not None and not mask.all():
            out = unwrap_phase(np.ma.array(psi, mask=~mask))
            return np.ma.filled(out.astype(np.float64), np.nan)
        return unwrap_phase(psi)
    raise ValueError(f"unknown unwrapping method {method!r}")


# --------------------------------------------------------------------------
# Background (piston / tilt / curvature) removal
# --------------------------------------------------------------------------

DETREND_MODES: dict[str, str] = {
    "none": "None",
    "piston": "Piston (mean)",
    "tilt": "Piston + tilt (plane)",
    "quadratic": "Piston + tilt + curvature",
}


def remove_background(phi: np.ndarray, mode: str = "tilt", mask: np.ndarray | None = None,
                      max_samples: int = 40000) -> np.ndarray:
    """Subtract a least-squares polynomial surface fitted to the valid pixels."""
    if mode == "none":
        return phi
    h, w = phi.shape
    valid = np.isfinite(phi) if mask is None else (mask & np.isfinite(phi))
    if valid.sum() < 6:
        return phi
    if mode == "piston":
        return phi - float(np.mean(phi[valid]))
    ys, xs = np.nonzero(valid)
    if ys.size > max_samples:
        sel = np.random.default_rng(0).choice(ys.size, max_samples, replace=False)
        ys, xs = ys[sel], xs[sel]
    # normalised coordinates in [-1, 1]
    xn = np.linspace(-1.0, 1.0, w) if w > 1 else np.zeros(1)
    yn = np.linspace(-1.0, 1.0, h) if h > 1 else np.zeros(1)
    x, y = xn[xs], yn[ys]
    cols = [np.ones_like(x), x, y]
    if mode == "quadratic":
        cols += [x * x, x * y, y * y]
    c, *_ = np.linalg.lstsq(np.stack(cols, axis=-1), phi[ys, xs], rcond=None)
    X, Y = xn[None, :], yn[:, None]
    surface = c[0] + c[1] * X + c[2] * Y
    if mode == "quadratic":
        surface = surface + c[3] * X * X + c[4] * X * Y + c[5] * Y * Y
    return phi - surface


# --------------------------------------------------------------------------
# Conversion to physical quantities
# --------------------------------------------------------------------------

OUTPUT_MODES: dict[str, str] = {
    "phase": "Phase (rad)",
    "opd": "Optical path difference",
    "reflection": "Surface height (reflection)",
    "transmission": "Thickness (transmission, from Δn)",
    "index": "Refractive-index change (known thickness)",
}


def to_physical(phi: np.ndarray, mode: str, wavelength_nm: float = 632.8,
                n_sample: float = 1.5, n_medium: float = 1.0,
                thickness_um: float = 1.0) -> tuple[np.ndarray, str, str]:
    """Convert unwrapped phase to ``(values, quantity, unit)``."""
    lam = float(wavelength_nm)
    if mode == "phase":
        return phi, "Phase", "rad"
    if mode == "opd":
        return phi * lam / TWO_PI, "Optical path difference", "nm"
    if mode == "reflection":  # double pass: phase = 4 pi h / lambda
        return phi * lam / (2.0 * TWO_PI), "Height", "nm"
    if mode == "transmission":  # phase = 2 pi (n_s - n_m) t / lambda
        dn = float(n_sample - n_medium)
        if abs(dn) < 1e-9:
            raise ValueError("n_sample and n_medium must differ for thickness conversion")
        return phi * lam / (TWO_PI * dn), "Thickness", "nm"
    if mode == "index":  # phase = 2 pi dn t / lambda
        t_nm = float(thickness_um) * 1000.0
        if t_nm <= 0:
            raise ValueError("thickness must be positive")
        return phi * lam / (TWO_PI * t_nm), "Δn", ""
    raise ValueError(f"unknown output mode {mode!r}")


# --------------------------------------------------------------------------
# Complete analysis
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseConfig:
    method: str = "off"  # "off", "ftp" or a key of ALGORITHMS
    n_steps: int = 4  # for N-step least squares
    unwrap: str = "least_squares"
    detrend: str = "tilt"
    modulation_threshold: float = 0.05  # fraction of the 99th percentile modulation
    output: str = "reflection"
    wavelength_nm: float = 632.8
    n_sample: float = 1.50
    n_medium: float = 1.00
    thickness_um: float = 1.0
    ftp_filter_radius: float = 0.5
    ftp_carrier: tuple[int, int] = (0, 0)  # (0, 0) = find automatically
    ftp_dc_exclude: int = 4
    # acquisition of phase-stepped sequences
    settle_ms: float = 30.0
    discard_frames: int = 0
    frames_per_step: int = 1

    @property
    def is_stepping(self) -> bool:
        return self.method in ALGORITHMS

    def shifts(self) -> np.ndarray:
        return ALGORITHMS[self.method].shifts(self.n_steps)

    def with_(self, **kw) -> "PhaseConfig":
        return replace(self, **kw)


@dataclass
class PhaseResult:
    wrapped: np.ndarray
    unwrapped: np.ndarray
    physical: np.ndarray  # detrended, converted, invalid pixels = NaN
    modulation: np.ndarray
    mask: np.ndarray  # True = valid
    quantity: str
    unit: str
    info: dict = field(default_factory=dict)
    config: PhaseConfig | None = None  # settings the result was computed with
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        parts = [self.info.get("method", "")]
        if "pv" in self.info:
            u = f" {self.unit}" if self.unit else ""
            parts.append(f"PV {self.info['pv']:.4g}{u}")
            parts.append(f"RMS {self.info['rms']:.4g}{u}")
        if "estimated_step_deg" in self.info:
            parts.append(f"measured step {self.info['estimated_step_deg']:.1f}°")
        if "carrier_px" in self.info:
            parts.append(f"carrier {self.info['carrier_px']}")
        parts.append(f"valid {100 * self.info.get('valid_fraction', 1):.0f}%")
        parts.append(f"{1000 * self.info.get('seconds', 0):.0f} ms")
        return "  |  ".join(p for p in parts if p)


def analyse(wrapped: np.ndarray, modulation: np.ndarray, cfg: PhaseConfig,
            info: dict | None = None) -> PhaseResult:
    t0 = time.perf_counter()
    info = dict(info or {})
    sample = modulation[:: max(1, modulation.shape[0] // 256), :: max(1, modulation.shape[1] // 256)]
    ref = float(np.nanpercentile(sample, 99)) or 1.0
    mask = modulation >= cfg.modulation_threshold * ref
    unwrapped = unwrap(wrapped, cfg.unwrap, mask=mask, weights=modulation / ref)
    if cfg.unwrap == "none":
        base = np.asarray(wrapped, dtype=np.float64)
    else:
        base = remove_background(unwrapped, cfg.detrend, mask)
    values, quantity, unit = to_physical(base, cfg.output, cfg.wavelength_nm, cfg.n_sample,
                                         cfg.n_medium, cfg.thickness_um)
    physical = np.where(mask, values, np.nan).astype(np.float32)
    finite = physical[np.isfinite(physical)]
    if finite.size:
        info["pv"] = float(finite.max() - finite.min())
        info["rms"] = float(np.std(finite))
    info["valid_fraction"] = float(mask.mean())
    info["seconds"] = info.get("seconds", 0.0) + time.perf_counter() - t0
    return PhaseResult(np.asarray(wrapped, np.float32), np.asarray(unwrapped, np.float32),
                       physical, np.asarray(modulation, np.float32), mask, quantity, unit, info, cfg)


def analyse_frame(img: np.ndarray, cfg: PhaseConfig) -> PhaseResult:
    """Fourier-transform method on a single interferogram."""
    t0 = time.perf_counter()
    carrier = cfg.ftp_carrier if cfg.ftp_carrier != (0, 0) else None
    wrapped, mod, info = phase_from_fourier(img, carrier, cfg.ftp_filter_radius, cfg.ftp_dc_exclude)
    info.update(method="Fourier transform (single frame)", seconds=time.perf_counter() - t0)
    return analyse(wrapped, mod, cfg, info)


def analyse_stack(frames, cfg: PhaseConfig) -> PhaseResult:
    """Phase-stepping analysis of a recorded sequence."""
    t0 = time.perf_counter()
    shapes = {np.shape(f) for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"phase-step frames have different shapes {sorted(shapes)} - "
                         "did the processing pipeline change during the sequence?")
    wrapped, mod, info = phase_from_steps(frames, cfg.method)
    info.update(method=ALGORITHMS[cfg.method].name, seconds=time.perf_counter() - t0)
    return analyse(wrapped, mod, cfg, info)
