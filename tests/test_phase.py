import numpy as np
import pytest

from phasescope import phase as P
from phasescope.camera import SimulatedInterferometer


def _surface(h=96, w=128):
    yy, xx = np.mgrid[:h, :w].astype(float)
    return 9.0 * np.exp(-((xx - 0.4 * w) ** 2 + (yy - 0.5 * h) ** 2) / (2 * (0.2 * w) ** 2)) + 0.03 * xx


def _same_up_to_constant(a, b, tol):
    d = np.asarray(a, float) - np.asarray(b, float)
    return np.ptp(d[np.isfinite(d)]) < tol


@pytest.mark.parametrize("key", list(P.ALGORITHMS))
def test_phase_stepping_recovers_phase(key):
    phi = _surface()
    algo = P.ALGORITHMS[key]
    frames = [100 + 60 * np.cos(phi + d) for d in algo.shifts(7)]
    wrapped, mod, info = P.phase_from_steps(frames, key)
    err = P.wrap(wrapped - phi)
    err = P.wrap(err - err.mean())
    assert np.abs(err).max() < 1e-4
    assert np.allclose(np.median(mod), 60, rtol=1e-3)


def test_carre_measures_step():
    phi = _surface()
    step = np.radians(80)
    frames = [100 + 60 * np.cos(phi + k * step) for k in range(4)]
    _, _, info = P.phase_from_steps(frames, "carre")
    assert info["estimated_step_deg"] == pytest.approx(80, abs=0.5)


def test_wrong_frame_count_rejected():
    with pytest.raises(ValueError):
        P.phase_from_steps(np.zeros((3, 8, 8)), "4step")


@pytest.mark.parametrize("method", ["itoh_rows", "itoh_cols", "least_squares", "weighted_ls", "quality_guided"])
def test_unwrapping(method):
    if method == "quality_guided" and not P.skimage_available():
        pytest.skip("scikit-image not installed")
    phi = _surface()
    out = P.unwrap(P.wrap(phi), method, weights=np.ones_like(phi))
    assert _same_up_to_constant(out, phi, 1e-3)


def test_fourier_transform_method():
    h, w = 128, 160
    phi = _surface(h, w)
    yy, xx = np.mgrid[:h, :w]
    img = 100 + 60 * np.cos(phi + 2 * np.pi * (24 * xx / w + 6 * yy / h))
    assert P.find_carrier(img) == (24, 6)
    wrapped, mod, info = P.phase_from_fourier(img)
    assert info["carrier_px"] == (24, 6)
    u = P.remove_background(P.unwrap(wrapped, "least_squares"), "tilt")
    ref = P.remove_background(phi, "tilt")
    d = (u - ref)[16:-16, 16:-16]
    assert np.std(d) < 0.05


def test_remove_background_plane_and_quadratic():
    yy, xx = np.mgrid[:50, :60].astype(float)
    plane = 3 + 0.2 * xx - 0.1 * yy
    assert np.abs(P.remove_background(plane, "tilt")).max() < 1e-9
    quad = plane + 0.01 * xx ** 2 + 0.003 * xx * yy
    assert np.abs(P.remove_background(quad, "quadratic")).max() < 1e-8
    assert abs(P.remove_background(plane, "piston").mean()) < 1e-9


def test_physical_conversion():
    phi = np.full((2, 2), np.pi)
    h, q, u = P.to_physical(phi, "reflection", 600.0)
    assert (q, u) == ("Height", "nm") and np.allclose(h, 150.0)  # lambda/4 for pi
    t, _, _ = P.to_physical(phi, "transmission", 600.0, n_sample=1.5, n_medium=1.0)
    assert np.allclose(t, 600.0)  # pi = half wave of (n-1) t
    dn, _, _ = P.to_physical(phi, "index", 600.0, thickness_um=3.0)
    assert np.allclose(dn, 0.1)
    with pytest.raises(ValueError):
        P.to_physical(phi, "transmission", 600.0, n_sample=1.0, n_medium=1.0)


def test_simulator_end_to_end_height_accuracy():
    cam = SimulatedInterferometer(width=320, height=240, fps=0, animate=False, noise=0.005, seed=0)
    truth = P.remove_background(cam.height_map().astype(float), "tilt")
    cfg = P.PhaseConfig(method="4step", unwrap="least_squares", detrend="tilt", output="reflection")
    frames = []
    for d in cfg.shifts():
        cam.set_phase_shift(d)
        frames.append(cam.grab())
    res = P.analyse_stack(frames, cfg)
    err = res.physical - truth
    err -= np.nanmean(err)
    assert np.nanstd(err) < 5.0  # nm
    assert res.unit == "nm" and res.info["valid_fraction"] > 0.9

    cam.set_phase_shift(0.0)
    ftp = P.analyse_frame(cam.grab(), cfg.with_(method="ftp"))
    err = ftp.physical - truth
    err -= np.nanmedian(err)
    assert np.nanmedian(np.abs(err)) < 5.0


def test_analyse_stack_rejects_mixed_shapes():
    cfg = P.PhaseConfig(method="3step")
    with pytest.raises(ValueError):
        P.analyse_stack([np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((5, 4))], cfg)
