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


def _carrier_image(h=128, w=160, a=100.0, b=60.0, fx=24, fy=6, noise=0.0, seed=0):
    yy, xx = np.mgrid[:h, :w]
    img = a + b * np.cos(_surface(h, w) + 2 * np.pi * (fx * xx / w + fy * yy / h))
    return img + np.random.default_rng(seed).normal(0, noise, img.shape)


def test_carrier_snr_separates_fringes_from_noise():
    _, _, info = P.phase_from_fourier(_carrier_image(noise=2.0))
    assert info["carrier_snr"] > 50
    noise = np.random.default_rng(1).normal(100, 10, (128, 160))
    _, _, info = P.phase_from_fourier(noise)
    assert info["carrier_snr"] < 20
    _, _, info = P.phase_from_fourier(noise, carrier=(24, 6))  # locked carrier on noise
    assert info["carrier_snr"] < 20


def test_contrast_frame_and_stack():
    cfg = P.PhaseConfig(method="ftp", output="phase")
    img = _carrier_image(a=100.0, b=60.0)
    res = P.analyse_frame(img, cfg)
    assert res.info["contrast"] == pytest.approx(0.6, abs=0.05)
    # level from the raw frame when it has the processed shape ...
    res = P.analyse_frame(img, cfg, raw=2 * img)
    assert res.info["contrast"] == pytest.approx(0.3, abs=0.03)
    # ... otherwise from the processed image
    res = P.analyse_frame(img, cfg, raw=np.ones((5, 5)))
    assert res.info["contrast"] == pytest.approx(0.6, abs=0.05)
    # no contrast when the level is not an intensity (background subtracted)
    assert "contrast" not in P.analyse_frame(img - np.median(img) - 1.0, cfg).info
    assert "contrast" not in P.analyse_frame(img - img.mean(), cfg).info

    phi = _surface()
    scfg = P.PhaseConfig(method="4step", output="phase")
    frames = [100 + 30 * np.cos(phi + d) for d in scfg.shifts()]
    assert P.analyse_stack(frames, scfg).info["contrast"] == pytest.approx(0.3, rel=1e-3)
    frames = [30 * np.cos(phi + d) for d in scfg.shifts()]
    assert "contrast" not in P.analyse_stack(frames, scfg).info


@pytest.mark.parametrize("info, level, words", [
    ({}, "unknown", ""),
    ({"carrier_snr": 300, "fringe_period_px": 10.66, "contrast": 0.78}, "good", "10.7 px apart, contrast 78 %"),
    ({"contrast": 0.78}, "good", "Fringe contrast 78 %"),
    ({"carrier_snr": float("inf"), "fringe_period_px": 8.0}, "good", "Fringes OK"),
    ({"carrier_snr": 5.0, "fringe_period_px": 10.0, "contrast": 0.8}, "poor", "mostly noise"),
    ({"carrier_snr": 35.0, "fringe_period_px": 10.0, "contrast": 0.8}, "fair", "usable"),
    ({"carrier_snr": 300, "fringe_period_px": 133.3}, "poor", "133 px apart"),
    ({"carrier_snr": 300, "fringe_period_px": 2.4}, "poor", "2.4 px apart"),
    ({"carrier_snr": 300, "fringe_period_px": 25.0}, "fair", "25.0 px"),
    ({"carrier_snr": 300, "fringe_period_px": 4.0}, "fair", "4.0 px"),
    ({"contrast": 0.09}, "poor", "9 %"),
    ({"contrast": 0.2}, "fair", "20 %"),
    # the worst metric wins
    ({"carrier_snr": 35.0, "fringe_period_px": 10.0, "contrast": 0.05}, "poor", "contrast (5 %)"),
])
def test_fringe_quality(info, level, words):
    got_level, message = P.fringe_quality(info)
    assert got_level == level
    assert words in message
    if level in ("fair",):
        assert message.startswith("Fringes usable")


def test_fringe_quality_on_real_results():
    cfg = P.PhaseConfig(method="ftp")
    assert P.fringe_quality(P.analyse_frame(_carrier_image(noise=1.0), cfg).info)[0] == "good"
    noise = np.random.default_rng(2).normal(100, 10, (128, 160))
    assert P.fringe_quality(P.analyse_frame(noise, cfg).info)[0] == "poor"


def _film_on_tilt(h=96, w=128, step=1.2):
    yy, xx = np.mgrid[:h, :w].astype(float)
    tilt = 0.05 * xx + 0.03 * yy + 0.4
    film = np.where(xx >= w // 2, step, 0.0)  # film over the right half of the field
    return tilt + film


def test_level_region_gives_true_step():
    h, w, step = 96, 128, 1.2
    phi = _film_on_tilt(h, w, step)
    cfg = P.PhaseConfig(method="4step", unwrap="least_squares", detrend="tilt", output="phase")
    sub = (slice(None), slice(4, w // 2 - 4))  # substrate, away from the edge
    film = (slice(None), slice(w // 2 + 4, w - 4))

    whole = P.analyse(P.wrap(phi), np.ones_like(phi), cfg)  # whole-image fit tilts the film
    assert np.ptp(whole.physical[sub]) > 0.3
    assert "levelled_on" not in whole.info

    res = P.analyse(P.wrap(phi), np.ones_like(phi), cfg.with_(level_region=(0, 0, w // 2 - 4, h)))
    assert res.info["levelled_on"] == "reference area" and "level_note" not in res.info
    assert np.abs(res.physical[sub]).max() < 0.02  # substrate flat at zero
    assert np.abs(res.physical[film] - step).max() < 0.02  # film levelled everywhere: true step
    assert "levelled on reference area" in res.summary()

    # region partly outside the map is clipped
    res = P.analyse(P.wrap(phi), np.ones_like(phi), cfg.with_(level_region=(-50, -50, w // 2 + 46, 500)))
    assert res.info["levelled_on"] == "reference area"
    assert np.abs(res.physical[film] - step).max() < 0.02

    # too few valid pixels -> whole image, with a note
    for region in [(w + 10, 0, 20, 20), (0, 0, 2, 2)]:
        res = P.analyse(P.wrap(phi), np.ones_like(phi), cfg.with_(level_region=region))
        assert "levelled_on" not in res.info and "too few valid pixels" in res.info["level_note"]
        assert np.allclose(res.physical, whole.physical, atol=1e-5)


def test_result_fields_and_summary():
    cfg = P.PhaseConfig(method="ftp", output="reflection")
    res = P.analyse_frame(_carrier_image(), cfg)
    labels = [f[0] for f in res.fields()]
    assert labels == ["Peak-to-valley", "RMS", "Valid pixels", "Fringe spacing", "Time", "Compute"]
    rows = {label: (value, tip) for label, value, tip in res.fields()}
    assert rows["Peak-to-valley"][0].endswith(" nm") and rows["Valid pixels"][0].endswith(" %")
    period = res.info["fringe_period_px"]
    assert rows["Fringe spacing"][0] == f"{period:.1f} px"
    assert all(tip for _, tip in rows.values())
    s = res.summary()
    assert f"fringe spacing {period:.1f} px" in s and "carrier" not in s

    res.info.update(pv=794.3, rms=122.4, valid_fraction=1.0, seconds=0.12, estimated_step_deg=88.37)
    rows = {label: value for label, value, _ in res.fields()}
    assert rows["Peak-to-valley"] == "794 nm" and rows["RMS"] == "122 nm"
    assert rows["Valid pixels"] == "100 %" and rows["Compute"] == "120 ms"
    assert rows["Measured step"] == "88.4°" and len(rows["Time"]) == 8 and "Measured" not in rows
    assert [f[0] for f in res.fields()].index("Measured step") == 4

    res.unit = ""
    assert dict((l, v) for l, v, _ in res.fields())["RMS"] == "122"


@pytest.mark.parametrize("unwrap", ["least_squares", "weighted_ls", "quality_guided", "none"])
def test_analyse_reuses_unwrapped(unwrap):
    if unwrap == "quality_guided" and not P.skimage_available():
        pytest.skip("scikit-image not installed")
    h, w = 96, 128
    phi = _film_on_tilt(h, w)
    mod = np.ones_like(phi)
    mod[:10, :20] = 0.0  # masked corner
    cfg = P.PhaseConfig(method="4step", unwrap=unwrap, detrend="tilt", output="reflection")
    full = P.analyse(P.wrap(phi), mod, cfg, {"method": "4-step"})
    again = P.analyse(full.wrapped, full.modulation, cfg, {"method": "4-step"}, unwrapped=full.unwrapped)
    assert np.array_equal(again.mask, full.mask) and not again.mask[:10, :20].any()
    assert np.allclose(again.physical, full.physical, atol=1e-3, equal_nan=True)
    assert np.array_equal(np.isnan(again.physical), np.isnan(full.physical))
    for key in ("pv", "rms", "valid_fraction"):
        assert again.info[key] == pytest.approx(full.info[key], rel=1e-5)
    assert (again.quantity, again.unit, again.info["method"]) == (full.quantity, full.unit, "4-step")


def test_analyse_with_unwrapped_redoes_levelling_and_units(monkeypatch):
    h, w, step = 96, 128, 1.2
    phi = _film_on_tilt(h, w, step)
    cfg = P.PhaseConfig(method="4step", unwrap="least_squares", detrend="tilt", output="phase")
    first = P.analyse(P.wrap(phi), np.ones_like(phi), cfg)
    calls = []
    monkeypatch.setattr(P, "unwrap", lambda *a, **k: calls.append(a))  # must not be needed
    new = cfg.with_(level_region=(0, 0, w // 2 - 4, h), output="reflection", wavelength_nm=500.0)
    res = P.analyse(first.wrapped, first.modulation, new, unwrapped=first.unwrapped)
    assert calls == [] and res.unit == "nm" and res.info["levelled_on"] == "reference area"
    film = (slice(None), slice(w // 2 + 4, w - 4))
    assert np.abs(res.physical[film] - step * 500.0 / (4 * np.pi)).max() < 1.0  # true step in nm
    with pytest.raises(ValueError):
        P.analyse(first.wrapped, first.modulation, cfg, unwrapped=first.unwrapped[:-1])
