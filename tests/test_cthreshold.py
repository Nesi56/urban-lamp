"""The Threshold Fourier filter: C library (c/threshold.c), its NumPy twin and the 'thres' program."""

import os
import shutil
import subprocess

import numpy as np
import pytest

from phasescope import cthreshold as T

CC = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
needs_cc = pytest.mark.skipif(CC is None, reason="no C compiler")


@pytest.fixture(scope="module")
def clib(tmp_path_factory):
    if CC is None:
        pytest.skip("no C compiler")
    path = T.build(str(tmp_path_factory.mktemp("clib")), compiler=CC)
    return T.load_library(path)


def _images():
    rng = np.random.default_rng(0)
    for shape in [(64, 64), (48, 80), (101, 77), (240, 320), (1, 8)]:
        yield rng.random(shape).astype(np.float32) * 255


@needs_cc
@pytest.mark.parametrize("rad, thres", [(24, 1.0), (10, 3.5), (0, 1.0), (1000, 0.0), (5, 2.0)])
def test_c_and_numpy_give_the_same_result(clib, rad, thres):
    for img in _images():
        out_c, spec_c = T.threshold_c(img, rad, thres, spectrum=True, lib=clib)
        out_n, spec_n = T.threshold_numpy(img, rad, thres, spectrum=True)
        assert out_c.shape == img.shape and out_c.dtype == np.float32
        assert np.abs(out_c - out_n).max() < 2e-3
        assert np.abs(spec_c.astype(int) - spec_n.astype(int)).max() <= 1


def test_no_filtering_gives_the_scaled_intensity():
    img = np.random.default_rng(1).random((48, 80)) * 255
    out = T.threshold_numpy(img, 1e6, 0.0)  # every frequency kept
    ref = img ** 2
    assert np.allclose(out, 255 * (ref - ref.min()) / (ref.max() - ref.min()), atol=1e-3)


def test_radius_and_threshold_remove_what_they_should():
    y, x = np.mgrid[:128, :128]
    slow = 100 + 50 * np.cos(2 * np.pi * x * 4 / 128)    # 4 px from the centre of the spectrum
    fast = 20 * np.cos(2 * np.pi * y * 40 / 128)          # 40 px from the centre
    weak = 0.01 * np.cos(2 * np.pi * x * 8 / 128)         # 8 px, far below the threshold
    expected = T.threshold_numpy(slow, 1e6, 0.0)          # |slow|^2 scaled
    assert np.allclose(T.threshold_numpy(slow + fast, 20, 0.0), expected, atol=0.05)  # outside the radius
    assert np.allclose(T.threshold_numpy(slow + weak, 20, 2.5), expected, atol=0.05)  # below the threshold
    flat = T.threshold_numpy(slow + fast, 0, 0.0)          # only the mean is left
    assert np.all(flat == 0)


@needs_cc
def test_thres_program_reads_and_writes_pgm(tmp_path):
    import cv2

    exe = str(tmp_path / "thres")
    subprocess.run([CC, "-O2", "-std=c99", "-o", exe, os.path.join(T.C_DIR, "thres_main.c"),
                    os.path.join(T.C_DIR, "threshold.c"), "-lm"], check=True)
    img = (np.random.default_rng(2).random((60, 90)) * 255).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "in.pgm"), img)
    run = subprocess.run([exe, str(tmp_path / "in.pgm"), str(tmp_path / "out.pgm"), "12", "2",
                          str(tmp_path / "spec.pgm")], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    out = cv2.imread(str(tmp_path / "out.pgm"), cv2.IMREAD_UNCHANGED)
    ref = np.floor(T.threshold_numpy(img, 12, 2) + 0.5)
    assert out.shape == img.shape and np.abs(out.astype(int) - ref).max() <= 1
    assert cv2.imread(str(tmp_path / "spec.pgm"), cv2.IMREAD_UNCHANGED).shape == img.shape
    # 16-bit input works too
    cv2.imwrite(str(tmp_path / "in16.pgm"), img.astype(np.uint16) * 200)
    assert subprocess.run([exe, str(tmp_path / "in16.pgm"), str(tmp_path / "out16.pgm")]).returncode == 0
    # bad input: a message, no crash
    (tmp_path / "bad.pgm").write_bytes(b"not an image")
    bad = subprocess.run([exe, str(tmp_path / "bad.pgm"), str(tmp_path / "x.pgm")], capture_output=True, text=True)
    assert bad.returncode != 0 and "PGM" in bad.stderr


def test_filter_is_in_the_fourier_menu():
    from phasescope.processing import REGISTRY, apply_operation

    op = REGISTRY["fourier_threshold"]
    assert op.group == "Fourier" and not op.phase_safe and [p.name for p in op.params] == ["rad", "thres"]
    out = apply_operation("fourier_threshold", np.random.default_rng(3).random((40, 50)) * 4000, rad=24, thres=1)
    assert out.shape == (40, 50) and 0 <= out.min() and out.max() <= 255.0001
