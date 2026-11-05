"""The "Threshold" Fourier filter of the C program (``c/threshold.c``) for PhaseScope.

``threshold(img, rad, thres)`` runs the C library when it has been built (``make`` in ``c/``),
and otherwise :func:`threshold_numpy`, which does exactly the same steps:

1. 2-D Fourier transform, zero frequency in the centre (``fft2dd``);
2. remove components weaker than ``thres`` (on log10(1 + |F|)) and everything farther than
   ``rad`` pixels from the centre (``filtcirct`` + ``filtcirc``);
3. inverse transform (``fft2di``);
4. intensity |.|^2 scaled to 0..255 (``getintens``, ``findmamime``, ``floattobyte``).

``python -m phasescope.cthreshold`` builds the library with the C compiler, if there is one.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

LIB_NAME = {"win32": "threshold.dll", "darwin": "libthreshold.dylib"}.get(sys.platform, "libthreshold.so")
C_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "c")

_lib: ctypes.CDLL | None | bool = False  # False: not looked for yet, None: not available


def library_path() -> str | None:
    """The built C library: $PHASESCOPE_THRESHOLD_LIB, next to this module, or in c/."""
    env = os.environ.get("PHASESCOPE_THRESHOLD_LIB")
    for path in ([env] if env else []) + [os.path.join(os.path.dirname(os.path.abspath(__file__)), LIB_NAME),
                                          os.path.join(C_DIR, LIB_NAME)]:
        if path and os.path.isfile(path):
            return path
    return None


def load_library(path: str | None = None) -> ctypes.CDLL | None:
    """Load the C library (cached); None when it is not built."""
    global _lib
    if path is None and _lib is not False:
        return _lib
    path = path or library_path()
    lib = None
    if path is not None:
        try:
            lib = ctypes.CDLL(path)
            fptr = ctypes.POINTER(ctypes.c_float)
            lib.ThresF.argtypes = [fptr, fptr, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int,
                                   ctypes.c_float, ctypes.c_float]
            lib.ThresF.restype = ctypes.c_int
        except (OSError, AttributeError):
            lib = None
    if path is None or lib is None or path == library_path():
        _lib = lib
    return lib


def backend() -> str:
    """'C' when the C library is used, else 'NumPy'."""
    return "C" if load_library() is not None else "NumPy"


def threshold_c(img: np.ndarray, rad: float, thres: float, spectrum: bool = False,
                lib: ctypes.CDLL | None = None):
    """Run ThresF of the C library; returns the 0..255 image (and the spectrum picture)."""
    lib = lib or load_library()
    if lib is None:
        raise RuntimeError("the C threshold library is not built - run 'make' in the c folder")
    src = np.ascontiguousarray(img, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError("the threshold filter needs a 2-D image")
    ny, nx = src.shape
    out = np.empty_like(src)
    spec = np.empty((ny, nx), np.uint8) if spectrum else None
    fptr = ctypes.POINTER(ctypes.c_float)
    status = lib.ThresF(src.ctypes.data_as(fptr), out.ctypes.data_as(fptr),
                        spec.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)) if spec is not None else None,
                        ny, nx, float(rad), float(thres))
    if status != 0:
        raise MemoryError("the threshold filter ran out of memory")
    return (out, spec) if spectrum else out


def _radius(shape) -> np.ndarray:
    ny, nx = shape
    y = np.arange(ny) - ny // 2
    x = np.arange(nx) - nx // 2
    return np.hypot(y[:, None], x[None, :])


def threshold_numpy(img: np.ndarray, rad: float, thres: float, spectrum: bool = False):
    """The same filter in NumPy (identical steps and conventions as c/threshold.c)."""
    src = np.asarray(img, dtype=np.float64)
    if src.ndim != 2:
        raise ValueError("the threshold filter needs a 2-D image")
    rad = max(float(rad), 0.0)
    f = np.fft.fftshift(np.fft.fft2(src))                       # fft2dd
    imf = np.log10(1.0 + np.abs(f))                              # imagfouf
    r = _radius(src.shape)
    r1, r2 = rad, rad + 10.0                                     # filtcirct(imf, b, thres, rad, rad + 10)
    w = np.where(r <= r1, 1.0, 0.5 * (1.0 + np.cos(np.pi * (r - r1) / (r2 - r1))))
    w[(r > r2) | ((r > 0) & (imf < thres))] = 0.0
    f = f * w
    f[r > rad] = 0.0                                             # filtcirc(b, rad)
    if spectrum:                                                 # imagfoumem
        s = np.log10(1.0 + np.abs(f))
        top = s.max()
        spec = (np.floor(255.0 * s / top + 0.5) if top > 0 else np.zeros_like(s)).astype(np.uint8)
    inten = np.abs(np.fft.ifft2(np.fft.ifftshift(f))) ** 2       # fft2di + getintens
    lo, hi = inten.min(), inten.max()                            # findmamime
    flat = hi - lo <= 1e-5 * abs(hi)  # constant apart from rounding noise
    out = (np.zeros_like(inten) if flat else 255.0 * (inten - lo) / (hi - lo)).astype(np.float32)
    return (out, spec) if spectrum else out


def threshold(img: np.ndarray, rad: float, thres: float) -> np.ndarray:
    """out = Thres(Im, rad, thres): the C library if built, else the NumPy version."""
    if load_library() is not None:
        return threshold_c(img, rad, thres)
    return threshold_numpy(img, rad, thres)


def build(out_dir: str | None = None, compiler: str | None = None) -> str:
    """Compile c/threshold.c into the shared library; returns its path."""
    cc = compiler or os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        raise RuntimeError("no C compiler found (install gcc, clang or MinGW, or run 'make' in the c folder)")
    out = os.path.join(out_dir or C_DIR, LIB_NAME)
    shared = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared"] + ([] if os.name == "nt" else ["-fPIC"])
    defines = ["-DTHRESHOLD_BUILD_DLL"] if os.name == "nt" else []
    subprocess.run([cc, "-O2", "-std=c99", *shared, *defines, "-o", out, os.path.join(C_DIR, "threshold.c"),
                    "-lm"], check=True)
    return out


if __name__ == "__main__":
    print(f"built {build()}")
