"""Image filters, geometric transformations and the live processing pipeline.

Every operation is a plain function ``func(img, state, **params) -> ndarray``
registered in :data:`REGISTRY` together with a description of its parameters.
The GUI builds its pop-up menus and parameter dialogs from that description,
so adding a new filter only requires writing one decorated function here.

All operations receive and return 2-D ``float32`` arrays in the camera's
intensity units.  ``state`` is a per-pipeline-step dictionary that stateful
operations (background subtraction, temporal averaging) use to remember
previous frames.  Operations that change the fringe pattern itself (edge
detectors, contrast stretches, ...) are registered with ``phase_safe=False`` so
the GUI can warn before they spoil phase / height results.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Param:
    """Description of one user-editable parameter of an operation."""

    name: str
    label: str
    kind: str  # "int" | "float" | "choice" | "bool" | "str"
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    odd: bool = False  # integer kernel sizes that must be odd
    decimals: int = 2
    tooltip: str = ""

    def coerce(self, value: Any) -> Any:
        if self.kind == "int":
            value = int(round(float(value)))
            if self.minimum is not None:
                value = max(int(self.minimum), value)
            if self.maximum is not None:
                value = min(int(self.maximum), value)
            if self.odd and value % 2 == 0:
                value += 1
            return value
        if self.kind == "float":
            value = float(value)
            if self.minimum is not None:
                value = max(float(self.minimum), value)
            if self.maximum is not None:
                value = min(float(self.maximum), value)
            return value
        if self.kind == "bool":
            return bool(value)
        if self.kind == "str":
            return str(value)
        if self.kind == "choice":
            value = str(value)
            if value not in self.choices:
                raise ValueError(f"{self.label}: {value!r} is not one of {self.choices}")
            return value
        raise ValueError(f"unknown parameter kind {self.kind!r}")


@dataclass(frozen=True)
class Operation:
    """A registered filter or transformation (built by :func:`register`)."""

    key: str
    name: str
    category: str  # "filter" | "transform"
    func: Callable[..., np.ndarray]
    params: tuple[Param, ...] = ()
    description: str = ""
    temporal: bool = False  # combines several frames (skipped inside phase-step sequences)
    group: str = ""  # sub-menu name in the pop-up menus
    phase_safe: bool = True  # False: distorts the fringes, phase / height results become wrong

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def coerce(self, params: dict[str, Any] | None) -> dict[str, Any]:
        out = self.defaults()
        for p in self.params:
            if params and p.name in params:
                out[p.name] = p.coerce(params[p.name])
        return out


REGISTRY: dict[str, Operation] = {}


def register(key: str, name: str, category: str, params: tuple[Param, ...] = (),
             description: str = "", temporal: bool = False, group: str = "",
             phase_safe: bool = True):
    """Decorator adding ``func(img, state, **params)`` to :data:`REGISTRY`.

    ``description`` is a plain-language sentence shown as tooltip and in the
    parameter dialog; ``phase_safe=False`` marks operations that change the
    fringe pattern and therefore corrupt phase / height results.
    """
    def deco(func):
        if key in REGISTRY:
            raise KeyError(f"operation {key!r} registered twice")
        REGISTRY[key] = Operation(key, name, category, func, tuple(params),
                                  description, temporal, group, phase_safe)
        return func
    return deco


def operations(category: str) -> list[Operation]:
    return [op for op in REGISTRY.values() if op.category == category]


def apply_operation(key: str, img: np.ndarray, state: dict | None = None, **params) -> np.ndarray:
    """Convenience wrapper used by tests and scripts."""
    op = REGISTRY[key]
    out = op.func(as_float(img), {} if state is None else state, **op.coerce(params))
    return np.ascontiguousarray(out, dtype=np.float32)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def as_float(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3:  # colour -> grey (interferometry works on intensity)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = img.mean(axis=2)
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {img.shape}")
    return np.ascontiguousarray(img, dtype=np.float32)


def _range(img: np.ndarray) -> tuple[float, float]:
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def to_uint8(img: np.ndarray) -> np.ndarray:
    lo, hi = _range(img)
    out = (np.nan_to_num(img, nan=lo) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def _to_uint16(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    lo, hi = _range(img)
    out = (np.nan_to_num(img, nan=lo) - lo) * (65535.0 / (hi - lo))
    return np.clip(out, 0, 65535).astype(np.uint16), lo, hi


_BORDERS = {"reflect": cv2.BORDER_REFLECT101, "replicate": cv2.BORDER_REPLICATE,
            "constant (0)": cv2.BORDER_CONSTANT}
_INTERP = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
           "cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA}


def _kernel(ksize: int) -> int:
    return max(1, int(ksize) | 1)


def _freq_radius(shape: tuple[int, int]) -> np.ndarray:
    """Radial spatial frequency (fraction of Nyquist, 0..~1.41) for an FFT grid."""
    fy = np.fft.fftfreq(shape[0])[:, None] * 2.0
    fx = np.fft.fftfreq(shape[1])[None, :] * 2.0
    return np.sqrt(fx * fx + fy * fy)


def _fourier_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    spec = np.fft.fft2(img)
    return np.real(np.fft.ifft2(spec * mask)).astype(np.float32)


def _lowpass_mask(r: np.ndarray, cutoff: float, profile: str) -> np.ndarray:
    cutoff = max(cutoff, 1e-6)
    if profile == "ideal":
        return (r <= cutoff).astype(np.float32)
    if profile == "butterworth":
        return 1.0 / (1.0 + (r / cutoff) ** 4)
    return np.exp(-0.5 * (r / cutoff) ** 2)  # gaussian


_FFT_PROFILE = Param("profile", "Window", "choice", "gaussian",
                     choices=("gaussian", "butterworth", "ideal"))


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------

@register("gaussian", "Gaussian blur", "filter", group="Smoothing", params=(
    Param("sigma", "Sigma (px)", "float", 2.0, 0.1, 50.0, 0.5),),
    description="Softens the image to reduce noise – a larger sigma blurs more.")
def gaussian(img, state, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101)


@register("median", "Median", "filter", group="Smoothing", params=(
    Param("ksize", "Kernel size (px)", "int", 5, 3, 31, 2, odd=True),),
    description="Removes isolated bright or dark specks (hot pixels, dust) while keeping edges.")
def median(img, state, ksize):
    ksize = _kernel(ksize)
    if ksize <= 5:
        return cv2.medianBlur(img, ksize)
    return ndimage.median_filter(img, size=ksize, mode="reflect")


@register("box", "Mean (box)", "filter", group="Smoothing", params=(
    Param("ksize", "Kernel size (px)", "int", 5, 1, 99, 2, odd=True),),
    description="Simple, fast smoothing that averages each pixel with its neighbours.")
def box(img, state, ksize):
    k = _kernel(ksize)
    return cv2.blur(img, (k, k), borderType=cv2.BORDER_REFLECT101)


@register("bilateral", "Bilateral (edge preserving)", "filter", group="Smoothing", params=(
    Param("diameter", "Diameter (px)", "int", 7, 3, 25),
    Param("sigma_color", "Intensity sigma (fraction of range)", "float", 0.1, 0.005, 1.0, 0.01, decimals=3),
    Param("sigma_space", "Spatial sigma (px)", "float", 5.0, 0.5, 50.0, 0.5)),
    description="Smooths noise but keeps sharp edges such as film steps.")
def bilateral(img, state, diameter, sigma_color, sigma_space):
    lo, hi = _range(img)
    norm = ((img - lo) / (hi - lo)).astype(np.float32)
    out = cv2.bilateralFilter(norm, int(diameter), sigma_color, sigma_space)
    return out * (hi - lo) + lo


@register("nlmeans", "Non-local means denoise", "filter", group="Smoothing", params=(
    Param("h", "Strength", "float", 10.0, 1.0, 50.0, 1.0),
    Param("template", "Template size", "int", 7, 3, 15, 2, odd=True),
    Param("search", "Search window", "int", 21, 7, 41, 2, odd=True)),
    description="Strong noise removal that keeps fine texture (slow on large frames, works on an 8-bit copy).")
def nlmeans(img, state, h, template, search):
    lo, hi = _range(img)
    out = cv2.fastNlMeansDenoising(to_uint8(img), None, h, template, search)
    return out.astype(np.float32) * ((hi - lo) / 255.0) + lo


@register("unsharp", "Sharpen (unsharp mask)", "filter", group="Sharpen / edges", params=(
    Param("sigma", "Sigma (px)", "float", 2.0, 0.1, 30.0, 0.5),
    Param("amount", "Amount", "float", 1.0, 0.0, 10.0, 0.1)),
    description="Makes fine detail and edges look crisper.")
def unsharp(img, state, sigma, amount):
    blur = cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101)
    return img + amount * (img - blur)


@register("laplacian", "Laplacian", "filter", group="Sharpen / edges", phase_safe=False, params=(
    Param("ksize", "Aperture", "int", 3, 1, 7, 2, odd=True),),
    description="Outlines edges and fine detail as bright and dark lines "
                "– for viewing only, not before phase analysis.")
def laplacian(img, state, ksize):
    return cv2.Laplacian(img, cv2.CV_32F, ksize=_kernel(ksize), borderType=cv2.BORDER_REFLECT101)


@register("sobel", "Sobel gradient", "filter", group="Sharpen / edges", phase_safe=False, params=(
    Param("direction", "Direction", "choice", "magnitude", choices=("magnitude", "x", "y")),
    Param("ksize", "Aperture", "int", 3, 1, 7, 2, odd=True)),
    description="Shows where the brightness changes quickly, highlighting edges "
                "– for viewing only, not before phase analysis.")
def sobel(img, state, direction, ksize):
    k = _kernel(ksize)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=k)
    if direction == "x":
        return gx
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=k)
    if direction == "y":
        return gy
    return cv2.magnitude(gx, gy)


@register("canny", "Canny edges", "filter", group="Sharpen / edges", phase_safe=False, params=(
    Param("low", "Low threshold (0-255)", "float", 50.0, 0.0, 255.0, 5.0),
    Param("high", "High threshold (0-255)", "float", 150.0, 0.0, 255.0, 5.0)),
    description="Draws edges as white lines – for viewing only, not before phase analysis.")
def canny(img, state, low, high):
    return cv2.Canny(to_uint8(img), low, high).astype(np.float32)


@register("hist_eq", "Histogram equalisation", "filter", group="Contrast", phase_safe=False,
          description="Spreads the brightness levels evenly to show more contrast "
                      "– distorts the fringes, not before phase analysis.")
def hist_eq(img, state):
    return cv2.equalizeHist(to_uint8(img)).astype(np.float32)


@register("clahe", "CLAHE (adaptive equalisation)", "filter", group="Contrast", phase_safe=False, params=(
    Param("clip", "Clip limit", "float", 2.0, 0.1, 40.0, 0.5),
    Param("tiles", "Tiles per side", "int", 8, 1, 64)),
    description="Boosts local contrast to reveal faint detail – distorts the fringes, not before phase analysis.")
def clahe(img, state, clip, tiles):
    u16, lo, hi = _to_uint16(img)
    out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(u16)
    return out.astype(np.float32) * ((hi - lo) / 65535.0) + lo


@register("gamma", "Gamma", "filter", group="Contrast", phase_safe=False, params=(
    Param("gamma", "Gamma", "float", 0.5, 0.05, 10.0, 0.05),),
    description="Brightens dark areas (gamma below 1) or darkens them (above 1) "
                "– distorts the fringes, not before phase analysis.")
def gamma(img, state, gamma):
    lo, hi = _range(img)
    return np.power(np.clip((img - lo) / (hi - lo), 0, 1), gamma) * (hi - lo) + lo


@register("normalize", "Normalise", "filter", group="Contrast", params=(
    Param("mode", "Mode", "choice", "min-max [0, 1]", choices=("min-max [0, 1]", "zero mean / unit std")),),
    description="Rescales the brightness to a standard range so images can be compared.")
def normalize(img, state, mode):
    if mode.startswith("min-max"):
        lo, hi = _range(img)
        return (img - lo) / (hi - lo)
    std = float(np.nanstd(img)) or 1.0
    return (img - float(np.nanmean(img))) / std


@register("invert", "Invert", "filter", group="Contrast",
          description="Swaps bright and dark, like a photographic negative.")
def invert(img, state):
    lo, hi = _range(img)
    return (hi + lo) - img


@register("threshold", "Threshold", "filter", group="Contrast", phase_safe=False, params=(
    Param("method", "Method", "choice", "otsu", choices=("otsu", "manual", "adaptive")),
    Param("level", "Manual level (fraction of range)", "float", 0.5, 0.0, 1.0, 0.05),
    Param("block", "Adaptive block size", "int", 31, 3, 255, 2, odd=True)),
    description="Turns the image into pure black and white – for viewing only, not before phase analysis.")
def threshold(img, state, method, level, block):
    u8 = to_uint8(img)
    if method == "otsu":
        _, out = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "manual":
        _, out = cv2.threshold(u8, level * 255.0, 255, cv2.THRESH_BINARY)
    else:
        out = cv2.adaptiveThreshold(u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, _kernel(block), 0)
    return out.astype(np.float32)


_MORPH = {"open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE, "erode": cv2.MORPH_ERODE,
          "dilate": cv2.MORPH_DILATE, "gradient": cv2.MORPH_GRADIENT,
          "top-hat": cv2.MORPH_TOPHAT, "black-hat": cv2.MORPH_BLACKHAT}


@register("morphology", "Morphology", "filter", group="Contrast", phase_safe=False, params=(
    Param("operation", "Operation", "choice", "open", choices=tuple(_MORPH)),
    Param("ksize", "Kernel size (px)", "int", 5, 1, 99, 2, odd=True),
    Param("shape", "Kernel shape", "choice", "ellipse", choices=("ellipse", "rectangle", "cross"))),
    description="Shape-based clean-up: removes small specks, fills small holes or traces outlines "
                "– distorts the fringes, not before phase analysis.")
def morphology(img, state, operation, ksize, shape):
    shapes = {"ellipse": cv2.MORPH_ELLIPSE, "rectangle": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}
    k = cv2.getStructuringElement(shapes[shape], (_kernel(ksize), _kernel(ksize)))
    return cv2.morphologyEx(img, _MORPH[operation], k)


@register("fft_lowpass", "FFT low-pass", "filter", group="Fourier", params=(
    Param("cutoff", "Cut-off (fraction of Nyquist)", "float", 0.2, 0.005, 1.5, 0.01, decimals=3),
    _FFT_PROFILE),
    description="Removes fine-grained noise and keeps larger features (too low a cut-off also removes the fringes).")
def fft_lowpass(img, state, cutoff, profile):
    return _fourier_mask(img, _lowpass_mask(_freq_radius(img.shape), cutoff, profile))


@register("fft_highpass", "FFT high-pass", "filter", group="Fourier", params=(
    Param("cutoff", "Cut-off (fraction of Nyquist)", "float", 0.02, 0.001, 1.5, 0.005, decimals=3),
    _FFT_PROFILE,
    Param("keep_mean", "Keep mean level", "bool", True)),
    description="Removes slowly varying background such as uneven illumination.")
def fft_highpass(img, state, cutoff, profile, keep_mean):
    mask = 1.0 - _lowpass_mask(_freq_radius(img.shape), cutoff, profile)
    out = _fourier_mask(img, mask)
    return out + float(img.mean()) if keep_mean else out


@register("fft_bandpass", "FFT band-pass", "filter", group="Fourier", params=(
    Param("low", "Low cut-off (fraction of Nyquist)", "float", 0.02, 0.0, 1.5, 0.005, decimals=3),
    Param("high", "High cut-off (fraction of Nyquist)", "float", 0.3, 0.005, 1.5, 0.01, decimals=3),
    _FFT_PROFILE),
    description="Keeps only features within a size range (removes noise and slow background).")
def fft_bandpass(img, state, low, high, profile):
    r = _freq_radius(img.shape)
    mask = _lowpass_mask(r, high, profile)
    if low > 0:
        mask = mask * (1.0 - _lowpass_mask(r, low, profile))
    return _fourier_mask(img, mask) + float(img.mean())


@register("background", "Background subtraction / flat-field", "filter", group="Background", params=(
    Param("mode", "Mode", "choice", "subtract", choices=("subtract", "divide (flat-field)")),
    Param("frames", "Frames averaged for background", "int", 10, 1, 500),
    Param("smooth", "Smooth background (sigma px, 0 = off)", "float", 0.0, 0.0, 200.0, 1.0)),
    description="Removes dust, fixed patterns and uneven illumination: the first N frames after adding "
                "(or re-capturing) this step are averaged and stored as background. Remove the sample, "
                "add the step, then put the sample back.")
def background(img, state, mode, frames, smooth):
    acc = state.get("acc")
    if state.get("shape") != img.shape or state.get("frames") != frames or state.get("smooth") != smooth:
        state.clear()
        state.update(shape=img.shape, frames=frames, smooth=smooth, count=0, acc=None, bg=None)
        acc = None
    if state["bg"] is None:
        acc = img.astype(np.float64) if acc is None else acc + img
        state["acc"] = acc
        state["count"] += 1
        if state["count"] < frames:
            return img  # still learning the background
        bg = (acc / state["count"]).astype(np.float32)
        if smooth > 0:
            bg = cv2.GaussianBlur(bg, (0, 0), smooth, borderType=cv2.BORDER_REFLECT101)
        state["bg"] = bg
        state["acc"] = None
        state["stored_at"] = time.strftime("%H:%M:%S")
    bg = state["bg"]
    if mode == "subtract":
        return img - bg
    return img / np.maximum(bg, 1e-6) * float(bg.mean())


@register("temporal_average", "Temporal average", "filter", group="Background", temporal=True, params=(
    Param("mode", "Mode", "choice", "sliding window", choices=("sliding window", "exponential")),
    Param("frames", "Frames", "int", 5, 2, 200)),
    description="Averages consecutive frames to reduce camera noise (adds lag for moving samples).")
def temporal_average(img, state, mode, frames):
    if state.get("shape") != img.shape or state.get("mode") != mode or state.get("frames") != frames:
        state.clear()
        state.update(shape=img.shape, mode=mode, frames=frames, count=0, buf=deque(), total=None, ema=None)
    state["count"] = min(state["count"] + 1, frames)  # frames in the average (for PipelineStep.status)
    if mode == "exponential":
        alpha = 1.0 / frames
        state["ema"] = img.copy() if state["ema"] is None else (1 - alpha) * state["ema"] + alpha * img
        return state["ema"]
    buf: deque = state["buf"]
    buf.append(img.astype(np.float64))
    state["total"] = buf[-1].copy() if state["total"] is None else state["total"] + buf[-1]
    while len(buf) > frames:
        state["total"] -= buf.popleft()
    return (state["total"] / len(buf)).astype(np.float32)


# --------------------------------------------------------------------------
# Transformations
# --------------------------------------------------------------------------

@register("flip_h", "Flip horizontal", "transform", group="Flip / rotate",
          description="Mirrors the image left to right.")
def flip_h(img, state):
    return img[:, ::-1]


@register("flip_v", "Flip vertical", "transform", group="Flip / rotate",
          description="Mirrors the image top to bottom.")
def flip_v(img, state):
    return img[::-1, :]


@register("rot90_cw", "Rotate 90° clockwise", "transform", group="Flip / rotate",
          description="Turns the image a quarter turn clockwise.")
def rot90_cw(img, state):
    return np.rot90(img, k=-1)


@register("rot90_ccw", "Rotate 90° counter-clockwise", "transform", group="Flip / rotate",
          description="Turns the image a quarter turn counter-clockwise.")
def rot90_ccw(img, state):
    return np.rot90(img, k=1)


@register("rot180", "Rotate 180°", "transform", group="Flip / rotate",
          description="Turns the image upside down (half turn).")
def rot180(img, state):
    return img[::-1, ::-1]


@register("transpose", "Transpose", "transform", group="Flip / rotate",
          description="Swaps rows and columns (mirrors along the diagonal).")
def transpose(img, state):
    return img.T


@register("rotate", "Rotate by angle…", "transform", group="Flip / rotate", params=(
    Param("angle", "Angle (deg, counter-clockwise)", "float", 10.0, -180.0, 180.0, 0.5),
    Param("expand", "Expand canvas to fit", "bool", False),
    Param("border", "Border", "choice", "reflect", choices=tuple(_BORDERS))),
    description="Rotates the image by any angle, e.g. to straighten fringes or edges.")
def rotate(img, state, angle, expand, border):
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    if expand:
        c, s = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(round(h * s + w * c)), int(round(h * c + w * s))
        m[0, 2] += nw / 2.0 - w / 2.0
        m[1, 2] += nh / 2.0 - h / 2.0
        w, h = nw, nh
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=_BORDERS[border])


@register("translate", "Translate (shift)", "transform", group="Geometry", params=(
    Param("dx", "Shift x (px)", "float", 0.0, -10000, 10000, 1.0),
    Param("dy", "Shift y (px)", "float", 0.0, -10000, 10000, 1.0),
    Param("border", "Border", "choice", "constant (0)", choices=tuple(_BORDERS))),
    description="Shifts the image sideways and / or up and down by a number of pixels.")
def translate(img, state, dx, dy, border):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=_BORDERS[border])


@register("resize", "Scale (resize)", "transform", group="Geometry", params=(
    Param("factor", "Scale factor", "float", 0.5, 0.05, 8.0, 0.05),
    Param("interpolation", "Interpolation", "choice", "area", choices=tuple(_INTERP))),
    description="Makes the image smaller or larger (smaller images process faster).")
def resize(img, state, factor, interpolation):
    h, w = img.shape
    size = (max(1, int(round(w * factor))), max(1, int(round(h * factor))))
    return cv2.resize(img, size, interpolation=_INTERP[interpolation])


@register("bin", "Pixel binning", "transform", group="Geometry", params=(
    Param("factor", "Bin factor (n x n)", "int", 2, 2, 16),),
    description="Averages n x n blocks: faster processing and lower noise.")
def bin_pixels(img, state, factor):
    h, w = (img.shape[0] // factor) * factor, (img.shape[1] // factor) * factor
    if h == 0 or w == 0:
        return img
    return img[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


@register("crop", "Crop", "transform", group="Geometry", params=(
    Param("x", "Left (px)", "int", 0, 0, 100000),
    Param("y", "Top (px)", "int", 0, 0, 100000),
    Param("width", "Width (px, 0 = to edge)", "int", 0, 0, 100000),
    Param("height", "Height (px, 0 = to edge)", "int", 0, 0, 100000)),
    description="Keeps only a rectangular part of the image.")
def crop(img, state, x, y, width, height):
    h, w = img.shape
    x0, y0 = min(x, w - 1), min(y, h - 1)
    x1 = w if width == 0 else min(w, x0 + width)
    y1 = h if height == 0 else min(h, y0 + height)
    return img[y0:y1, x0:x1]


@register("center_crop", "Centre crop (square)", "transform", group="Geometry", params=(
    Param("size", "Side (px)", "int", 512, 8, 100000),),
    description="Keeps a square from the middle of the image.")
def center_crop(img, state, size):
    h, w = img.shape
    s = min(size, h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


@register("fft_spectrum", "Fourier spectrum (log |F|)", "transform", group="Intensity maps", phase_safe=False,
          description="Shows the spatial frequencies, e.g. to check fringe spacing and direction "
                      "– for viewing only, not before phase analysis.")
def fft_spectrum(img, state):
    spec = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    return np.log1p(np.abs(spec)).astype(np.float32)


@register("log", "Logarithm", "transform", group="Intensity maps", phase_safe=False,
          description="Strongly brightens dim regions to show detail in the dark "
                      "– distorts the fringes, not before phase analysis.")
def log_transform(img, state):
    lo, _ = _range(img)
    return np.log1p(np.maximum(img - lo, 0))


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    return f"{v:g}" if isinstance(v, (float, np.floating)) else str(v)


_UNITS = {"px": "px", "deg": "°", "°": "°", "µm": "µm", "um": "µm", "nm": "nm", "mm": "mm",
          "ms": "ms", "s": "s", "%": "%"}


def _short_param(label: str) -> tuple[str, str, str]:
    """``(short name, unit, meaning of 0)`` from a parameter label.

    "Sigma (px)" -> ("sigma", "px", ""), "Intensity sigma (fraction of range)" ->
    ("intensity σ", "", ""), "Width (px, 0 = to edge)" -> ("width", "px", "to edge").
    Parenthesised notes that are not units are dropped.
    """
    base, _, note = label.partition("(")
    words = base.split(" for ")[0].split()  # "Frames averaged for background" -> "frames averaged"
    if len(words) > 1 and words[-1].lower() == "size":  # "Kernel size" -> "kernel"
        words = words[:-1]
    if words and not words[0].isupper():
        words[0] = words[0][:1].lower() + words[0][1:]
    if len(words) > 1:
        words = ["σ" if w.lower() == "sigma" else w for w in words]
    parts = [part.strip() for part in note.rstrip().rstrip(")").split(",")]
    first = parts[0].split()
    unit = _UNITS.get(first[-1].lower(), "") if first else ""
    zero = next((part[3:].strip() for part in parts if part.startswith("0 =")), "")
    return " ".join(words), unit, zero


def _param_text(p: Param, value: Any, op_name: str) -> str:
    """One parameter as friendly text such as "sigma 2 px" ("" = leave out)."""
    name, unit, zero = _short_param(p.label)
    if p.kind == "bool":
        return name if value else ""
    if zero and p.kind in ("int", "float") and value == 0:
        return f"{name} {zero}"
    text = _fmt(value)
    if unit:
        text += unit if unit == "°" else f" {unit}"
    if not name or name.lower() == op_name.lower():  # "Gamma · 0.5", not "Gamma · gamma 0.5"
        return text
    return f"{name} {text}"


def _background_status(state: dict, params: dict) -> str:
    frames = params.get("frames")
    if state.get("frames") != frames or state.get("smooth") != params.get("smooth"):
        return "waiting for frames"  # no state yet, or re-learning after a parameter change
    if state.get("bg") is not None:
        return f"background stored {state.get('stored_at', '')}".rstrip()
    count = state.get("count", 0)
    return f"learning background {count}/{frames}" if count else "waiting for frames"


def _temporal_status(state: dict, params: dict) -> str:
    frames = params.get("frames")
    same = state.get("mode") == params.get("mode") and state.get("frames") == frames
    count = state.get("count", 0) if same else 0
    return f"averaging {count}/{frames} frames" if frames and count < frames else ""


_STATUS: dict[str, Callable[[dict, dict], str]] = {
    "background": _background_status,
    "temporal_average": _temporal_status,
}


@dataclass
class PipelineStep:
    op: Operation
    params: dict[str, Any]
    enabled: bool = True
    state: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def label(self) -> str:
        """Friendly summary, e.g. "Gaussian blur · sigma 2 px" (just the name without parameters)."""
        if not self.params:
            return self.op.name
        texts = [_param_text(p, self.params.get(p.name, p.default), self.op.name) for p in self.op.params]
        args = ", ".join(t for t in texts if t)
        return f"{self.op.name} · {args}" if args else self.op.name

    @property
    def status(self) -> str:
        """Progress of stateful steps, e.g. "learning background 3/10" ("" for most steps)."""
        func = _STATUS.get(self.op.key)
        if func is None:
            return ""
        try:
            return func(self.state, self.params)
        except Exception:  # state is being replaced by the processing thread
            return ""

    def apply(self, img: np.ndarray) -> np.ndarray:
        return self.op.func(img, self.state, **self.params)


class Pipeline:
    """Thread-safe ordered list of processing steps applied to every frame.

    Editing never waits for a frame being processed: :meth:`apply` takes ``_lock`` only to
    snapshot the steps, and runs them under ``_run_lock`` (which serialises :meth:`apply` calls
    from different threads, as stateful steps are not re-entrant).
    """

    def __init__(self) -> None:
        self._steps: list[PipelineStep] = []
        self._lock = threading.RLock()  # guards the step list and step attributes, held only briefly
        self._run_lock = threading.RLock()
        self.bypass = False  # True: apply() returns the unprocessed frame (raw vs. filtered comparison)

    # -- editing ----------------------------------------------------------
    def add(self, key: str, params: dict[str, Any] | None = None) -> PipelineStep:
        op = REGISTRY[key]
        step = PipelineStep(op, op.coerce(params))
        with self._lock:
            self._steps.append(step)
        return step

    def remove(self, index: int) -> None:
        with self._lock:
            del self._steps[index]

    def move(self, index: int, delta: int) -> int:
        with self._lock:
            new = max(0, min(len(self._steps) - 1, index + delta))
            self._steps.insert(new, self._steps.pop(index))
            return new

    def clear(self) -> None:
        with self._lock:
            self._steps.clear()

    def signature(self) -> tuple:
        """Which steps, parameters and on/off states there are (tells results' processing apart)."""
        with self._lock:
            return tuple((st.op.key, repr(st.params), st.enabled) for st in self._steps)

    def set_params(self, index: int, params: dict[str, Any]) -> None:
        """Change a step's parameters.  Its state is kept: stateful operations reset
        themselves when a relevant parameter changes (so e.g. switching the background
        mode keeps the learned background, changing its frame count re-learns it)."""
        with self._lock:
            step = self._steps[index]
            step.params = step.op.coerce(params)
            step.error = None

    def set_enabled(self, index: int, enabled: bool) -> None:
        with self._lock:
            self._steps[index].enabled = enabled
            self._steps[index].error = None

    def reset_step(self, index: int) -> None:
        """Forget one step's state, e.g. to re-capture the background.

        The state is replaced by a fresh dict (not cleared), so a frame being processed
        right now finishes with the old one."""
        with self._lock:
            self._steps[index].state = {}
            self._steps[index].error = None

    def reset_state(self) -> None:
        with self._lock:
            for step in self._steps:
                step.state = {}
                step.error = None

    @property
    def steps(self) -> list[PipelineStep]:
        with self._lock:
            return list(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def has_unsafe_steps(self) -> list[PipelineStep]:
        """Enabled steps that distort the fringes (``op.phase_safe`` is False)."""
        return [s for s in self.steps if s.enabled and not s.op.phase_safe]

    def has_temporal_steps(self) -> bool:
        """True if an enabled step combines several frames (``op.temporal``, e.g. temporal average)."""
        return any(s.enabled and s.op.temporal for s in self.steps)

    # -- processing -------------------------------------------------------
    def apply_split(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One pass that returns ``(processed, processed_without_temporal_steps)``.

        Steps before the first temporal step run once for both results.  After it, the other
        steps also process the non-temporal branch, on a copy of their state, so stateful steps
        (e.g. a background being learned) still see every frame exactly once."""
        out = as_float(img)
        if self.bypass:
            return out, out
        with self._run_lock:
            with self._lock:
                steps = [s for s in self._steps if s.enabled]
            nt = out  # branch without temporal steps; the same array until one has run
            for step in steps:
                try:
                    res = np.ascontiguousarray(step.apply(out), dtype=np.float32)
                    if res.ndim != 2 or res.size == 0:
                        raise ValueError(f"produced an invalid image of shape {res.shape}")
                    if not step.op.temporal:
                        nt = res if nt is out else np.ascontiguousarray(
                            step.op.func(nt, dict(step.state), **step.params), dtype=np.float32)
                    out = res
                    step.error = None
                except Exception as exc:  # keep the live view running
                    step.error = f"{type(exc).__name__}: {exc}"
        return out, nt

    def apply(self, img: np.ndarray, include_temporal: bool = True) -> np.ndarray:
        """Run all enabled steps.  A failing step is bypassed and its error recorded.

        The enabled steps are snapshot under the lock and run without it, so editing the
        pipeline meanwhile affects the next frame.  With :attr:`bypass` set the frame is
        returned unprocessed (as float32)."""
        out = as_float(img)
        if self.bypass:
            return out
        with self._run_lock:
            with self._lock:
                steps = [s for s in self._steps
                         if s.enabled and not (s.op.temporal and not include_temporal)]
            for step in steps:
                try:
                    res = np.ascontiguousarray(step.apply(out), dtype=np.float32)
                    if res.ndim != 2 or res.size == 0:
                        raise ValueError(f"produced an invalid image of shape {res.shape}")
                    out = res
                    step.error = None
                except Exception as exc:  # keep the live view running
                    step.error = f"{type(exc).__name__}: {exc}"
        return out
