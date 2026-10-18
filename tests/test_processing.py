import numpy as np
import pytest

from phasescope.processing import REGISTRY, Param, Pipeline, apply_operation, operations, register

UNSAFE = {"canny", "sobel", "laplacian", "threshold", "hist_eq", "clahe", "gamma", "morphology", "log",
          "fft_spectrum"}


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:60, :80]
    return (2000 + 1500 * np.cos(xx / 3.0) + rng.normal(0, 50, xx.shape)).astype(np.uint16)


@pytest.mark.parametrize("key", list(REGISTRY))
def test_every_operation_runs(key, image):
    out = apply_operation(key, image)
    assert out.dtype == np.float32 and out.ndim == 2 and out.size > 0
    assert np.isfinite(out).all()


def test_shapes_of_transformations(image):
    assert apply_operation("rot90_cw", image).shape == (80, 60)
    assert apply_operation("transpose", image).shape == (80, 60)
    assert apply_operation("bin", image, factor=4).shape == (15, 20)
    assert apply_operation("resize", image, factor=0.5).shape == (30, 40)
    assert apply_operation("crop", image, x=10, y=5, width=20, height=30).shape == (30, 20)
    assert apply_operation("center_crop", image, size=40).shape == (40, 40)
    assert apply_operation("rotate", image, angle=90, expand=True).shape == (80, 60)
    flipped = apply_operation("flip_h", image)
    assert np.array_equal(flipped, image[:, ::-1].astype(np.float32))


def test_menu_categories_are_populated():
    assert len(operations("filter")) >= 15
    assert len(operations("transform")) >= 10


def test_param_coercion():
    op = REGISTRY["median"]
    assert op.coerce({"ksize": 4})["ksize"] == 5  # kernel sizes are forced odd
    assert op.coerce({"ksize": 1000})["ksize"] == 31  # clamped to the maximum
    with pytest.raises(ValueError):
        REGISTRY["sobel"].coerce({"direction": "diagonal"})


def test_pipeline_editing_and_error_bypass(image):
    p = Pipeline()
    p.add("gaussian", {"sigma": 1.0})
    p.add("flip_v")
    out = p.apply(image)
    assert out.shape == image.shape
    p.move(1, -1)
    assert [s.op.key for s in p.steps] == ["flip_v", "gaussian"]
    p.set_enabled(0, False)
    assert np.allclose(p.apply(image), apply_operation("gaussian", image, sigma=1.0))
    # a failing step is bypassed and reports its error instead of killing the live view
    p.steps[1].params["sigma"] = -1.0
    assert np.array_equal(p.apply(image), image.astype(np.float32))
    assert p.steps[1].error
    p.remove(0)
    p.clear()
    assert len(p) == 0


def test_temporal_average_and_background_state(image):
    p = Pipeline()
    p.add("temporal_average", {"frames": 3})
    frames = [image.astype(np.float32) + k for k in range(3)]
    for f in frames:
        out = p.apply(f)
    assert np.allclose(out, np.mean(frames, axis=0), atol=1e-3)
    # temporal steps can be skipped (used inside phase-stepping sequences)
    assert np.array_equal(p.apply(image, include_temporal=False), image.astype(np.float32))

    p = Pipeline()
    p.add("background", {"frames": 2, "mode": "subtract"})
    p.apply(image)
    p.apply(image)  # background learned from the first two frames
    assert np.allclose(p.apply(image.astype(np.float32) + 5), 5, atol=1e-3)
    p.reset_state()
    assert np.array_equal(p.apply(image), image.astype(np.float32))  # learning again


@pytest.mark.parametrize("key", list(REGISTRY))
def test_every_operation_has_a_plain_description(key):
    op = REGISTRY[key]
    assert op.description and op.description[0].isupper() and op.description.endswith(".")
    if not op.phase_safe:  # the menu tooltip must say why it is a bad idea before phase analysis
        assert "phase analysis" in op.description


def test_phase_safe_flags():
    assert {k for k, op in REGISTRY.items() if not op.phase_safe} == UNSAFE
    assert REGISTRY["gaussian"].phase_safe and REGISTRY["background"].phase_safe

    @register("_test_view_only", "Test view only", "filter", phase_safe=False)
    def _view_only(img, state):
        return img
    try:
        assert REGISTRY["_test_view_only"].phase_safe is False
    finally:
        del REGISTRY["_test_view_only"]


def test_has_unsafe_steps():
    p = Pipeline()
    p.add("gaussian")
    assert p.has_unsafe_steps() == []
    canny = p.add("canny")
    p.add("fft_spectrum")
    assert [s.op.key for s in p.has_unsafe_steps()] == ["canny", "fft_spectrum"]
    p.set_enabled(2, False)  # unticked steps do not matter
    assert p.has_unsafe_steps() == [canny]


def test_step_labels():
    p = Pipeline()
    assert p.add("gaussian").label == "Gaussian blur · sigma 2 px"
    assert p.add("median").label == "Median · kernel 5 px"
    assert p.add("bilateral").label == \
        "Bilateral (edge preserving) · diameter 7 px, intensity σ 0.1, spatial σ 5 px"
    assert p.add("invert").label == "Invert"  # no parameters: just the name
    assert p.add("gamma", {"gamma": 0.8}).label == "Gamma · 0.8"  # not "gamma 0.8" twice
    assert p.add("rotate", {"angle": 12.5}).label == "Rotate by angle… · angle 12.5°, border reflect"
    assert p.add("crop", {"x": 10, "width": 64}).label == "Crop · left 10 px, top 0 px, width 64 px, height to edge"
    assert p.add("fft_highpass").label == "FFT high-pass · cut-off 0.02, window gaussian, keep mean level"
    assert p.add("fft_highpass", {"keep_mean": False}).label == "FFT high-pass · cut-off 0.02, window gaussian"
    assert p.add("background", {"smooth": 3}).label == \
        "Background subtraction / flat-field · mode subtract, frames averaged 10, smooth background 3 px"
    for key in REGISTRY:  # no raw "key=value" dumps or parameter notes left in any label
        label = Pipeline().add(key).label
        assert "=" not in label and "(px" not in label and "fraction" not in label


def test_label_for_custom_operation():
    @register("_test_custom", "Custom", "filter", params=(
        Param("strength", "Strength", "float", 1.5),
        Param("size", "Window size (µm)", "int", 3),
        Param("note", "Comment (anything goes)", "str", "hi")))
    def _custom(img, state, strength, size, note):
        return img
    try:
        assert Pipeline().add("_test_custom").label == "Custom · strength 1.5, window 3 µm, comment hi"
    finally:
        del REGISTRY["_test_custom"]


def test_background_status_and_recapture(image):
    p = Pipeline()
    step = p.add("background", {"frames": 3})
    assert step.status == "waiting for frames"
    p.apply(image)
    assert step.status == "learning background 1/3"
    p.apply(image)
    assert step.status == "learning background 2/3"
    p.apply(image)
    assert step.status.startswith("background stored ")
    assert step.status == f"background stored {step.state['stored_at']}"
    assert len(step.state["stored_at"]) == 8  # HH:MM:SS
    p.reset_step(0)  # "Re-capture background"
    assert step.state == {} and step.status == "waiting for frames"
    assert np.array_equal(p.apply(image), image.astype(np.float32))  # learning again
    assert step.status == "learning background 1/3"


def test_temporal_average_status(image):
    p = Pipeline()
    step = p.add("temporal_average", {"frames": 3})
    assert step.status == "averaging 0/3 frames"
    p.apply(image)
    assert step.status == "averaging 1/3 frames"
    p.apply(image)
    assert step.status == "averaging 2/3 frames"
    p.apply(image)
    assert step.status == ""
    p.apply(image)
    assert step.status == ""
    p.set_params(0, {"frames": 4})  # a longer window starts filling again
    assert step.status == "averaging 0/4 frames"
    p.apply(image)
    assert step.status == "averaging 1/4 frames"
    p.set_params(0, {"frames": 2, "mode": "exponential"})
    for k in range(2):
        assert step.status == f"averaging {k}/2 frames"
        p.apply(image)
    assert step.status == ""
    assert p.add("gaussian").status == ""


def test_set_params_keeps_the_learned_background(image):
    p = Pipeline()
    step = p.add("background", {"frames": 2, "mode": "subtract"})
    p.apply(image)
    p.apply(image)
    bg = step.state["bg"]
    p.steps[0].error = "old error"
    p.set_params(0, {"frames": 2, "mode": "divide (flat-field)"})  # only the mode changes
    assert step.error is None and step.state["bg"] is bg
    assert step.status.startswith("background stored")
    assert np.allclose(p.apply(image), float(bg.mean()), rtol=1e-4)  # flat-fielded with the same background
    p.set_params(0, {"frames": 2, "mode": "subtract"})
    assert np.allclose(p.apply(image.astype(np.float32) + 5), 5, atol=1e-3)

    p.set_params(0, {"frames": 3, "mode": "subtract"})  # more frames: learn again
    assert step.status == "waiting for frames"
    assert np.array_equal(p.apply(image + 7), (image + 7).astype(np.float32))
    assert step.status == "learning background 1/3"

    for _ in range(2):
        p.apply(image)
    assert step.status.startswith("background stored")
    p.set_params(0, {"frames": 3, "smooth": 4.0})  # smoothing changes the background: learn again
    assert step.status == "waiting for frames"
    p.apply(image)
    assert step.status == "learning background 1/3"


def test_set_params_keeps_temporal_buffer(image):
    p = Pipeline()
    step = p.add("temporal_average", {"frames": 3})
    p.apply(image)
    p.apply(image)
    p.set_params(0, {"frames": 3})  # unchanged values (e.g. dialog re-applied) keep the buffer
    assert len(step.state["buf"]) == 2 and step.status == "averaging 2/3 frames"


def test_bypass(image):
    p = Pipeline()
    p.add("flip_h")
    step = p.add("background", {"frames": 1})
    assert p.bypass is False
    p.bypass = True
    out = p.apply(image)
    assert out.dtype == np.float32 and np.array_equal(out, image.astype(np.float32))
    assert step.state == {}  # nothing ran, so nothing was learned
    p.bypass = False
    assert np.allclose(p.apply(image), 0)  # processing again: background learned from the first frame
    assert step.status.startswith("background stored")


def test_reset_step_only_touches_one_step(image):
    p = Pipeline()
    p.add("background", {"frames": 1})
    p.add("temporal_average", {"frames": 2})
    p.apply(image)
    p.steps[0].error = "boom"
    p.reset_step(0)
    assert p.steps[0].state == {} and p.steps[0].error is None
    assert len(p.steps[1].state["buf"]) == 1
    with pytest.raises(IndexError):
        p.reset_step(5)


def test_has_temporal_steps():
    p = Pipeline()
    p.add("gaussian")
    p.add("background")
    assert p.has_temporal_steps() is False  # background is stateful, but not temporal
    p.add("temporal_average")
    assert p.has_temporal_steps() is True
    p.set_enabled(2, False)  # unticked steps do not matter
    assert p.has_temporal_steps() is False


def test_reset_swaps_in_a_fresh_state_dict(image):
    p = Pipeline()
    bg = p.add("background", {"frames": 1})
    avg = p.add("temporal_average", {"frames": 3})
    p.apply(image)
    old_bg, old_avg = bg.state, avg.state
    p.reset_step(0)
    assert bg.state == {} and bg.state is not old_bg
    assert old_bg["bg"] is not None  # a frame still using the old dict keeps working on it
    assert avg.state is old_avg
    p.reset_state()
    assert avg.state == {} and avg.state is not old_avg and len(old_avg["buf"]) == 1


def test_apply_runs_steps_without_holding_the_lock(image):
    import threading
    import time

    started, release = threading.Event(), threading.Event()
    calls = []

    @register("_test_slow", "Test slow", "filter")
    def _slow(img, state):
        calls.append(threading.current_thread().name)
        started.set()
        release.wait(5.0)
        state["frames"] = state.get("frames", 0) + 1  # writes into the dict it was given
        return img + 1

    try:
        p = Pipeline()
        slow = p.add("_test_slow")
        p.add("gaussian")
        results = {}
        workers = [threading.Thread(target=lambda n=n: results.update({n: p.apply(image)}), name=f"w{n}")
                   for n in (1, 2)]
        workers[0].start()
        assert started.wait(5.0)
        workers[1].start()  # a second apply waits for the first (stateful steps are not re-entrant)

        t0 = time.perf_counter()  # the GUI edits the pipeline while a frame is being processed
        p.set_enabled(1, False)
        p.set_params(1, {"sigma": 3.0})
        p.reset_step(0)
        assert len(p.steps) == 2 and p.has_unsafe_steps() == [] and slow.status == ""
        p.add("flip_h")
        p.remove(2)
        assert time.perf_counter() - t0 < 0.5
        time.sleep(0.05)
        assert workers[0].is_alive() and calls == ["w1"]  # still running, the edits did not wait for it

        release.set()
        for w in workers:
            w.join(5.0)
        assert not any(w.is_alive() for w in workers) and calls == ["w1", "w2"]
        assert slow.error is None
        # frame 1 used the snapshot (gaussian still on); frame 2 saw the gaussian unticked
        assert not np.allclose(results[1], image + 1) and np.allclose(results[2], image + 1)
        # reset_step swapped the dict under the running frame: it finished on the old one
        assert slow.state == {"frames": 1}
    finally:
        release.set()
        del REGISTRY["_test_slow"]


def test_apply_split_runs_stateful_steps_once(image):
    p = Pipeline()
    p.add("gaussian", {"sigma": 1.0})
    p.add("temporal_average", {"frames": 3})
    bg = p.add("background", {"frames": 4})
    full, nt = p.apply_split(image)
    assert bg.state["count"] == 1  # the copy used for the non-temporal branch does not count
    assert np.allclose(nt, apply_operation("gaussian", image, sigma=1.0))
    for _ in range(3):
        p.apply_split(image)
    assert bg.status.startswith("background stored")  # after exactly 4 frames
    q = Pipeline()
    q.add("gaussian", {"sigma": 1.0})
    a, b = q.apply_split(image)
    assert a is b  # no temporal step: one result
