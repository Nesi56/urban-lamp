import numpy as np
import pytest

from phasescope.processing import REGISTRY, Pipeline, apply_operation, operations


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
