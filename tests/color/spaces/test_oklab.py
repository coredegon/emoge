import numpy as np
import pytest

from emoge.color.spaces import OKLAB, oklab_to_rgb, rgb_to_oklab

# Colors in Oklab as published with it.
KNOWN = {
    "white": ([1, 1, 1], [1, 0, 0]),
    "black": ([0, 0, 0], [0, 0, 0]),
    "red": ([1, 0, 0], [0.627955, 0.224863, 0.125846]),
    "green": ([0, 1, 0], [0.866440, -0.233888, 0.179498]),
    "blue": ([0, 0, 1], [0.452014, -0.032457, -0.311528]),
}


@pytest.mark.parametrize(("rgb", "oklab"), KNOWN.values(), ids=KNOWN.keys())
def test_oklab_has_the_published_values(rgb, oklab):
    assert rgb_to_oklab(np.array(rgb, dtype=float)) == pytest.approx(oklab, abs=1e-6)


def test_oklab_goes_back_to_rgb():
    rgb = np.random.default_rng(0).uniform(0, 1, (50, 40, 3))
    assert oklab_to_rgb(rgb_to_oklab(rgb)) == pytest.approx(rgb, abs=1e-9)


def test_oklab_sees_lightness_as_people_do():
    # One HSI intensity, a third, yet olive looks far lighter than blue.
    olive, blue = rgb_to_oklab(np.array([[128 / 255, 128 / 255, 0], [0, 0, 1]]))
    assert olive[0] > blue[0] + 0.1


def test_the_oklab_space():
    assert [c.name for c in OKLAB.channels] == ["lightness", "a", "b"]
