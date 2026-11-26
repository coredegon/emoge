import colorsys

import numpy as np
import pytest

from emoge.fuzzy_color.spaces import HSV, hsv_to_rgb, rgb_to_hsv

KNOWN = {
    "red": ([1, 0, 0], [0, 1, 1]),
    "yellow": ([1, 1, 0], [60, 1, 1]),
    "cyan": ([0, 1, 1], [180, 1, 1]),
    "magenta": ([1, 0, 1], [300, 1, 1]),
    "gray": ([0.5, 0.5, 0.5], [0, 0, 0.5]),
    "black": ([0, 0, 0], [0, 0, 0]),
}


@pytest.mark.parametrize(("rgb", "hsv"), KNOWN.values(), ids=KNOWN.keys())
def test_hsv_follows_its_definition(rgb, hsv):
    assert rgb_to_hsv(np.array(rgb, dtype=float)) == pytest.approx(hsv, abs=1e-12)


def test_hsv_agrees_with_the_standard_library_and_goes_back():
    rgb = np.random.default_rng(0).uniform(0, 1, (500, 3))
    hsv = rgb_to_hsv(rgb)
    expected = np.array([colorsys.rgb_to_hsv(*color) for color in rgb.tolist()])
    expected[:, 0] *= 360.0
    assert hsv == pytest.approx(expected, abs=1e-9)
    assert hsv_to_rgb(hsv) == pytest.approx(rgb, abs=1e-9)


def test_the_hsv_space():
    assert [c.name for c in HSV.channels] == ["hue", "saturation", "value"]
    assert HSV.channel("hue").periodic
