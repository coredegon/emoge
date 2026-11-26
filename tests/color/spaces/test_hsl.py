import numpy as np
import pytest

from emoge.color.spaces import HSL, hsl_to_rgb, rgb_to_hsl

# Colors whose hue, saturation and lightness follow from the definition.
KNOWN = {
    "red": ([1, 0, 0], [0, 1, 0.5]),
    "dark green": ([0, 0.5, 0], [120, 1, 0.25]),
    "pink": ([1, 0.5, 0.5], [0, 1, 0.75]),
    "gray": ([0.5, 0.5, 0.5], [0, 0, 0.5]),
    "white": ([1, 1, 1], [0, 0, 1]),
    "black": ([0, 0, 0], [0, 0, 0]),
}


@pytest.mark.parametrize(("rgb", "hsl"), KNOWN.values(), ids=KNOWN.keys())
def test_hsl_follows_its_definition(rgb, hsl):
    assert rgb_to_hsl(np.array(rgb, dtype=float)) == pytest.approx(hsl, abs=1e-12)


def test_hsl_goes_back_to_rgb():
    rgb = np.random.default_rng(0).uniform(0, 1, (50, 40, 3))
    assert hsl_to_rgb(rgb_to_hsl(rgb)) == pytest.approx(rgb, abs=1e-9)


def test_the_hsl_space():
    assert [c.name for c in HSL.channels] == ["hue", "saturation", "lightness"]
    assert HSL.channel("hue").periodic
