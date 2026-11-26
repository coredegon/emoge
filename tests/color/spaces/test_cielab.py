import numpy as np
import pytest

from emoge.color.spaces import CIELAB, cielab_to_rgb, rgb_to_cielab

# Colors in CIELAB against D65, as the tables of color science give them.
KNOWN = {
    "white": ([1, 1, 1], [100, 0, 0]),
    "black": ([0, 0, 0], [0, 0, 0]),
    "red": ([1, 0, 0], [53.2408, 80.0925, 67.2032]),
    "green": ([0, 1, 0], [87.7347, -86.1827, 83.1793]),
    "blue": ([0, 0, 1], [32.2970, 79.1875, -107.8602]),
}


@pytest.mark.parametrize(("rgb", "lab"), KNOWN.values(), ids=KNOWN.keys())
def test_cielab_has_the_values_of_the_tables(rgb, lab):
    assert rgb_to_cielab(np.array(rgb, dtype=float)) == pytest.approx(lab, abs=1e-4)


def test_cielab_goes_back_to_rgb():
    rgb = np.random.default_rng(0).uniform(0, 1, (50, 40, 3))
    assert cielab_to_rgb(rgb_to_cielab(rgb)) == pytest.approx(rgb, abs=1e-9)
    near_black = np.array([[0.001, 0.002, 0.0]])  # where the line replaces the root
    assert cielab_to_rgb(rgb_to_cielab(near_black)) == pytest.approx(near_black)


def test_the_cielab_space():
    assert [c.name for c in CIELAB.channels] == ["lightness", "a", "b"]
