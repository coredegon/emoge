import numpy as np
import pytest

from emoge.color.spaces import (
    WHITE,
    XYZ,
    from_linear,
    rgb_to_xyz,
    to_linear,
    xyz_to_rgb,
)


def test_xyz_has_the_values_of_the_srgb_standard():
    assert pytest.approx([0.95047, 1.0, 1.08883], abs=1e-5) == WHITE
    red = rgb_to_xyz(np.array([1.0, 0, 0]))
    assert red == pytest.approx([0.4124564, 0.2126729, 0.0193339])
    assert rgb_to_xyz(np.array([1.0, 1, 1])) == pytest.approx(WHITE)


def test_xyz_goes_back_to_rgb():
    rgb = np.random.default_rng(0).uniform(0, 1, (50, 40, 3))
    assert xyz_to_rgb(rgb_to_xyz(rgb)) == pytest.approx(rgb, abs=1e-9)
    assert [c.name for c in XYZ.channels] == ["x", "y", "z"]


def test_linear_light_goes_both_ways_and_keeps_signs():
    levels = np.linspace(-0.2, 1.2, 141)
    assert from_linear(to_linear(levels)) == pytest.approx(levels)
    assert to_linear(np.array([0.5])) == pytest.approx([0.214041], abs=1e-6)
    assert to_linear(np.array([-0.5])) == pytest.approx([-0.214041], abs=1e-6)
