import numpy as np
import pytest

from emoge.color.spaces import OKLCH, oklch_to_rgb, rgb_to_oklch


def test_oklch_has_the_published_values():
    red = rgb_to_oklch(np.array([1.0, 0, 0]))
    assert red == pytest.approx([0.627955, 0.257683, 29.233885], abs=1e-6)


def test_grays_have_no_chroma_and_hue_0():
    grays = rgb_to_oklch(np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]]))
    assert grays[:, 1:] == pytest.approx(0, abs=1e-6)
    assert (grays[:, 2] == 0).all()


def test_oklch_goes_back_to_rgb():
    rgb = np.random.default_rng(0).uniform(0, 1, (50, 40, 3))
    lch = rgb_to_oklch(rgb)
    assert (lch[..., 2] >= 0).all()
    assert (lch[..., 2] < 360).all()
    assert oklch_to_rgb(lch) == pytest.approx(rgb, abs=1e-9)


def test_the_oklch_space():
    assert [c.name for c in OKLCH.channels] == ["lightness", "chroma", "hue"]
    assert OKLCH.channel("hue").periodic
