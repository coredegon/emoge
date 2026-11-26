import numpy as np
import pytest

from emoge.color.spaces import CIELCH, cielch_to_rgb, rgb_to_cielch


def test_cielch_has_the_values_of_the_tables():
    red = rgb_to_cielch(np.array([1.0, 0, 0]))
    assert red == pytest.approx([53.2408, 104.5518, 39.9990], abs=1e-4)


def test_grays_have_no_chroma_and_hue_0():
    grays = rgb_to_cielch(np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]]))
    assert grays[:, 1:] == pytest.approx(0, abs=1e-4)
    assert (grays[:, 2] == 0).all()


def test_cielch_goes_back_to_rgb():
    rgb = np.random.default_rng(0).uniform(0, 1, (50, 40, 3))
    lch = rgb_to_cielch(rgb)
    assert (lch[..., 2] >= 0).all()
    assert (lch[..., 2] < 360).all()
    assert cielch_to_rgb(lch) == pytest.approx(rgb, abs=1e-9)


def test_the_cielch_space():
    assert [c.name for c in CIELCH.channels] == ["lightness", "chroma", "hue"]
    assert CIELCH.channel("hue").periodic
