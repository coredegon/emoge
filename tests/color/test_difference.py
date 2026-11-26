import numpy as np
import pytest

from emoge.color import COLIBRI, HSI, OKLAB, RGB, difference


def test_a_difference_for_every_channel():
    assert difference([255, 0, 0], [0, 255, 255], RGB).tolist() == [1, 1, 1]
    assert difference([1.0, 0, 0], [1.0, 0, 0]).tolist() == [0, 0, 0]


def test_hues_go_the_shorter_way_round():
    red, magenta = [255, 0, 0], [255, 0, 255]  # hues 0 and 300 in HSI
    assert difference(red, magenta, HSI)[0] == pytest.approx(60)
    hues = difference(red, np.random.default_rng(0).integers(0, 256, (100, 3)))[:, 2]
    assert hues.min() >= 0
    assert hues.max() <= 180


def test_colors_broadcast_against_each_other():
    prototypes = COLIBRI.prototypes()
    gaps = difference(prototypes[:, None], prototypes[None])
    assert gaps.shape == (86, 86, 3)
    assert gaps == pytest.approx(gaps.swapaxes(0, 1))
    assert np.diagonal(gaps, axis1=0, axis2=1) == pytest.approx(0, abs=1e-9)


def test_in_oklab_the_length_is_the_distance():
    a, b = np.array([0.2, 0.5, 0.7]), np.array([0.6, 0.1, 0.3])
    length = np.linalg.norm(difference(a, b, OKLAB))
    expected = np.linalg.norm(OKLAB.from_rgb(a) - OKLAB.from_rgb(b))
    assert length == pytest.approx(expected)
