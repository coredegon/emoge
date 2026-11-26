import numpy as np
import pytest

from emoge.color.spaces import RGB, fit_into_rgb, to_float_rgb


def test_whole_numbers_are_8_bit_levels():
    assert to_float_rgb(np.array([0, 51, 255], dtype=np.uint8)).tolist() == [0, 0.2, 1]
    assert to_float_rgb([0, 51, 255]).tolist() == [0, 0.2, 1]
    assert to_float_rgb([0.0, 0.2, 1.0]).tolist() == [0, 0.2, 1]
    with pytest.raises(ValueError, match="0 to 255"):
        to_float_rgb(np.array([0, 1000, 65535], dtype=np.uint16))
    with pytest.raises(ValueError, match="3 components"):
        to_float_rgb([[1, 2], [3, 4]])
    with pytest.raises(TypeError, match="whole numbers or floats"):
        to_float_rgb(np.array(["a", "b", "c"]))


def test_too_bright_colors_are_darkened_and_keep_their_hue():
    rgb = np.array([[2.0, 1.0, 0.5], [0.5, 0.2, -0.1]])
    assert fit_into_rgb(rgb).tolist() == [[1.0, 0.5, 0.25], [0.5, 0.2, 0.0]]


def test_the_rgb_space_changes_nothing():
    rgb = np.array([0.1, 0.5, 0.9])
    assert RGB.from_rgb(rgb) is rgb
    assert [c.name for c in RGB.channels] == ["red", "green", "blue"]
