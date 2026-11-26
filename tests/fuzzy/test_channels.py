import pytest

from emoge.fuzzy import Channel


def test_a_channel_needs_a_range():
    with pytest.raises(ValueError, match="low < high"):
        Channel("flat", 1.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        Channel("endless", 0.0, float("inf"))


def test_channels_of_the_same_axis_are_equal():
    assert Channel("hue", 0, 360, periodic=True) == Channel("hue", 0, 360, True)
    assert Channel("hue", 0, 360) != Channel("hue", 0, 360, periodic=True)
