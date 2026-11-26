import numpy as np
import pytest

from emoge.color import ColorModel, Space
from emoge.fuzzy import Channel, partition

GRAY = Space(
    "gray",
    (Channel("lightness", 0.0, 1.0),),
    lambda rgb: rgb.mean(axis=-1, keepdims=True),
    lambda gray: np.repeat(gray, 3, axis=-1),
)


def test_a_space_finds_its_channels_by_name():
    assert GRAY.channel("lightness") == Channel("lightness", 0.0, 1.0)
    with pytest.raises(KeyError, match="no channel 'hue'"):
        GRAY.channel("hue")


def test_a_space_of_your_own_carries_a_model():
    lightness = partition(GRAY.channel("lightness"), {"dark": 0, "light": 1})
    model = ColorModel("grays", GRAY, {"gray": [lightness]})
    assert model.label([[0, 0, 0], [255, 255, 255]]).tolist() == ["dark", "light"]
    assert model.prototypes().tolist() == [[0, 0, 0], [1, 1, 1]]
