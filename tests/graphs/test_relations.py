import numpy as np
import pytest

from emoge.color import OKLCH
from emoge.fuzzy import Channel, partition
from emoge.graphs import relate

HARMONY = partition(
    Channel("hue difference", 0, 180),
    {"same": 0, "analogous": 30, "triadic": 120, "complementary": 180},
)


def test_every_two_colors_in_every_word_of_a_relation():
    rgb = np.array([[1.0, 0, 0], [0, 0, 1], [1.0, 0.5, 0]])  # red, blue, orange
    names, degrees = relate(rgb, {"harmony": (OKLCH, "hue", HARMONY)})
    assert names == (
        "harmony/same",
        "harmony/analogous",
        "harmony/triadic",
        "harmony/complementary",
    )
    assert degrees.shape == (3, 3, 4)
    assert degrees.sum(axis=-1) == pytest.approx(1.0)
    assert degrees[0, 1].argmax() == 2  # red and blue: mostly triadic
    assert degrees[0, 2].argmax() == 1  # red and orange: mostly analogous
    assert degrees[0, 0].tolist() == [1, 0, 0, 0]


def test_no_relations_add_no_features():
    names, degrees = relate(np.zeros((4, 3)), {})
    assert names == ()
    assert degrees.shape == (4, 4, 0)
