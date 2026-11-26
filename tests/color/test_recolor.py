import numpy as np

from emoge.color import COLIBRI, HSI, recolor
from emoge.color.models.colibri import HUE, INTENSITY

# Red, orange, a reddish color and green.
IMAGE = np.array([[[255, 0, 0], [255, 128, 0], [200, 60, 50], [30, 120, 40]]], np.uint8)


def test_what_is_source_becomes_target():
    recolored = recolor(IMAGE, HSI, HUE, "red", "blue")
    assert recolored.dtype == np.uint8
    assert COLIBRI.label(recolored).tolist() == [
        [
            "blue/high/dark",
            "orange/high/medium",
            "blue/medium/medium",
            "green/medium/dark",
        ]
    ]
    assert (recolored[0, [1, 3]] == IMAGE[0, [1, 3]]).all()  # neither red at all


def test_weights_keep_the_change_to_a_region():
    recolored = recolor(IMAGE, HSI, HUE, "red", "blue", weights=[[1, 1, 0, 1]])
    assert COLIBRI.label(recolored)[0, 0] == "blue/high/dark"
    assert (recolored[0, 2] == IMAGE[0, 2]).all()


def test_a_channel_that_is_no_circle_stays_in_its_range():
    grays = np.array([[[20, 20, 20], [240, 240, 240]]], dtype=np.uint8)
    lighter = recolor(grays, HSI, INTENSITY, "black", "white")
    assert lighter.tolist() == [[[255, 255, 255], [240, 240, 240]]]


def test_floats_come_back_as_floats():
    recolored = recolor(IMAGE / 255, HSI, HUE, "red", "blue")
    assert recolored.dtype == np.float64
    assert recolored.min() >= 0
    assert recolored.max() <= 1
