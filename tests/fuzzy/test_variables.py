import numpy as np
import pytest

from emoge.fuzzy import Channel, FuzzySet, Variable
from emoge.fuzzy.math import membership as mf

UNIT = Channel("unit", 0.0, 1.0)
HUE = Channel("hue", 0.0, 360.0, periodic=True)
LEVEL = Variable(
    UNIT,
    {"low": FuzzySet(mf.falling, (0.2, 0.4)), "high": FuzzySet(mf.rising, (0.6, 0.8))},
)


def test_a_variable_gives_the_degree_in_every_term():
    assert LEVEL.degrees(0.3).tolist() == pytest.approx([0.5, 0.0])
    assert LEVEL.degrees([[0.1, 0.7]]).shape == (2, 1, 2)
    assert repr(LEVEL) == "Variable(unit: low, high)"


def test_select_keeps_the_named_terms():
    assert list(LEVEL.select("high", "low").terms) == ["high", "low"]
    with pytest.raises(ValueError, match="no terms"):
        LEVEL.select("medium")
    with pytest.raises(ValueError, match="needs terms"):
        Variable(UNIT, {})


def test_on_a_periodic_channel_values_come_round_into_the_range():
    # A term across the seam is two pieces joined with |; ~ then holds everywhere.
    red = FuzzySet(mf.trapezoid, (330, 350, 365, 380)) | FuzzySet(
        mf.trapezoid, (-30, -10, 5, 20)
    )
    hue = Variable(HUE, {"red": red, "not red": ~red})
    expected = np.array([[1, 2 / 3, 1, 2 / 3, 1], [0, 1 / 3, 0, 1 / 3, 0]])
    assert hue.degrees([0.0, 10.0, 355.0, 370.0, -5.0]) == pytest.approx(expected)
    x = np.random.default_rng(0).uniform(-720, 720, 1000)
    assert hue.degrees(x) == pytest.approx(hue.degrees(x % 360))


def test_core_centers_are_the_middles_of_the_cores():
    assert LEVEL.core_centers() == pytest.approx([0.1, 0.9], abs=1e-3)
    red = FuzzySet(mf.trapezoid, (330, 350, 360, 361)) | FuzzySet(
        mf.trapezoid, (-1, 0, 10, 20)
    )  # core 350 to 10, across the seam
    hue = Variable(HUE, {"red": red})
    assert min(hue.core_centers()[0], 360 - hue.core_centers()[0]) == pytest.approx(
        0, abs=0.1
    )
