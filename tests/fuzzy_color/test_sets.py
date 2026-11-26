import numpy as np
import pytest

from emoge.fuzzy_color import FuzzySet
from emoge.fuzzy_color.math import membership as mf

LOW = FuzzySet(mf.falling, (0, 10))
HIGH = FuzzySet(mf.rising, (0, 10))
X = np.array([2.0, 5.0, 8.0])  # LOW: 0.8, 0.5, 0.2; HIGH: 0.2, 0.5, 0.8


def test_a_set_is_a_function_and_its_parameters():
    warm = FuzzySet(mf.triangle, (10, 20, 30))
    assert warm.degrees([5, 15, 20]).tolist() == [0, 0.5, 1]
    assert warm.degrees(15).shape == ()
    assert repr(warm) == "triangle(10, 20, 30)"
    assert warm == FuzzySet(mf.triangle, (10, 20, 30))
    assert repr(FuzzySet(mf.gaussian, (180, 20), name="cyanish")) == "cyanish"


def test_any_function_of_your_own_makes_a_set():
    spike = FuzzySet(lambda x, width: np.exp(-np.abs(x) / width), (2.0,))
    assert float(spike.degrees(0.0)) == 1.0


def test_wrong_parameters_fail_when_the_set_is_made():
    with pytest.raises(ValueError, match="ascending"):
        FuzzySet(mf.triangle, (30, 20, 10))


def test_sets_combine_value_by_value():
    assert (LOW & HIGH).degrees(X) == pytest.approx([0.2, 0.5, 0.2])  # the smaller
    assert (LOW | HIGH).degrees(X) == pytest.approx([0.8, 0.5, 0.8])  # the larger
    assert (~LOW).degrees(X) == pytest.approx([0.2, 0.5, 0.8])  # 1 - degree
    assert (LOW | ~LOW).degrees(X) == pytest.approx([0.8, 0.5, 0.8])


def test_a_combined_set_prints_as_it_is_written():
    assert repr(LOW & HIGH) == "(falling(0, 10) & rising(0, 10))"
    assert repr(LOW | ~HIGH) == "(falling(0, 10) | ~rising(0, 10))"
    assert LOW & HIGH == LOW & HIGH


def test_sets_have_no_truth_value():
    with pytest.raises(TypeError, match="no truth value"):
        _ = LOW and HIGH
