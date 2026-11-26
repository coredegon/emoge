import pytest

from emoge.fuzzy import rising, truth

MOST = rising(0.5, 0.8)
DARK = [0.9, 0.8, 0.2, 1.0]  # how dark each of four images is
FEAR = [1, 1, 0, 1]  # which of them are fearful


def test_a_sentence_is_as_true_as_its_quantifier_of_the_share():
    assert truth(MOST, DARK) == pytest.approx(0.75)  # a share of 0.725
    assert truth(MOST, [0, 0, 0, 0]) == 0.0


def test_among_keeps_the_sentence_to_a_group():
    assert truth(MOST, DARK, among=FEAR) == 1.0  # a share of 0.9
    soft = [0.5, 0.5, 1, 0.5]  # the smaller degrees sum to 1.7, the group to 2.5
    assert truth(MOST, DARK, among=soft) == pytest.approx(0.6)  # a share of 0.68


def test_a_sentence_needs_items():
    with pytest.raises(ValueError, match="items"):
        truth(MOST, DARK, among=[0, 0, 0, 0])
