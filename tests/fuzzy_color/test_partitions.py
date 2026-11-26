import numpy as np
import pytest

from emoge.fuzzy_color import HSI, Channel, partition

UNIT = Channel("unit", 0.0, 1.0)
HUE = HSI.channel("hue")


def test_a_partition_sums_to_one():
    level = partition(UNIT, {"low": 0.1, "medium": (0.4, 0.6), "high": 0.9})
    x = np.linspace(-0.5, 1.5, 801)
    assert level.degrees(x).sum(axis=0) == pytest.approx(1.0)
    assert level.degrees(0.25).tolist() == pytest.approx([0.5, 0.5, 0.0])
    assert level.core_centers() == pytest.approx([0.05, 0.5, 0.95], abs=1e-3)
    assert [repr(term) for term in level.terms.values()] == [
        "falling(0.1, 0.4)",
        "trapezoid(0.1, 0.4, 0.6, 0.9)",
        "rising(0.6, 0.9)",
    ]


def test_a_periodic_partition_wraps_around():
    hue = partition(HUE, {"red": (350, 10), "green": 120, "blue": 240})
    assert hue.degrees(0.0).tolist() == [1.0, 0.0, 0.0]
    assert hue.degrees(295.0) == pytest.approx([0.5, 0.0, 0.5])
    assert hue.degrees([-10.0, 10.0, 350.0, 359.9]).tolist()[0] == [1, 1, 1, 1]
    x = np.random.default_rng(0).uniform(-720, 720, 5000)
    assert hue.degrees(x) == pytest.approx(hue.degrees(x % 360))
    assert hue.degrees(x).sum(axis=0) == pytest.approx(1.0)


BAD = {
    "one term": lambda: partition(UNIT, {"only": 0.5}),
    "touching cores": lambda: partition(UNIT, {"a": (0.1, 0.5), "b": 0.5}),
    "core outside": lambda: partition(UNIT, {"a": 0.1, "b": 1.2}),
    "circle out of order": lambda: partition(HUE, {"a": 10, "b": 200, "c": 100}),
}


@pytest.mark.parametrize("build", BAD.values(), ids=BAD.keys())
def test_bad_partitions_are_rejected(build):
    with pytest.raises(ValueError, match=r"must|needs"):
        build()
