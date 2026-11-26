import numpy as np
import pytest

from emoge.regions import adjacency, grid


def test_regions_that_touch_are_neighbours():
    cells = grid((4, 6), 1, 3)  # three columns of cells, two pixels wide
    touch = adjacency(cells)
    assert touch.shape == (3, 3)
    assert (touch == touch.T).all()
    assert touch.sum() == pytest.approx(1.0)
    assert touch[0, 1] > 0
    assert touch[1, 2] > 0
    assert touch[0, 2] == 0  # not side by side
    # 20 side by side and 18 one above the other: 4 pairs cross each border.
    assert touch[0, 1] == pytest.approx(4 / (2 * 38))


def test_neighbours_need_room():
    with pytest.raises(ValueError, match="no neighbours"):
        adjacency(np.ones((1, 1, 1)))
