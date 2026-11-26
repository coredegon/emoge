import numpy as np
import pytest

from emoge.regions import grid


def test_a_crisp_grid_cuts_the_image_into_cells():
    cells = grid((4, 6), 2, 3)
    assert cells.shape == (6, 4, 6)
    assert (cells.sum(axis=0) == 1).all()
    assert cells[0, :2, :2].all()  # the top left cell
    assert cells[5, 2:, 4:].all()  # the bottom right one
    assert (cells.sum(axis=(1, 2)) == 4).all()


def test_soft_cells_pass_into_their_neighbours_and_sum_to_1():
    cells = grid((30, 40), 3, 3, soft=True)
    assert cells.shape == (9, 30, 40)
    assert cells.sum(axis=0) == pytest.approx(1.0)
    assert 0 < cells[0, 9, 13] < 1  # between the first cell and the next
    assert cells[:, 15, 20].argmax() == 4  # by the middle, the middle cell
    assert cells[4, 15, 20] > 0.9


def test_a_grid_has_cells():
    assert (grid((5, 5), 1, 1, soft=True) == np.ones((1, 5, 5))).all()
    with pytest.raises(ValueError, match="at least one cell"):
        grid((5, 5), 0, 2)
