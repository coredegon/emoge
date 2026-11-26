import numpy as np
import pytest

from emoge.regions import GEOMETRY, geometry


def test_area_middle_and_spread_of_every_region():
    regions = np.zeros((3, 4, 8))
    regions[0, :2, :4] = 1  # the top left quarter
    regions[1] = 0.5  # half of everything
    shapes = geometry(regions)
    assert shapes.shape == (3, len(GEOMETRY))
    area, y, x, spread_y, spread_x = shapes[0]
    assert area == 0.25
    assert (y, x) == (0.25, 0.25)
    assert spread_y == pytest.approx(0.125)  # rows at 1/8 and 3/8
    assert spread_x == pytest.approx(np.std([1, 3, 5, 7]) / 16)
    assert shapes[1, :3] == pytest.approx([0.5, 0.5, 0.5])
    assert shapes[2].tolist() == [0, 0, 0, 0, 0]  # an empty region
