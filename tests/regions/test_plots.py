import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from emoge.regions import grid, plot_regions

IMAGE = np.random.default_rng(0).integers(0, 256, (20, 30, 3), dtype=np.uint8)
CELLS = grid(IMAGE.shape, 2, 3)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_borders_are_drawn_over_the_image():
    ax = plot_regions(IMAGE, CELLS)
    assert len(ax.images) == 3  # the image, a black halo and white lines
    lines = np.asarray(ax.images[2].get_array())[..., 3] > 0
    assert lines[:, 9].all()  # between the first and second column of cells
    assert not lines[:9, 3].any()  # inside the first cell, above the row of borders
    assert len(plot_regions(IMAGE, CELLS, borders=False).images) == 1


def test_filled_regions_take_their_mean_color():
    ax = plot_regions(IMAGE, CELLS, fill=True, borders=False)
    filled = np.asarray(ax.images[0].get_array())
    first = filled[:10, :10].reshape(-1, 3)
    assert np.allclose(first, first[0])  # one color all over a crisp region
    assert not np.allclose(filled[0, 0], filled[0, 15])


def test_regions_bear_their_numbers():
    ax = plot_regions(IMAGE, CELLS, numbers=True)
    assert [text.get_text() for text in ax.texts] == ["0", "1", "2", "3", "4", "5"]
    _, given = plt.subplots()
    assert plot_regions(IMAGE, CELLS, given) is given
