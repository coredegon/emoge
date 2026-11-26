import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection, PathCollection

from emoge.color import COLIBRI
from emoge.graphs import Graph, color_graph, plot_graph, region_graph
from emoge.regions import grid

FLAG = np.zeros((40, 60, 3), dtype=np.uint8)
FLAG[:, :30], FLAG[:, 30:] = [200, 30, 30], [30, 60, 200]  # red beside blue
CELLS = grid(FLAG.shape, 2, 3)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def dots(ax):
    return next(c for c in ax.collections if isinstance(c, PathCollection))


def lines(ax):
    return next(c for c in ax.collections if isinstance(c, LineCollection))


def test_a_graph_of_colors_stands_round_a_circle():
    graph = color_graph(COLIBRI, FLAG)
    ax = plot_graph(graph)
    share = graph.nodes[:, 0]
    shown = share >= 0.01 * share.max()
    assert len(dots(ax).get_offsets()) == shown.sum()
    assert np.allclose(np.hypot(*dots(ax).get_offsets().T), 1.0)  # on the circle
    x, y = dots(ax).get_offsets().T
    turns = np.diff(np.unwrap(np.arctan2(y, x)))
    assert np.allclose(turns, -2 * np.pi / shown.sum())  # evenly, clockwise
    assert len(ax.texts) == shown.sum()  # few enough to name
    strong = (share >= 0.5 * share.max()).sum()
    assert len(plot_graph(graph, least=0.5).texts) == strong == 3


def test_a_graph_of_regions_stands_over_its_image():
    graph = region_graph(COLIBRI, FLAG, CELLS)
    ax = plot_graph(graph, image=FLAG)
    assert len(ax.images) == 1
    places = dots(ax).get_offsets()
    assert len(places) == 6
    assert places[0].tolist() == pytest.approx([9.5, 9.5])  # middle of the first cell
    assert len(lines(ax).get_segments()) == 7  # cells side by side, or one above
    assert not ax.texts


def test_edges_are_drawn_by_the_feature_named():
    edges = np.zeros((3, 3, 2))
    edges[0, 1] = edges[1, 0] = [1.0, 0.0]  # only the first feature joins 0 and 1
    graph = Graph(np.ones((3, 1)), edges, ("a", "b", "c"), ("x1",), ("p", "q"))
    assert len(lines(plot_graph(graph)).get_segments()) == 1
    assert not any(
        isinstance(c, LineCollection) for c in plot_graph(graph, edges="q").collections
    )
