import numpy as np
import pytest

from emoge.color import COLIBRI, OKLCH
from emoge.fuzzy import Channel, partition
from emoge.graphs import region_graph
from emoge.regions import GEOMETRY, grid

FLAG = np.zeros((40, 60, 3), dtype=np.uint8)
FLAG[:, :30], FLAG[:, 30:] = [200, 30, 30], [30, 60, 200]  # red beside blue
CELLS = grid(FLAG.shape, 2, 2)


def test_a_node_for_every_region():
    graph = region_graph(COLIBRI, FLAG, CELLS)
    assert graph.names == ("0", "1", "2", "3")
    assert graph.node_features == (*COLIBRI.colors, *GEOMETRY, "lightness", "a", "b")
    assert graph.nodes[:, :86] == pytest.approx(COLIBRI.palette(FLAG, CELLS))
    red = COLIBRI.colors.index("red/high/dark")
    assert graph.nodes[0, :86].argmax() == red  # the top left cell is red
    assert graph.nodes[0, 86] == 0.25  # a quarter of the image


def test_regions_that_touch_share_an_edge():
    graph = region_graph(COLIBRI, FLAG, CELLS)
    assert graph.edge_features == ("border", "distance")
    touching = np.abs(graph.edges).sum(axis=-1) > 0
    assert touching.astype(int).tolist() == [
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ]  # not across the diagonal
    assert graph.edges[0, 1, 1] > graph.edges[0, 2, 1]  # red to blue looks farther


def test_relations_add_their_words_to_the_edges():
    lightness = partition(
        Channel("lightness difference", 0, 1), {"alike": 0, "apart": 1}
    )
    graph = region_graph(
        COLIBRI, FLAG, CELLS, {"light": (OKLCH, "lightness", lightness)}
    )
    assert graph.edge_features == ("border", "distance", "light/alike", "light/apart")
    assert graph.edges[0, 2, 2:].sum() == pytest.approx(1.0)  # two red cells
    assert graph.edges[0, 3, 2:].sum() == 0  # no edge, no words
