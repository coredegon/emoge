import numpy as np
import pytest

from emoge.color import COLIBRI, OKLAB
from emoge.graphs import color_graph

IMAGE = np.random.default_rng(0).integers(0, 256, (30, 40, 3), dtype=np.uint8)


def test_a_node_for_every_color_of_the_model():
    graph = color_graph(COLIBRI, IMAGE)
    assert graph.names == COLIBRI.colors
    assert graph.node_features == ("share", "beside itself", "lightness", "a", "b")
    assert graph.nodes[:, 0] == pytest.approx(COLIBRI.palette(IMAGE))
    assert graph.nodes[:, 1] == pytest.approx(np.diagonal(COLIBRI.cooccurrence(IMAGE)))
    assert graph.nodes[:, 2:] == pytest.approx(OKLAB.from_rgb(COLIBRI.prototypes()))


def test_every_two_colors_share_an_edge():
    graph = color_graph(COLIBRI, IMAGE)
    assert graph.edge_features == ("neighbours", "distance")
    meets = COLIBRI.cooccurrence(IMAGE)
    np.fill_diagonal(meets, 0)
    assert graph.edges[..., 0] == pytest.approx(meets)
    assert (graph.edges == graph.edges.transpose(1, 0, 2)).all()
    assert (np.diagonal(graph.edges, axis1=0, axis2=1) == 0).all()
    lab = OKLAB.from_rgb(COLIBRI.prototypes())
    looks = np.linalg.norm(lab[:, None] - lab[None], axis=-1)
    assert graph.edges[..., 1] == pytest.approx(looks)
