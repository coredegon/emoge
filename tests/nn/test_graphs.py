import numpy as np
import pytest

torch = pytest.importorskip("torch")

from emoge.color import COLIBRI, HSI  # noqa: E402
from emoge.graphs import Graph, color_graph  # noqa: E402
from emoge.nn import FuzzyColors, GraphLayer, batch, readout  # noqa: E402
from emoge.nn import color_graph as tensor_color_graph  # noqa: E402

IMAGES = np.random.default_rng(0).integers(0, 256, (2, 12, 16, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)  # a layer starts at random; the same start every run


def graph(size, value):
    edges = np.zeros((size, size, 1))
    edges[0, 1] = edges[1, 0] = 1.0
    return Graph(
        np.full((size, 2), value), edges, tuple("abcd"[:size]), ("p", "q"), ("e",)
    )


def test_a_batch_fills_smaller_graphs_with_0():
    nodes, edges, mask = batch([graph(2, 1.0), graph(4, 2.0)])
    assert nodes.shape == (2, 4, 2)
    assert edges.shape == (2, 4, 4, 1)
    assert mask.tolist() == [[True, True, False, False], [True] * 4]
    assert nodes[0, 2:].abs().sum() == 0
    other = Graph(np.ones((2, 1)), np.zeros((2, 2, 1)), ("a", "b"), ("p",), ("e",))
    with pytest.raises(ValueError, match="share their features"):
        batch([graph(2, 1.0), other])


def test_a_layer_sends_along_edges_and_keeps_padding_empty():
    nodes, edges, mask = batch([graph(2, 1.0), graph(4, 2.0)])
    layer = GraphLayer(2, 1, 5)
    out = layer(nodes, edges, mask)
    assert out.shape == (2, 4, 5)
    assert out[0, 2:].abs().sum() == 0
    alone = layer(nodes, torch.zeros_like(edges), mask)  # no edges, no messages
    assert not torch.allclose(out[1, 0], alone[1, 0])
    assert torch.allclose(out[1, 2], alone[1, 2])  # node c has no edge
    assert readout(out, mask).shape == (2, 5)
    assert torch.allclose(readout(out, mask)[0], out[0, :2].mean(dim=0))


def test_the_tensor_color_graph_is_that_of_emoge_graphs():
    degrees = torch.as_tensor(np.moveaxis(COLIBRI.degrees(IMAGES), 0, -1))
    nodes, edges = tensor_color_graph(degrees)
    expected = color_graph(COLIBRI, IMAGES[1])
    assert torch.allclose(nodes[1], torch.as_tensor(expected.nodes[:, :2]))
    assert torch.allclose(edges[1, ..., 0], torch.as_tensor(expected.edges[..., 0]))


def test_the_colors_learn_through_the_graph():
    colors, layer = FuzzyColors(COLIBRI), GraphLayer(2, 1, 4)
    values = torch.as_tensor(HSI.from_rgb(IMAGES / 255), dtype=torch.float32)
    nodes, edges = tensor_color_graph(colors(values))
    readout(layer(nodes, edges)).sum().backward()
    grads = [p.grad for p in colors.parameters()]
    assert sum(float(g.abs().sum()) for g in grads if g is not None) > 0
