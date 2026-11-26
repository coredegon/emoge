import numpy as np
import pytest

torch = pytest.importorskip("torch")

from emoge.color import COLIBRI, HSI  # noqa: E402
from emoge.graphs import Graph, color_graph, load, save  # noqa: E402
from emoge.nn import FuzzyColors, GraphEncoder, GraphLayer, batch, readout  # noqa: E402
from emoge.nn import color_graph as tensor_color_graph  # noqa: E402

IMAGES = np.random.default_rng(0).integers(0, 256, (2, 12, 16, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)  # a layer starts at random; the same start every run


def graph(size, value):
    edges = np.zeros((size, size, 1))
    edges[0, 1] = edges[1, 0] = 1.0
    return Graph(
        np.full((size, 2), value),
        edges,
        tuple(map(str, range(size))),
        ("p", "q"),
        ("e",),
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


def ring(size, seed=0, copies=1):
    """`copies` rings of `size` nodes each, the same random nodes in every ring."""
    rng = np.random.default_rng(seed)
    nodes = np.tile(rng.random((size, 2)), (copies, 1))
    edges = np.zeros((size * copies, size * copies, 3))
    for start in range(0, size * copies, size):
        for i in range(size):
            j = start + (i + 1) % size
            edges[start + i, j] = edges[j, start + i] = [0.5, 0.2, 1.0]
    names = tuple(map(str, range(size * copies)))
    return Graph(nodes, edges, names, ("p", "q"), ("e", "f", "g"))


def test_pairs_give_their_labels_too():
    nodes, _, mask, labels = batch([(graph(2, 1.0), 0), (graph(4, 2.0), 1)])
    assert nodes.shape == (2, 4, 2)
    assert mask.tolist() == [[True, True, False, False], [True] * 4]
    assert labels.tolist() == [0, 1]
    scores = batch(
        [(graph(2, 1.0), np.array([0.1, 0.9])), (graph(3, 1.0), np.zeros(2))]
    )
    assert scores[3].shape == (2, 2)


def test_a_loader_of_saved_graphs(tmp_path):
    save([ring(3), ring(5, seed=1), ring(4, seed=2)], tmp_path)
    pairs = torch.utils.data.StackDataset(load(tmp_path), [2, 0, 1])
    loader = torch.utils.data.DataLoader(pairs, batch_size=2, collate_fn=batch)
    nodes, edges, mask, labels = next(iter(loader))
    assert nodes.shape == (2, 5, 2)
    assert edges.shape == (2, 5, 5, 3)
    assert mask.sum(dim=1).tolist() == [3, 5]
    assert labels.tolist() == [2, 0]


def test_the_encoder_gives_one_vector_a_graph_whatever_its_size():
    encoder = GraphEncoder(2, 3, width=8, depth=2)
    assert encoder.out_features == 16
    alone = encoder(*batch([ring(4)]))
    assert alone.shape == (1, 16)
    together = encoder(*batch([ring(4), ring(9, seed=1)]))  # filled up to 9 nodes
    assert torch.allclose(alone[0], together[0], atol=1e-6)
    twice = encoder(*batch([ring(4, copies=2)]))  # the same graph twice over
    assert torch.allclose(alone, twice, atol=1e-6)


def test_the_encoder_learns_to_tell_graphs_apart():
    pairs = [(ring(size, seed=size), size % 2) for size in range(3, 11)]
    nodes, edges, mask, labels = batch(pairs)
    encoder = GraphEncoder(2, 3, width=16, depth=2)
    head = torch.nn.Linear(encoder.out_features, 2)
    optimizer = torch.optim.Adam([*encoder.parameters(), *head.parameters()], lr=0.01)
    losses = []
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(
            head(encoder(nodes, edges, mask)), labels
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] / 2
