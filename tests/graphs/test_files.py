import pickle

import numpy as np
import pytest

from emoge.graphs import Graph, load, save


def numbered(count, seed=0):
    """A graph of `count` nodes, numbered, with an edge between every next two."""
    rng = np.random.default_rng(seed)
    edges = np.zeros((count, count, 2))
    for i in range(count - 1):
        edges[i, i + 1] = edges[i + 1, i] = rng.random(2) + 0.1
    names = tuple(str(i) for i in range(count))
    return Graph(rng.random((count, 3)), edges, names, ("a", "b", "c"), ("x", "y"))


def test_graphs_of_every_size_come_back(tmp_path):
    written = [numbered(count, seed) for seed, count in enumerate([5, 1, 12, 3])]
    assert save(iter(written), tmp_path) == 4  # one pass is enough
    saved = load(tmp_path)
    assert len(saved) == 4
    assert saved.sizes.tolist() == [5, 1, 12, 3]
    for back, graph in zip(saved, written, strict=True):
        assert back.names == graph.names
        assert (back.node_features, back.edge_features) == (("a", "b", "c"), ("x", "y"))
        assert back.nodes == pytest.approx(graph.nodes, abs=1e-6)
        assert back.edges == pytest.approx(graph.edges, abs=1e-6)


def test_shared_names_are_kept(tmp_path):
    first, second = numbered(3), numbered(3, seed=1)
    names = ("red", "green", "blue")
    shared = [
        Graph(g.nodes, g.edges, names, g.node_features, g.edge_features)
        for g in (first, second)
    ]
    save(shared, tmp_path)
    assert [graph.names for graph in load(tmp_path)] == [names, names]


def test_indexing_as_a_sequence(tmp_path):
    save([numbered(2), numbered(4), numbered(6)], tmp_path)
    saved = load(tmp_path)
    assert len(saved[-1].names) == 6
    assert [len(graph.names) for graph in saved[1:]] == [4, 6]
    with pytest.raises(IndexError):
        saved[3]


def test_graphs_without_edges(tmp_path):
    lonely = Graph(
        np.ones((1, 3)), np.zeros((1, 1, 2)), ("0",), ("a", "b", "c"), ("x", "y")
    )
    save([lonely, lonely], tmp_path)
    back = load(tmp_path)[1]
    assert back.edges.tolist() == [[[0.0, 0.0]]]
    assert back.nodes.tolist() == [[1.0, 1.0, 1.0]]


def test_graphs_go_to_other_processes_without_their_files(tmp_path):
    save([numbered(4)], tmp_path)
    saved = load(tmp_path)
    saved[0]  # maps the files
    copy = pickle.loads(pickle.dumps(saved))
    assert copy._maps is None
    assert copy[0].edges == pytest.approx(saved[0].edges)


def test_what_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError, match="no graphs"):
        save([], tmp_path)
    other = numbered(3)
    other = Graph(other.nodes, other.edges, other.names, ("a", "b", "d"), ("x", "y"))
    with pytest.raises(ValueError, match="features"):
        save([numbered(3), other], tmp_path / "features")
    named = Graph(
        other.nodes, other.edges, ("r", "g", "b"), ("a", "b", "c"), ("x", "y")
    )
    renamed = Graph(
        other.nodes, other.edges, ("r", "g", "y"), ("a", "b", "c"), ("x", "y")
    )
    with pytest.raises(ValueError, match="names"):
        save([named, renamed], tmp_path / "names")
