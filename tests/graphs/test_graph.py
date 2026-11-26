import numpy as np
import pytest

from emoge.graphs import Graph


def test_a_graph_keeps_its_edges_dense_and_gives_them_sparse():
    edges = np.zeros((3, 3, 2))
    edges[0, 1] = edges[1, 0] = [0.5, 1.0]
    graph = Graph(np.ones((3, 1)), edges, ("a", "b", "c"), ("x",), ("p", "q"))
    pairs, features = graph.sparse()
    assert pairs.tolist() == [[0, 1], [1, 0]]
    assert features.tolist() == [[0.5, 1.0], [0.5, 1.0]]
    assert repr(graph) == "Graph(3 nodes, 1 edges)"


def test_a_graph_checks_its_shapes():
    with pytest.raises(ValueError, match="nodes need"):
        Graph(np.ones((2, 1)), np.zeros((3, 3, 0)), ("a", "b", "c"), ("x",), ())
    with pytest.raises(ValueError, match="edges need"):
        Graph(np.ones((3, 1)), np.zeros((3, 3, 1)), ("a", "b", "c"), ("x",), ())
