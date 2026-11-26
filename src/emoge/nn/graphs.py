"""Graphs in a network: tensors of graphs, messages along their edges, a readout.

A batch of graphs is `nodes`, `(..., nodes, features)`, `edges`, `(..., nodes,
nodes, features)`, 0 where two nodes share no edge, and `mask`, `(..., nodes)`,
true for the nodes that are there.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast, overload

import torch
from torch.utils.data import default_collate

from emoge.graphs.graph import Graph
from emoge.nn.pooling import cooccurrence, palette


@overload
def batch(
    items: Sequence[Graph],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
@overload
def batch(
    items: Sequence[tuple[Graph, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]: ...
def batch(
    items: Sequence[Graph] | Sequence[tuple[Graph, Any]],
) -> tuple[Any, ...]:
    """Return graphs as tensors, filled with 0 up to the most nodes among them.

    ```python
    nodes, edges, mask = batch([region_graph(...), ...])
    nodes, edges, mask, labels = batch([(graph, label), ...])
    ```

    The graphs share their features, as graphs made the same way do. Pairs of
    a graph and its label, as `torch.utils.data.StackDataset(graphs, labels)`
    gives them, give the labels too, stacked as a `DataLoader` stacks them: so
    `batch` is the `collate_fn` of a loader of graphs.
    """
    if not items:
        raise ValueError("a batch needs graphs")
    if isinstance(items[0], Graph):
        return _tensors(cast("Sequence[Graph]", items))
    graphs, labels = zip(*items, strict=True)
    return *_tensors(graphs), default_collate(list(labels))


def _tensors(
    graphs: Sequence[Graph],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The nodes, edges and mask of graphs, filled with 0 up to the largest."""
    first = graphs[0]
    if any(
        (g.node_features, g.edge_features) != (first.node_features, first.edge_features)
        for g in graphs
    ):
        raise ValueError("the graphs of a batch share their features")
    count = max(len(g.names) for g in graphs)
    nodes = torch.zeros(len(graphs), count, len(first.node_features))
    edges = torch.zeros(len(graphs), count, count, len(first.edge_features))
    mask = torch.zeros(len(graphs), count, dtype=torch.bool)
    for i, graph in enumerate(graphs):
        size = len(graph.names)
        nodes[i, :size] = torch.as_tensor(graph.nodes)
        edges[i, :size, :size] = torch.as_tensor(graph.edges)
        mask[i, :size] = True
    return nodes, edges, mask


def color_graph(
    degrees: torch.Tensor, distance: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the color graph of degree maps, `(..., height, width, colors)`.

    The nodes, `(..., colors, 2)`, are how much of the image every color takes
    and how often it lies beside itself; the edges, `(..., colors, colors, 1)`,
    how often two colors are neighbours. Gradients pass through, so the colors
    of a `FuzzyColors` learn along with what reads the graph.
    """
    meets = cooccurrence(degrees, distance)
    nodes = torch.stack([palette(degrees), torch.diagonal(meets, dim1=-2, dim2=-1)], -1)
    apart = 1.0 - torch.eye(meets.shape[-1], dtype=meets.dtype, device=meets.device)
    return nodes, (meets * apart).unsqueeze(-1)


class GraphLayer(torch.nn.Module):
    """One round of messages along the edges: `(..., nodes, features)` to `out`.

    ```python
    layer = GraphLayer(node_features, edge_features, 32)
    nodes = layer(nodes, edges, mask)
    ```

    Every node takes the mean of what its neighbours send it, each message
    weighed by a gate the features of its edge open, and adds what it makes of
    itself. Nodes outside `mask` neither send nor keep anything.
    """

    def __init__(
        self, node_features: int, edge_features: int, out_features: int
    ) -> None:
        super().__init__()
        self.own = torch.nn.Linear(node_features, out_features)
        self.send = torch.nn.Linear(node_features, out_features)
        self.gate = torch.nn.Linear(edge_features, 1)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the nodes after one round of messages: `(..., nodes, out)`."""
        there = edges.abs().sum(dim=-1) > 0
        if mask is not None:
            there = there & mask[..., None, :] & mask[..., :, None]
        gates = torch.sigmoid(self.gate(edges)).squeeze(-1) * there
        neighbours = there.sum(dim=-1, keepdim=True).clamp(min=1)
        updated = torch.relu(self.own(nodes) + gates @ self.send(nodes) / neighbours)
        return updated if mask is None else updated * mask[..., None]


def readout(nodes: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return the mean of the nodes in `mask` of every graph: `(..., features)`."""
    if mask is None:
        return nodes.mean(dim=-2)
    kept = mask[..., None].to(nodes.dtype)
    return (nodes * kept).sum(dim=-2) / kept.sum(dim=-2).clamp(min=1)


class GraphEncoder(torch.nn.Module):
    """Graphs of any size, one vector each: `(..., 2 * width)`.

    ```python
    encoder = GraphEncoder(node_features, edge_features)
    vectors = encoder(nodes, edges, mask)  # (graphs, encoder.out_features)
    ```

    A layer takes every node to `width` features, `depth` rounds of
    `GraphLayer` follow, each added to what was there, and the vector of a
    graph is the mean and the largest of every feature over its nodes. Messages
    are means over the neighbours, so a graph of 20 nodes and one of 500 give
    vectors alike in scale, and the nodes filling a batch change nothing.
    """

    def __init__(
        self, node_features: int, edge_features: int, width: int = 64, depth: int = 3
    ) -> None:
        super().__init__()
        self.first = torch.nn.Linear(node_features, width)
        self._norms = [torch.nn.LayerNorm(width) for _ in range(depth)]
        self._layers = [GraphLayer(width, edge_features, width) for _ in range(depth)]
        self.norms = torch.nn.ModuleList(self._norms)  # hold their parameters
        self.layers = torch.nn.ModuleList(self._layers)
        self.out_features = 2 * width

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a vector of every graph: `(..., out_features)`."""
        state = torch.relu(self.first(nodes))
        if mask is not None:
            state = state * mask[..., None]
        for norm, layer in zip(self._norms, self._layers, strict=True):
            state = state + layer(norm(state), edges, mask)
        largest = state.amax(dim=-2)  # states are never below 0, nor the filling
        return torch.cat([readout(state, mask), largest], dim=-1)
