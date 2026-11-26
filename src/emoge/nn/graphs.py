"""Graphs in a network: tensors of graphs, messages along their edges, a readout.

A batch of graphs is `nodes`, `(..., nodes, features)`, `edges`, `(..., nodes,
nodes, features)`, 0 where two nodes share no edge, and `mask`, `(..., nodes)`,
true for the nodes that are there.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from emoge.graphs.graph import Graph
from emoge.nn.pooling import cooccurrence, palette


def batch(graphs: Sequence[Graph]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return graphs as tensors, filled with 0 up to the most nodes among them.

    ```python
    nodes, edges, mask = batch([region_graph(...), ...])
    ```

    The graphs share their features, as graphs made the same way do.
    """
    if not graphs:
        raise ValueError("a batch needs graphs")
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
