"""`Graph`: nodes and the edges between every two of them, with named features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Graph:
    """Nodes and the edges between every two of them, with named features.

    `nodes` is `(nodes, len(node_features))`; `edges` is `(nodes, nodes,
    len(edge_features))`, all 0 where two nodes have no edge, and on the
    diagonal. `names` names the nodes.
    """

    nodes: NDArray[np.float64]
    edges: NDArray[np.float64]
    names: tuple[str, ...]
    node_features: tuple[str, ...]
    edge_features: tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.names)
        if self.nodes.shape != (count, len(self.node_features)):
            raise ValueError(
                f"nodes need the shape {(count, len(self.node_features))}, "
                f"got {self.nodes.shape}"
            )
        if self.edges.shape != (count, count, len(self.edge_features)):
            raise ValueError(
                f"edges need the shape {(count, count, len(self.edge_features))}, "
                f"got {self.edges.shape}"
            )

    def sparse(self) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Return the edges there are, each way: `(2, edges)`, `(edges, features)`.

        ```python
        pairs, features = graph.sparse()   # as PyTorch Geometric takes them
        ```
        """
        there = np.abs(self.edges).sum(axis=-1) > 0
        first, second = np.nonzero(there)
        return np.stack([first, second]), self.edges[first, second]

    def __repr__(self) -> str:
        pairs = int((np.abs(self.edges).sum(axis=-1) > 0).sum()) // 2
        return f"Graph({len(self.names)} nodes, {pairs} edges)"
