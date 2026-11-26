"""`save` and `load`: the graphs of a whole dataset in one folder, read one at a time.

A folder holds the nodes of all graphs, their edges as pairs of nodes and their
features, all in float32 and int32 written raw, and a JSON file of what they
are. Reading maps the files and makes a graph only when it is asked for.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

from emoge.graphs.graph import Graph

_FILES = {"nodes": np.float32, "pairs": np.int32, "edges": np.float32}


def save(graphs: Iterable[Graph], folder: str | os.PathLike[str]) -> int:
    """Write graphs into a folder, one after another, and return how many.

    ```python
    save(each(images, make_graph), "graphs")   # never all of them in memory
    ```

    The graphs share their features, and their names too, as graphs of colors
    do, or are numbered from 0, as graphs of regions are. Numbers are kept as
    float32 and the edges sparse: a graph of 100 superpixels takes about 45 kB,
    a graph of colors about 120 kB.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    counts: list[tuple[int, int]] = []
    first: Graph | None = None
    shared = numbered = True
    with (
        open(folder / "nodes.raw", "wb") as nodes,
        open(folder / "pairs.raw", "wb") as pairs,
        open(folder / "edges.raw", "wb") as edges,
    ):
        for graph in graphs:
            if first is None:
                first = graph
            if (graph.node_features, graph.edge_features) != (
                first.node_features,
                first.edge_features,
            ):
                raise ValueError("the graphs of a folder share their features")
            shared &= graph.names == first.names
            numbered &= graph.names == tuple(map(str, range(len(graph.names))))
            if not (shared or numbered):
                raise ValueError(
                    "saved graphs share their names, or are numbered from 0"
                )
            pair, features = graph.sparse()
            graph.nodes.astype(np.float32).tofile(nodes)
            pair.T.astype(np.int32).tofile(pairs)
            features.astype(np.float32).tofile(edges)
            counts.append((len(graph.names), pair.shape[1]))
    if first is None:
        raise ValueError("there are no graphs to save")
    about = {
        "node_features": first.node_features,
        "edge_features": first.edge_features,
        "names": first.names if shared and not numbered else None,
        "counts": counts,
    }
    (folder / "graphs.json").write_text(json.dumps(about), encoding="utf-8")
    return len(counts)


def load(folder: str | os.PathLike[str]) -> Saved:
    """Return the graphs a folder holds, each made when it is asked for."""
    return Saved(folder)


class Saved(Sequence[Graph]):
    """The graphs of a folder that `save` wrote: a sequence, read as asked.

    `sizes` is the number of nodes of every graph, to batch graphs of a size
    together. The files are mapped, not read, and mapped again in every process
    a `torch.utils.data.DataLoader` starts.
    """

    def __init__(self, folder: str | os.PathLike[str]) -> None:
        self.folder = Path(folder)
        about = json.loads((self.folder / "graphs.json").read_text(encoding="utf-8"))
        self.node_features = tuple(about["node_features"])
        self.edge_features = tuple(about["edge_features"])
        self.names = None if about["names"] is None else tuple(about["names"])
        counts = np.array(about["counts"], dtype=np.int64).reshape(-1, 2)
        self.sizes = counts[:, 0]
        self._node_starts = np.concatenate([[0], np.cumsum(counts[:, 0])])
        self._edge_starts = np.concatenate([[0], np.cumsum(counts[:, 1])])
        self._maps: dict[str, NDArray[Any]] | None = None

    def __len__(self) -> int:
        return len(self.sizes)

    @overload
    def __getitem__(self, index: int) -> Graph: ...
    @overload
    def __getitem__(self, index: slice) -> list[Graph]: ...
    def __getitem__(self, index: int | slice) -> Graph | list[Graph]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if not -len(self) <= index < len(self):
            raise IndexError(f"{len(self)} graphs, no graph {index}")
        index %= len(self)
        maps = self._mapped()
        nodes = slice(self._node_starts[index], self._node_starts[index + 1])
        edges = slice(self._edge_starts[index], self._edge_starts[index + 1])
        count = int(self.sizes[index])
        dense = np.zeros((count, count, len(self.edge_features)))
        first, second = maps["pairs"][edges].T
        dense[first, second] = maps["edges"][edges]
        names = self.names or tuple(map(str, range(count)))
        return Graph(
            maps["nodes"][nodes].astype(np.float64),
            dense,
            names,
            self.node_features,
            self.edge_features,
        )

    def _mapped(self) -> dict[str, NDArray[Any]]:
        """The files of the folder, mapped the first time they are needed."""
        if self._maps is None:
            widths = {
                "nodes": len(self.node_features),
                "pairs": 2,
                "edges": len(self.edge_features),
            }
            rows = {
                "nodes": self._node_starts[-1],
                "pairs": self._edge_starts[-1],
                "edges": self._edge_starts[-1],
            }
            self._maps = {}
            for name, kind in _FILES.items():
                shape = (int(rows[name]), widths[name])
                if shape[0] * shape[1] == 0:  # nothing to map
                    self._maps[name] = np.zeros(shape, dtype=kind)
                else:
                    path = self.folder / f"{name}.raw"
                    self._maps[name] = np.memmap(
                        path, dtype=kind, mode="r", shape=shape
                    )
        return self._maps

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_maps": None}  # map again, never copy the files
