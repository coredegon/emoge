"""`color_graph`: the colors of an image as a graph, those of one model in all."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from emoge.color.model import ColorModel
from emoge.color.spaces.oklab import OKLAB
from emoge.graphs.graph import Graph
from emoge.graphs.relations import Relations, relate


def color_graph(
    model: ColorModel,
    image: ArrayLike,
    relations: Relations | None = None,
    distance: int = 1,
) -> Graph:
    """Return the colors of an image as a graph: a node for every color of the model.

    ```python
    color_graph(COLIBRI, image)   # 86 nodes, whatever the image
    ```

    A node is how much of the image the color takes, how often it lies beside
    itself, and its prototype in Oklab. Every two colors share an edge: how
    often they are neighbours, `distance` apart, and how far apart their
    prototypes look, in Oklab; `relations` add their words, as `relate` gives
    them. Every image of one model gives the same nodes.
    """
    meets = model.cooccurrence(image, distance)
    prototypes = model.prototypes()
    lab = OKLAB.from_rgb(prototypes)
    nodes = np.column_stack([model.palette(image), np.diagonal(meets), lab])
    looks = np.linalg.norm(lab[:, None] - lab[None], axis=-1)
    words, degrees = relate(prototypes, relations or {})
    edges = np.concatenate([np.stack([meets, looks], axis=-1), degrees], axis=-1)
    edges[np.arange(len(edges)), np.arange(len(edges))] = 0.0
    return Graph(
        nodes,
        edges,
        model.colors,
        ("share", "beside itself", "lightness", "a", "b"),
        ("neighbours", "distance", *words),
    )
