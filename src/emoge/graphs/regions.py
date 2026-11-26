"""`region_graph`: the regions of an image as a graph, those that touch joined."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from emoge.color.model import ColorModel
from emoge.color.spaces.oklab import OKLAB
from emoge.color.spaces.rgb import fit_into_rgb, to_float_rgb
from emoge.graphs.graph import Graph
from emoge.graphs.relations import Relations, relate
from emoge.regions.adjacency import adjacency
from emoge.regions.geometry import GEOMETRY, geometry


def region_graph(
    model: ColorModel,
    image: ArrayLike,
    masks: ArrayLike,
    relations: Relations | None = None,
) -> Graph:
    """Return the regions of an image as a graph: a node for every mask.

    ```python
    region_graph(COLIBRI, image, superpixels(image, 100))
    ```

    A node is the palette of its region in the colors of the model, where the
    region is and how large, its `GEOMETRY`, and its mean color in Oklab. Two
    regions that touch share an edge: how much they border each other, how far
    apart their mean colors look, and the words of `relations`, as `relate`
    gives them. Regions that do not touch have no edge.
    """
    regions = np.asarray(masks, dtype=np.float64)
    palettes = model.palette(image, regions)  # refuses a region with no pixels
    lab = OKLAB.from_rgb(to_float_rgb(image))
    mean_lab = (
        np.einsum("rhw,hwc->rc", regions, lab) / regions.sum(axis=(1, 2))[:, None]
    )
    nodes = np.column_stack([palettes, geometry(regions), mean_lab])
    touch = adjacency(regions)
    looks = np.linalg.norm(mean_lab[:, None] - mean_lab[None], axis=-1)
    words, degrees = relate(fit_into_rgb(OKLAB.to_rgb(mean_lab)), relations or {})
    edges = np.concatenate([np.stack([touch, looks], axis=-1), degrees], axis=-1)
    edges *= (touch > 0)[..., None]
    edges[np.arange(len(edges)), np.arange(len(edges))] = 0.0
    return Graph(
        nodes,
        edges,
        tuple(str(i) for i in range(len(regions))),
        (*model.colors, *GEOMETRY, "lightness", "a", "b"),
        ("border", "distance", *words),
    )
