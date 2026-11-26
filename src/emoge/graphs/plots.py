"""Plots of graphs: nodes and edges, over their image or round a circle.

Every function returns Axes, never the Figure, so a notebook shows the plot once.
Matplotlib is imported by the first plot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from emoge.color.spaces.oklab import OKLAB
from emoge.color.spaces.rgb import fit_into_rgb, to_float_rgb
from emoge.graphs.graph import Graph

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_INK = (0.33, 0.32, 0.30)  # edges on a white ground
_BLUE = (0.165, 0.471, 0.839)  # nodes with no color of their own
_RIM = (0.72, 0.71, 0.69)  # round nodes, white ones too, on a white ground


def plot_graph(
    graph: Graph,
    ax: Axes | None = None,
    *,
    image: ArrayLike | None = None,
    edges: str | None = None,
    labels: bool | None = None,
    least: float = 0.01,
) -> Axes:
    """Draw a graph: its nodes as dots, its edges as lines.

    A graph whose nodes have the features "y" and "x", as a graph of regions
    does, stands where they say, over `image` if given; any other stands round
    a circle, the nodes shown spread evenly in their order, its edges bowing to
    the middle. A node is
    painted in its color when it has "lightness", "a" and "b", in Oklab, and
    is as large as its "share" or "area"; one smaller than `least` of the
    largest is left out, with its edges. An edge is as thick and as strong as
    its feature `edges`, the first unless named. `labels` names the nodes: by
    default only round a circle of at most 30 nodes shown.
    """
    import matplotlib.pyplot as plt

    count = len(graph.names)
    size = next((n for n in ("share", "area") if n in graph.node_features), None)
    weight = np.ones(count) if size is None else _feature(graph, size)
    shown = weight >= least * weight.max(initial=0) if size else np.ones(count, bool)
    shown &= weight > 0
    placed = {"y", "x"} <= set(graph.node_features)
    over = placed and image is not None
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 6), layout="constrained")
    if placed:
        height, width = np.shape(image)[:2] if image is not None else (1, 1)
        nudge = 0.5 if over else 0.0  # pixels have their middles at whole numbers
        places = np.column_stack(
            [
                _feature(graph, "x") * width - nudge,
                _feature(graph, "y") * height - nudge,
            ]
        )
        if image is not None:
            ax.imshow(to_float_rgb(image))
        else:
            ax.invert_yaxis()  # y runs down an image
    else:  # the shown nodes in order, evenly, clockwise from the top
        turn = (np.cumsum(shown) - 1) / max(int(shown.sum()), 1)
        angle = np.pi / 2 - 2 * np.pi * turn
        places = np.column_stack([np.cos(angle), np.sin(angle)])
    index = graph.edge_features.index(edges) if edges else 0
    _draw_edges(
        ax, graph.edges[..., index], places, shown, curved=not placed, over=over
    )
    largest = max(weight[shown].max(initial=0), 1e-12)
    areas = (12 + 200 * weight / largest) if placed else (18 + 520 * weight / largest)
    ax.scatter(
        places[shown, 0],
        places[shown, 1],
        s=areas[shown],
        c=_paint(graph)[shown],
        edgecolors="black" if placed else _RIM,
        linewidths=0.6,
        zorder=2,
    )
    named = labels if labels is not None else (not placed and shown.sum() <= 30)
    if named:
        _label(ax, graph, places, shown, np.sqrt(areas) / 2, round_a_circle=not placed)
    if not placed:
        reach = 1.6 if named else 1.15
        ax.set(xlim=(-reach, reach), ylim=(-reach, reach))
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def _feature(graph: Graph, name: str) -> NDArray[np.float64]:
    return graph.nodes[:, list(graph.node_features).index(name)]


def _paint(graph: Graph) -> NDArray[np.float64]:
    """The color of every node, from its Oklab features, or one blue for all."""
    if {"lightness", "a", "b"} <= set(graph.node_features):
        lab = np.column_stack([_feature(graph, n) for n in ("lightness", "a", "b")])
        return fit_into_rgb(OKLAB.to_rgb(lab))
    return np.tile(_BLUE, (len(graph.names), 1))


def _draw_edges(
    ax: Axes,
    strength: NDArray[np.float64],
    places: NDArray[np.float64],
    shown: NDArray[np.bool_],
    *,
    curved: bool,
    over: bool,
) -> None:
    """Draw every edge between shown nodes, the stronger thicker, bolder, on top."""
    import matplotlib.patheffects as effects
    from matplotlib.collections import LineCollection

    first, second = np.nonzero(
        np.triu(strength > 0, k=1) & shown[:, None] & shown[None]
    )
    if not len(first):
        return
    share = strength[first, second] / strength[first, second].max()
    start, end = places[first], places[second]
    if curved:  # a quadratic curve through a point pulled toward the middle
        t = np.linspace(0.0, 1.0, 24)[:, None, None]
        bend = (start + end) / 2 * 0.3
        segments = (1 - t) ** 2 * start + 2 * (1 - t) * t * bend + t**2 * end
        segments = segments.transpose(1, 0, 2)
    else:
        segments = np.stack([start, end], axis=1)
    order = np.argsort(share)
    colors = np.column_stack(
        [
            np.tile((1.0, 1.0, 1.0) if over else _INK, (len(share), 1)),
            0.05 + 0.85 * share,
        ]
    )
    lines = LineCollection(
        list(segments[order]),
        linewidths=(0.4 + 3.0 * share)[order],
        colors=colors[order],
        capstyle="round",
        zorder=1,
    )
    if over:  # white edged in black, to show on any image
        lines.set_path_effects(
            [effects.withStroke(linewidth=2.4, foreground="black", alpha=0.6)]
        )
    ax.add_collection(lines)


def _label(
    ax: Axes,
    graph: Graph,
    places: NDArray[np.float64],
    shown: NDArray[np.bool_],
    radii: NDArray[np.float64],
    *,
    round_a_circle: bool,
) -> None:
    """Name the shown nodes: just outside a circle, or on them where they stand."""
    for i in np.nonzero(shown)[0]:
        x, y = places[i]
        away = (radii[i] + 4) * places[i] if round_a_circle else (0.0, 0.0)
        side = "left" if x > 0.05 else "right" if x < -0.05 else "center"
        ax.annotate(
            graph.names[i],
            (x, y),
            xytext=tuple(away),
            textcoords="offset points",
            fontsize="x-small",
            ha=side if round_a_circle else "center",
            va="center",
        )
