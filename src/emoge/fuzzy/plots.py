"""Plots of variables: the terms of a variable along its channel.

Every function returns Axes, never the Figure, so a notebook shows the plot once.
Matplotlib is imported by the first plot.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from emoge.fuzzy.variables import Variable

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_EDGE_COLOR = "#898781"  # outlines pale colors against a white background


def plot_variable(variable: Variable, ax: Axes | None = None) -> Axes:
    """Draw the terms of a variable, named in a legend below."""
    import matplotlib.pyplot as plt

    if ax is None:
        size = (2.4 + 0.5 * len(variable.terms), 3.0)
        _, ax = plt.subplots(figsize=size, layout="constrained")
    colors = [f"C{i}" for i in range(len(variable.terms))]
    draw_terms(ax, variable, colors)
    ax.set_ylabel("membership")
    return ax


def draw_terms(ax: Axes, variable: Variable, colors: Sequence[Any]) -> None:
    """Draw every term of a variable in its color, with a legend below.

    What `plot_variable` draws, for plots that choose the colors of the terms
    themselves.
    """
    import matplotlib.patheffects as effects

    channel = variable.channel
    x = np.linspace(channel.low, channel.high, 721)
    outline = [effects.Stroke(linewidth=3.0, foreground=_EDGE_COLOR), effects.Normal()]
    lines = []
    for name, degrees, color in zip(
        variable.terms, variable.degrees(x), colors, strict=True
    ):
        # Only where the term is above 0, so the zeros of all terms do not pile up
        # along the bottom.
        near = np.convolve(degrees > 0, [1, 1, 1], "same") > 0
        shown = np.where(near, degrees, np.nan)
        lines += ax.plot(
            x, shown, color=color, linewidth=2, path_effects=outline, label=name
        )
    # A legend fills column by column, at most five columns; deal the lines so
    # that they read in order along the rows.
    columns = math.ceil(len(lines) / math.ceil(len(lines) / 5))
    by_rows = [lines[i] for c in range(columns) for i in range(c, len(lines), columns)]
    ax.legend(
        handles=by_rows,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=columns,
        frameon=False,
        fontsize="small",
    )
    ax.set(xlim=(channel.low, channel.high), ylim=(0, 1.05), xlabel=channel.name)
    ax.spines[["top", "right"]].set_visible(False)
