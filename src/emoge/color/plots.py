"""Plots of color models: their terms, their colors and the palettes of images.

Every function returns Axes, never the Figure, so a notebook shows the plot once.
Matplotlib is imported by the first plot. A single variable is drawn by
`emoge.fuzzy.plot_variable`.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from emoge.color.model import ColorModel
from emoge.color.spaces.rgb import fit_into_rgb
from emoge.fuzzy.channels import Channel
from emoge.fuzzy.plots import draw_terms
from emoge.fuzzy.variables import Variable

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_EDGE_COLOR = "#898781"  # outlines pale colors against a white background


def plot_terms(model: ColorModel) -> NDArray[np.object_]:
    """Draw the terms of every variable: a row per family, a column per channel.

    Every term is painted in the color it stands for; a channel a family does
    not describe has its Axes turned off.
    """
    import matplotlib.pyplot as plt

    channels = model.space.channels
    # A channel with more terms gets a wider column, room for its legend.
    widths = [
        2.4 + 0.5 * max((len(v.terms) for v in _variables_on(model, c)), default=0)
        for c in channels
    ]
    figure, axes = plt.subplots(
        len(model.families),
        len(channels),
        squeeze=False,
        figsize=(sum(widths), 3.3 * len(model.families)),
        width_ratios=widths,
        layout="constrained",
    )
    for row, (family, variables) in zip(axes, model.families.items(), strict=True):
        by_channel = {variable.channel: variable for variable in variables}
        for ax, channel in zip(row, channels, strict=True):
            if channel in by_channel:
                variable = by_channel[channel]
                colors = _term_colors(model, variables, variable).tolist()
                draw_terms(ax, variable, colors)
            else:
                ax.set_axis_off()
        next(ax for ax in row if ax.axison).set_ylabel(f"{family}\nmembership")
    figure.suptitle(f"{model.name}: {len(model.colors)} colors")
    return axes


def plot_swatches(model: ColorModel) -> NDArray[np.object_]:
    """Draw the typical color of every color as tiles, a panel per family.

    The terms of a family's first variable run across, the combinations of the
    terms of its other variables run down.
    """
    import matplotlib.pyplot as plt

    families = list(model.families.items())
    columns = [len(variables[0].terms) for _, variables in families]
    rows = [len(_combinations(variables[1:])) for _, variables in families]
    figure, axes = plt.subplots(
        len(families),
        1,
        squeeze=False,
        figsize=(1.1 * max(columns) + 2.0, 0.45 * sum(rows) + 0.9 * len(rows)),
        height_ratios=rows,
        layout="constrained",
    )
    prototypes = model.prototypes()
    first_color = 0
    for ax, (family, variables), across, down in zip(
        axes[:, 0], families, columns, rows, strict=True
    ):
        colors = prototypes[first_color : first_color + across * down]
        first_color += across * down
        # Colors go with the first variable slowest: (across, down, rgb) as stored.
        ax.imshow(colors.reshape(across, down, 3).swapaxes(0, 1), aspect="auto")
        ax.set_xticks(range(across), labels=list(variables[0].terms))
        row_names = [", ".join(names) for names in _combinations(variables[1:])]
        ax.set_yticks(range(down), labels=row_names)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        ax.set_title(family, loc="left")
    figure.suptitle(f"{model.name}: {len(model.colors)} colors")
    return axes[:, 0]


def plot_palette(
    model: ColorModel, weights: ArrayLike, ax: Axes | None = None, top: int = 12
) -> Axes:
    """Draw the largest shares of a palette as bars painted in their colors."""
    import matplotlib.pyplot as plt

    shares = np.asarray(weights, dtype=np.float64)
    if shares.shape != (len(model.colors),):
        raise ValueError(
            f"a palette of {model.name} has {len(model.colors)} weights, "
            f"got the shape {shares.shape}"
        )
    largest = np.argsort(-shares, kind="stable")[:top]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.3 * len(largest) + 1), layout="constrained")
    rows = np.arange(len(largest))
    ax.barh(
        rows,
        shares[largest],
        color=model.prototypes()[largest],
        edgecolor=_EDGE_COLOR,
        linewidth=0.5,
    )
    ax.set_yticks(rows, labels=[model.colors[i] for i in largest])
    ax.invert_yaxis()  # the largest share on top
    for row, share in enumerate(shares[largest].tolist()):
        ax.text(share, row, f" {share:.0%}", va="center", fontsize="small")
    ax.set_xlabel("share of the image")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def _variables_on(model: ColorModel, channel: Channel) -> list[Variable]:
    """The variables of all families on one channel."""
    return [
        v
        for variables in model.families.values()
        for v in variables
        if v.channel == channel
    ]


def _combinations(variables: tuple[Variable, ...]) -> list[tuple[str, ...]]:
    return list(itertools.product(*(variable.terms for variable in variables)))


def _term_colors(
    model: ColorModel, family: tuple[Variable, ...], variable: Variable
) -> NDArray[np.float64]:
    """The RGB color of every term of a variable among the colors of its family.

    The other channels stay at the only term of a variable of one term, and at
    the middle of the channel otherwise.
    """
    point = {c.name: (c.low + c.high) / 2.0 for c in model.space.channels}
    for other in family:
        if len(other.terms) == 1:
            point[other.channel.name] = float(other.core_centers()[0])
    rows = []
    for center in variable.core_centers():
        point[variable.channel.name] = float(center)
        rows.append([point[channel.name] for channel in model.space.channels])
    return fit_into_rgb(model.space.to_rgb(np.array(rows)))
