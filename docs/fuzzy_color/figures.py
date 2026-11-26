"""Draw the figures of the fuzzy_color guide: `uv run docs/fuzzy_color/figures.py`."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import emoge.fuzzy_color as fc
from emoge.fuzzy_color.models.colibri import HUE

IMAGES = Path(__file__).parent / "img"
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, GRAY = "#333333", "#aaaaaa"
# Every membership function as a set whose parameters show its shape on [0, 10].
GALLERY = [
    fc.triangle(2, 5, 8),
    fc.trapezoid(1, 3, 6, 9),
    fc.rising(3, 7),
    fc.falling(3, 7),
    fc.rectangle(3, 7),
    fc.singleton(5),
    fc.piecewise_linear(((1, 0), (3, 1), (6, 0.4), (9, 0))),
    fc.gaussian(5, 1.5),
    fc.two_sided_gaussian(4, 1, 6, 2),
    fc.generalized_bell(5, 2, 3),
    fc.sigmoid(5, 2),
    fc.sigmoid_difference(3, 4, 7, 4),
    fc.sigmoid_product(3, 4, 7, -4),
    fc.s_shape(2, 8),
    fc.z_shape(2, 8),
    fc.pi_shape(1, 4, 6, 9),
]


def save(figure, name):
    figure.savefig(IMAGES / name, dpi=144)
    plt.close(figure)


def membership_gallery():
    figure, axes = plt.subplots(
        4, 4, figsize=(9, 5.8), sharex=True, sharey=True, layout="constrained"
    )
    x = np.linspace(0, 10, 2001)
    for ax, fuzzy_set in zip(axes.flat, GALLERY, strict=True):
        degrees = fuzzy_set.degrees(x)
        ax.plot(x, degrees, color=BLUE, linewidth=2)
        ax.fill_between(x, degrees, color=BLUE, alpha=0.1, linewidth=0)
        title = repr(fuzzy_set)
        if len(title) > 34:  # the points of piecewise_linear
            title = title[: title.index("(")] + "(points)"
        ax.set_title(title, fontsize=8, family="monospace")
        ax.set(xlim=(0, 10), ylim=(0, 1.05), xticks=(0, 5, 10), yticks=(0, 1))
        ax.spines[["top", "right"]].set_visible(False)
    save(figure, "membership.png")


def combining():
    """What & | ~ make of dark and light intensities."""
    dark = fc.falling(0.2, 0.4)
    light = fc.rising(0.6, 0.8)
    panels = {
        "dark | light": dark | light,
        "~dark": ~dark,
        "~dark & ~light": ~dark & ~light,
    }
    figure, axes = plt.subplots(
        1, 3, figsize=(9, 2.4), sharey=True, layout="constrained"
    )
    x = np.linspace(0, 1, 801)
    for ax, (title, result) in zip(axes, panels.items(), strict=True):
        ax.plot(x, dark.degrees(x), "--", color=INK, linewidth=1, label="dark")
        if "light" in title:
            ax.plot(x, light.degrees(x), "--", color=GRAY, linewidth=1, label="light")
        ax.fill_between(x, result.degrees(x), color=BLUE, alpha=0.25, linewidth=0)
        ax.plot(x, result.degrees(x), color=BLUE, linewidth=2, label="result")
        _bare(ax, title, (0, 1))
        ax.set_xlabel("intensity")
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    save(figure, "combining.png")


def union_of_neighbours():
    """red | orange dips where they cross; joined in a partition, they add up."""
    figure, ax = plt.subplots(figsize=(5, 2.4), layout="constrained")
    hue = np.linspace(-30, 60, 721)  # -30 is 330: hue goes round
    red, orange = HUE.select("red", "orange").degrees(hue)
    ax.plot(hue, red, "--", color=RED, linewidth=1, label="red")
    ax.plot(hue, orange, "--", color=ORANGE, linewidth=1, label="orange")
    ax.fill_between(hue, np.maximum(red, orange), color=BLUE, alpha=0.25, linewidth=0)
    ax.plot(hue, np.maximum(red, orange), color=BLUE, linewidth=2, label="red | orange")
    ax.plot(
        hue, red + orange, ":", color=INK, linewidth=1.5, label="joined in a partition"
    )
    _bare(ax, "red | orange", (-30, 60))
    figure.legend(loc="outside right center", frameon=False, fontsize=8)
    save(figure, "union_of_neighbours.png")


def _bare(ax, title, limits):
    ax.set_title(title, family="monospace", fontsize=10)
    ax.set(xlim=limits, ylim=(0, 1.05), yticks=(0, 1))
    ax.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    membership_gallery()
    combining()
    union_of_neighbours()
    save(fc.plot_terms(fc.COLIBRI)[0, 0].figure, "colibri_terms.png")
    save(fc.plot_swatches(fc.COLIBRI)[0].figure, "colibri_swatches.png")
