import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from emoge.fuzzy_color import plot_palette, plot_swatches, plot_terms, plot_variable
from emoge.fuzzy_color.models import COLIBRI

HUE_TERMS = COLIBRI.families["chromatic"][0].terms


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_terms_draws_every_variable_of_every_family():
    axes = plot_terms(COLIBRI)
    assert axes.shape == (2, 3)  # a row per family, a column per channel
    assert not axes[1, 0].axison  # the achromatic colors have no hue
    shown = [ax for ax in axes.flat if ax.axison]
    assert [ax.get_xlabel() for ax in shown] == [
        "hue",
        "saturation",
        "intensity",
        "saturation",
        "intensity",
    ]
    assert [len(ax.lines) for ax in shown] == [9, 3, 3, 1, 5]
    legend = shown[0].get_legend()
    assert legend is not None
    shown[0].figure.canvas.draw()
    # The legend reads in order along its rows: five hues, then four.
    marks = [handle.get_window_extent() for handle in legend.legend_handles]
    places = [(-round(box.y0), round(box.x0)) for box in marks]
    names = [text.get_text() for text in legend.get_texts()]
    assert [name for _, name in sorted(zip(places, names, strict=True))] == list(
        HUE_TERMS
    )


def test_plot_swatches_shows_every_color_once():
    axes = plot_swatches(COLIBRI)
    chromatic, achromatic = (np.asarray(ax.images[0].get_array()) for ax in axes)
    assert chromatic.shape == (9, 9, 3)
    assert achromatic.shape == (1, 5, 3)
    prototypes = COLIBRI.prototypes()
    assert chromatic[1, 0].tolist() == prototypes[1].tolist()  # red, low, medium
    assert chromatic[0, 1].tolist() == prototypes[9].tolist()  # orange, low, dark


def test_plot_palette_draws_the_largest_shares_first():
    weights = np.zeros(86)
    weights[[0, 40, 85]] = 0.5, 0.3, 0.2
    ax = plot_palette(COLIBRI, weights, top=2)
    assert isinstance(ax, Axes)
    assert len(ax.patches) == 2
    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert labels == [COLIBRI.colors[0], COLIBRI.colors[40]]
    with pytest.raises(ValueError, match="86 weights"):
        plot_palette(COLIBRI, weights[:5])


def test_plot_variable_draws_every_term():
    hue = COLIBRI.families["chromatic"][0]
    ax = plot_variable(hue)
    assert len(ax.lines) == 9
    assert ax.get_xlabel() == "hue"
    _, given = plt.subplots()
    assert plot_variable(hue, given) is given
