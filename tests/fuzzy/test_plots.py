import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from emoge.fuzzy import Channel, partition, plot_variable
from emoge.fuzzy.plots import draw_terms

VERTICAL = partition(Channel("y", 0.0, 1.0), {"top": 0, "middle": 0.5, "bottom": 1})


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_variable_draws_every_term():
    ax = plot_variable(VERTICAL)
    assert len(ax.lines) == 3
    assert ax.get_xlabel() == "y"
    assert ax.get_ylabel() == "membership"
    _, given = plt.subplots()
    assert plot_variable(VERTICAL, given) is given


def test_draw_terms_paints_every_term_in_its_color():
    _, ax = plt.subplots()
    draw_terms(ax, VERTICAL, ["red", "green", "blue"])
    assert [to_hex(line.get_color()) for line in ax.lines] == [
        "#ff0000",
        "#008000",
        "#0000ff",
    ]
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == list(VERTICAL.terms)
    assert ax.get_ylabel() == ""  # left to the plot that draws the terms
