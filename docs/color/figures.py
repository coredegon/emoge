"""Draw the figures of the color guide: `uv run docs/color/figures.py`."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emoge import color

IMAGES = Path(__file__).parent / "img"


def save(figure, name):
    figure.savefig(IMAGES / name, dpi=144)
    plt.close(figure)


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    save(color.plot_terms(color.COLIBRI)[0, 0].figure, "colibri_terms.png")
    save(color.plot_swatches(color.COLIBRI)[0].figure, "colibri_swatches.png")
