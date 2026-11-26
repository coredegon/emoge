"""Draw the figures of the graphs guide: `uv run docs/graphs/figures.py`."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emoge import color, data, graphs, regions

IMAGES = Path(__file__).parent / "img"
PHOTO = Path(__file__).parent.parent / "coffee.jpg"  # Rachel Michetti, CC0


def two_graphs():
    """One photo, two graphs: of its regions over it, of its colors round a circle."""
    photo = data.load_image(PHOTO)
    pieces = regions.superpixels(photo, 150)
    figure, axes = plt.subplots(
        1, 2, figsize=(10, 4), layout="constrained", width_ratios=[3, 2]
    )
    graphs.plot_graph(
        graphs.region_graph(color.COLIBRI, photo, pieces), axes[0], image=photo
    )
    graphs.plot_graph(graphs.color_graph(color.COLIBRI, photo), axes[1])
    axes[0].set_title("its regions", fontsize=10)
    axes[1].set_title("its colors", fontsize=10)
    figure.savefig(IMAGES / "graphs.png", dpi=144)
    plt.close(figure)


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    two_graphs()
