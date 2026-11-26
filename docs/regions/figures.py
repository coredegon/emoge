"""Draw the figures of the regions guide: `uv run docs/regions/figures.py`."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emoge import data, regions

IMAGES = Path(__file__).parent / "img"
PHOTO = Path(__file__).parent.parent / "coffee.jpg"  # Rachel Michetti, CC0


def superpixels():
    """The same photo cut into superpixels: borders, then filled, crisp and soft."""
    photo = data.load_image(PHOTO)
    crisp = regions.superpixels(photo, 150)
    soft = regions.superpixels(photo, 150, soft=True)
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 2.6), layout="constrained")
    regions.plot_regions(photo, crisp, axes[0], numbers=True)
    regions.plot_regions(photo, crisp, axes[1], fill=True)
    regions.plot_regions(photo, soft, axes[2], fill=True, borders=False)
    titles = ["borders and numbers", "filled", "soft, filled"]
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title, fontsize=10)
    figure.savefig(IMAGES / "superpixels.png", dpi=144)
    plt.close(figure)


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    superpixels()
