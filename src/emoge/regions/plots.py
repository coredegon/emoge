"""Plots of regions: an image with the borders of its regions, filled or not.

Every function returns Axes, never the Figure, so a notebook shows the plot once.
Matplotlib is imported by the first plot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from emoge.color.spaces.oklab import OKLAB
from emoge.color.spaces.rgb import fit_into_rgb, to_float_rgb
from emoge.regions.geometry import geometry

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_regions(
    image: ArrayLike,
    masks: ArrayLike,
    ax: Axes | None = None,
    *,
    fill: bool = False,
    borders: bool = True,
    numbers: bool = False,
) -> Axes:
    """Draw an image with the borders of its regions.

    A border runs between pixels whose strongest region differs: white, edged
    in black, to show on any image. With `fill`, every pixel takes the mean
    color of its regions, each by its degree: flat in crisp regions, blended in
    soft ones; colors are averaged in Oklab, as people see them. `borders=False`
    leaves out the borders, which soft regions blur. With `numbers`, every
    region bears its number at its middle, as the nodes of its graph are named.
    """
    import matplotlib.patheffects as effects
    import matplotlib.pyplot as plt

    rgb = to_float_rgb(image)
    regions = np.asarray(masks, dtype=np.float64)
    height, width = rgb.shape[:2]
    if ax is None:
        scale = 6.0 / max(height, width)
        size = (width * scale, height * scale)
        _, ax = plt.subplots(figsize=size, layout="constrained")
    ax.imshow(_filled(rgb, regions) if fill else rgb, interpolation="nearest")
    if borders:
        border = _border(regions.argmax(axis=0))
        halo = border.copy()  # one pixel wider, black under the white
        halo[1:] |= border[:-1]
        halo[:-1] |= border[1:]
        halo[:, 1:] |= border[:, :-1]
        halo[:, :-1] |= border[:, 1:]
        for lines, shade, alpha in [(halo, 0.0, 0.45), (border, 1.0, 0.9)]:
            overlay = np.zeros((height, width, 4))
            overlay[..., :3] = shade
            overlay[..., 3] = lines * alpha
            ax.imshow(overlay, interpolation="nearest")
    if numbers:
        outline = [effects.withStroke(linewidth=2.0, foreground="black")]
        for number, (area, y, x, _, _) in enumerate(geometry(regions).tolist()):
            if area > 0:
                ax.text(
                    x * width - 0.5,
                    y * height - 0.5,
                    str(number),
                    color="white",
                    fontsize="x-small",
                    ha="center",
                    va="center",
                    path_effects=outline,
                )
    ax.set_axis_off()
    return ax


def _filled(
    rgb: NDArray[np.float64], regions: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Every pixel in the mean color of its regions, each by its degree, in Oklab."""
    lab = OKLAB.from_rgb(rgb)
    mass = np.maximum(regions.sum(axis=(1, 2)), 1e-12)
    means = np.einsum("rhw,hwc->rc", regions, lab) / mass[:, None]
    weight = np.maximum(regions.sum(axis=0), 1e-12)[..., None]
    return fit_into_rgb(OKLAB.to_rgb(np.einsum("rhw,rc->hwc", regions, means) / weight))


def _border(labels: NDArray[np.intp]) -> NDArray[np.bool_]:
    """The pixels whose region differs from that of the pixel right or below."""
    border = np.zeros(labels.shape, dtype=bool)
    border[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    border[:-1] |= labels[:-1] != labels[1:]
    return border
