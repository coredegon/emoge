"""`geometry`: where every region is, and how large and spread."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# The columns of `geometry`; places are shares of the height and the width.
GEOMETRY = ("area", "y", "x", "spread y", "spread x")


def geometry(masks: ArrayLike) -> NDArray[np.float64]:
    """Return the area, middle and spread of every region: `(regions, 5)`.

    The area is the share of the image a region takes; its middle, `y` down
    and `x` across, and its spread, the standard deviation of its pixels, are
    shares of the height and the width. A soft region counts every pixel by its
    degree; an empty one is all 0.
    """
    regions = np.asarray(masks, dtype=np.float64)
    count, height, width = regions.shape
    y = (np.arange(height) + 0.5) / height
    x = (np.arange(width) + 0.5) / width
    rows, columns = regions.sum(axis=2), regions.sum(axis=1)  # along y and along x
    mass = rows.sum(axis=1)

    def mean(totals: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.divide(totals, mass, out=np.zeros(count), where=mass > 0)

    middle_y, middle_x = mean(rows @ y), mean(columns @ x)
    spread_y = np.sqrt(np.maximum(mean(rows @ y**2) - middle_y**2, 0.0))
    spread_x = np.sqrt(np.maximum(mean(columns @ x**2) - middle_x**2, 0.0))
    area = mass / (height * width)
    return np.stack([area, middle_y, middle_x, spread_y, spread_x], axis=-1)
