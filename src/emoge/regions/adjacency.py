"""`adjacency`: how much every two regions border each other."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def adjacency(masks: ArrayLike, distance: int = 1) -> NDArray[np.float64]:
    """Return how much every two regions are neighbours: `(regions, regions)`.

    Two pixels `distance` apart, side by side or one above the other, are
    neighbours. Their degrees in the two regions are multiplied and averaged
    over all neighbours both ways round: the result is symmetric, its diagonal
    is the inside of a region and the rest are the borders between regions.
    """
    regions = np.asarray(masks, dtype=np.float32)  # exact for counts of crisp ones
    count, height, width = regions.shape
    if distance < 1 or (height <= distance and width <= distance):
        raise ValueError(
            f"an image of {height} x {width} has no neighbours {distance} apart"
        )
    side_by_side = regions[:, :, :-distance], regions[:, :, distance:]
    one_above = regions[:, :-distance], regions[:, distance:]
    totals = np.zeros((count, count))
    pairs = 0
    for one, other in (side_by_side, one_above):
        one, other = one.reshape(count, -1), other.reshape(count, -1)
        totals += one @ other.T
        pairs += one.shape[1]
    return (totals + totals.T) / (2 * pairs)
