"""`difference`: how far apart colors are on every channel of a space."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from emoge.color.space import Space
from emoge.color.spaces.oklch import OKLCH
from emoge.color.spaces.rgb import to_float_rgb


def difference(a: ArrayLike, b: ArrayLike, space: Space = OKLCH) -> NDArray[np.float64]:
    """Return how far apart colors are on every channel of a space.

    ```python
    difference([255, 0, 0], [0, 0, 255])  # lightness, chroma and hue of Oklch
    ```

    `a` and `b` are pixels as a `ColorModel` takes them, and broadcast against
    each other: the result has their shape, with a difference for every
    channel last. A difference is never below 0; on a periodic channel it goes
    the shorter way round, so two hues are at most 180 apart.
    """
    gap = np.abs(space.from_rgb(to_float_rgb(a)) - space.from_rgb(to_float_rgb(b)))
    for i, channel in enumerate(space.channels):
        if channel.periodic:
            around = np.mod(gap[..., i], channel.high - channel.low)
            gap[..., i] = np.minimum(around, channel.high - channel.low - around)
    return gap
