"""`recolor`: an image with one word of its colors made another, for what-ifs."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from emoge.color.space import Space
from emoge.color.spaces.rgb import fit_into_rgb, to_float_rgb
from emoge.fuzzy.variables import Variable


def recolor(
    image: ArrayLike,
    space: Space,
    variable: Variable,
    source: str,
    target: str,
    weights: ArrayLike | None = None,
) -> NDArray[np.uint8] | NDArray[np.float64]:
    """Return the image with what is `source` in it made `target`.

    ```python
    recolor(image, HSI, HUE, "red", "blue")   # what if the reds were blue?
    ```

    Every pixel moves along the channel of `variable`, from the middle of the
    core of `source` towards that of `target`, as far as it is `source`: a red
    pixel becomes blue, a half red one moves half the way, an orange one stays.
    A hue goes the shorter way round; other channels stay in their range.
    `weights`, from 0 to 1 for every pixel, keep the change to a region. The
    image comes back as it came, bytes or floats.
    """
    rgb = to_float_rgb(image)
    index = space.channels.index(variable.channel)
    names = list(variable.terms)
    centers = variable.core_centers()
    shift = centers[names.index(target)] - centers[names.index(source)]
    channel = variable.channel
    span = channel.high - channel.low
    if channel.periodic:
        shift = (shift + span / 2) % span - span / 2  # the shorter way round
    values = space.from_rgb(rgb)
    moved = values[..., index]
    amount = variable.degrees(moved)[names.index(source)]
    if weights is not None:
        amount = amount * np.asarray(weights, dtype=np.float64)
    moved = moved + amount * shift
    if not channel.periodic:
        moved = np.clip(moved, channel.low, channel.high)
    values[..., index] = moved
    recolored = fit_into_rgb(space.to_rgb(values))
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        return np.round(recolored * 255.0).astype(np.uint8)
    return recolored
