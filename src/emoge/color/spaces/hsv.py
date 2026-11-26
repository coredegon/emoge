"""HSV: hue, saturation and value, the largest of the three components."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from emoge.color.space import Space
from emoge.fuzzy.channels import Channel


def rgb_to_hsv(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert RGB in [0, 1] to hue in [0, 360), saturation and value in [0, 1].

    Grays get hue 0, and black gets saturation 0.
    """
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    value = np.maximum(np.maximum(red, green), blue)
    chroma = value - np.minimum(np.minimum(red, green), blue)
    saturation = np.divide(chroma, value, out=np.zeros_like(value), where=value > 0)
    # The hue sector, 0 to 6, from the component that is largest.
    divisor = np.where(chroma > 0, chroma, 1.0)
    sector = np.select(
        [chroma == 0, value == red, value == green],
        [0.0, np.mod((green - blue) / divisor, 6.0), (blue - red) / divisor + 2.0],
        (red - green) / divisor + 4.0,
    )
    hue = sector * 60.0
    hue = np.where(hue < 360.0, hue, 0.0)  # mod rounds tiny negative sectors up to 6
    return np.stack([hue, saturation, value], axis=-1)


def hsv_to_rgb(hsv: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert hue in degrees, saturation and value in [0, 1] to RGB."""
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    chroma = value * saturation
    sector = np.mod(hue, 360.0) / 60.0
    middle = chroma * (1.0 - np.abs(np.mod(sector, 2.0) - 1.0))
    zero = np.zeros_like(chroma)
    which = np.floor(sector).astype(np.intp) % 6
    # In each sector one component leads, one is zero and one lies between.
    red = np.choose(which, [chroma, middle, zero, zero, middle, chroma])
    green = np.choose(which, [middle, chroma, chroma, middle, zero, zero])
    blue = np.choose(which, [zero, zero, middle, chroma, chroma, middle])
    lowest = value - chroma
    return np.stack([red + lowest, green + lowest, blue + lowest], axis=-1)


HSV = Space(
    "HSV",
    (
        Channel("hue", 0.0, 360.0, periodic=True),
        Channel("saturation", 0.0, 1.0),
        Channel("value", 0.0, 1.0),
    ),
    rgb_to_hsv,
    hsv_to_rgb,
)
