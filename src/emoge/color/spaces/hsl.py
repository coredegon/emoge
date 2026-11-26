"""HSL: hue, saturation and lightness, the middle of the largest and smallest."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from emoge.color.space import Space
from emoge.color.spaces.hsv import hsv_to_rgb, rgb_to_hsv
from emoge.fuzzy.channels import Channel


def rgb_to_hsl(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert RGB in [0, 1] to hue in [0, 360), saturation and lightness in [0, 1].

    The hue is that of HSV. Grays get hue 0 and saturation 0.
    """
    largest, smallest = rgb.max(axis=-1), rgb.min(axis=-1)
    lightness = (largest + smallest) / 2.0
    room = 1.0 - np.abs(2.0 * lightness - 1.0)  # the most chroma at this lightness
    chroma = largest - smallest
    saturation = np.divide(chroma, room, out=np.zeros_like(chroma), where=room > 0)
    return np.stack([rgb_to_hsv(rgb)[..., 0], saturation, lightness], axis=-1)


def hsl_to_rgb(hsl: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert hue in degrees, saturation and lightness in [0, 1] to RGB."""
    hue, saturation, lightness = hsl[..., 0], hsl[..., 1], hsl[..., 2]
    value = lightness + saturation * np.minimum(lightness, 1.0 - lightness)
    ratio = np.divide(lightness, value, out=np.ones_like(value), where=value > 0)
    return hsv_to_rgb(np.stack([hue, 2.0 * (1.0 - ratio), value], axis=-1))


HSL = Space(
    "HSL",
    (
        Channel("hue", 0.0, 360.0, periodic=True),
        Channel("saturation", 0.0, 1.0),
        Channel("lightness", 0.0, 1.0),
    ),
    rgb_to_hsl,
    hsl_to_rgb,
)
