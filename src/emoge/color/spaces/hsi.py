"""HSI: hue, saturation and intensity, the mean of the three components."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from emoge.color.space import Space
from emoge.fuzzy.channels import Channel

_SQRT_3 = math.sqrt(3.0)


def rgb_to_hsi(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert RGB in [0, 1] to hue in [0, 360), saturation and intensity in [0, 1].

    Hue is the angle of the color around the gray axis, 0 for red. Grays get
    hue 0, and black gets saturation 0.
    """
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    intensity = (red + green + blue) / 3.0
    weakest = np.minimum(np.minimum(red, green), blue)
    gray_share = np.divide(
        weakest, intensity, out=np.ones_like(intensity), where=intensity > 0
    )
    saturation = np.clip(1.0 - gray_share, 0.0, 1.0)
    angle = np.arctan2(_SQRT_3 * (green - blue), 2.0 * red - green - blue)
    hue = np.mod(np.degrees(angle), 360.0)
    hue = np.where(hue < 360.0, hue, 0.0)  # mod rounds tiny negative angles up to 360
    return np.stack([hue, saturation, intensity], axis=-1)


def hsi_to_rgb(hsi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert hue in degrees, saturation and intensity in [0, 1] to RGB.

    Not every combination is a color: a light one cannot be saturated. Such
    combinations give components above 1.
    """
    hue, saturation, intensity = hsi[..., 0], hsi[..., 1], hsi[..., 2]
    # Each third of the hue circle has its own weakest primary: blue, red, green.
    third = np.floor_divide(hue, 120.0) % 3
    angle = np.radians(hue - 120.0 * third)
    weakest = intensity * (1.0 - saturation)
    leading = intensity * (
        1.0 + saturation * np.cos(angle) / np.cos(math.pi / 3 - angle)
    )
    remaining = 3.0 * intensity - (weakest + leading)
    thirds = [third == 0, third == 1]
    red = np.select(thirds, [leading, weakest], remaining)
    green = np.select(thirds, [remaining, leading], weakest)
    blue = np.select(thirds, [weakest, remaining], leading)
    return np.stack([red, green, blue], axis=-1)


HSI = Space(
    "HSI",
    (
        Channel("hue", 0.0, 360.0, periodic=True),
        Channel("saturation", 0.0, 1.0),
        Channel("intensity", 0.0, 1.0),
    ),
    rgb_to_hsi,
    hsi_to_rgb,
)
