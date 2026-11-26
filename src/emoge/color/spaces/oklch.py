"""Oklch: Oklab as lightness, chroma and hue, the angle around the gray axis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from emoge.color.space import Space
from emoge.color.spaces.oklab import oklab_to_rgb, rgb_to_oklab
from emoge.fuzzy.channels import Channel

# The chroma grays come out with, from the rounding of the published matrices.
_GRAY_CHROMA = 1e-6


def rgb_to_oklch(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert RGB in [0, 1] to lightness, chroma, and hue in [0, 360).

    Hue is 0 for red-pink, as in Oklab's `a`; grays get hue 0.
    """
    oklab = rgb_to_oklab(rgb)
    lightness, a, b = oklab[..., 0], oklab[..., 1], oklab[..., 2]
    chroma = np.hypot(a, b)
    hue = np.mod(np.degrees(np.arctan2(b, a)), 360.0)
    # mod rounds tiny negative angles up to 360
    hue = np.where((chroma > _GRAY_CHROMA) & (hue < 360.0), hue, 0.0)
    return np.stack([lightness, chroma, hue], axis=-1)


def oklch_to_rgb(oklch: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert lightness, chroma and hue in degrees to RGB.

    Not every combination is a color: those outside RGB give components below
    0 or above 1.
    """
    lightness, chroma, angle = oklch[..., 0], oklch[..., 1], np.radians(oklch[..., 2])
    a, b = chroma * np.cos(angle), chroma * np.sin(angle)
    return oklab_to_rgb(np.stack([lightness, a, b], axis=-1))


OKLCH = Space(
    "Oklch",
    (
        Channel("lightness", 0.0, 1.0),
        Channel("chroma", 0.0, 0.4),
        Channel("hue", 0.0, 360.0, periodic=True),
    ),
    rgb_to_oklch,
    oklch_to_rgb,
)
