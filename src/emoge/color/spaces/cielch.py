"""CIELCh: CIELAB as lightness, chroma and hue, the angle around the gray axis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from emoge.color.space import Space
from emoge.color.spaces.cielab import cielab_to_rgb, rgb_to_cielab
from emoge.fuzzy.channels import Channel

# The chroma grays come out with, from rounding.
_GRAY_CHROMA = 1e-4


def rgb_to_cielch(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert RGB in [0, 1] to lightness, chroma, and hue in [0, 360).

    Hue is 0 for red-pink, as in CIELAB's `a`; grays get hue 0.
    """
    lab = rgb_to_cielab(rgb)
    lightness, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    hue = np.mod(np.degrees(np.arctan2(b, a)), 360.0)
    # mod rounds tiny negative angles up to 360
    hue = np.where((chroma > _GRAY_CHROMA) & (hue < 360.0), hue, 0.0)
    return np.stack([lightness, chroma, hue], axis=-1)


def cielch_to_rgb(lch: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert lightness, chroma and hue in degrees to RGB.

    Not every combination is a color: those outside RGB give components below
    0 or above 1.
    """
    lightness, chroma, angle = lch[..., 0], lch[..., 1], np.radians(lch[..., 2])
    a, b = chroma * np.cos(angle), chroma * np.sin(angle)
    return cielab_to_rgb(np.stack([lightness, a, b], axis=-1))


CIELCH = Space(
    "CIELCh",
    (
        Channel("lightness", 0.0, 100.0),
        Channel("chroma", 0.0, 150.0),
        Channel("hue", 0.0, 360.0, periodic=True),
    ),
    rgb_to_cielch,
    cielch_to_rgb,
)
