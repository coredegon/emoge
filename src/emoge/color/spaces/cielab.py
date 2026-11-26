"""CIELAB: lightness and two opposing axes of color, the classic perceptual space.

CIE 1976 L*a*b*, against the white of sRGB, D65. Lightness runs from 0 to 100;
`a` from green to red and `b` from blue to yellow. Much of the research on color
and emotion is written in it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from emoge.color.space import Space
from emoge.color.spaces.xyz import WHITE, rgb_to_xyz, xyz_to_rgb
from emoge.fuzzy.channels import Channel

_EDGE = 6.0 / 29.0  # where the cube root gives way to a straight line, near black


def rgb_to_cielab(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert RGB in [0, 1] to lightness in [0, 100] and the axes `a` and `b`."""
    ratio = rgb_to_xyz(rgb) / WHITE
    f = np.where(ratio > _EDGE**3, np.cbrt(ratio), ratio / (3 * _EDGE**2) + 4 / 29)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def cielab_to_rgb(lab: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert lightness, `a` and `b` to RGB.

    Not every combination is a color: those outside RGB give components below
    0 or above 1.
    """
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    ratio = np.where(f > _EDGE, f**3, 3 * _EDGE**2 * (f - 4 / 29))
    return xyz_to_rgb(ratio * WHITE)


CIELAB = Space(
    "CIELAB",
    (
        Channel("lightness", 0.0, 100.0),
        Channel("a", -128.0, 128.0),
        Channel("b", -128.0, 128.0),
    ),
    rgb_to_cielab,
    cielab_to_rgb,
)
