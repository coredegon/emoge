"""XYZ: the CIE 1931 color space, the light of a color as the eye takes it in.

Y is luminance; the white of sRGB, D65, is `WHITE`. CIELAB is made from it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from emoge.color.space import Space
from emoge.color.spaces.rgb import from_linear, to_linear
from emoge.fuzzy.channels import Channel

# Linear sRGB to XYZ, as the sRGB standard gives it.
_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_FROM_XYZ = np.linalg.inv(_TO_XYZ)
# The XYZ of the white of sRGB, D65.
WHITE = _TO_XYZ.sum(axis=1)


def rgb_to_xyz(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert RGB in [0, 1] to X, Y and Z, white being `WHITE`."""
    return to_linear(rgb) @ _TO_XYZ.T


def xyz_to_rgb(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert X, Y and Z to RGB; light outside RGB gives components outside [0, 1]."""
    return from_linear(xyz @ _FROM_XYZ.T)


XYZ = Space(
    "XYZ",
    tuple(
        Channel(name, 0.0, float(top)) for name, top in zip("xyz", WHITE, strict=True)
    ),
    rgb_to_xyz,
    xyz_to_rgb,
)
