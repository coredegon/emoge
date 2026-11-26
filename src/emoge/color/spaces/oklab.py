"""Oklab: lightness and two opposing axes of color, as people see them.

B. Ottosson, "A perceptual color space for image processing", 2020,
https://bottosson.github.io/posts/oklab/. Equal distances look about equally
different, which makes it the space to tell how far apart colors are: `a` runs
from green to red, `b` from blue to yellow.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from emoge.color.space import Space
from emoge.color.spaces.rgb import from_linear, to_linear
from emoge.fuzzy.channels import Channel

# Linear RGB to the responses of the three kinds of cones, and their cube roots
# to Oklab; the matrices are the published ones.
_TO_CONES = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_TO_OKLAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_FROM_CONES = np.linalg.inv(_TO_CONES)
_FROM_OKLAB = np.linalg.inv(_TO_OKLAB)


def rgb_to_oklab(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert RGB in [0, 1] to lightness in [0, 1] and the axes `a` and `b`."""
    return np.cbrt(to_linear(rgb) @ _TO_CONES.T) @ _TO_OKLAB.T


def oklab_to_rgb(oklab: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert lightness, `a` and `b` to RGB.

    Not every combination is a color: those outside RGB give components below
    0 or above 1.
    """
    return from_linear((oklab @ _FROM_OKLAB.T) ** 3 @ _FROM_CONES.T)


OKLAB = Space(
    "Oklab",
    (
        Channel("lightness", 0.0, 1.0),
        Channel("a", -0.4, 0.4),
        Channel("b", -0.4, 0.4),
    ),
    rgb_to_oklab,
    oklab_to_rgb,
)
