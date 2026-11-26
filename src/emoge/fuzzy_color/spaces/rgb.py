"""RGB: the exchange format of colors, with components in [0, 1]."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from emoge.fuzzy_color.spaces.space import Channel, Space


def to_float_rgb(rgb: ArrayLike) -> NDArray[np.float64]:
    """Return pixels as float RGB in [0, 1].

    Whole numbers are 8-bit levels from 0 to 255; floats are taken as they are.
    """
    pixels = np.asarray(rgb)
    if pixels.ndim == 0 or pixels.shape[-1] != 3:
        raise ValueError(f"pixels need a last axis of 3 components, got {pixels.shape}")
    if np.issubdtype(pixels.dtype, np.floating):
        return pixels.astype(np.float64, copy=False)
    if not np.issubdtype(pixels.dtype, np.integer):
        raise TypeError(f"pixels are whole numbers or floats, got {pixels.dtype}")
    if (
        pixels.dtype != np.uint8
        and pixels.size
        and (pixels.min() < 0 or pixels.max() > 255)
    ):
        raise ValueError("whole-number pixels are 8-bit levels from 0 to 255")
    return pixels / 255.0


def fit_into_rgb(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Darken colors too bright for RGB until they fit, and clip negatives to 0.

    A color whose largest component is above 1 is scaled down to make it 1, which
    keeps its hue.
    """
    largest = np.maximum(rgb.max(axis=-1, keepdims=True), 1.0)
    return np.clip(rgb / largest, 0.0, 1.0)


def _unchanged(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    return rgb


RGB = Space(
    "RGB",
    (Channel("red", 0.0, 1.0), Channel("green", 0.0, 1.0), Channel("blue", 0.0, 1.0)),
    _unchanged,
    _unchanged,
)
