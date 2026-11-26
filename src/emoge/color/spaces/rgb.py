"""RGB: the exchange format of colors, with components in [0, 1]."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from emoge.color.space import Space
from emoge.fuzzy.channels import Channel


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


def to_linear(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return sRGB components as linear light, which adds up as light does.

    A component outside [0, 1] keeps its sign, as colors outside RGB need.
    """
    size = np.abs(rgb)
    linear = np.where(size <= 0.04045, size / 12.92, ((size + 0.055) / 1.055) ** 2.4)
    return np.sign(rgb) * linear


def from_linear(linear: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return linear light as sRGB components: the inverse of `to_linear`."""
    size = np.abs(linear)
    encoded = np.where(
        size <= 0.0031308, 12.92 * size, 1.055 * size ** (1 / 2.4) - 0.055
    )
    return np.sign(linear) * encoded


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
