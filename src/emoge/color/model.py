"""`ColorModel`: colors as terms joined by a t-norm, one term from every variable."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from emoge.color.space import Space
from emoge.color.spaces.rgb import to_float_rgb
from emoge.fuzzy.math.norms import Norm, product
from emoge.fuzzy.variables import Variable

# Pixels worked on at once: bounds the (colors x pixels) array of degrees.
_PIXELS_PER_BLOCK = 1 << 16
# Above this share of distinct colors among the pixels, finding them does not pay.
_MOST_DISTINCT = 0.8
# How far outside [0, 1] a component of a prototype may stray, from rounding.
_SLACK = 1e-9


class ColorModel:
    """A fuzzy color model: families of colors on the channels of a color space.

    ```python
    model = ColorModel("toy", HSI, {
        "tint": [hue, saturation.select("pale", "vivid")],
        "gray": [intensity, saturation.select("gray")],
    })
    model.colors                  # ("red/pale", "red/vivid", ..., "dark", "light")
    model.label([200, 30, 30])    # array('red/vivid')
    ```

    A family is a list of variables on different channels of the space. Its
    colors take one term from every variable, and the degree of a color is the
    degrees of its terms joined by `and_norm`. With the product and partitions,
    the degrees of all colors of a pixel sum to 1. A color is named after its
    terms joined by "/", leaving out a variable of one term; a family of one
    color takes its name.

    Pixels have shape `(..., 3)`: 8-bit levels or floats in [0, 1].
    """

    def __init__(
        self,
        name: str,
        space: Space,
        families: Mapping[str, Sequence[Variable]],
        and_norm: Norm = product,
    ) -> None:
        for family, variables in families.items():
            channels = [variable.channel for variable in variables]
            if not variables or len(set(channels)) < len(channels):
                raise ValueError(f"{family!r} needs variables on different channels")
            strangers = [c.name for c in channels if c not in space.channels]
            if strangers:
                raise ValueError(
                    f"{family!r}: {strangers} are not channels of {space.name}"
                )
        colors = tuple(
            color
            for family, variables in families.items()
            for color in _color_names(family, variables)
        )
        if not colors or len(set(colors)) < len(colors):
            raise ValueError(f"{name!r} needs colors that differ, got {colors}")
        self.name = name
        self.space = space
        self.families = {
            family: tuple(variables) for family, variables in families.items()
        }
        self.and_norm = and_norm
        self.colors = colors
        self._colors_then_empty = np.array([*colors, ""])  # index len(colors): none
        self._prototypes: NDArray[np.float64] | None = None

    def degrees(self, rgb: ArrayLike) -> NDArray[np.float64]:
        """Return the degree of the pixels in every color: `(len(colors), *pixels)`."""
        values = self._channel_values(rgb)
        pixel_shape = np.shape(rgb)[:-1]
        blocks = []
        for variables in self.families.values():
            degrees = np.ones((1, *pixel_shape))  # 1 joins as nothing under a t-norm
            for variable in variables:
                term_degrees = variable.degrees(values[variable.channel.name])
                # Every color so far joined with every term of this variable.
                joined = self.and_norm(degrees[:, None], term_degrees[None])
                degrees = joined.reshape(-1, *pixel_shape)
            blocks.append(degrees)
        return np.concatenate(blocks)

    def classify(self, rgb: ArrayLike) -> NDArray[np.intp]:
        """Return the index in `colors` of the strongest color of every pixel.

        A pixel of no color, which a model with gaps between its terms leaves,
        gets `len(colors)`; of equally strong colors the first wins.
        """
        return self.strongest(rgb)[0]

    def strongest(self, rgb: ArrayLike) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Return the strongest color of every pixel as in `classify`, and its degree.

        A pixel of no color has degree 0.
        """
        values = self._channel_values(rgb)
        pixel_shape = np.shape(rgb)[:-1]
        best_index = np.full(pixel_shape, len(self.colors), dtype=np.intp)
        best_degree = np.zeros(pixel_shape)
        first_color = 0
        for variables in self.families.values():
            # A t-norm never falls when one of its degrees grows, so the strongest
            # color of a family is made of the strongest term of every variable.
            index = np.zeros(pixel_shape, dtype=np.intp)
            degree = np.ones(pixel_shape)
            for variable in variables:
                term_degrees = variable.degrees(values[variable.channel.name])
                index = index * len(variable.terms) + term_degrees.argmax(axis=0)
                degree = self.and_norm(degree, term_degrees.max(axis=0))
            stronger = degree > best_degree
            best_index = np.where(stronger, first_color + index, best_index)
            best_degree = np.where(stronger, degree, best_degree)
            first_color += math.prod(len(variable.terms) for variable in variables)
        return best_index, best_degree

    def label(self, rgb: ArrayLike) -> NDArray[np.str_]:
        """Return the name of the strongest color of every pixel; "" for no color."""
        index = self.classify(rgb)
        return self._colors_then_empty[index.reshape(-1)].reshape(index.shape)

    def palette(
        self, image: ArrayLike, weights: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Return the mean degree of every color over the pixels of an image.

        `weights`, one for every pixel, weigh the mean, as a soft mask of a
        region does. Masks stacked along a first axis give a palette each,
        `(len(masks), len(colors))`, from degrees worked out once. In an 8-bit
        image colors repeat, and each is worked out once for all its pixels.
        """
        pixels = np.asarray(image)
        pixel_shape = pixels.shape[:-1]
        pixels = pixels.reshape(-1, *pixels.shape[-1:])  # `degrees` checks it
        if not len(pixels):
            raise ValueError("an image without pixels has no palette")
        # Masks stay as given, often bytes or bools; a block at a time becomes floats.
        masks = np.ones(pixel_shape) if weights is None else np.asarray(weights)
        stacked = masks.shape != pixel_shape
        if stacked and masks.shape[1:] != pixel_shape:
            raise ValueError(
                f"weights need the shape of the pixels, {pixel_shape}, "
                f"or a stack of it; got {masks.shape}"
            )
        masks = masks.reshape(-1, len(pixels))
        mass = masks.sum(axis=1, dtype=np.float64)
        if masks.min() < 0 or not (mass > 0).all():
            raise ValueError("weights are at least 0, and above 0 somewhere in a mask")
        distinct = _distinct(pixels)
        if distinct is not None:
            # Every color once, weighed by the weights of its pixels.
            pixels, where, counts = distinct
            if weights is None:
                masks = counts[None]
            else:
                colors = len(pixels)
                masks = np.stack([np.bincount(where, mask, colors) for mask in masks])
        totals = np.zeros((len(masks), len(self.colors)))
        for start in range(0, len(pixels), _PIXELS_PER_BLOCK):
            block = slice(start, start + _PIXELS_PER_BLOCK)
            weights_here = masks[:, block].astype(np.float64)
            totals += weights_here @ self.degrees(pixels[block]).T
        palettes = totals / mass[:, None]
        return palettes if stacked else palettes[0]

    def cooccurrence(self, image: ArrayLike, distance: int = 1) -> NDArray[np.float64]:
        """Return how often every two colors are neighbours in an image.

        Two pixels `distance` apart, side by side or one above the other, are
        neighbours. Their degrees are multiplied and averaged over all
        neighbours both ways round: the result, `(len(colors), len(colors))`, is
        symmetric and sums to 1 where the degrees of a pixel do. Its diagonal is
        a color beside itself, the rest are borders between colors.
        """
        pixels = np.asarray(image)
        if pixels.ndim != 3:
            raise ValueError(f"an image is (height, width, 3), got {pixels.shape}")
        if distance < 1:
            raise ValueError(f"neighbours are at least 1 apart, got {distance}")
        height, width = pixels.shape[:2]
        totals = np.zeros((len(self.colors), len(self.colors)))
        pairs = 0
        step = max(1, _PIXELS_PER_BLOCK // max(width, 1))
        for top in range(0, height, step):
            # The rows of this block and `distance` more, for their neighbours below.
            block = self.degrees(pixels[top : top + step + distance])
            rows = min(step, height - top)
            below = max(0, min(rows, block.shape[1] - distance))
            side_by_side = block[:, :rows, :-distance], block[:, :rows, distance:]
            one_above = block[:, :below], block[:, distance : distance + below]
            for one, other in (side_by_side, one_above):
                one, other = one.reshape(len(one), -1), other.reshape(len(other), -1)
                totals += one @ other.T
                pairs += one.shape[1]
        if not pairs:
            raise ValueError(
                f"an image of {height} x {width} has no neighbours {distance} apart"
            )
        return (totals + totals.T) / (2 * pairs)

    def prototypes(self) -> NDArray[np.float64]:
        """Return a typical RGB color for every color: shape `(len(colors), 3)`.

        A channel takes the middle of the core of the color's term, or the middle
        of the channel if the family does not describe it. Where RGB cannot show
        that, as for a light vivid red, the color takes the RGB color it is
        strongest in, the one nearest to that middle, among 33 levels of every
        component.
        """
        if self._prototypes is None:
            self._prototypes = self._typical_colors()
        return self._prototypes.copy()

    def _typical_colors(self) -> NDArray[np.float64]:
        middles = {c.name: (c.low + c.high) / 2.0 for c in self.space.channels}
        rows = []
        for variables in self.families.values():
            all_centers = [variable.core_centers() for variable in variables]
            for centers in itertools.product(*all_centers):
                point = dict(middles)
                for variable, center in zip(variables, centers, strict=True):
                    point[variable.channel.name] = float(center)
                rows.append([point[channel.name] for channel in self.space.channels])
        rgb = self.space.to_rgb(np.array(rows))
        outside = ((rgb < -_SLACK) | (rgb > 1.0 + _SLACK)).any(axis=-1)
        if outside.any():
            levels = np.linspace(0.0, 1.0, 33)
            shown = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), -1)
            shown = shown.reshape(-1, 3)
            degrees = self.degrees(shown)[outside]
            strongest = degrees >= degrees.max(axis=1, keepdims=True) - _SLACK
            away = ((shown[None] - rgb[outside][:, None]) ** 2).sum(axis=-1)
            rgb[outside] = shown[np.where(strongest, away, np.inf).argmin(axis=1)]
        return np.clip(rgb, 0.0, 1.0)

    def _channel_values(self, rgb: ArrayLike) -> dict[str, NDArray[np.float64]]:
        """The values of every channel of the space for the pixels, by name."""
        values = self.space.from_rgb(to_float_rgb(rgb))
        return {c.name: values[..., i] for i, c in enumerate(self.space.channels)}

    def __repr__(self) -> str:
        sizes = ", ".join(
            f"{math.prod(len(v.terms) for v in variables)} {family}"
            for family, variables in self.families.items()
        )
        return f"ColorModel({self.name!r}, {self.space.name}: {sizes})"


def _distinct(
    rgb: ArrayLike,
) -> tuple[NDArray[np.uint8], NDArray[np.intp], NDArray[np.intp]] | None:
    """The distinct colors of 8-bit pixels, which of them each pixel is, how many.

    None for pixels that are not bytes, or barely repeat: those are quicker
    worked out as they are.
    """
    pixels = np.asarray(rgb)
    if pixels.dtype != np.uint8 or pixels.ndim < 2 or pixels.shape[-1] != 3:
        return None
    flat = pixels.reshape(-1, 3).astype(np.int32)
    packed = flat[:, 0] << 16 | flat[:, 1] << 8 | flat[:, 2]
    unique, where, counts = np.unique(packed, return_inverse=True, return_counts=True)
    if len(unique) > _MOST_DISTINCT * len(packed):
        return None
    colors = np.stack([unique >> 16, unique >> 8 & 255, unique & 255], axis=-1)
    return colors.astype(np.uint8), where, counts


def _color_names(family: str, variables: Sequence[Variable]) -> list[str]:
    """Name every combination of terms, leaving out variables of one term."""
    names = []
    for terms in itertools.product(*(variable.terms for variable in variables)):
        named = [
            term for term, v in zip(terms, variables, strict=True) if len(v.terms) > 1
        ]
        names.append("/".join(named) or family)
    return names
