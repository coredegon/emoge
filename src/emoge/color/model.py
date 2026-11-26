"""`ColorModel`: colors as terms joined by a t-norm, one term from every variable."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from emoge.color.space import Space
from emoge.color.spaces.rgb import fit_into_rgb, to_float_rgb
from emoge.fuzzy.math.norms import Norm, product
from emoge.fuzzy.variables import Variable

# Pixels worked on at once: bounds the (colors x pixels) array of degrees.
_PIXELS_PER_BLOCK = 1 << 16


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
        `(len(masks), len(colors))`, from degrees worked out once.
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
        totals = np.zeros((len(masks), len(self.colors)))
        for start in range(0, len(pixels), _PIXELS_PER_BLOCK):
            block = slice(start, start + _PIXELS_PER_BLOCK)
            weights_here = masks[:, block].astype(np.float64)
            totals += weights_here @ self.degrees(pixels[block]).T
        palettes = totals / mass[:, None]
        return palettes if stacked else palettes[0]

    def prototypes(self) -> NDArray[np.float64]:
        """Return a typical RGB color for every color: shape `(len(colors), 3)`.

        A channel takes the middle of the core of the color's term, or the middle
        of the channel if the family does not describe it. A color too bright for
        RGB is darkened until it fits.
        """
        middles = {c.name: (c.low + c.high) / 2.0 for c in self.space.channels}
        rows = []
        for variables in self.families.values():
            all_centers = [variable.core_centers() for variable in variables]
            for centers in itertools.product(*all_centers):
                point = dict(middles)
                for variable, center in zip(variables, centers, strict=True):
                    point[variable.channel.name] = float(center)
                rows.append([point[channel.name] for channel in self.space.channels])
        return fit_into_rgb(self.space.to_rgb(np.array(rows)))

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


def _color_names(family: str, variables: Sequence[Variable]) -> list[str]:
    """Name every combination of terms, leaving out variables of one term."""
    names = []
    for terms in itertools.product(*(variable.terms for variable in variables)):
        named = [
            term for term, v in zip(terms, variables, strict=True) if len(v.terms) > 1
        ]
        names.append("/".join(named) or family)
    return names
