"""`relate`: the words of relations between every two colors, as edge features."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from emoge.color.space import Space
from emoge.fuzzy.variables import Variable

# A relation: the words of `Variable` on the difference of the colors on a channel
# of a space, such as {"harmony": (OKLCH, "hue", harmony)}.
type Relations = Mapping[str, tuple[Space, str, Variable]]


def relate(
    rgb: NDArray[np.float64], relations: Relations
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """Return the degree of every two colors in every word of the relations.

    `rgb` is a color for every node, `(nodes, 3)`; the result is the names,
    `"relation/word"`, and the degrees, `(nodes, nodes, words)`. A difference
    is as `emoge.color.difference` takes it: never below 0, the shorter way
    round on a periodic channel.
    """
    names: list[str] = []
    degrees = []
    for relation, (space, name, variable) in relations.items():
        channel = space.channel(name)
        values = space.from_rgb(rgb)[:, space.channels.index(channel)]
        gap = np.abs(values[:, None] - values[None])  # each color worked out once
        if channel.periodic:
            span = channel.high - channel.low
            gap = np.minimum(gap % span, span - gap % span)
        degrees.extend(variable.degrees(gap))
        names.extend(f"{relation}/{term}" for term in variable.terms)
    count = len(rgb)
    if not degrees:
        return (), np.zeros((count, count, 0))
    return tuple(names), np.stack(degrees, axis=-1)
