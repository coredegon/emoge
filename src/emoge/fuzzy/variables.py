"""`Variable`: a channel described by named fuzzy sets."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from emoge.fuzzy.channels import Channel
from emoge.fuzzy.sets import FuzzySet

_SAMPLES = 4001  # how finely a term is sampled to find its core


class Variable:
    """A channel described by named fuzzy sets, its terms.

    ```python
    level = Variable(channel, {"low": falling(0.2, 0.4), ...})
    level.degrees(0.3)        # the degree in every term
    ```

    On a periodic channel, like hue, values are first brought into its range,
    `[low, high)`, and every term describes the whole circle there: a term
    across the seam is two pieces joined with `|`, as `partition` builds it.
    """

    def __init__(self, channel: Channel, terms: Mapping[str, FuzzySet]) -> None:
        if not terms:
            raise ValueError(f"a variable on {channel.name!r} needs terms")
        self.channel = channel
        self.terms = dict(terms)

    def degrees(self, x: ArrayLike) -> NDArray[np.float64]:
        """Return the degree of `x` in every term: shape `(len(terms), *x.shape)`."""
        values = np.asarray(x, dtype=np.float64)
        if self.channel.periodic:
            low, high = self.channel.low, self.channel.high
            with np.errstate(invalid="ignore"):  # an infinite value has no place: NaN
                values = low + np.mod(values - low, high - low)
        return np.stack([term.degrees(values) for term in self.terms.values()])

    def select(self, *names: str) -> Variable:
        """Return a variable with only the named terms, in the given order."""
        unknown = [name for name in names if name not in self.terms]
        if unknown or not names:
            raise ValueError(f"no terms {unknown} among {list(self.terms)}")
        return Variable(self.channel, {name: self.terms[name] for name in names})

    def core_centers(self) -> NDArray[np.float64]:
        """Return the typical value of every term: the middle of where it is highest.

        On a periodic channel the middle is found going around the circle.
        """
        low, high = self.channel.low, self.channel.high
        x = np.linspace(low, high, _SAMPLES, endpoint=not self.channel.periodic)
        degrees = self.degrees(x)
        highest = degrees.max(axis=1, keepdims=True)
        in_core = (degrees >= highest - 1e-9).astype(np.float64)
        if not self.channel.periodic:
            return (in_core @ x) / in_core.sum(axis=1)
        # Average as angles, so that a core across the seam (350 to 10) gives 0.
        angle = (x - low) * (math.tau / (high - low))
        mean_angle = np.arctan2(in_core @ np.sin(angle), in_core @ np.cos(angle))
        return low + (mean_angle % math.tau) * ((high - low) / math.tau)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.channel.name}: {', '.join(self.terms)})"
