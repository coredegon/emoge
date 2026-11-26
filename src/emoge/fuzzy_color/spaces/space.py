"""`Channel` and `Space`: what every color space is made of."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Channel:
    """One coordinate of a color space and its range.

    A periodic channel is a circle, like hue: `high` is `low` again.
    """

    name: str
    low: float
    high: float
    periodic: bool = False

    def __post_init__(self) -> None:
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError(f"the channel {self.name!r} needs a finite range")
        if not self.low < self.high:
            raise ValueError(f"the channel {self.name!r} needs low < high")


@dataclass(frozen=True)
class Space:
    """A color space: its channels, in the order of the last axis, and conversions.

    `from_rgb` and `to_rgb` convert float arrays of shape `(..., 3)`, with RGB
    in [0, 1].
    """

    name: str
    channels: tuple[Channel, ...]
    from_rgb: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    to_rgb: Callable[[NDArray[np.float64]], NDArray[np.float64]]

    def channel(self, name: str) -> Channel:
        """Return the channel of this name."""
        for channel in self.channels:
            if channel.name == name:
                return channel
        names = [channel.name for channel in self.channels]
        raise KeyError(f"{self.name} has no channel {name!r}, only {names}")
