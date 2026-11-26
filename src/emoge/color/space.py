"""`Space`: a color space, its channels and its conversions.

A channel is an `emoge.fuzzy.Channel`, the axis every variable describes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from emoge.fuzzy.channels import Channel


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
