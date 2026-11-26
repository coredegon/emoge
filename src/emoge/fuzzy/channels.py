"""`Channel`: an axis of values and its range."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Channel:
    """An axis of values and its range.

    A coordinate of a color space, a place in an image, the difference between
    two values: whatever a variable describes. A periodic channel is a circle,
    like hue: `high` is `low` again.
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
