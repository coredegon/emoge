"""`partition`: a variable whose terms sum to 1, built from the core of every term."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping

from emoge.fuzzy_color.math import membership
from emoge.fuzzy_color.sets import FuzzySet
from emoge.fuzzy_color.spaces.space import Channel
from emoge.fuzzy_color.variables import Variable

# The values that belong to a term completely: one value or a (low, high) range.
type Core = float | tuple[float, float]


def partition(channel: Channel, cores: Mapping[str, Core]) -> Variable:
    """Build a variable whose terms sum to 1, from the core of every term.

    ```python
    partition(HSI.channel("saturation"), {"low": 0.15, "medium": (0.4, 0.5), "high": 1})
    ```

    A term is 1 on its core and passes linearly to its neighbours; the outer
    terms of a regular channel stay at 1 up to its ends. Cores ascend and leave
    gaps between neighbours. On a periodic channel the last term blends into the
    first, and one core may wrap around, like `(350, 10)` on hue: that term is
    two pieces, one on either side of the seam, joined with `|`.
    """
    if len(cores) < 2:
        raise ValueError(f"a partition needs at least two terms, got {list(cores)}")
    ranges = [
        core if isinstance(core, tuple) else (core, core) for core in cores.values()
    ]
    if channel.periodic:
        terms = _terms_around_circle(ranges, channel.low, channel.high)
    else:
        terms = _terms_along_line(ranges, channel.low, channel.high)
    return Variable(channel, dict(zip(cores, terms, strict=True)))


def _term_around_core(
    rise_start: float, core_start: float, core_end: float, fall_end: float
) -> FuzzySet:
    if core_start == core_end:
        return FuzzySet(membership.triangle, (rise_start, core_start, fall_end))
    return FuzzySet(membership.trapezoid, (rise_start, core_start, core_end, fall_end))


def _terms_along_line(
    cores: list[tuple[float, float]], low: float, high: float
) -> list[FuzzySet]:
    """Blend neighbouring cores along a line; the outer terms are shoulders."""
    bounds = [low, *itertools.chain.from_iterable(cores), high]
    apart = all(left[1] < right[0] for left, right in itertools.pairwise(cores))
    if bounds != sorted(bounds) or not apart:
        raise ValueError(
            f"cores must ascend inside ({low:g}, {high:g}) and leave gaps, got {cores}"
        )
    inner = [
        _term_around_core(previous[1], *core, following[0])
        for previous, core, following in zip(cores, cores[1:], cores[2:], strict=False)
    ]
    first = FuzzySet(membership.falling, (cores[0][1], cores[1][0]))
    last = FuzzySet(membership.rising, (cores[-2][1], cores[-1][0]))
    return [first, *inner, last]


def _terms_around_circle(
    cores: list[tuple[float, float]], low: float, high: float
) -> list[FuzzySet]:
    """Blend neighbouring cores around a circle; the last blends into the first."""
    period = high - low
    # Corners in one direction around the circle, each after the one before, so a
    # term may run past `high`.
    all_corners = []
    for i, (core_low, core_high) in enumerate(cores):
        previous_end = cores[i - 1][1]
        next_start = cores[(i + 1) % len(cores)][0]
        rise_start = low + (previous_end - low) % period
        core_start = rise_start + (core_low - previous_end) % period
        core_end = core_start + (core_high - core_low) % period
        fall_end = core_end + (next_start - core_high) % period
        all_corners.append((rise_start, core_start, core_end, fall_end))
    # In order, the stretches from one core to the next go around exactly once.
    covered = sum(fall_end - core_start for _, core_start, _, fall_end in all_corners)
    touching = any(core_end == fall_end for *_, core_end, fall_end in all_corners)
    if not math.isclose(covered, period) or touching:
        raise ValueError(
            f"cores must go once around ({low:g}, {high:g}) in ascending order "
            f"and leave gaps, got {cores}"
        )
    terms = []
    for corners in all_corners:
        term = _term_around_core(*corners)
        if (
            corners[-1] > high
        ):  # past `high` the circle goes on at `low`: add that piece
            term = term | _term_around_core(*(corner - period for corner in corners))
        terms.append(term)
    return terms
