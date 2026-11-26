"""Fuzzy sets of every shape, made at once: `rising(0.25, 0.75)`.

Each function makes the set of the membership function of its name:
`rising(a, b)` is `FuzzySet(membership.rising, (a, b))`, and prints so. The
shape is told in `membership` and drawn in the guide.
"""

from __future__ import annotations

from collections.abc import Sequence

from emoge.fuzzy.math import membership
from emoge.fuzzy.sets import FuzzySet

# --- straight lines -------------------------------------------------------------------


def rising(a: float, b: float) -> FuzzySet:
    """The set of `membership.rising`."""
    return FuzzySet(membership.rising, (a, b))


def falling(a: float, b: float) -> FuzzySet:
    """The set of `membership.falling`."""
    return FuzzySet(membership.falling, (a, b))


def triangle(a: float, b: float, c: float) -> FuzzySet:
    """The set of `membership.triangle`."""
    return FuzzySet(membership.triangle, (a, b, c))


def trapezoid(a: float, b: float, c: float, d: float) -> FuzzySet:
    """The set of `membership.trapezoid`."""
    return FuzzySet(membership.trapezoid, (a, b, c, d))


def rectangle(a: float, b: float) -> FuzzySet:
    """The set of `membership.rectangle`."""
    return FuzzySet(membership.rectangle, (a, b))


def singleton(a: float) -> FuzzySet:
    """The set of `membership.singleton`."""
    return FuzzySet(membership.singleton, (a,))


def piecewise_linear(points: Sequence[tuple[float, float]]) -> FuzzySet:
    """The set of `membership.piecewise_linear`."""
    return FuzzySet(membership.piecewise_linear, (points,))


# --- smooth curves --------------------------------------------------------------------


def gaussian(center: float, sigma: float) -> FuzzySet:
    """The set of `membership.gaussian`."""
    return FuzzySet(membership.gaussian, (center, sigma))


def two_sided_gaussian(
    center1: float, sigma1: float, center2: float, sigma2: float
) -> FuzzySet:
    """The set of `membership.two_sided_gaussian`."""
    return FuzzySet(membership.two_sided_gaussian, (center1, sigma1, center2, sigma2))


def generalized_bell(center: float, width: float, slope: float) -> FuzzySet:
    """The set of `membership.generalized_bell`."""
    return FuzzySet(membership.generalized_bell, (center, width, slope))


def sigmoid(center: float, slope: float) -> FuzzySet:
    """The set of `membership.sigmoid`."""
    return FuzzySet(membership.sigmoid, (center, slope))


def sigmoid_difference(
    center1: float, slope1: float, center2: float, slope2: float
) -> FuzzySet:
    """The set of `membership.sigmoid_difference`."""
    return FuzzySet(membership.sigmoid_difference, (center1, slope1, center2, slope2))


def sigmoid_product(
    center1: float, slope1: float, center2: float, slope2: float
) -> FuzzySet:
    """The set of `membership.sigmoid_product`."""
    return FuzzySet(membership.sigmoid_product, (center1, slope1, center2, slope2))


def s_shape(a: float, b: float) -> FuzzySet:
    """The set of `membership.s_shape`."""
    return FuzzySet(membership.s_shape, (a, b))


def z_shape(a: float, b: float) -> FuzzySet:
    """The set of `membership.z_shape`."""
    return FuzzySet(membership.z_shape, (a, b))


def pi_shape(a: float, b: float, c: float, d: float) -> FuzzySet:
    """The set of `membership.pi_shape`."""
    return FuzzySet(membership.pi_shape, (a, b, c, d))
