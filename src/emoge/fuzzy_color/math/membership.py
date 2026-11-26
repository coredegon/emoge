"""Membership functions: the shapes of fuzzy sets, as plain NumPy functions.

Every function takes an array of values `x` first, then the parameters of its
shape, and returns the degree of every value: in [0, 1], in the shape of `x`.
`FuzzySet(function, parameters)` makes a set of one.

Parameters that mark places on the axis ascend. Equal neighbours make a
vertical edge, where the higher degree holds.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def _require_ascending(*parameters: float) -> None:
    if not all(map(math.isfinite, parameters)) or list(parameters) != sorted(
        parameters
    ):
        raise ValueError(f"parameters must be finite and ascending, got {parameters}")


def _require_positive(**parameters: float) -> None:
    for name, value in parameters.items():
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive and finite, got {value}")


# --- straight lines -------------------------------------------------------------------


def rising(x: NDArray[np.float64], a: float, b: float) -> NDArray[np.float64]:
    """0 up to `a`, a straight line, 1 from `b` on."""
    _require_ascending(a, b)
    if a == b:
        return (x >= a).astype(np.float64)
    return np.clip((x - a) / (b - a), 0.0, 1.0)


def falling(x: NDArray[np.float64], a: float, b: float) -> NDArray[np.float64]:
    """1 up to `a`, a straight line, 0 from `b` on."""
    _require_ascending(a, b)
    if a == b:
        return (x <= a).astype(np.float64)
    return np.clip((b - x) / (b - a), 0.0, 1.0)


def triangle(
    x: NDArray[np.float64], a: float, b: float, c: float
) -> NDArray[np.float64]:
    """Rises from `a` to the peak at `b`, falls to `c`."""
    _require_ascending(a, b, c)
    return np.minimum(rising(x, a, b), falling(x, b, c))


def trapezoid(
    x: NDArray[np.float64], a: float, b: float, c: float, d: float
) -> NDArray[np.float64]:
    """Rises from `a` to `b`, is 1 until `c`, falls to `d`."""
    _require_ascending(a, b, c, d)
    return np.minimum(rising(x, a, b), falling(x, c, d))


def rectangle(x: NDArray[np.float64], a: float, b: float) -> NDArray[np.float64]:
    """1 on `[a, b]`, 0 elsewhere."""
    return trapezoid(x, a, a, b, b)


def singleton(x: NDArray[np.float64], a: float) -> NDArray[np.float64]:
    """1 at `a`, 0 elsewhere."""
    return trapezoid(x, a, a, a, a)


def piecewise_linear(
    x: NDArray[np.float64], points: Sequence[tuple[float, float]]
) -> NDArray[np.float64]:
    """Straight lines through `(value, degree)` points, flat before and after them.

    The values strictly ascend.
    """
    values = [value for value, _ in points]
    degrees = [degree for _, degree in points]
    if not values or any(left >= right for left, right in itertools.pairwise(values)):
        raise ValueError(f"the values of the points must strictly ascend, got {values}")
    if not all(0.0 <= degree <= 1.0 for degree in degrees):
        raise ValueError(f"the degrees of the points must lie in [0, 1], got {degrees}")
    return np.interp(x, values, degrees)


# --- smooth curves --------------------------------------------------------------------


def gaussian(
    x: NDArray[np.float64], center: float, sigma: float
) -> NDArray[np.float64]:
    """`exp(-(x - center)**2 / (2 * sigma**2))`."""
    _require_positive(sigma=sigma)
    return np.exp(-((x - center) ** 2) / (2.0 * sigma**2))


def two_sided_gaussian(
    x: NDArray[np.float64], center1: float, sigma1: float, center2: float, sigma2: float
) -> NDArray[np.float64]:
    """A gaussian up to `center1`, 1 until `center2`, a gaussian after it."""
    _require_positive(sigma1=sigma1, sigma2=sigma2)
    left = np.minimum(x, center1) - center1  # 0 from center1 on
    right = np.maximum(x, center2) - center2  # 0 up to center2
    return np.exp(-(left**2) / (2.0 * sigma1**2) - right**2 / (2.0 * sigma2**2))


def generalized_bell(
    x: NDArray[np.float64], center: float, width: float, slope: float
) -> NDArray[np.float64]:
    """`1 / (1 + |(x - center) / width|**(2 * slope))`."""
    _require_positive(width=width, slope=slope)
    return 1.0 / (1.0 + np.abs((x - center) / width) ** (2.0 * slope))


def sigmoid(x: NDArray[np.float64], center: float, slope: float) -> NDArray[np.float64]:
    """`1 / (1 + exp(-slope * (x - center)))`; a negative slope makes it fall."""
    with np.errstate(over="ignore"):  # exp overflows to inf, which gives 0
        return 1.0 / (1.0 + np.exp(-slope * (x - center)))


def sigmoid_difference(
    x: NDArray[np.float64], center1: float, slope1: float, center2: float, slope2: float
) -> NDArray[np.float64]:
    """`|sigmoid(center1, slope1) - sigmoid(center2, slope2)|`."""
    return np.abs(sigmoid(x, center1, slope1) - sigmoid(x, center2, slope2))


def sigmoid_product(
    x: NDArray[np.float64], center1: float, slope1: float, center2: float, slope2: float
) -> NDArray[np.float64]:
    """`sigmoid(center1, slope1) * sigmoid(center2, slope2)`."""
    return sigmoid(x, center1, slope1) * sigmoid(x, center2, slope2)


def s_shape(x: NDArray[np.float64], a: float, b: float) -> NDArray[np.float64]:
    """Smooth rise from 0 at `a` to 1 at `b`: `2t**2`, then `1 - 2(1 - t)**2`."""
    _require_ascending(a, b)
    if a == b:
        raise ValueError(f"a smooth rise needs a < b, got {a} and {b}")
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    lower_half = 2.0 * np.minimum(t, 1.0 - t) ** 2
    return np.where(t <= 0.5, lower_half, 1.0 - lower_half)


def z_shape(x: NDArray[np.float64], a: float, b: float) -> NDArray[np.float64]:
    """Smooth fall from 1 at `a` to 0 at `b`: `1 - s_shape(a, b)`."""
    return 1.0 - s_shape(x, a, b)


def pi_shape(
    x: NDArray[np.float64], a: float, b: float, c: float, d: float
) -> NDArray[np.float64]:
    """Smooth rise from `a` to `b`, 1 until `c`, smooth fall to `d`."""
    _require_ascending(a, b, c, d)
    return s_shape(x, a, b) * z_shape(x, c, d)
