"""Norms: ways to join two degrees into one.

A t-norm joins degrees for "and", an s-norm for "or". Every norm takes two
broadcastable arrays of degrees and returns the joined degrees.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

type Norm = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


# --- t-norms: "and" -------------------------------------------------------------------


def minimum(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """`min(a, b)`: the Goedel t-norm, the largest of all."""
    return np.minimum(a, b)


def product(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """`a * b`: the product t-norm."""
    return a * b


def bounded_difference(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """`max(a + b - 1, 0)`: the Lukasiewicz t-norm."""
    return np.maximum(a + b - 1.0, 0.0)


def drastic_product(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """`b` if `a` is 1, `a` if `b` is 1, else 0: the smallest of all t-norms."""
    return np.where(a == 1.0, b, np.where(b == 1.0, a, 0.0))


def nilpotent_minimum(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """`min(a, b)` if `a + b > 1`, else 0."""
    return np.where(a + b > 1.0, np.minimum(a, b), 0.0)


def hamacher_product(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """`a * b / (a + b - a * b)`, and 0 where both are 0."""
    joined = a * b
    divisor = a + b - joined
    return np.divide(joined, divisor, out=np.zeros_like(divisor), where=divisor > 0.0)


def einstein_product(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """`a * b / (2 - (a + b - a * b))`."""
    return a * b / (2.0 - (a + b - a * b))


# --- s-norms: "or" --------------------------------------------------------------------


def maximum(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """`max(a, b)`: the Goedel s-norm, the smallest of all."""
    return np.maximum(a, b)


def probabilistic_sum(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """`a + b - a * b`: the s-norm of the product."""
    return a + b - a * b


def bounded_sum(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """`min(a + b, 1)`: the Lukasiewicz s-norm."""
    return np.minimum(a + b, 1.0)


def drastic_sum(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """`b` if `a` is 0, `a` if `b` is 0, else 1: the largest of all s-norms."""
    return np.where(a == 0.0, b, np.where(b == 0.0, a, 1.0))


def nilpotent_maximum(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """`max(a, b)` if `a + b < 1`, else 1."""
    return np.where(a + b < 1.0, np.maximum(a, b), 1.0)


def hamacher_sum(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """`(a + b - 2ab) / (1 - ab)`, and 1 where both are 1."""
    joined = a * b
    divisor = 1.0 - joined
    return np.divide(
        a + b - 2.0 * joined, divisor, out=np.ones_like(divisor), where=divisor > 0.0
    )


def einstein_sum(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """`(a + b) / (1 + a * b)`."""
    return (a + b) / (1.0 + a * b)
