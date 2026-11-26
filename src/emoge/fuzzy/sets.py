"""`FuzzySet`: a membership function and its parameters.

Sets combine value by value, and the result is a set again:

- `a & b`, both: the smaller of the two degrees;
- `a | b`, either: the larger of the two degrees;
- `~a`, not: one minus the degree.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class FuzzySet:
    """A fuzzy set: a membership function and its parameters.

    ```python
    warm = FuzzySet(membership.triangle, (10, 20, 30))
    warm.degrees([15, 20])      # array([0.5, 1. ])
    ```

    `function(x, *parameters)` gives the degree of every value of `x`. Any
    function of `emoge.fuzzy.math.membership` fits, and so does one of
    your own. Wrong parameters fail when the set is made, not later. `name`,
    if given, is how the set prints.
    """

    function: Callable[..., ArrayLike]
    parameters: tuple[object, ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        self.degrees(np.zeros(1))  # let the function check its parameters

    def degrees(self, x: ArrayLike) -> NDArray[np.float64]:
        """Return the degree of every value, in the shape of `x`."""
        values = np.asarray(x, dtype=np.float64)
        return np.asarray(self.function(values, *self.parameters), dtype=np.float64)

    def __and__(self, other: FuzzySet) -> FuzzySet:
        return FuzzySet(_both, (self, other), f"({self} & {other})")

    def __or__(self, other: FuzzySet) -> FuzzySet:
        return FuzzySet(_either, (self, other), f"({self} | {other})")

    def __invert__(self) -> FuzzySet:
        return FuzzySet(_not, (self,), f"~{self}")

    def __bool__(self) -> bool:
        raise TypeError("a FuzzySet has no truth value: combine sets with & | ~")

    def __repr__(self) -> str:
        if self.name:
            return self.name
        parameters = ", ".join(_describe(parameter) for parameter in self.parameters)
        return f"{_describe(self.function)}({parameters})"


def _both(x: NDArray[np.float64], a: FuzzySet, b: FuzzySet) -> NDArray[np.float64]:
    return np.minimum(a.degrees(x), b.degrees(x))


def _either(x: NDArray[np.float64], a: FuzzySet, b: FuzzySet) -> NDArray[np.float64]:
    return np.maximum(a.degrees(x), b.degrees(x))


def _not(x: NDArray[np.float64], a: FuzzySet) -> NDArray[np.float64]:
    return 1.0 - a.degrees(x)


def _describe(parameter: object) -> str:
    if isinstance(parameter, int | float):
        return f"{parameter:g}"
    if isinstance(parameter, FuzzySet):
        return repr(parameter)
    return getattr(parameter, "__name__", repr(parameter))
