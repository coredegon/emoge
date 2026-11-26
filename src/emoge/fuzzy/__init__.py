"""Fuzzy logic on any axis: every value belongs to every word of the axis to a degree.

```python
from emoge import fuzzy
```

An axis can be anything: a place in an image, the direction of a line, a share
of the pixels, a channel of a color.
"""

from emoge.fuzzy.channels import Channel
from emoge.fuzzy.math import membership, norms
from emoge.fuzzy.partitions import partition
from emoge.fuzzy.plots import plot_variable
from emoge.fuzzy.sets import FuzzySet
from emoge.fuzzy.shapes import (
    falling,
    gaussian,
    generalized_bell,
    pi_shape,
    piecewise_linear,
    rectangle,
    rising,
    s_shape,
    sigmoid,
    sigmoid_difference,
    sigmoid_product,
    singleton,
    trapezoid,
    triangle,
    two_sided_gaussian,
    z_shape,
)
from emoge.fuzzy.summaries import truth
from emoge.fuzzy.variables import Variable

__all__ = [
    "Channel",
    "FuzzySet",
    "Variable",
    "falling",
    "gaussian",
    "generalized_bell",
    "membership",
    "norms",
    "partition",
    "pi_shape",
    "piecewise_linear",
    "plot_variable",
    "rectangle",
    "rising",
    "s_shape",
    "sigmoid",
    "sigmoid_difference",
    "sigmoid_product",
    "singleton",
    "trapezoid",
    "triangle",
    "truth",
    "two_sided_gaussian",
    "z_shape",
]
