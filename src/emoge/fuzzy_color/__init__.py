"""Fuzzy color models: every pixel belongs to every named color to a degree.

```python
import emoge.fuzzy_color as fc
```
"""

from emoge.fuzzy_color.lut import lookup_table
from emoge.fuzzy_color.math import membership, norms
from emoge.fuzzy_color.model import ColorModel
from emoge.fuzzy_color.models import COLIBRI
from emoge.fuzzy_color.partitions import partition
from emoge.fuzzy_color.plots import (
    plot_palette,
    plot_swatches,
    plot_terms,
    plot_variable,
)
from emoge.fuzzy_color.sets import FuzzySet
from emoge.fuzzy_color.shapes import (
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
from emoge.fuzzy_color.spaces import HSI, HSV, RGB, Channel, Space
from emoge.fuzzy_color.variables import Variable

__all__ = [
    "COLIBRI",
    "HSI",
    "HSV",
    "RGB",
    "Channel",
    "ColorModel",
    "FuzzySet",
    "Space",
    "Variable",
    "falling",
    "gaussian",
    "generalized_bell",
    "lookup_table",
    "membership",
    "norms",
    "partition",
    "pi_shape",
    "piecewise_linear",
    "plot_palette",
    "plot_swatches",
    "plot_terms",
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
    "two_sided_gaussian",
    "z_shape",
]
