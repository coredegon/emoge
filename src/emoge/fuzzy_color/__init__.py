"""Fuzzy color models: every pixel belongs to every named color to a degree.

- `math`: membership functions and norms, as plain NumPy functions;
- `sets`: `FuzzySet`, combined with `&`, `|` and `~`;
- `variables`: `Variable`, a channel described by named fuzzy sets;
- `partitions`: `partition`, a variable whose terms sum to 1;
- `spaces`: `Channel`, `Space` and the color spaces, one module each;
- `model`: `ColorModel`;
- `lut`: `lookup_table`, the class of every 8-bit pixel worked out once;
- `models`: the models built with all of the above, one module each;
- `plots`: variables, terms, colors and palettes.
"""

from emoge.fuzzy_color.lut import lookup_table
from emoge.fuzzy_color.model import ColorModel
from emoge.fuzzy_color.partitions import partition
from emoge.fuzzy_color.plots import (
    plot_palette,
    plot_swatches,
    plot_terms,
    plot_variable,
)
from emoge.fuzzy_color.sets import FuzzySet
from emoge.fuzzy_color.spaces import HSI, HSV, RGB, Channel, Space
from emoge.fuzzy_color.variables import Variable

__all__ = [
    "HSI",
    "HSV",
    "RGB",
    "Channel",
    "ColorModel",
    "FuzzySet",
    "Space",
    "Variable",
    "lookup_table",
    "partition",
    "plot_palette",
    "plot_swatches",
    "plot_terms",
    "plot_variable",
]
