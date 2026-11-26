"""Fuzzy color models: every pixel belongs to every named color to a degree.

```python
from emoge import color
```

Color spaces, color models such as COLIBRI and their files, how far apart colors
are, and plots. The fuzzy logic the models are made of (sets, channels,
partitions and norms) is `emoge.fuzzy`.
"""

from emoge.color.difference import difference
from emoge.color.files import load, save
from emoge.color.lut import lookup_table
from emoge.color.model import ColorModel
from emoge.color.models import COLIBRI
from emoge.color.plots import plot_palette, plot_swatches, plot_terms
from emoge.color.recolor import recolor
from emoge.color.space import Space
from emoge.color.spaces import CIELAB, CIELCH, HSI, HSL, HSV, OKLAB, OKLCH, RGB, XYZ

__all__ = [
    "CIELAB",
    "CIELCH",
    "COLIBRI",
    "HSI",
    "HSL",
    "HSV",
    "OKLAB",
    "OKLCH",
    "RGB",
    "XYZ",
    "ColorModel",
    "Space",
    "difference",
    "load",
    "lookup_table",
    "plot_palette",
    "plot_swatches",
    "plot_terms",
    "recolor",
    "save",
]
