"""Fuzzy color models: every pixel belongs to every named color to a degree.

```python
from emoge import color
```

Color spaces, color models such as COLIBRI, and plots of them. The fuzzy logic
the models are made of (sets, channels, partitions and norms) is `emoge.fuzzy`.
"""

from emoge.color.lut import lookup_table
from emoge.color.model import ColorModel
from emoge.color.models import COLIBRI
from emoge.color.plots import plot_palette, plot_swatches, plot_terms
from emoge.color.space import Space
from emoge.color.spaces import HSI, HSV, RGB

__all__ = [
    "COLIBRI",
    "HSI",
    "HSV",
    "RGB",
    "ColorModel",
    "Space",
    "lookup_table",
    "plot_palette",
    "plot_swatches",
    "plot_terms",
]
