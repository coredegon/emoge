"""Regions of an image as masks, `(regions, height, width)`: crisp 0 and 1, or soft.

```python
from emoge import regions
```

Masks come from any segmentation, from a grid, or from superpixels; then where
the regions are, and how much they border each other.
"""

from emoge.regions.adjacency import adjacency
from emoge.regions.geometry import GEOMETRY, geometry
from emoge.regions.grids import grid
from emoge.regions.masks import masks
from emoge.regions.superpixels import superpixels

__all__ = ["GEOMETRY", "adjacency", "geometry", "grid", "masks", "superpixels"]
