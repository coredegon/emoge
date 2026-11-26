"""Graphs of the colors and of the regions of an image, with named features.

```python
from emoge import graphs
```
"""

from emoge.graphs.colors import color_graph
from emoge.graphs.files import load, save
from emoge.graphs.graph import Graph
from emoge.graphs.plots import plot_graph
from emoge.graphs.regions import region_graph
from emoge.graphs.relations import relate

__all__ = [
    "Graph",
    "color_graph",
    "load",
    "plot_graph",
    "region_graph",
    "relate",
    "save",
]
