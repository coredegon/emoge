"""Color models that learn, and graphs, as PyTorch layers: `pip install emoge[torch]`.

```python
from emoge import nn
```
"""

try:
    import torch  # noqa: F401
except ImportError:
    raise ImportError("emoge.nn needs PyTorch: pip install emoge[torch]") from None

from emoge.nn.colors import FuzzyColors
from emoge.nn.graphs import GraphEncoder, GraphLayer, batch, color_graph, readout
from emoge.nn.pooling import cooccurrence, palette

__all__ = [
    "FuzzyColors",
    "GraphEncoder",
    "GraphLayer",
    "batch",
    "color_graph",
    "cooccurrence",
    "palette",
    "readout",
]
