"""Color models that learn, as PyTorch layers: `pip install emoge[torch]`.

```python
from emoge import nn
```
"""

try:
    import torch  # noqa: F401
except ImportError:
    raise ImportError("emoge.nn needs PyTorch: pip install emoge[torch]") from None

from emoge.nn.colors import FuzzyColors
from emoge.nn.pooling import cooccurrence, palette

__all__ = ["FuzzyColors", "cooccurrence", "palette"]
