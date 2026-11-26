"""Datasets of images: one sample format, whatever the layout of the files.

```python
from emoge import data
```
"""

from emoge.data.dataset import Dataset, load_image
from emoge.data.features import features
from emoge.data.sources import folders, table

__all__ = ["Dataset", "features", "folders", "load_image", "table"]
