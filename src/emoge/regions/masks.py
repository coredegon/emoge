"""`masks`: the regions of a label map, a mask each."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def masks(labels: ArrayLike) -> NDArray[np.bool_]:
    """Return a mask for every region of a label map: `(regions, *labels.shape)`.

    ```python
    masks(segmentation)   # the label map of any segmentation
    ```

    Region `i` is where the label is `i`, from 0 up to the largest label; a
    negative label is in no region.
    """
    labels = np.asarray(labels)
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"labels are whole numbers, got {labels.dtype}")
    count = int(labels.max(initial=-1)) + 1
    return labels == np.arange(count).reshape(-1, *[1] * labels.ndim)
