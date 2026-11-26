"""`features`: something of every image of a dataset, as one array."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from emoge.data.dataset import Dataset


def features(
    dataset: Dataset,
    function: Callable[[NDArray[np.uint8]], ArrayLike],
    workers: int = 4,
) -> NDArray[Any]:
    """Return `function` of every image of a dataset, stacked: `(len(dataset), ...)`.

    ```python
    palettes = features(images, COLIBRI.palette)   # (len(images), 86)
    ```

    Images load and `function` runs in `workers` threads, which share the
    work, as decoding images and NumPy let go of the GIL.
    """

    def one(i: int) -> ArrayLike:
        return function(dataset[i]["image"])

    with ThreadPoolExecutor(max(1, workers)) as pool:
        return np.stack(list(pool.map(one, range(len(dataset)))))
