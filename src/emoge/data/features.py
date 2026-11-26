"""Something of every image of a dataset: one by one, or stacked into one array."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from emoge.data.dataset import Dataset


def each[T](
    dataset: Dataset, function: Callable[[NDArray[np.uint8]], T], workers: int = 4
) -> Iterator[T]:
    """Yield `function` of every image of a dataset, in order, as each is ready.

    ```python
    for graph in each(images, make_graph):   # one at a time, never all at once
        ...
    ```

    Images load and `function` runs in `workers` threads, which share the work,
    as decoding images and NumPy let go of the GIL; only a few results wait at
    a time, so a large dataset never fills the memory.
    """
    workers = max(1, workers)
    with ThreadPoolExecutor(workers) as pool:
        waiting: deque[Future[T]] = deque()
        for i in range(len(dataset)):
            waiting.append(pool.submit(lambda i=i: function(dataset[i]["image"])))
            if len(waiting) > 2 * workers:
                yield waiting.popleft().result()
        while waiting:
            yield waiting.popleft().result()


def features(
    dataset: Dataset,
    function: Callable[[NDArray[np.uint8]], ArrayLike],
    workers: int = 4,
) -> NDArray[Any]:
    """Return `function` of every image of a dataset, stacked: `(len(dataset), ...)`.

    ```python
    palettes = features(images, COLIBRI.palette)   # (len(images), 86)
    ```

    The images go through `each`, in `workers` threads.
    """
    return np.stack([np.asarray(result) for result in each(dataset, function, workers)])
