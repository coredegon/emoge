"""`grid`: the cells of a grid over an image, crisp or soft."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from emoge.fuzzy.channels import Channel
from emoge.fuzzy.partitions import partition


def grid(
    shape: tuple[int, ...], rows: int, columns: int, *, soft: bool = False
) -> NDArray[np.float64]:
    """Return the cells of a grid over an image: `(rows * columns, height, width)`.

    ```python
    grid(image.shape, 3, 3, soft=True)
    ```

    Cells go row by row. A crisp cell is 1 inside, 0 outside. A soft one passes
    into its neighbours, the words of two partitions joined with the product,
    one down the image and one across, with a core at the middle of each cell;
    the cells of a pixel still sum to 1.
    """
    height, width = shape[:2]
    if rows < 1 or columns < 1:
        raise ValueError(f"a grid has at least one cell, got {rows} x {columns}")
    down = _cells((np.arange(height) + 0.5) / height, rows, soft)
    across = _cells((np.arange(width) + 0.5) / width, columns, soft)
    cells = down[:, None, :, None] * across[None, :, None, :]
    return cells.reshape(rows * columns, height, width)


def _cells(x: NDArray[np.float64], count: int, soft: bool) -> NDArray[np.float64]:
    """The degree of every place along one side in every one of `count` cells."""
    if count == 1:
        return np.ones((1, len(x)))
    if not soft:
        return (
            np.minimum(x * count, count - 1).astype(int) == np.arange(count)[:, None]
        ).astype(float)
    middles = {str(i): (i + 0.5) / count for i in range(count)}
    return partition(Channel("place", 0.0, 1.0), middles).degrees(x)
