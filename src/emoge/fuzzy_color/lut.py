"""`lookup_table`: the strongest color of every 8-bit pixel, worked out once."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from emoge.fuzzy_color.model import ColorModel

# The strongest color of every pixel: its index, or its index and degree.
type Entries = NDArray[np.unsignedinteger | np.void]


def lookup_table(
    model: ColorModel,
    path: str | os.PathLike[str] | None = None,
    *,
    degree: bool = False,
) -> Entries:
    """Return `model.classify` of every 8-bit pixel, as `table[red, green, blue]`.

    ```python
    table = lookup_table(COLIBRI, "colibri.npy")  # built and saved, or loaded
    table[image[..., 0], image[..., 1], image[..., 2]]  # COLIBRI.classify(image)
    ```

    With `path`, the table is loaded from that .npy file, or built in seconds
    and saved there when there is none yet. A file that disagrees with the
    model on a sample of pixels is refused: delete it to build anew.

    An index takes the smallest unsigned integer that holds `len(colors)`, the
    index of no color: a byte up to 255 colors, a 16 MB table.

    With `degree=True`, every entry also keeps the degree of its color, as a
    float16: `table["color"]` and `table["degree"]`, three bytes a pixel.
    """
    entry = np.min_scalar_type(len(model.colors))
    if degree:
        entry = np.dtype([("color", entry), ("degree", np.float16)])
    if path is None:
        return _build(model, entry)
    path = Path(path)
    if path.suffix != ".npy":
        path = path.with_name(path.name + ".npy")  # as np.save names it
    if not path.exists():
        table = _build(model, entry)
        np.save(path, table)
        return table
    table = np.load(path)
    sample = np.arange(0, 256, 17, dtype=np.uint8)  # 16 levels a channel
    red, green, blue = np.meshgrid(sample, sample, sample, indexing="ij")
    expected = _entries(model, np.stack([red, green, blue], axis=-1), entry)
    if (
        table.dtype != entry
        or table.shape != (256, 256, 256)
        or not np.array_equal(table[red, green, blue], expected)
    ):
        raise ValueError(f"{path} holds a table of another model or kind")
    return table


def _build(model: ColorModel, entry: np.dtype) -> Entries:
    table = np.empty((256, 256, 256), dtype=entry)
    levels = np.arange(256, dtype=np.uint8)
    green, blue = np.meshgrid(levels, levels, indexing="ij")

    def fill(red: int) -> None:  # one red level: 65536 pixels
        pixels = np.stack([np.full_like(green, red), green, blue], axis=-1)
        table[red] = _entries(model, pixels, entry)

    # NumPy lets go of the GIL while it computes, so threads share the levels;
    # past 4 of them memory, not the cores, holds the work back.
    with ThreadPoolExecutor(min(4, os.cpu_count() or 1)) as pool:
        list(pool.map(fill, range(256)))  # list() lets an error out
    return table


def _entries(model: ColorModel, pixels: NDArray[np.uint8], entry: np.dtype) -> Entries:
    """The strongest color of every pixel, and its degree if the entry keeps it."""
    color, degree = model.strongest(pixels)
    if entry.names is None:
        return color.astype(entry)
    entries = np.empty(color.shape, dtype=entry)
    entries["color"] = color
    entries["degree"] = degree
    return entries
