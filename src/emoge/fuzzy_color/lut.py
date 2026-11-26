"""`lookup_table`: the strongest color of every 8-bit pixel, worked out once."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

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
    table = lookup_table(COLIBRI, "colibri.npz")  # built and saved, or loaded
    table[image[..., 0], image[..., 1], image[..., 2]]  # COLIBRI.classify(image)
    ```

    With `path`, the table is loaded from that compressed .npz file, or built
    in seconds and saved there when there is none yet. A file that disagrees
    with the model on a sample of pixels is refused: delete it to build anew.

    A color takes the smallest unsigned integer that holds `len(colors)`, the
    index of no color: a byte up to 255 colors. With `degree=True` every entry
    also keeps the degree of its color, in steps of 1/255, the byte it takes in
    a file: `table["color"]` and `table["degree"]`.
    """
    if path is None:
        return _build(model, degree)
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_name(path.name + ".npz")  # as np.savez names it
    if not path.exists():
        table = _build(model, degree)
        _save(path, table)
        return table
    table = _load(path)
    sample = np.arange(0, 256, 17, dtype=np.uint8)  # 16 levels a channel
    red, green, blue = np.meshgrid(sample, sample, sample, indexing="ij")
    expected = _entries(model, np.stack([red, green, blue], axis=-1), degree)
    if (
        table.dtype != expected.dtype
        or table.shape != (256, 256, 256)
        or not np.array_equal(table[red, green, blue], expected)
    ):
        raise ValueError(f"{path} holds a table of another model or kind")
    return table


def _build(model: ColorModel, degree: bool) -> Entries:
    levels = np.arange(256, dtype=np.uint8)
    green, blue = np.meshgrid(levels, levels, indexing="ij")

    def level(red: int) -> Entries:  # one red level: 65536 pixels
        pixels = np.stack([np.full_like(green, red), green, blue], axis=-1)
        return _entries(model, pixels, degree)

    # NumPy lets go of the GIL while it computes, so threads share the levels;
    # past 4 of them memory, not the cores, holds the work back.
    with ThreadPoolExecutor(min(4, os.cpu_count() or 1)) as pool:
        return np.stack(list(pool.map(level, range(256))))


def _entries(model: ColorModel, pixels: NDArray[np.uint8], degree: bool) -> Entries:
    """The strongest color of every pixel, and its degree if it is kept."""
    color, color_degree = model.strongest(pixels)
    color = color.astype(np.min_scalar_type(len(model.colors)))
    if not degree:
        return color
    return _join(color, np.round(color_degree * 255.0) / 255.0)


def _join(color: Entries, degree: NDArray[np.floating]) -> Entries:
    table = np.empty(
        color.shape, dtype=[("color", color.dtype), ("degree", np.float16)]
    )
    table["color"] = color
    table["degree"] = degree
    return table


# A file keeps the color and the degree apart, the degree in a byte, which
# compresses far better than a float16.


def _save(path: Path, table: NDArray[Any]) -> None:
    if table.dtype.names is None:
        np.savez_compressed(path, color=table)
        return
    in_bytes = np.round(table["degree"].astype(np.float32) * 255.0).astype(np.uint8)
    np.savez_compressed(path, color=table["color"], degree=in_bytes)


def _load(path: Path) -> Entries:
    with np.load(path) as file:
        if "degree" not in file.files:
            return file["color"]
        return _join(file["color"], file["degree"] / 255.0)
