"""`Dataset`: images and their annotations, one sample format for all of them."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image


class Dataset:
    """Images and their annotations: a sequence of samples.

    ```python
    images = Dataset(rows, root="photos", label="emotion", size=224)
    images[0]   # {"image": (224, 224, 3) uint8, "path": ..., "label": 2, ...}
    ```

    Every row is a mapping with the path of an image under `image`, relative
    to `root`, and any other fields. A sample is the row with its image loaded
    as RGB bytes, `(height, width, 3)`, and its full path under `"path"`. With
    `label`, the values of that field become indices into `classes`, sorted,
    under `"label"`. With `size`, every image is made `size` x `size`: its
    shorter side shrinks to `size`, every new pixel the mean of those it
    covers, and the middle is cut out.

    A dataset is a sequence, so a `torch.utils.data.DataLoader` takes it as it
    is.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        root: str | os.PathLike[str] | None = None,
        *,
        image: str = "image",
        label: str | None = None,
        size: int | None = None,
    ) -> None:
        missing = [i for i, row in enumerate(rows) if image not in row]
        if missing:
            raise KeyError(f"rows {missing[:5]} have no {image!r}")
        if size is not None and size < 1:
            raise ValueError(f"an image is at least 1 pixel, got {size}")
        self.rows = [dict(row) for row in rows]
        self.root = Path(root) if root is not None else None
        self.image, self.label, self.size = image, label, size
        self.classes: list[Any] = []
        if label is not None:
            self.classes = sorted({row[label] for row in self.rows})

    @property
    def labels(self) -> NDArray[np.intp]:
        """The label of every sample, without loading the images."""
        if self.label is None:
            raise ValueError("a dataset without `label` has no labels")
        index = {name: i for i, name in enumerate(self.classes)}
        return np.array([index[row[self.label]] for row in self.rows], dtype=np.intp)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.rows[i]
        path = Path(row[self.image])
        if self.root is not None:
            path = self.root / path
        sample = {key: value for key, value in row.items() if key != self.image}
        if self.label is not None:
            sample["label"] = self.classes.index(sample.pop(self.label))
        sample["image"] = load_image(path, self.size)
        sample["path"] = str(path)
        return sample

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return (self[i] for i in range(len(self)))

    def __repr__(self) -> str:
        labelled = f", {len(self.classes)} classes" if self.label else ""
        return f"Dataset({len(self)} images{labelled})"


def load_image(
    path: str | os.PathLike[str], size: int | None = None
) -> NDArray[np.uint8]:
    """Load an image as RGB bytes, `(height, width, 3)`, made `size` x `size`."""
    with Image.open(path) as opened:
        if size is not None:
            opened.draft("RGB", (size, size))  # a JPEG decodes smaller, faster
        picture = opened.convert("RGB")
    if size is not None:
        scale = size / min(picture.size)
        width, height = (max(size, round(side * scale)) for side in picture.size)
        picture = picture.resize((width, height), Image.Resampling.BOX)
        left, top = (width - size) // 2, (height - size) // 2
        picture = picture.crop((left, top, left + size, top + size))
    return np.array(picture, dtype=np.uint8)  # a copy: writable, as torch wants
