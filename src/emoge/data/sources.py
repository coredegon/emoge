"""Datasets from the usual layouts: a folder per class, or a table of images."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from emoge.data.dataset import Dataset

# The files taken for images in a folder.
IMAGE_SUFFIXES = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


def folders(root: str | os.PathLike[str], *, size: int | None = None) -> Dataset:
    """Return the images in a folder per class, named after its class.

    ```python
    folders("photos")   # photos/calm/1.jpg, photos/fear/2.jpg, ...
    ```

    The label of an image is its folder, an index into `classes`.
    """
    root = Path(root)
    rows = [
        {"image": f"{folder.name}/{file.name}", "class": folder.name}
        for folder in sorted(p for p in root.iterdir() if p.is_dir())
        for file in sorted(folder.iterdir())
        if file.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not rows:
        raise ValueError(f"{root} has no folders of images")
    return Dataset(rows, root, label="class", size=size)


def table(
    path: str | os.PathLike[str],
    root: str | os.PathLike[str] | None = None,
    *,
    image: str = "image",
    label: str | None = None,
    size: int | None = None,
) -> Dataset:
    """Return the images a table lists, a row each, with its other columns.

    ```python
    table("train.csv", image="path", label="emotion")
    ```

    The table is CSV, whose values stay text, or JSON: a list of objects, or
    an object on every line in `.jsonl`. Paths of images are relative to
    `root`, the folder of the table unless given.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
    elif path.suffix == ".json":
        rows = json.loads(text)
    elif path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raise ValueError(f"a table is .csv, .json or .jsonl, got {path.name}")
    return Dataset(
        rows, path.parent if root is None else root, image=image, label=label, size=size
    )
