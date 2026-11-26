"""`save` and `load`: color models in JSON files, to keep or share them."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from emoge.color.model import ColorModel
from emoge.color.space import Space
from emoge.color.spaces import HSI, HSV, OKLAB, OKLCH, RGB
from emoge.fuzzy.math import norms
from emoge.fuzzy.partitions import Partition, partition
from emoge.fuzzy.variables import Variable

_SPACES = {space.name: space for space in (HSI, HSV, OKLAB, OKLCH, RGB)}


def save(model: ColorModel, path: str | os.PathLike[str]) -> None:
    """Write a color model to a JSON file, which `load` reads back.

    ```python
    save(COLIBRI, "colibri.json")
    ```

    The model is made of partitions on a space of `emoge.color`, as COLIBRI
    is; the file holds their cores, each written as `partition` takes it.
    """
    if _SPACES.get(model.space.name) is not model.space:
        raise ValueError(f"only the spaces of emoge.color are saved, not {model.space}")
    and_norm = getattr(model.and_norm, "__name__", "")
    if getattr(norms, and_norm, None) is not model.and_norm:
        raise ValueError("only the norms of emoge.fuzzy.norms are saved")
    saved = {
        "name": model.name,
        "space": model.space.name,
        "and_norm": and_norm,
        "families": {
            family: [_saved(variable) for variable in variables]
            for family, variables in model.families.items()
        },
    }
    text = json.dumps(saved, indent=2, separators=(", ", ": "))
    # A core or a list of terms on one line: no string holds a line break.
    text = re.sub(r"\[[^\[\]{}]*\]", lambda list_: re.sub(r"\n *", "", list_[0]), text)
    text = re.sub(r" +\n", "\n", text)
    Path(path).write_text(text + "\n", encoding="utf-8")


def load(path: str | os.PathLike[str]) -> ColorModel:
    """Read a color model that `save` wrote."""
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    space = _SPACES[saved["space"]]
    families = {
        family: [_partition(space, variable) for variable in variables]
        for family, variables in saved["families"].items()
    }
    and_norm = getattr(norms, saved["and_norm"])
    return ColorModel(saved["name"], space, families, and_norm=and_norm)


def _saved(variable: Variable) -> dict[str, Any]:
    """A partition as JSON: its channel, its cores, and the terms it selects."""
    if not isinstance(variable, Partition):
        raise TypeError(f"{variable} is not made by partition: it has no cores")
    cores = {
        name: start if start == end else [start, end]
        for name, (start, end) in variable.cores.items()
    }
    saved: dict[str, Any] = {"channel": variable.channel.name, "cores": cores}
    if list(variable.terms) != list(variable.cores):
        saved["terms"] = list(variable.terms)
    return saved


def _partition(space: Space, saved: dict[str, Any]) -> Partition:
    cores = {
        name: tuple(core) if isinstance(core, list) else core
        for name, core in saved["cores"].items()
    }
    made = partition(space.channel(saved["channel"]), cores)
    return made.select(*saved.get("terms", made.terms))
