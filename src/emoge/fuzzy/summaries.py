"""`truth`: how true a sentence like "most of the images are dark" is.

R. R. Yager, "A new approach to the summarization of data", Information
Sciences 28, 1982.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from emoge.fuzzy.sets import FuzzySet


def truth(
    quantifier: FuzzySet, degrees: ArrayLike, among: ArrayLike | None = None
) -> float:
    """Return how true it is that `quantifier` of the items are as `degrees` say.

    ```python
    truth(most, dark)               # most of the images are dark
    truth(most, dark, among=fear)   # most of the fearful images are dark
    ```

    `quantifier` is a fuzzy set on the share of the items, from 0 to 1;
    `degrees` give how much each item is what the sentence says. `among`, the
    degree of each item in a group, keeps the sentence to that group: an item
    counts as much as the smaller of its two degrees.
    """
    said = np.asarray(degrees, dtype=np.float64)
    group = np.ones_like(said) if among is None else np.asarray(among, dtype=float)
    said, group = np.broadcast_arrays(said, group)
    size = group.sum()
    if not size > 0:
        raise ValueError("a sentence needs items to speak of")
    share = np.minimum(said, group).sum() / size
    return float(quantifier.degrees(share))
