"""The mathematics of fuzzy sets as plain NumPy functions; nothing here knows colors.

- `membership`: the shapes of fuzzy sets;
- `norms`: t-norms and s-norms, ways to join degrees.
"""

from emoge.fuzzy_color.math import membership, norms

__all__ = ["membership", "norms"]
