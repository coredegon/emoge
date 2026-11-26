"""Palettes and neighbours of degree maps, batched and differentiable.

Degree maps are `(..., height, width, colors)`, as `FuzzyColors` gives them for
images; the results match `ColorModel.palette` and `ColorModel.cooccurrence`.
"""

from __future__ import annotations

import torch


def palette(degrees: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Return the mean degree of every color over the pixels: `(..., colors)`.

    `weights`, `(..., height, width)`, weigh the mean as a soft mask of a region
    does; masks stacked as `(..., masks, height, width)` give a palette each,
    `(..., masks, colors)`.
    """
    if weights is None:
        return degrees.mean(dim=(-3, -2))
    if weights.dim() == degrees.dim() - 1:
        return palette(degrees, weights.unsqueeze(-3)).squeeze(-2)
    totals = torch.einsum("...mhw,...hwc->...mc", weights.to(degrees.dtype), degrees)
    return totals / weights.sum(dim=(-2, -1)).unsqueeze(-1)


def cooccurrence(degrees: torch.Tensor, distance: int = 1) -> torch.Tensor:
    """Return how often every two colors are neighbours: `(..., colors, colors)`.

    Two pixels `distance` apart, side by side or one above the other, are
    neighbours; their degrees are multiplied and averaged over all neighbours
    both ways round.
    """
    height, width = degrees.shape[-3], degrees.shape[-2]
    if distance < 1 or (height <= distance and width <= distance):
        raise ValueError(
            f"an image of {height} x {width} has no neighbours {distance} apart"
        )
    pair = "...hwi,...hwj->...ij"
    side_by_side = degrees[..., :, :-distance, :], degrees[..., :, distance:, :]
    one_above = degrees[..., :-distance, :, :], degrees[..., distance:, :, :]
    totals = torch.einsum(pair, *side_by_side) + torch.einsum(pair, *one_above)
    pairs = height * max(width - distance, 0) + max(height - distance, 0) * width
    return (totals + totals.transpose(-1, -2)) / (2 * pairs)
