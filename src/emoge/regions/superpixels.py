"""`superpixels`: regions of like color close together, crisp or soft."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from emoge.color.spaces.oklab import rgb_to_oklab
from emoge.color.spaces.rgb import to_float_rgb

# Added to a squared distance, so that a pixel on a center does not divide by 0.
_NEAR = 1e-9

type Window = tuple[slice, slice]


def superpixels(
    image: ArrayLike,
    count: int = 100,
    *,
    compactness: float = 10.0,
    soft: bool = False,
    steps: int = 10,
) -> NDArray[np.bool_] | NDArray[np.float64]:
    """Return regions of like color close together: `(regions, height, width)`.

    ```python
    superpixels(image, 100)              # crisp: every pixel in one region
    superpixels(image, 100, soft=True)   # soft: degrees that sum to 1
    ```

    The regions start on a grid of about `count` cells. For `steps` rounds,
    every pixel joins the nearest region around it, by color in Oklab and by
    place, and every region moves to its pixels, as SLIC does; the larger
    `compactness`, the more place counts against color and the rounder the
    regions. A soft region is a fuzzy c-means cluster among the regions around
    a pixel: the pixel belongs to each by how near it is, against the others.
    A crisp region left with no pixels is dropped.
    """
    rgb = to_float_rgb(image)
    height, width = rgb.shape[:2]
    if count < 1:
        raise ValueError(f"superpixels number at least 1, got {count}")
    side = math.sqrt(height * width / count)  # between neighbouring starts
    scale = compactness / side
    ys, xs = np.mgrid[0:height, 0:width]
    # Oklab a hundredfold is about as large as CIELab, which SLIC weighs place against.
    lab = 100.0 * rgb_to_oklab(rgb)
    points = np.concatenate([lab, np.stack([ys, xs], axis=-1) * scale], axis=-1)
    rows, columns = max(1, round(height / side)), max(1, round(width / side))
    start_y = ((np.arange(rows) + 0.5) * height / rows).astype(int)
    start_x = ((np.arange(columns) + 0.5) * width / columns).astype(int)
    centers = points[start_y[:, None], start_x[None, :]].reshape(-1, 5)
    reach = math.ceil(side)  # a region looks one cell around its center
    for _ in range(steps):
        windows = _windows(centers, scale, reach)
        if soft:
            # Fuzzy c-means weighs a pixel by its membership squared.
            weights = [share**2 for share in _shares(points, centers, windows)]
            mass = np.array([weight.sum() for weight in weights])
            sums = np.stack(
                [
                    np.einsum("hw,hwd->d", weight, points[window])
                    for weight, window in zip(weights, windows, strict=True)
                ]
            )
        else:
            labels = _labels(points, centers, windows).ravel()
            mass = np.bincount(labels, minlength=len(centers)).astype(float)
            sums = np.stack(
                [
                    np.bincount(labels, axis, len(centers))
                    for axis in points.reshape(-1, 5).T
                ],
                axis=1,
            )
        kept = mass > 0
        centers[kept] = sums[kept] / mass[kept, None]
    windows = _windows(centers, scale, reach)
    if soft:
        return _memberships(points, centers, windows)
    regions = (
        _labels(points, centers, windows) == np.arange(len(centers))[:, None, None]
    )
    return regions[regions.any(axis=(1, 2))]


def _windows(centers: NDArray[np.float64], scale: float, reach: int) -> list[Window]:
    """The pixels every region looks at: `reach` around its center on every side."""
    windows = []
    for y, x in (centers[:, 3:] / scale).astype(int):
        top, left = max(0, y - reach), max(0, x - reach)
        windows.append((slice(top, y + reach + 1), slice(left, x + reach + 1)))
    return windows


def _distances(
    points: NDArray[np.float64], center: NDArray[np.float64], window: Window
) -> NDArray[np.float64]:
    return ((points[window] - center) ** 2).sum(axis=-1)


def _nearest_of_all(
    points: NDArray[np.float64], centers: NDArray[np.float64]
) -> NDArray[np.intp]:
    """For pixels no window holds, as centers moved away: the nearest region."""
    return ((points[:, None, :] - centers) ** 2).sum(axis=-1).argmin(axis=1)


def _labels(
    points: NDArray[np.float64], centers: NDArray[np.float64], windows: list[Window]
) -> NDArray[np.intp]:
    """The nearest region of every pixel among those whose window holds it."""
    best = np.full(points.shape[:2], np.inf)
    labels = np.full(points.shape[:2], -1)
    for region, window in enumerate(windows):
        distances = _distances(points, centers[region], window)
        closer = distances < best[window]
        best[window] = np.where(closer, distances, best[window])
        labels[window] = np.where(closer, region, labels[window])
    alone = labels < 0
    if alone.any():
        labels[alone] = _nearest_of_all(points[alone], centers)
    return labels


def _shares(
    points: NDArray[np.float64], centers: NDArray[np.float64], windows: list[Window]
) -> list[NDArray[np.float64]]:
    """The fuzzy c-means membership of the pixels of every window in its region.

    With the fuzzifier 2: a pixel belongs to every region whose window holds it
    as 1 over the squared distance, against the others.
    """
    nearness = [
        1.0 / (_distances(points, centers[region], window) + _NEAR)
        for region, window in enumerate(windows)
    ]
    total = np.zeros(points.shape[:2])
    for window, near in zip(windows, nearness, strict=True):
        total[window] += near
    return [
        near / total[window] for window, near in zip(windows, nearness, strict=True)
    ]


def _memberships(
    points: NDArray[np.float64], centers: NDArray[np.float64], windows: list[Window]
) -> NDArray[np.float64]:
    """Fuzzy c-means memberships of every pixel: `(regions, height, width)`."""
    memberships = np.zeros((len(centers), *points.shape[:2]))
    for region, (window, share) in enumerate(
        zip(windows, _shares(points, centers, windows), strict=True)
    ):
        memberships[region][window] = share
    alone = memberships.sum(axis=0) == 0
    if alone.any():
        ys, xs = np.nonzero(alone)
        memberships[_nearest_of_all(points[alone], centers), ys, xs] = 1.0
    return memberships
