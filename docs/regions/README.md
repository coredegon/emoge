# regions

The regions of an image are masks, `(regions, height, width)`. A crisp mask is
1 inside its region and 0 outside; a soft one holds degrees, which sum to 1 in
every pixel. Whatever takes masks takes either kind.

```python
import numpy as np
from emoge import color, regions

# An image to cut up: red beside blue.
flag = np.zeros((40, 60, 3), dtype=np.uint8)
flag[:, :30], flag[:, 30:] = [200, 30, 30], [30, 60, 200]
```

Everything below is at `regions.`.

## Masks

`grid(shape, rows, columns)` cuts an image into cells, row by row; with
`soft=True`, every cell passes into its neighbours.

```python
cells = regions.grid(flag.shape, 2, 3)
print(cells.shape)  # (6, 40, 60)
soft = regions.grid(flag.shape, 2, 3, soft=True)
print(soft[:, 10, 25].round(2).tolist())  # [0.22, 0.76, 0.0, 0.01, 0.02, 0.0]
```

`superpixels(image, count)` gathers pixels of like color close together, about
`count` regions, as SLIC does; `compactness` weighs place against color. With
`soft=True` the regions are fuzzy: a pixel belongs to each region around it by
how near it is.

```python
pieces = regions.superpixels(flag, 12)
print(len(pieces), (pieces.sum(axis=0) == 1).all())  # 12 True
```

A segmentation of any other kind gives a label map, which `masks(labels)`
turns into masks:

```python
print(regions.masks([[0, 0, 1], [2, 2, 1]]).shape)  # (3, 2, 3)
```

## Where the regions are

`geometry(masks)` gives the area of every region, its middle and its spread,
in the columns `GEOMETRY`; places are shares of the height and the width.

```python
print(regions.GEOMETRY)  # ('area', 'y', 'x', 'spread y', 'spread x')
print(
    regions.geometry(cells)[0].round(3).tolist()
)  # [0.167, 0.25, 0.167, 0.144, 0.096]
```

`adjacency(masks)` tells how much every two regions are neighbours: pixels side
by side or one above the other, as `cooccurrence` does for colors. Its diagonal
is the inside of a region, the rest are the borders.

```python
touch = regions.adjacency(cells)
print((touch[0] > 0).astype(int).tolist())  # [1, 1, 0, 1, 0, 0]
```

The first cell borders the one beside it and the one below.

## The colors of regions

A palette of [`emoge.color`](../color/README.md) takes masks:

```python
palettes = color.COLIBRI.palette(flag, cells)
print(palettes.shape)  # (6, 86)
```
