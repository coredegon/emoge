# nn

A color model that learns. `FuzzyColors` starts as a model of
[`emoge.color`](../color/README.md), such as COLIBRI, and learns its cores, the
values that are fully each word, along with the rest of a network. It needs
PyTorch: `pip install emoge[torch]`.

```python
import numpy as np
import torch
from emoge import color, nn
```

Everything below is at `nn.`.

## Degrees of colors

The layer takes the channels of the model's space and gives the degree of every
color, both along the last axis. Turn pixels into channels where the images are
loaded; the layer then runs wherever the network does.

```python
images = np.random.default_rng(0).integers(0, 256, (2, 8, 8, 3))  # two images
values = torch.as_tensor(color.HSI.from_rgb(images / 255), dtype=torch.float32)

layer = nn.FuzzyColors(color.COLIBRI)
degrees = layer(values)
print(tuple(degrees.shape))  # (2, 8, 8, 86)
```

Before it learns, the layer is the model it started from.

## Learning

The cores learn with the rest of a network, as any parameters do:

```python
optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
loss = -layer(values)[..., 0].mean()  # any loss of the degrees
loss.backward()
optimizer.step()
```

Every word learns where its core starts and ends. The cores keep their order,
and the degrees of a pixel still sum to 1. A partition that several families
share, as saturation in COLIBRI, learns once.

Free cores may drift from their meaning: a red that moves into orange is no
longer red. `drift()` tells how far they have moved, and added to the loss it
holds them, the more the larger its weight:

```python
loss = -layer(values)[..., 0].mean() + 10 * layer.drift()
```

## Palettes and neighbours

`palette(degrees, weights)` and `cooccurrence(degrees, distance)` are those of
`emoge.color` for a batch of degree maps, `(..., height, width, colors)`, and
gradients pass through them:

```python
print(tuple(nn.palette(degrees).shape), tuple(nn.cooccurrence(degrees).shape))
# (2, 86) (2, 86, 86)
```

## Graphs

`color_graph(degrees)` makes the color graph of degree maps: a node for every
color, how much of the image it takes and how often it lies beside itself, and
an edge for how often two colors are neighbours. `GraphLayer` passes messages
along the edges, every one weighed by a gate its features open, and `readout`
takes the mean of the nodes, one vector a graph. From pixels to emotions, the
colors learn with everything after them:

```python
graph_layer = nn.GraphLayer(2, 1, 16)
emotions = torch.nn.Linear(16, 8)

nodes, edges = nn.color_graph(layer(values))
logits = emotions(nn.readout(graph_layer(nodes, edges)))
print(tuple(logits.shape))  # (2, 8)
```

Graphs made by [`emoge.graphs`](../graphs/README.md), as of regions, become
tensors with `batch(graphs)`: `nodes`, `edges`, and a `mask` of the nodes that
are there, the smaller graphs filled with 0. `GraphLayer` and `readout` take
the mask.

## Back to a color model

`to_model()` gives the model with its cores as learned, for everything
`emoge.color` does with a model: palettes, plots, lookup tables, and files.

```python
learned = layer.to_model()
print(learned.label([255, 0, 0]))  # red/high/dark
color.save(learned, "learned.json")  # to keep it
```
