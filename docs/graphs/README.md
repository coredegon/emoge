# graphs

The colors of an image, or its regions, as a graph: nodes with features, and
edges between them with features of their own, every feature named.

```python
import numpy as np
from emoge import color, fuzzy, graphs, regions

# An image to draw graphs of: red beside blue.
flag = np.zeros((40, 60, 3), dtype=np.uint8)
flag[:, :30], flag[:, 30:] = [200, 30, 30], [30, 60, 200]
```

Everything below is at `graphs.`.

## A graph

A `Graph` holds `nodes`, `(nodes, features)`, and `edges`, `(nodes, nodes,
features)`, which are 0 where two nodes share no edge; `names` names the nodes,
and `node_features` and `edge_features` the features. `sparse()` gives the edges
there are as pairs of nodes and their features, as PyTorch Geometric takes them.

## The colors of an image

`color_graph(model, image)` makes a node of every color of a model of
[`emoge.color`](../color/README.md), so every image gives the same nodes. A
node is how much of the image the color takes, how often it lies beside itself,
and its prototype in Oklab; an edge, how often two colors are neighbours and how
far apart their prototypes look.

```python
colors = graphs.color_graph(color.COLIBRI, flag)
print(colors)  # Graph(86 nodes, 3655 edges)
print(colors.node_features)  # ('share', 'beside itself', 'lightness', 'a', 'b')
print(colors.edge_features)  # ('neighbours', 'distance')
share = colors.nodes[:, 0]
print([colors.names[i] for i in np.argsort(-share)[:2]])
# ['blue/high/medium', 'red/high/dark']
```

## The regions of an image

`region_graph(model, image, masks)` makes a node of every region of
[`emoge.regions`](../regions/README.md): its palette, where it is and how large,
and its mean color in Oklab. Two regions that touch share an edge: how much
they border each other and how far apart their colors look.

```python
cells = regions.grid(flag.shape, 2, 2)
parts = graphs.region_graph(color.COLIBRI, flag, cells)
print(parts, parts.edge_features)  # Graph(4 nodes, 4 edges) ('border', 'distance')
```

The four cells touch side by side and one above the other, not across.

## Relations

`relations=` adds the words of relations of colors to the edges, given as data:
a name, a space, a channel, and the words of a partition of `emoge.fuzzy` on how
far apart two colors are on that channel.

```python
harmony = fuzzy.partition(
    fuzzy.Channel("hue difference", 0, 180),
    {"same": 0, "analogous": 30, "triadic": 120, "complementary": 180},
)
relations = {"harmony": (color.OKLCH, "hue", harmony)}
parts = graphs.region_graph(color.COLIBRI, flag, cells, relations)
print(parts.edge_features[2:])
# ('harmony/same', 'harmony/analogous', 'harmony/triadic', 'harmony/complementary')
print(parts.edges[0, 1, 2:].round(2).tolist())  # [0.0, 0.0, 0.98, 0.02]
```

The red cell and the blue one beside it are triadic.
