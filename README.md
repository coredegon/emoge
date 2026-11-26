# emoge

emoge is a package for research purposes: the analysis of images and the
emotions they evoke.

## Documentation

| Package | What it is for | Guide |
|---|---|---|
| `emoge.fuzzy` | Fuzzy logic on any axis: fuzzy sets, the words of a channel, and the ways to join their degrees | [docs/fuzzy](https://github.com/coredegon/emoge/tree/main/docs/fuzzy) |
| `emoge.color` | Fuzzy color models: the degree to which every pixel belongs to every named color, and the color composition of an image | [docs/color](https://github.com/coredegon/emoge/tree/main/docs/color) |
| `emoge.regions` | Regions of an image as masks, crisp or soft: grids, superpixels, label maps; where regions are and how they border | [docs/regions](https://github.com/coredegon/emoge/tree/main/docs/regions) |
| `emoge.graphs` | Graphs of the colors and of the regions of an image, with named features and relations given as data; the graphs of a whole dataset saved and read one at a time | [docs/graphs](https://github.com/coredegon/emoge/tree/main/docs/graphs) |
| `emoge.data` | Datasets of images in one format: a folder per class, a table, or rows from anywhere; features of a whole dataset | [docs/data](https://github.com/coredegon/emoge/tree/main/docs/data) |
| `emoge.nn` | Color models that learn, and networks on graphs of any size, as PyTorch layers; `pip install emoge[torch]` | [docs/nn](https://github.com/coredegon/emoge/tree/main/docs/nn) |

## Disclaimer

Not everything may be implemented yet. Missing parts can be added on your own,
or suggested in an issue or a pull request.
