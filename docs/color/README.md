# color

People name colors loosely: an orange-red pixel is mostly orange, a little red.
A fuzzy color model says it in numbers, a degree from 0 to 1 for every color.

```python
import numpy as np
from emoge import color
```

Everything below is at `color.`.

## Naming colors

COLIBRI names 86 colors, after a survey of how people name them.

```python
image = np.array([[[200, 30, 30], [250, 220, 40]], [[30, 60, 200], [20, 20, 20]]])
print(color.COLIBRI.label(image).tolist())
# [['red/high/dark', 'yellow/high/light'], ['blue/high/medium', 'black']]

degrees = color.COLIBRI.degrees([230, 90, 40])  # an orange-red pixel
for i in np.argsort(-degrees)[:2]:
    print(color.COLIBRI.colors[i], round(float(degrees[i]), 2))
# orange/high/medium 0.48
# red/high/medium 0.26
```

| Method | Gives |
|---|---|
| `degrees(rgb)` | the degree of every pixel in every color |
| `label(rgb)`, `classify(rgb)` | the strongest color: its name, or its index in `colors` |
| `strongest(rgb)` | the index of the strongest color and its degree |
| `palette(image)` | how much of the image each color takes |
| `prototypes()` | a typical RGB color for every color, as below |

![The 86 colors of COLIBRI](img/colibri_swatches.png)

For many images, `color.lookup_table(color.COLIBRI, "colibri.npz")` classifies
every 8-bit pixel once and keeps the table in that file, 0.6 MB compressed; then
`table[red, green, blue]` classifies a whole image at once. `degree=True` keeps
the degree of each color too, in steps of 1/255.

## How a model works

A pixel first becomes hue, saturation and intensity, the channels of `HSI`.
Each channel is described by words, its terms, such as gray, pale and vivid
for saturation. A color takes one word per channel, like red, high, dark, and its
degree is the degrees of its words multiplied.

The words are made with [`emoge.fuzzy`](../fuzzy/README.md); its guide tells
what a word is and how `partition` makes the words of a channel.

### Colors from words

A family takes one word from each of its channels; a model is its families.

```python
from emoge import fuzzy

hue = fuzzy.partition(
    color.HSI.channel("hue"), {"red": (350, 10), "green": 120, "blue": 240}
)
saturation = fuzzy.partition(
    color.HSI.channel("saturation"), {"gray": 0, "pale": 0.5, "vivid": 1}
)
intensity = fuzzy.partition(color.HSI.channel("intensity"), {"dark": 0, "light": 1})
toy = color.ColorModel(
    "toy",
    color.HSI,
    {
        "tint": [hue, saturation.select("pale", "vivid")],
        "gray": [intensity, saturation.select("gray")],
    },
)
print(toy.colors)
# ('red/pale', 'red/vivid', 'green/pale', 'green/vivid', 'blue/pale', 'blue/vivid', 'dark', 'light')
print(toy.label([0, 0, 255]))  # blue/vivid
```

Gray has no hue, so its family leaves hue out. `and_norm=` joins the degrees of
the words in another way than the product: any t-norm of `fuzzy.norms`.

## Color spaces

| Space | Channels |
|---|---|
| `HSI` | `hue` 0–360 around, `saturation`, `intensity` 0–1 |
| `HSV` | `hue` 0–360 around, `saturation`, `value` 0–1 |
| `RGB` | `red`, `green`, `blue` 0–1 |

`Space(name, channels, from_rgb, to_rgb)` adds one of your own, with channels
made by `fuzzy.Channel`.

## Plots

`plot_terms(model)`, `plot_swatches(model)` and `plot_palette(model, weights)`;
each returns its Axes.

![The words of COLIBRI](img/colibri_terms.png)
