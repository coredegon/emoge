"""COLIBRI: the color categories of people, on HSI.

P. Shamoi et al., "COLIBRI Fuzzy Model: Color Linguistic-Based Representation
and Interpretation", IEEE Access 13, 2025 (arXiv:2507.11488). The cores of the
terms come from a large survey of how people name colors.

A color of very low saturation is achromatic and named after its intensity
alone. Every other color is chromatic and named after its hue, its saturation
and a coarser intensity, its shade: 9 x 3 x 3 + 5 = 86 colors.
"""

from emoge.fuzzy_color.model import ColorModel
from emoge.fuzzy_color.partitions import partition
from emoge.fuzzy_color.spaces.hsi import HSI

HUE = partition(
    HSI.channel("hue"),
    {
        "red": (350, 5),
        "orange": (20, 35),
        "yellow": 45,
        "green": (65, 130),
        "cyan": 175,
        "light blue": 190,
        "blue": (215, 240),
        "violet": (270, 280),
        "magenta": (320, 330),
    },
)

SATURATION = partition(
    HSI.channel("saturation"),
    {"very low": 0, "low": 0.15, "medium": (0.375, 0.45), "high": (0.725, 1)},
)

INTENSITY = partition(
    HSI.channel("intensity"),
    {
        "black": (0, 30 / 255),
        "dark gray": 75 / 255,
        "gray": (110 / 255, 115 / 255),
        "light gray": (190 / 255, 215 / 255),
        "white": 1,
    },
)

# INTENSITY with neighbours merged: dark = black + dark gray, medium = gray,
# light = light gray + white.
SHADE = partition(
    HSI.channel("intensity"),
    {"dark": (0, 75 / 255), "medium": (110 / 255, 115 / 255), "light": (190 / 255, 1)},
)

COLIBRI = ColorModel(
    "COLIBRI",
    HSI,
    {
        "chromatic": [HUE, SATURATION.select("low", "medium", "high"), SHADE],
        "achromatic": [INTENSITY, SATURATION.select("very low")],
    },
)
