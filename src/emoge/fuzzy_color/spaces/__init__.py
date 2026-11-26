"""Color spaces, one module each; RGB with components in [0, 1] is the exchange format.

- `space`: `Channel` and `Space`, what every space is made of;
- `rgb`: `RGB`, `to_float_rgb` and `fit_into_rgb`;
- `hsi`: `HSI`, hue, saturation and intensity;
- `hsv`: `HSV`, hue, saturation and value.
"""

from emoge.fuzzy_color.spaces.hsi import HSI, hsi_to_rgb, rgb_to_hsi
from emoge.fuzzy_color.spaces.hsv import HSV, hsv_to_rgb, rgb_to_hsv
from emoge.fuzzy_color.spaces.rgb import RGB, fit_into_rgb, to_float_rgb
from emoge.fuzzy_color.spaces.space import Channel, Space

__all__ = [
    "HSI",
    "HSV",
    "RGB",
    "Channel",
    "Space",
    "fit_into_rgb",
    "hsi_to_rgb",
    "hsv_to_rgb",
    "rgb_to_hsi",
    "rgb_to_hsv",
    "to_float_rgb",
]
