"""Color spaces built as `emoge.color.Space`, one module each.

RGB with components in [0, 1] is the exchange format.

- `rgb`: `RGB`, `to_float_rgb` and `fit_into_rgb`;
- `hsi`: `HSI`, hue, saturation and intensity;
- `hsv`: `HSV`, hue, saturation and value;
- `oklab`: `OKLAB`, lightness and the axes `a` and `b`, as people see them;
- `oklch`: `OKLCH`, Oklab as lightness, chroma and hue.
"""

from emoge.color.spaces.hsi import HSI, hsi_to_rgb, rgb_to_hsi
from emoge.color.spaces.hsv import HSV, hsv_to_rgb, rgb_to_hsv
from emoge.color.spaces.oklab import OKLAB, oklab_to_rgb, rgb_to_oklab
from emoge.color.spaces.oklch import OKLCH, oklch_to_rgb, rgb_to_oklch
from emoge.color.spaces.rgb import RGB, fit_into_rgb, to_float_rgb

__all__ = [
    "HSI",
    "HSV",
    "OKLAB",
    "OKLCH",
    "RGB",
    "fit_into_rgb",
    "hsi_to_rgb",
    "hsv_to_rgb",
    "oklab_to_rgb",
    "oklch_to_rgb",
    "rgb_to_hsi",
    "rgb_to_hsv",
    "rgb_to_oklab",
    "rgb_to_oklch",
    "to_float_rgb",
]
