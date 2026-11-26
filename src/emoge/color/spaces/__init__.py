"""Color spaces built as `emoge.color.Space`, one module each.

RGB with components in [0, 1] is the exchange format.

- `rgb`: `RGB`, `to_float_rgb`, `fit_into_rgb`, and `to_linear` and
  `from_linear` for linear light;
- `hsi`: `HSI`, hue, saturation and intensity;
- `hsv`: `HSV`, hue, saturation and value;
- `hsl`: `HSL`, hue, saturation and lightness;
- `oklab`: `OKLAB`, lightness and the axes `a` and `b`, as people see them;
- `oklch`: `OKLCH`, Oklab as lightness, chroma and hue;
- `xyz`: `XYZ`, the CIE 1931 space, and `WHITE`, the white of sRGB;
- `cielab`: `CIELAB`, the classic perceptual space;
- `cielch`: `CIELCH`, CIELAB as lightness, chroma and hue.
"""

from emoge.color.spaces.cielab import CIELAB, cielab_to_rgb, rgb_to_cielab
from emoge.color.spaces.cielch import CIELCH, cielch_to_rgb, rgb_to_cielch
from emoge.color.spaces.hsi import HSI, hsi_to_rgb, rgb_to_hsi
from emoge.color.spaces.hsl import HSL, hsl_to_rgb, rgb_to_hsl
from emoge.color.spaces.hsv import HSV, hsv_to_rgb, rgb_to_hsv
from emoge.color.spaces.oklab import OKLAB, oklab_to_rgb, rgb_to_oklab
from emoge.color.spaces.oklch import OKLCH, oklch_to_rgb, rgb_to_oklch
from emoge.color.spaces.rgb import (
    RGB,
    fit_into_rgb,
    from_linear,
    to_float_rgb,
    to_linear,
)
from emoge.color.spaces.xyz import WHITE, XYZ, rgb_to_xyz, xyz_to_rgb

__all__ = [
    "CIELAB",
    "CIELCH",
    "HSI",
    "HSL",
    "HSV",
    "OKLAB",
    "OKLCH",
    "RGB",
    "WHITE",
    "XYZ",
    "cielab_to_rgb",
    "cielch_to_rgb",
    "fit_into_rgb",
    "from_linear",
    "hsi_to_rgb",
    "hsl_to_rgb",
    "hsv_to_rgb",
    "oklab_to_rgb",
    "oklch_to_rgb",
    "rgb_to_cielab",
    "rgb_to_cielch",
    "rgb_to_hsi",
    "rgb_to_hsl",
    "rgb_to_hsv",
    "rgb_to_oklab",
    "rgb_to_oklch",
    "rgb_to_xyz",
    "to_float_rgb",
    "to_linear",
    "xyz_to_rgb",
]
