import numpy as np
import pytest

torch = pytest.importorskip("torch")

from emoge.color import COLIBRI  # noqa: E402
from emoge.nn import cooccurrence, palette  # noqa: E402

IMAGES = np.random.default_rng(0).integers(0, 256, (2, 12, 16, 3), dtype=np.uint8)
DEGREES = torch.as_tensor(np.moveaxis(COLIBRI.degrees(IMAGES), 0, -1))


def test_palette_matches_the_color_model():
    assert palette(DEGREES).shape == (2, 86)
    assert torch.allclose(
        palette(DEGREES)[1], torch.as_tensor(COLIBRI.palette(IMAGES[1]))
    )
    masks = np.random.default_rng(1).random((2, 3, 12, 16))
    weighed = palette(DEGREES, torch.as_tensor(masks))
    assert weighed.shape == (2, 3, 86)
    expected = COLIBRI.palette(IMAGES[0], masks[0])
    assert torch.allclose(weighed[0], torch.as_tensor(expected))
    one = palette(DEGREES, torch.as_tensor(masks[:, 0]))
    assert torch.allclose(one, weighed[:, 0])


def test_cooccurrence_matches_the_color_model():
    meets = cooccurrence(DEGREES, distance=2)
    assert meets.shape == (2, 86, 86)
    expected = COLIBRI.cooccurrence(IMAGES[1], distance=2)
    assert torch.allclose(meets[1], torch.as_tensor(expected))
    with pytest.raises(ValueError, match="no neighbours"):
        cooccurrence(DEGREES[:, :1, :1])
