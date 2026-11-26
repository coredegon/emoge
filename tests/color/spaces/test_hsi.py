import numpy as np
import pytest

from emoge.color.spaces import HSI, hsi_to_rgb, rgb_to_hsi

# Colors whose hue, saturation and intensity follow from the definition.
KNOWN = {
    "red": ([1, 0, 0], [0, 1, 1 / 3]),
    "green": ([0, 1, 0], [120, 1, 1 / 3]),
    "blue": ([0, 0, 1], [240, 1, 1 / 3]),
    "yellow": ([1, 1, 0], [60, 1, 2 / 3]),
    "gray": ([0.5, 0.5, 0.5], [0, 0, 0.5]),
    "black": ([0, 0, 0], [0, 0, 0]),
    "pink": ([1, 0.5, 0.5], [0, 0.25, 2 / 3]),
}


@pytest.mark.parametrize(("rgb", "hsi"), KNOWN.values(), ids=KNOWN.keys())
def test_hsi_follows_its_definition(rgb, hsi):
    assert rgb_to_hsi(np.array(rgb, dtype=float)) == pytest.approx(hsi, abs=1e-12)


def test_hsi_goes_back_to_rgb():
    rgb = np.random.default_rng(0).uniform(0, 1, (50, 40, 3))
    hsi = rgb_to_hsi(rgb)
    assert (hsi[..., 0] >= 0).all()
    assert (hsi[..., 0] < 360).all()
    assert hsi_to_rgb(hsi) == pytest.approx(rgb, abs=1e-9)


def test_the_hsi_space():
    assert [c.name for c in HSI.channels] == ["hue", "saturation", "intensity"]
    assert HSI.channel("hue").periodic
    with pytest.raises(KeyError, match="no channel 'value'"):
        HSI.channel("value")
