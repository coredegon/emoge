import numpy as np
import pytest

torch = pytest.importorskip("torch")

from emoge.color import COLIBRI, HSI, ColorModel  # noqa: E402
from emoge.fuzzy import Channel, FuzzySet, Variable, norms, partition  # noqa: E402
from emoge.fuzzy.math import membership  # noqa: E402
from emoge.fuzzy.partitions import Partition  # noqa: E402
from emoge.nn import FuzzyColors  # noqa: E402

IMAGE = np.random.default_rng(0).integers(0, 256, (2, 12, 16, 3), dtype=np.uint8)
VALUES = torch.as_tensor(HSI.from_rgb(IMAGE / 255), dtype=torch.float32)


def as_colors_last(degrees):
    return torch.as_tensor(np.moveaxis(degrees, 0, -1), dtype=torch.float32)


def test_before_learning_the_layer_is_its_model():
    layer = FuzzyColors(COLIBRI)
    degrees = layer(VALUES)
    assert degrees.shape == (2, 12, 16, 86)
    assert torch.allclose(degrees, as_colors_last(COLIBRI.degrees(IMAGE)), atol=1e-4)
    assert layer.colors == COLIBRI.colors


def test_a_partition_shared_by_families_learns_once():
    # Where red starts and the lengths of hue, then saturation for both
    # families, shade, and the intensity of the grays.
    shapes = [tuple(p.shape) for p in FuzzyColors(COLIBRI).parameters()]
    assert shapes == [(), (18,), (9,), (7,), (11,)]


def test_learned_cores_keep_a_partition():
    layer = FuzzyColors(COLIBRI)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.05)
    for _ in range(20):
        optimizer.zero_grad()
        (-layer(VALUES)[..., 0].mean()).backward()  # more of the first color
        optimizer.step()
    degrees = layer(VALUES).detach()
    assert torch.allclose(degrees.sum(dim=-1), torch.ones(2, 12, 16), atol=1e-5)
    learned = layer.to_model()
    assert learned.colors == COLIBRI.colors
    hue = learned.families["chromatic"][0]
    assert isinstance(hue, Partition)
    assert hue.cores["red"] != (350, 5)
    assert torch.allclose(degrees, as_colors_last(learned.degrees(IMAGE)), atol=1e-4)


def test_only_partitions_joined_with_the_product_learn():
    level = partition(Channel("intensity", 0, 1), {"dark": 0, "light": 1})
    with pytest.raises(ValueError, match="product"):
        FuzzyColors(ColorModel("m", HSI, {"f": [level]}, and_norm=norms.minimum))
    shaped = Variable(level.channel, {"dark": FuzzySet(membership.falling, (0, 1))})
    with pytest.raises(TypeError, match="partition"):
        FuzzyColors(ColorModel("m", HSI, {"f": [shaped]}))
