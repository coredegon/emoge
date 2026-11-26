import numpy as np
from PIL import Image

from emoge.color import COLIBRI
from emoge.data import Dataset, features


def test_features_of_every_image_in_order(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        pixels = rng.integers(0, 256, (10, 12, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(tmp_path / f"{i}.png")
        rows.append({"image": f"{i}.png"})
    images = Dataset(rows, tmp_path)
    palettes = features(images, COLIBRI.palette)
    assert palettes.shape == (6, 86)
    expected = np.stack([COLIBRI.palette(sample["image"]) for sample in images])
    assert palettes.tolist() == expected.tolist()
    assert features(images, COLIBRI.palette, workers=1).tolist() == expected.tolist()
