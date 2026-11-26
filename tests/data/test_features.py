import numpy as np
from PIL import Image

from emoge.color import COLIBRI
from emoge.data import Dataset, each, features


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


def test_each_gives_results_in_order_a_few_at_a_time(tmp_path):
    for i in range(8):
        Image.fromarray(np.full((4, 4, 3), i * 30, dtype=np.uint8)).save(
            tmp_path / f"{i}.png"
        )
    images = Dataset([{"image": f"{i}.png"} for i in range(8)], tmp_path)
    called = []

    def brightness(image):
        called.append(1)
        return int(image[0, 0, 0])

    results = each(images, brightness, workers=1)
    assert next(results) == 0
    assert len(called) <= 3  # never the whole dataset at once
    assert list(results) == [i * 30 for i in range(1, 8)]
