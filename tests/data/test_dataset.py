import numpy as np
import pytest
from PIL import Image

from emoge.data import Dataset, load_image

ROWS = [
    {"image": "a.png", "emotion": "fear", "valence": 0.2},
    {"image": "b.png", "emotion": "awe", "valence": 0.8},
    {"image": "c.png", "emotion": "fear", "valence": 0.1},
]


@pytest.fixture
def photos(tmp_path):
    for name, rgb in [("a", (20, 20, 30)), ("b", (60, 90, 200)), ("c", (10, 10, 10))]:
        Image.new("RGB", (40, 30), rgb).save(tmp_path / f"{name}.png")
    return tmp_path


def test_an_image_loads_as_rgb_bytes(tmp_path):
    Image.new("L", (40, 30), 128).save(tmp_path / "gray.png")
    image = load_image(tmp_path / "gray.png")
    assert image.shape == (30, 40, 3)
    assert image.dtype == np.uint8
    assert (image == 128).all()
    assert image.flags.writeable  # for augmentation in place, and for torch


def test_size_shrinks_the_shorter_side_and_cuts_out_the_middle(tmp_path):
    flag = np.zeros((40, 80, 3), dtype=np.uint8)
    flag[:, :40], flag[:, 40:] = [255, 0, 0], [0, 0, 255]
    Image.fromarray(flag).save(tmp_path / "flag.png")
    image = load_image(tmp_path / "flag.png", size=20)
    # 40 x 20 after shrinking; the middle 20 columns are half red, half blue.
    assert image.shape == (20, 20, 3)
    assert (image[:, :10] == [255, 0, 0]).all()
    assert (image[:, 10:] == [0, 0, 255]).all()


def test_a_sample_is_its_image_and_its_fields(photos):
    images = Dataset(ROWS, photos, label="emotion", size=16)
    assert len(images) == 3
    assert images.classes == ["awe", "fear"]
    assert images.labels.tolist() == [1, 0, 1]
    sample = images[1]
    assert sample["image"].shape == (16, 16, 3)
    assert (sample["image"] == [60, 90, 200]).all()
    assert sample["label"] == 0
    assert sample["valence"] == 0.8
    assert sample["path"] == str(photos / "b.png")
    assert "emotion" not in sample
    assert [s["label"] for s in images] == [1, 0, 1]
    assert repr(images) == "Dataset(3 images, 2 classes)"


def test_a_dataset_checks_its_rows(photos):
    with pytest.raises(KeyError, match="'image'"):
        Dataset([{"path": "a.png"}], photos)
    with pytest.raises(ValueError, match="no labels"):
        _ = Dataset(ROWS, photos).labels


def test_a_dataloader_takes_a_dataset_as_it_is(photos):
    torch = pytest.importorskip("torch")
    images = Dataset(ROWS, photos, label="emotion", size=16)
    batch = next(iter(torch.utils.data.DataLoader(images, batch_size=3)))
    assert batch["image"].shape == (3, 16, 16, 3)
    assert batch["label"].tolist() == [1, 0, 1]
