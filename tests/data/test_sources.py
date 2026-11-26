import json

import pytest
from PIL import Image

from emoge.data import folders, table


def test_a_folder_per_class(tmp_path):
    for path in ["calm/1.png", "calm/2.jpg", "fear/1.png"]:
        (tmp_path / path).parent.mkdir(exist_ok=True)
        Image.new("RGB", (8, 8), (90, 90, 90)).save(tmp_path / path)
    (tmp_path / "fear" / "notes.txt").write_text("not an image")
    images = folders(tmp_path, size=4)
    assert len(images) == 3
    assert images.classes == ["calm", "fear"]
    assert images.labels.tolist() == [0, 0, 1]
    assert images[2]["image"].shape == (4, 4, 3)
    with pytest.raises(ValueError, match="no folders"):
        folders(tmp_path / "calm")


@pytest.fixture
def photos(tmp_path):
    (tmp_path / "img").mkdir()
    for name in ["a", "b"]:
        Image.new("RGB", (8, 8), (200, 30, 30)).save(tmp_path / "img" / f"{name}.png")
    return tmp_path


def test_a_csv_table(photos):
    (photos / "train.csv").write_text(
        "path,emotion,valence\nimg/a.png,fear,0.2\nimg/b.png,awe,0.8\n"
    )
    images = table(photos / "train.csv", image="path", label="emotion")
    assert images.classes == ["awe", "fear"]
    sample = images[0]
    assert sample["label"] == 1
    assert sample["valence"] == "0.2"  # CSV values stay text
    assert sample["image"].shape == (8, 8, 3)


def test_json_tables(photos):
    rows = [{"image": "img/a.png", "valence": 0.2}, {"image": "img/b.png"}]
    (photos / "train.json").write_text(json.dumps(rows))
    (photos / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    for name in ["train.json", "train.jsonl"]:
        images = table(photos / name)
        assert len(images) == 2
        assert images[0]["valence"] == 0.2
    (photos / "train.txt").write_text("")
    with pytest.raises(ValueError, match="csv"):
        table(photos / "train.txt")
