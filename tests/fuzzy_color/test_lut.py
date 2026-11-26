import numpy as np
import pytest

from emoge.fuzzy_color import RGB, ColorModel, lookup_table, partition
from emoge.fuzzy_color.models import COLIBRI

RED = partition(RGB.channel("red"), {"dark": 0, "bright": 1})
BRIGHT = ColorModel("bright", RGB, {"r": [RED.select("bright")]})  # none at red 0
IMAGE = np.random.default_rng(0).integers(0, 256, (40, 50, 3), dtype=np.uint8)


def look_up(table, image):
    return table[image[..., 0], image[..., 1], image[..., 2]]


def test_the_table_answers_as_classify_in_a_byte():
    table = lookup_table(COLIBRI)
    assert table.shape == (256, 256, 256)
    assert table.dtype == np.uint8
    assert look_up(table, IMAGE).tolist() == COLIBRI.classify(IMAGE).tolist()


def test_the_table_can_keep_the_degree_of_each_color():
    table = lookup_table(BRIGHT, degree=True)
    assert table.dtype.names == ("color", "degree")
    looked_up = look_up(table, IMAGE)
    assert looked_up["color"].tolist() == BRIGHT.classify(IMAGE).tolist()
    degree = BRIGHT.degrees(IMAGE).max(axis=0)
    assert looked_up["degree"] == pytest.approx(degree, abs=1e-3)
    assert table[0, 9, 9].tolist() == (1, 0.0)  # no color: the index past the colors


def test_a_table_is_saved_once_and_then_loaded(tmp_path):
    table = lookup_table(BRIGHT, tmp_path / "bright")
    saved = tmp_path / "bright.npy"
    written = saved.stat().st_mtime_ns
    assert np.array_equal(lookup_table(BRIGHT, saved), table)
    assert saved.stat().st_mtime_ns == written


def test_a_file_of_another_model_or_kind_is_refused(tmp_path):
    path = tmp_path / "bright.npy"
    lookup_table(BRIGHT, path)
    with pytest.raises(ValueError, match="another model or kind"):
        lookup_table(ColorModel("red", RGB, {"r": [RED]}), path)
    with pytest.raises(ValueError, match="another model or kind"):
        lookup_table(BRIGHT, path, degree=True)
