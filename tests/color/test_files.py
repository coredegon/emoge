import json

import numpy as np
import pytest

from emoge.color import CIELCH, COLIBRI, HSI, HSL, XYZ, ColorModel, Space, load, save
from emoge.fuzzy import Channel, FuzzySet, Variable, norms, partition
from emoge.fuzzy.math import membership

PIXELS = np.random.default_rng(0).integers(0, 256, (500, 3), dtype=np.uint8)


def test_a_saved_model_loads_as_it_was(tmp_path):
    save(COLIBRI, tmp_path / "colibri.json")
    loaded = load(tmp_path / "colibri.json")
    assert loaded.colors == COLIBRI.colors
    assert loaded.space is HSI
    assert loaded.degrees(PIXELS).tolist() == COLIBRI.degrees(PIXELS).tolist()


def test_the_file_reads_as_the_cores(tmp_path):
    save(COLIBRI, tmp_path / "colibri.json")
    saved = json.loads((tmp_path / "colibri.json").read_text(encoding="utf-8"))
    hue, saturation, _ = saved["families"]["chromatic"]
    assert hue["cores"]["red"] == [350, 5]
    assert hue["cores"]["yellow"] == 45
    assert "terms" not in hue  # all of them
    assert saturation["terms"] == ["low", "medium", "high"]


def test_the_norm_is_saved_too(tmp_path):
    level = partition(HSI.channel("intensity"), {"dark": 0, "light": 1})
    model = ColorModel("m", HSI, {"f": [level]}, and_norm=norms.minimum)
    save(model, tmp_path / "m.json")
    assert load(tmp_path / "m.json").and_norm is norms.minimum


def test_only_partitions_on_spaces_of_emoge_color_are_saved(tmp_path):
    shaped = Variable(
        HSI.channel("intensity"), {"dark": FuzzySet(membership.falling, (0, 1))}
    )
    with pytest.raises(TypeError, match="partition"):
        save(ColorModel("m", HSI, {"f": [shaped]}), tmp_path / "m.json")
    gray = Space(
        "gray", (Channel("level", 0, 1),), lambda rgb: rgb[..., :1], lambda v: v
    )
    level = partition(gray.channel("level"), {"dark": 0, "light": 1})
    with pytest.raises(ValueError, match=r"spaces of emoge\.color"):
        save(ColorModel("m", gray, {"f": [level]}), tmp_path / "m.json")


@pytest.mark.parametrize("space", [CIELCH, HSL, XYZ])
def test_models_on_every_space_are_saved(tmp_path, space):
    lightness = space.channels[0]
    middle = (lightness.low + lightness.high) / 2
    level = partition(lightness, {"dark": lightness.low, "light": middle})
    model = ColorModel("levels", space, {"level": [level]})
    save(model, tmp_path / "levels.json")
    loaded = load(tmp_path / "levels.json")
    assert loaded.space is space
    assert loaded.degrees(PIXELS).tolist() == model.degrees(PIXELS).tolist()
