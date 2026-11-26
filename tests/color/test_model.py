import numpy as np
import pytest

from emoge.color import HSI, HSV, ColorModel
from emoge.fuzzy import Channel, norms, partition

HUE = partition(HSI.channel("hue"), {"red": 0, "green": 120, "blue": 240})
SATURATION = partition(HSI.channel("saturation"), {"gray": 0, "pale": 0.5, "vivid": 1})
INTENSITY = partition(HSI.channel("intensity"), {"dark": 0, "light": 1})
FAMILIES = {
    "tint": [HUE, SATURATION.select("pale", "vivid")],
    "gray": [INTENSITY, SATURATION.select("gray")],
}
MODEL = ColorModel("toy", HSI, FAMILIES)
IMAGE = np.random.default_rng(0).integers(0, 256, (30, 40, 3), dtype=np.uint8)


def test_a_family_names_every_combination_of_terms():
    assert MODEL.colors == (
        "red/pale",
        "red/vivid",
        "green/pale",
        "green/vivid",
        "blue/pale",
        "blue/vivid",
        "dark",  # one saturation: left out of the name
        "light",
    )
    alone = ColorModel("m", HSI, {"gray": [SATURATION.select("gray")]})
    assert alone.colors == ("gray",)
    assert repr(MODEL) == "ColorModel('toy', HSI: 6 tint, 2 gray)"


def test_the_degrees_of_a_pixel_sum_to_one():
    red = dict(zip(MODEL.colors, MODEL.degrees([255, 0, 0]).tolist(), strict=True))
    assert red["red/vivid"] == 1.0
    assert sum(red.values()) == 1.0
    pink = dict(zip(MODEL.colors, MODEL.degrees([1.0, 0.5, 0.5]).tolist(), strict=True))
    assert pink["red/pale"] == pytest.approx(0.5)  # saturation 0.25
    assert pink["light"] == pytest.approx(0.5 * 2 / 3)
    degrees = MODEL.degrees(IMAGE)
    assert degrees.shape == (8, 30, 40)
    assert degrees.sum(axis=0) == pytest.approx(1.0)
    assert MODEL.degrees(IMAGE / 255.0) == pytest.approx(degrees)


def test_classify_and_label_name_the_strongest_color():
    assert MODEL.label([255, 0, 0]) == "red/vivid"
    assert MODEL.label([255, 0, 0]).shape == ()
    assert MODEL.label([[255, 0, 0], [20, 20, 20]]).tolist() == ["red/vivid", "dark"]
    assert MODEL.colors[MODEL.classify([0, 255, 0])] == "green/vivid"
    best = MODEL.degrees(IMAGE).argmax(axis=0)
    assert MODEL.classify(IMAGE).tolist() == best.tolist()


def test_strongest_gives_the_color_and_its_degree():
    color, degree = MODEL.strongest(IMAGE)
    assert color.tolist() == MODEL.classify(IMAGE).tolist()
    assert degree.tolist() == MODEL.degrees(IMAGE).max(axis=0).tolist()


def test_another_t_norm_still_finds_the_strongest_color():
    by_minimum = ColorModel("toy", HSI, FAMILIES, and_norm=norms.minimum)
    degrees = by_minimum.degrees(IMAGE)
    best = by_minimum.classify(IMAGE)
    strongest = np.take_along_axis(degrees, best[None], axis=0)[0]
    assert strongest.tolist() == degrees.max(axis=0).tolist()


def test_a_model_can_live_in_another_space():
    value = partition(HSV.channel("value"), {"dark": 0, "bright": 1})
    model = ColorModel("bright", HSV, {"v": [value]})
    assert model.label([[255, 0, 0], [10, 10, 10]]).tolist() == ["bright", "dark"]


def test_a_pixel_of_no_color_gets_the_index_past_the_colors():
    vivid = ColorModel("vivid", HSI, {"v": [SATURATION.select("vivid")]})
    assert vivid.classify([[255, 0, 0], [90, 90, 90]]).tolist() == [0, 1]
    assert vivid.strongest([90, 90, 90])[1] == 0.0
    assert vivid.label([[255, 0, 0], [90, 90, 90]]).tolist() == ["v", ""]


def test_a_palette_is_the_mean_degree_of_the_pixels():
    weights = MODEL.palette(IMAGE)
    assert weights == pytest.approx(MODEL.degrees(IMAGE).mean(axis=(1, 2)))
    large = np.random.default_rng(1).integers(0, 256, (300, 300, 3), dtype=np.uint8)
    assert MODEL.palette(large) == pytest.approx(MODEL.degrees(large).mean(axis=(1, 2)))
    with pytest.raises(ValueError, match="without pixels"):
        MODEL.palette(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="3 components"):
        MODEL.palette(np.zeros((4, 4)))


def test_weights_give_the_palette_of_a_region():
    degrees = MODEL.degrees(IMAGE)
    left = np.zeros((30, 40), dtype=bool)
    left[:, :20] = True
    soft = np.random.default_rng(2).random((30, 40))
    assert MODEL.palette(IMAGE, left) == pytest.approx(degrees[..., :20].mean((1, 2)))
    both = MODEL.palette(IMAGE, np.stack([left, soft]))
    assert both.shape == (2, 8)
    assert both[1] == pytest.approx((degrees * soft).sum((1, 2)) / soft.sum())
    large = np.random.default_rng(3).integers(0, 256, (300, 300, 3), dtype=np.uint8)
    mask = np.random.default_rng(4).random((300, 300))
    expected = (MODEL.degrees(large) * mask).sum((1, 2)) / mask.sum()
    assert MODEL.palette(large, mask) == pytest.approx(expected)
    with pytest.raises(ValueError, match="shape of the pixels"):
        MODEL.palette(IMAGE, soft[:10])
    with pytest.raises(ValueError, match="above 0"):
        MODEL.palette(IMAGE, np.zeros((30, 40)))


def test_prototypes_are_typical_colors():
    typical = dict(zip(MODEL.colors, MODEL.prototypes().tolist(), strict=True))
    assert typical["dark"] == pytest.approx([0, 0, 0], abs=1e-3)
    assert typical["light"] == pytest.approx([1, 1, 1], abs=1e-3)
    assert typical["red/vivid"] == pytest.approx([1, 0, 0], abs=1e-3)


def test_a_model_checks_its_families():
    value = partition(Channel("value", 0, 1), {"a": 0, "b": 1})
    with pytest.raises(ValueError, match="not channels of HSI"):
        ColorModel("m", HSI, {"f": [value]})
    with pytest.raises(ValueError, match="different channels"):
        ColorModel("m", HSI, {"twice": [HUE, HUE]})
    with pytest.raises(ValueError, match="different channels"):
        ColorModel("m", HSI, {"empty": []})
    with pytest.raises(ValueError, match="colors that differ"):
        ColorModel("m", HSI, {"a": [INTENSITY], "b": [INTENSITY]})
