import numpy as np
import pytest

from emoge.regions import superpixels

# Red beside blue, with a little noise: every superpixel keeps to one color.
FLAG = np.zeros((40, 60, 3), dtype=np.uint8)
FLAG[:, :30], FLAG[:, 30:] = [200, 30, 30], [30, 60, 200]
FLAG = np.clip(FLAG + np.random.default_rng(0).integers(-8, 8, FLAG.shape), 0, 255)
FLAG = FLAG.astype(np.uint8)


def test_every_pixel_is_in_one_crisp_superpixel():
    regions = superpixels(FLAG, 24)
    assert regions.dtype == bool
    assert regions.shape[1:] == (40, 60)
    assert 12 <= len(regions) <= 24
    assert (regions.sum(axis=0) == 1).all()


def test_superpixels_follow_the_colors():
    for region in superpixels(FLAG, 24):
        columns = np.nonzero(region.any(axis=0))[0]
        assert columns.max() < 30 or columns.min() >= 30


def pieces(region):
    """How many connected pieces a mask is in, side by side or one above the other."""
    left, count = set(zip(*np.nonzero(region), strict=True)), 0
    while left:
        count += 1
        frontier = [left.pop()]
        while frontier:
            y, x = frontier.pop()
            for near in [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]:
                if near in left:
                    left.remove(near)
                    frontier.append(near)
    return count


def test_a_crisp_superpixel_keeps_together():
    noisy = np.random.default_rng(1).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    regions = superpixels(noisy, 12)
    assert (regions.sum(axis=0) == 1).all()
    assert max(pieces(region) for region in regions) <= 2
    assert sum(pieces(region) == 1 for region in regions) >= len(regions) - 2


def test_soft_superpixels_are_degrees_that_sum_to_1():
    regions = superpixels(FLAG, 24, soft=True)
    assert regions.dtype == np.float64
    assert regions.sum(axis=0) == pytest.approx(1.0)
    assert (regions > 0).sum(axis=0).max() <= 9  # only the regions around a pixel
    assert regions.max(axis=0).mean() > 0.5


def test_one_superpixel_is_the_whole_image():
    assert superpixels(FLAG, 1).all()
    with pytest.raises(ValueError, match="at least 1"):
        superpixels(FLAG, 0)


def test_what_superpixels_refuse():
    with pytest.raises(ValueError, match="at least 1"):
        superpixels(FLAG, 0)
    with pytest.raises(ValueError, match="compactness"):
        superpixels(FLAG, 24, compactness=0)
