import numpy as np
import pytest

from emoge.regions import masks


def test_a_mask_for_every_label():
    regions = masks([[0, 1, 1], [2, -1, 1]])
    assert regions.shape == (3, 2, 3)
    assert regions.astype(int).tolist() == [
        [[1, 0, 0], [0, 0, 0]],
        [[0, 1, 1], [0, 0, 1]],
        [[0, 0, 0], [1, 0, 0]],
    ]  # a negative label is in no region


def test_labels_are_whole_numbers():
    assert masks(np.full((2, 2), -1)).shape == (0, 2, 2)
    with pytest.raises(TypeError, match="whole numbers"):
        masks([[0.5]])
