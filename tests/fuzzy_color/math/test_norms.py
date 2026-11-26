import numpy as np
import pytest

from emoge.fuzzy_color.math import norms

T_NORMS = [
    norms.minimum,
    norms.product,
    norms.bounded_difference,
    norms.drastic_product,
    norms.nilpotent_minimum,
    norms.hamacher_product,
    norms.einstein_product,
]
S_NORMS = [
    norms.maximum,
    norms.probabilistic_sum,
    norms.bounded_sum,
    norms.drastic_sum,
    norms.nilpotent_maximum,
    norms.hamacher_sum,
    norms.einstein_sum,
]
GRID = np.linspace(0, 1, 11)
A, B = GRID[:, None], GRID[None, :]


def test_known_values():
    a, b = np.array([0.2, 0.9]), np.array([0.5, 0.6])
    assert norms.product(a, b) == pytest.approx([0.1, 0.54])
    assert norms.probabilistic_sum(a, b) == pytest.approx([0.6, 0.96])
    assert norms.bounded_sum(a, b) == pytest.approx([0.7, 1.0])
    assert norms.bounded_difference(a, b) == pytest.approx([0.0, 0.5])
    assert norms.nilpotent_minimum(a, b) == pytest.approx([0.0, 0.6])


@pytest.mark.parametrize("t_norm", T_NORMS, ids=lambda norm: norm.__name__)
def test_t_norms_keep_their_laws(t_norm):
    joined = t_norm(A, B)
    assert joined == pytest.approx(t_norm(B, A))  # commutative
    assert t_norm(GRID, np.ones(11)) == pytest.approx(GRID)  # 1 is neutral
    assert joined.min() >= 0.0
    assert (joined <= norms.minimum(A, B) + 1e-12).all()  # the minimum is the largest


@pytest.mark.parametrize("s_norm", S_NORMS, ids=lambda norm: norm.__name__)
def test_s_norms_keep_their_laws(s_norm):
    joined = s_norm(A, B)
    assert joined == pytest.approx(s_norm(B, A))  # commutative
    assert s_norm(GRID, np.zeros(11)) == pytest.approx(GRID)  # 0 is neutral
    assert joined.max() <= 1.0 + 1e-12
    assert (joined >= norms.maximum(A, B) - 1e-12).all()  # the maximum is the smallest
