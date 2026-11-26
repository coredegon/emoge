import numpy as np
import pytest

from emoge.fuzzy_color.math import membership as mf

# Every function with values that follow from its definition: (function, x, degrees).
SHAPES = {
    "triangle": (
        lambda x: mf.triangle(x, 0, 49, 98),
        [-1, 24.5, 49, 73.5, 99],
        [0, 0.5, 1, 0.5, 0],
    ),
    "trapezoid": (
        lambda x: mf.trapezoid(x, 0, 2, 5, 9),
        [0, 1, 2, 5, 7, 9],
        [0, 0.5, 1, 1, 0.5, 0],
    ),
    "vertical edge": (
        lambda x: mf.trapezoid(x, 0, 0, 2, 4),
        [-1e-300, 0, 2, 3],
        [0, 1, 1, 0.5],
    ),
    "rising": (lambda x: mf.rising(x, 2, 4), [1, 2, 3, 4, 5], [0, 0, 0.5, 1, 1]),
    "falling": (lambda x: mf.falling(x, 2, 4), [1, 2, 3, 4, 5], [1, 1, 0.5, 0, 0]),
    "rectangle": (
        lambda x: mf.rectangle(x, 1, 3),
        [0.999, 1, 2, 3, 3.001],
        [0, 1, 1, 1, 0],
    ),
    "singleton": (lambda x: mf.singleton(x, 2), [1.999, 2, 2.001], [0, 1, 0]),
    "piecewise_linear": (
        lambda x: mf.piecewise_linear(x, ((0, 0), (2, 1), (4, 0.5))),
        [-1, 1, 2, 3, 9],
        [0, 0.5, 1, 0.75, 0.5],
    ),
    "gaussian": (
        lambda x: mf.gaussian(x, 5, 2),
        [3, 5, 7],
        [np.exp(-0.5), 1, np.exp(-0.5)],
    ),
    "two_sided_gaussian": (
        lambda x: mf.two_sided_gaussian(x, 3, 1, 6, 2),
        [2, 3, 4.5, 6, 8],
        [np.exp(-0.5), 1, 1, 1, np.exp(-0.5)],
    ),
    "generalized_bell": (
        lambda x: mf.generalized_bell(x, 5, 2, 3),
        [3, 5, 7],
        [0.5, 1, 0.5],
    ),
    "sigmoid": (lambda x: mf.sigmoid(x, 5, 2), [-1e4, 5, 1e4], [0, 0.5, 1]),
    "sigmoid_difference": (
        lambda x: mf.sigmoid_difference(x, 2, 50, 7, 50),
        [0, 2, 4.5, 7, 9],
        [0, 0.5, 1, 0.5, 0],
    ),
    "sigmoid_product": (
        lambda x: mf.sigmoid_product(x, 2, 50, 7, -50),
        [0, 2, 4.5, 7, 9],
        [0, 0.5, 1, 0.5, 0],
    ),
    "s_shape": (
        lambda x: mf.s_shape(x, 1, 8),
        [0, 2.75, 4.5, 6.25, 9],
        [0, 0.125, 0.5, 0.875, 1],
    ),
    "z_shape": (
        lambda x: mf.z_shape(x, 1, 8),
        [0, 2.75, 4.5, 6.25, 9],
        [1, 0.875, 0.5, 0.125, 0],
    ),
    "pi_shape": (
        lambda x: mf.pi_shape(x, 1, 4, 5, 10),
        [1, 2.5, 4, 7.5, 10],
        [0, 0.5, 1, 0.5, 0],
    ),
}


@pytest.mark.parametrize(("shape", "x", "degrees"), SHAPES.values(), ids=SHAPES.keys())
def test_a_membership_function_follows_its_definition(shape, x, degrees):
    assert shape(np.array(x, dtype=float)) == pytest.approx(degrees, abs=1e-12)


@pytest.mark.parametrize(("shape", "x", "degrees"), SHAPES.values(), ids=SHAPES.keys())
def test_a_membership_function_keeps_the_shape_and_stays_in_0_1(shape, x, degrees):
    values = np.random.default_rng(0).uniform(-2, 12, (40, 50))
    result = shape(values)
    assert result.shape == values.shape
    assert result.min() >= 0.0
    assert result.max() <= 1.0


BAD = {
    "descending": lambda: mf.triangle(np.zeros(1), 3, 2, 1),
    "infinite": lambda: mf.trapezoid(np.zeros(1), 0, 1, np.inf, 3),
    "flat gaussian": lambda: mf.gaussian(np.zeros(1), 0, 0),
    "negative slope": lambda: mf.generalized_bell(np.zeros(1), 0, 1, -1),
    "vertical spline": lambda: mf.s_shape(np.zeros(1), 2, 2),
    "repeated point": lambda: mf.piecewise_linear(np.zeros(1), ((0, 0), (0, 1))),
    "degree above 1": lambda: mf.piecewise_linear(np.zeros(1), ((0, 0), (1, 1.5))),
}


@pytest.mark.parametrize("build", BAD.values(), ids=BAD.keys())
def test_bad_parameters_are_rejected(build):
    with pytest.raises(ValueError, match=r"must|needs"):
        build()
