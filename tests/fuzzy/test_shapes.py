import inspect

import pytest

from emoge.fuzzy import FuzzySet, membership, shapes

FUNCTIONS = [
    name
    for name, function in inspect.getmembers(membership, inspect.isfunction)
    if function.__module__ == membership.__name__ and not name.startswith("_")
]


@pytest.mark.parametrize("name", FUNCTIONS)
def test_every_membership_function_makes_its_set(name):
    parameters = list(inspect.signature(getattr(membership, name)).parameters)[1:]
    assert list(inspect.signature(getattr(shapes, name)).parameters) == parameters


def test_a_shape_is_the_set_of_its_function():
    assert shapes.triangle(10, 20, 30) == FuzzySet(membership.triangle, (10, 20, 30))
    assert repr(shapes.rising(0.25, 0.75)) == "rising(0.25, 0.75)"
