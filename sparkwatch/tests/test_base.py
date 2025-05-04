import pytest

from sparkwatch.base import (
    _is_categorical,
    _is_numerical,
)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("bigint", False),
        ("smallint", False),
        ("int", False),
        ("float", False),
        ("double", False),
        ("string", True),
        ("boolean", True),
    ],
)
def test_is_categorical(test_input, expected):
    assert _is_categorical(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("bigint", True),
        ("smallint", True),
        ("int", True),
        ("float", True),
        ("double", True),
        ("string", False),
        ("boolean", False),
    ],
)
def test_is_numerical(test_input, expected):
    assert _is_numerical(test_input) == expected
