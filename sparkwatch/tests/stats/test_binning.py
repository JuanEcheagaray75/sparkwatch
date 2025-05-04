import hypothesis.strategies as st
import numpy as np
from numpy.lib._histograms_impl import _hist_bin_selectors  # type: ignore
import pytest
from hypothesis import given, settings
from pyspark.sql import SparkSession
from collections.abc import Callable

from sparkwatch.stats.binning import (
    _data_range,
    _doane_width,
    _freedman_diaconis_width,
    _sturges_width,
    _ptp,
    InsufficientDataError,
)

unique_float_list = st.lists(
    st.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False, width=32),
    unique=True,
    min_size=10,
    max_size=100,
)
small_float_list = st.lists(
    st.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False, width=32),
    unique=True,
    min_size=1,
    max_size=2,
)
MAX_EXAMPLES = 10


@pytest.mark.parametrize(
    "method,func",
    [
        ("doane", _doane_width),
        ("sturges", _sturges_width),
        ("fd", _freedman_diaconis_width),
    ],
)
@given(data=unique_float_list)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
def test_bin_width_matches_numpy(
    spark: SparkSession, data: list[float], method: str, func: Callable
):
    arr = np.asarray(data).flatten()
    df = spark.createDataFrame(arr, ["value"])
    calculated = func(df, col="value")
    expected = float(_hist_bin_selectors[method](arr, None))

    np.testing.assert_allclose(calculated, expected)


@pytest.mark.parametrize(
    "method,func",
    [
        ("doane", _doane_width),
        ("sturges", _sturges_width),
        ("fd", _freedman_diaconis_width),
    ],
)
@given(data=small_float_list)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
def test_bin_width_insufficient_data(
    spark: SparkSession, data: list[float], method: str, func: Callable
):
    arr = np.asarray(data).flatten()
    df = spark.createDataFrame(arr, ["value"])
    with pytest.raises(InsufficientDataError):
        func(df, col="value")


@given(data=unique_float_list)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
def test_data_range(spark: SparkSession, data: list[float]):
    arr = np.asarray(data).flatten()
    df = spark.createDataFrame(arr, ["value"])

    maxx, minx = _data_range(df, "value")

    assert np.allclose([maxx, minx], [np.max(arr), np.min(arr)])


@given(data=unique_float_list)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
def test_ptp(spark: SparkSession, data: list[float]):
    arr = np.asarray(data).flatten()
    df = spark.createDataFrame(arr, ["value"])
    actual = _ptp(df, "value")
    expected = np.lib._histograms_impl._ptp(arr)  # type: ignore

    np.testing.assert_allclose(actual, expected)
