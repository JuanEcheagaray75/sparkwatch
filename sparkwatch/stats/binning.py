import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame


class InsufficientDataError(Exception):
    def __init__(self, n: int, required: int = 3):
        message = f"Only {n} samples provided, but at least {required} are required."
        super().__init__(message)


def _data_range(data: DataFrame, col: str) -> tuple[float, float]:
    return data.agg(F.max(col), F.min(col)).first()  # type: ignore


def _ptp(data: DataFrame, col: str) -> float:
    return np.subtract(*_data_range(data, col))


def _is_sufficient_data(n: int) -> bool:
    return n > 2


def _doane_width(data: DataFrame, col: str) -> float:
    n, mean, std = data.agg(
        F.count("*").alias("n"),
        F.mean(col).alias("mean"),
        F.stddev_pop(col).alias("std"),
    ).first()  # type: ignore

    if not _is_sufficient_data(n):
        raise InsufficientDataError(n)

    fishers_skewness = (
        data.select((F.pow((F.col(col) - mean) / std, 3).alias("z3")))
        .agg(F.mean("z3"))
        .first()[0]  # type: ignore
    )  # type: ignore
    sigma_skew = np.sqrt(6.0 * (n - 2) / ((n + 1.0) * (n + 3)))
    n_bins = 1.0 + np.log2(n) + np.log2(1.0 + np.abs(fishers_skewness) / sigma_skew)
    bin_width = _ptp(data, col) / n_bins
    return bin_width


def _freedman_diaconis_width(data: DataFrame, col: str) -> float:
    n, q1, q3 = data.agg(
        F.count("*"),
        F.percentile(col, 0.25),
        F.percentile(col, 0.75),
    ).first()  # type: ignore

    if not _is_sufficient_data(n):
        raise InsufficientDataError(n)

    iqr = q3 - q1
    bin_width = 2 * iqr / np.cbrt(n)
    return bin_width


def _sturges_width(data: DataFrame, col: str) -> float:
    n = data.count()

    if not _is_sufficient_data(n):
        raise InsufficientDataError(n)

    n_bins = np.log2(n) + 1
    bin_width = _ptp(data, col) / n_bins

    return bin_width
