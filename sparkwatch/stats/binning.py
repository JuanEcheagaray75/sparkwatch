import math

import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def _data_range(data: DataFrame, col: str) -> tuple[float, float]:
    val_range = (
        data.agg(F.min(col).alias("min"), F.max(col).alias("max")).collect()[0].asDict()
    )
    min_val = val_range["min"]
    max_val = val_range["max"]
    return min_val, max_val


def _doane_width(data: DataFrame, col: str) -> float:
    min_val, max_val = _data_range(data, col)
    n_bins = (
        data.agg(F.count("*").alias("n"), F.kurtosis(col).alias("kurt"))
        .withColumn(
            "sigma",
            F.sqrt((6 * (F.col("n") - 2)) / ((F.col("n") + 1) * (F.col("n") + 3))),
        )
        .withColumn(
            "n_bins",
            F.ceil(1 + F.log2("n") + F.log2(1 + F.abs("kurt") / F.col("sigma"))),
        )
        .collect()[0]["n_bins"]
    )
    bin_width = (max_val - min_val) / n_bins
    return bin_width


def _freedman_diaconis_width(data: DataFrame, col: str) -> float:
    bin_width = (
        data.agg(
            F.count("*").alias("n"),
            F.percentile_approx(col, 0.25).alias("q1"),
            F.percentile_approx(col, 0.75).alias("q3"),
        )
        .withColumn("iqr", F.col("q3") - F.col("q1"))
        .withColumn("bin_width", 2 * F.col("iqr") / F.cbrt("n"))
        .collect()[0]["bin_width"]
    )
    return bin_width


def _sturges_width(data: DataFrame, col: str) -> float:
    n = data.count()
    n_bins = math.floor(math.log(n, 2) + 1)
    min_val, max_val = _data_range(data, col)
    bin_width = (max_val - min_val) / n_bins

    return bin_width