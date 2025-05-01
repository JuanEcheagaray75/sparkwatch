import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer
from pyspark.sql import DataFrame

from ._utils import _get_bin_edges
from .binning import _doane_width, _freedman_diaconis_width, _sturges_width

_hist_bin_selector = {
    "doane": _doane_width,
    "sturges": _sturges_width,
    "fd": _freedman_diaconis_width,
}


def _histogram_numeric(data: DataFrame, col: str, edges: list[float]) -> pd.DataFrame:
    bucketizer = Bucketizer(splits=edges, inputCol=col, outputCol="bin")
    hist_count = (
        bucketizer.transform(data)
        .groupby("bin")
        .agg(F.count("*").alias("count"))
        .sort(F.asc("bin"))
        .toPandas()
    )
    hist_df = (
        pd.merge(
            pd.DataFrame(
                {
                    "bin": np.arange(len(edges) - 1),
                    "bin_min": edges[:-1],
                    "bin_max": edges[1:],
                }
            ),
            hist_count,
            how="left",
            on="bin",
        )
        .fillna(value=0)
        .drop(columns="bin")
    )
    return hist_df


def get_bin_edges(data: DataFrame, col: str, method: str = "doane") -> list[float]:
    if method not in _hist_bin_selector:
        raise ValueError(
            f"Select a valid method from {list(_hist_bin_selector.keys())}"
        )

    bin_width = _hist_bin_selector[method](data, col)
    edges = _get_bin_edges(data, col, bin_width)
    return edges


def histogram_numeric(
    data: DataFrame,
    col: str,
    method: str = "doane",
) -> pd.DataFrame:
    edges = get_bin_edges(data, col, method)
    hist_df = _histogram_numeric(data, col, edges)
    return hist_df


def histogram_categorical(data: DataFrame, col: str) -> pd.DataFrame:
    return (
        data.groupby(F.col(col).alias("category"))
        .agg(F.count("*").alias("count"))
        .toPandas()
    )


def get_proba_from_hist(histogram: pd.DataFrame) -> np.ndarray:
    counts = np.asarray(histogram["count"])
    return counts / np.sum(counts)
