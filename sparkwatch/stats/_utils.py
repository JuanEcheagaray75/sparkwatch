import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def _data_range(data: DataFrame, col: str) -> tuple[float, float]:
    val_range = (
        data.agg(F.min(col).alias("min"), F.max(col).alias("max")).collect()[0].asDict()
    )
    min_val = val_range["min"]
    max_val = val_range["max"]
    return min_val, max_val


def _get_bin_edges(data: DataFrame, col: str, bin_width: float) -> list[float]:
    min_val, max_val = _data_range(data, col)
    edges = np.arange(min_val, max_val + bin_width, bin_width)
    return edges.tolist()


def _get_edges_from_hist(histogram: pd.DataFrame) -> list[float]:
    edges = histogram["bin_min"].tolist()
    edges.append(float(histogram["bin_max"].iloc[-1]))
    return edges


def _expand_hist_edges(edges: list[float]) -> list[float]:
    edge_cp = edges.copy()
    edge_cp.append(float("inf"))
    edge_cp.insert(0, -float("inf"))
    return edge_cp
