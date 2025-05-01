import pandas as pd
from pyspark.sql import DataFrame


def approximate_cdf(
    data: DataFrame, num_col: str, num_perc_splits: int = 100, rel_error: float = 1e-3
) -> pd.DataFrame:
    probas = [i / num_perc_splits for i in range(num_perc_splits + 1)]
    val_cdf = data.approxQuantile(num_col, probas, rel_error)

    return pd.DataFrame({"val": val_cdf, "proba": probas})
