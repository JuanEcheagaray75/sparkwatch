from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from pyspark.sql import DataFrame


class FeatureType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


@dataclass
class Threshold:
    lower_threshold: float = 0
    upper_threshold: float = float("inf")

    def __post_init__(self):
        if self.lower_threshold > self.upper_threshold:
            raise ValueError(
                f"Lower threshold ({self.lower_threshold}) cannot be greater than upper threshold ({self.upper_threshold})."
            )

    def alert(self, value: float) -> bool:
        return value < self.lower_threshold or value > self.upper_threshold


def _is_numerical(dtype: str) -> bool:
    return dtype in ["bigint", "smallint", "int", "float", "double"]


def _is_categorical(dtype) -> bool:
    return dtype in ["string", "boolean"]


def _split_features_by_type(data: DataFrame, feature_column_names: Sequence[str]):
    dataset_columns = data.columns
    available_cols = [c for c in dataset_columns if c in feature_column_names]
    df_dtypes = data.select(*available_cols).dtypes

    numerical_columns = [name for name, dtype in df_dtypes if _is_numerical(dtype)]
    categorical_columns = [name for name, dtype in df_dtypes if _is_categorical(dtype)]

    return numerical_columns, categorical_columns


def _get_feature_type(data: DataFrame, column: str) -> FeatureType:
    num, _ = _split_features_by_type(data, [column])

    return FeatureType.NUMERICAL if num else FeatureType.CATEGORICAL


def _collect_unique_columns(
    found: Sequence[str], treat: Sequence[str] | None
) -> list[str]:
    return list(set(found) | set(treat if treat else {}))


def _detect_column_types(
    data: DataFrame,
    available_columns: Sequence[str],
    treat_as_numerical: Sequence[str] | None = None,
    treat_as_categorical: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    found_numerical, found_categorical = _split_features_by_type(
        data, available_columns
    )
    numerical_columns = _collect_unique_columns(found_numerical, treat_as_numerical)
    categorical_columns = _collect_unique_columns(
        found_categorical, treat_as_categorical
    )

    return numerical_columns, categorical_columns
