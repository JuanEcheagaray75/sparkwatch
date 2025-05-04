import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.Builder().appName("pytest-spark").master("local[1]").getOrCreate()
    )
    yield spark
    spark.stop()
