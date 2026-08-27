"""Structural reshaping that has to happen before cleaning logic can apply -
cBioPortal's clinical API returns one row per (sample, attribute), not one
row per sample.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def pivot_long_to_wide(df: DataFrame, id_cols: list[str], attribute_col: str, value_col: str) -> DataFrame:
    return df.groupBy(*id_cols).pivot(attribute_col).agg(F.first(value_col))
