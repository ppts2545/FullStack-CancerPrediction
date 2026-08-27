"""Shared writer: land any extractor's pandas output into the bronze Delta layer.

Idempotent by `ingestion_date` partition - replaceWhere overwrites only today's
slice, so a DAG retry never duplicates rows.
"""
import datetime as dt
import tempfile
from pathlib import Path

import pandas as pd
from delta.tables import DeltaTable
from pyspark.sql import SparkSession

from common.config import bronze_path


def write_bronze(spark: SparkSession, df: pd.DataFrame, domain: str, ingestion_date: str | None = None) -> str:
    if df.empty:
        raise ValueError(f"refusing to write empty dataframe for domain={domain}")

    ingestion_date = ingestion_date or dt.date.today().isoformat()
    df = df.copy()
    df["ingestion_date"] = ingestion_date

    # Round-trip through a local parquet file instead of spark.createDataFrame:
    # createDataFrame pulls the whole pandas frame through the driver and OOMs
    # the local[*] JVM on large extracts (RNA-seq is ~1.5M rows).
    with tempfile.TemporaryDirectory() as tmp:
        staged = str(Path(tmp) / f"{domain}.parquet")
        df.to_parquet(staged, index=False)
        spark_df = spark.read.parquet(staged)
        return _write(spark, spark_df, domain, ingestion_date)


def _write(spark: SparkSession, spark_df, domain: str, ingestion_date: str) -> str:
    path = bronze_path(domain)

    writer = spark_df.write.format("delta").partitionBy("ingestion_date")

    #write only new data for the current ingestion_date partition, if it already exists
    if DeltaTable.isDeltaTable(spark, path):
        writer = writer.mode("overwrite").option("replaceWhere", f"ingestion_date = '{ingestion_date}'")
    else:
        writer = writer.mode("overwrite")
    writer.save(path)
    return path
