"""Silver/gold writes are full recomputes of a derived table, not append logs -
plain overwrite is correct here (unlike bronze's replaceWhere-by-partition)."""
from pyspark.sql import DataFrame


def overwrite_delta(df: DataFrame, path: str, partition_by: list[str] | None = None) -> str:
    writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(path)
    return path


def read_delta(spark, path: str) -> DataFrame:
    return spark.read.format("delta").load(path)
