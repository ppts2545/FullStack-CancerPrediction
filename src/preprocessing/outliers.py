"""Outlier handling, split by whether the tail is a measurement artifact or biology.

Lab/vital numerics use IQR capping - clinically implausible values are almost
always transcription/instrument error. Gene expression uses median/MAD instead
of mean/std because RNA-Seq counts are long-tailed by design; a mean-based
z-score would flag real high-expression biology as "outliers" and strip signal
the model needs.
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def cap_iqr(df: DataFrame, columns: list[str], group_cols: list[str] | None = None, k: float = 1.5) -> DataFrame:
    group_cols = group_cols or []
    for col in columns:
        if group_cols:
            q1 = F.percentile_approx(col, 0.25).over(Window.partitionBy(*group_cols))
            q3 = F.percentile_approx(col, 0.75).over(Window.partitionBy(*group_cols))
            lower = F.expr(f"q1 - {k} * (q3 - q1)")
            upper = F.expr(f"q3 + {k} * (q3 - q1)")
            df = (
                df.withColumn("q1", q1)
                .withColumn("q3", q3)
                .withColumn(col, F.greatest(F.least(F.col(col), upper), lower))
                .drop("q1", "q3")
            )
        else:
            q1, q3 = df.approxQuantile(col, [0.25, 0.75], 0.01)
            iqr = q3 - q1
            lower, upper = q1 - k * iqr, q3 + k * iqr
            df = df.withColumn(col, F.greatest(F.least(F.col(col), F.lit(upper)), F.lit(lower)))
    return df


def flag_robust_zscore_outliers(df: DataFrame, columns: list[str], threshold: float = 3.5) -> DataFrame:
    """Median/MAD z-score (Iglewicz-Hoaglin) per column - flags, doesn't remove.
    Downstream jobs decide whether a flagged value is a batch-effect artifact
    or genuine biological extremity; blind removal here would destroy signal.
    """
    mad_const = 1.4826
    for col in columns:
        median = df.approxQuantile(col, [0.5], 0.01)[0]
        abs_dev = F.abs(F.col(col) - F.lit(median))
        mad = df.select(F.percentile_approx(abs_dev, 0.5)).first()[0] or 1e-9
        robust_z = F.abs(F.col(col) - F.lit(median)) / F.lit(mad_const * mad)
        df = df.withColumn(f"{col}_is_outlier", robust_z > threshold)
    return df
