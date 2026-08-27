"""Missing-value handling - strategy differs per modality, see architecture notes.

Clinical/lab numerics are imputed *within* a stratum (e.g. cancer_type) because
pooling across cancer types before imputing biases the mean toward whichever
type is most common in the cohort. Genomic missingness is never imputed - a
null mutation call means "not tested", not "no mutation", and imputing it
would silently fabricate a wild-type call.
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def impute_numeric_stratified(df: DataFrame, columns: list[str], strata_cols: list[str]) -> DataFrame:
    """Fill numeric nulls with the per-stratum median."""
    window = Window.partitionBy(*strata_cols)
    for col in columns:
        stratum_median = F.percentile_approx(col, 0.5).over(window)
        df = df.withColumn(col, F.coalesce(F.col(col), stratum_median))
    return df


def add_missing_indicators(df: DataFrame, columns: list[str]) -> DataFrame:
    """Add a `{col}_was_missing` boolean *before* imputing - the fact that a lab
    test was never ordered is itself clinically informative and must survive
    imputation as a feature, not get erased by it."""
    for col in columns:
        df = df.withColumn(f"{col}_was_missing", F.col(col).isNull())
    return df


def fill_categorical_unknown(df: DataFrame, columns: list[str], placeholder: str = "Unknown") -> DataFrame:
    for col in columns:
        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(placeholder)))
    return df


def flag_genomic_not_tested(df: DataFrame, mutation_status_col: str) -> DataFrame:
    """Null mutation call -> explicit 'not_tested' category, distinct from 'wild_type'."""
    return df.withColumn(
        mutation_status_col,
        F.when(F.col(mutation_status_col).isNull(), F.lit("not_tested")).otherwise(F.col(mutation_status_col)),
    )


def fill_environmental_nearest(df: DataFrame, value_col: str, county_col: str, date_col: str) -> DataFrame:
    """Fall back to the same county's nearest-in-time reading when a monitor
    station has a gap, rather than dropping the row or imputing a global mean
    that ignores local air-quality patterns."""
    window = Window.partitionBy(county_col).orderBy(date_col).rowsBetween(Window.unboundedPreceding, Window.currentRow)
    forward_filled = F.last(value_col, ignorenulls=True).over(window)
    return df.withColumn(value_col, F.coalesce(F.col(value_col), forward_filled))
