"""HIPAA Safe Harbor de-identification - runs once, right after patient linkage,
before any data crosses into the feature-engineering / analyst-facing zone.

Safe Harbor requires removing 18 identifier categories; this module covers the
ones that actually show up in this pipeline's sources (direct identifiers,
dates, geography finer than state). It does not cover free-text notes -
scrubbing those needs a separate NLP-based redaction pass, not string ops.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# The 18 Safe Harbor identifier categories, as they appear in this pipeline's schemas.
DIRECT_IDENTIFIER_COLUMNS = [
    "patient_name", "mrn", "ssn", "phone", "email", "fax",
    "street_address", "device_id", "vehicle_id", "biometric_id",
    "photo_url", "url", "ip_address", "account_number", "certificate_number",
]


def drop_direct_identifiers(df: DataFrame) -> DataFrame:
    present = [c for c in DIRECT_IDENTIFIER_COLUMNS if c in df.columns]
    return df.drop(*present)


def pseudonymize_patient_id(df: DataFrame, id_col: str, salt: str, output_col: str = "patient_key") -> DataFrame:
    """Replace the real patient ID with a salted hash. The salt lives in a
    secrets manager, never in code/config committed to the repo - anyone with
    only the hashed key and no salt cannot reverse it to the source ID.
    """
    return df.withColumn(output_col, F.sha2(F.concat(F.col(id_col), F.lit(salt)), 256)).drop(id_col)


def shift_dates(df: DataFrame, date_columns: list[str], patient_key_col: str, max_shift_days: int = 365) -> DataFrame:
    """Shift every date for a given patient by the same pseudo-random offset,
    derived deterministically from their patient_key. This preserves the
    relative timeline between events (diagnosis -> treatment -> follow-up)
    that survival models need, while the absolute calendar date is no longer
    the real one.
    """
    offset_days = (F.abs(F.hash(F.col(patient_key_col))) % F.lit(2 * max_shift_days)) - F.lit(max_shift_days)
    for col in date_columns:
        df = df.withColumn(col, F.date_add(F.col(col), offset_days.cast("int")))
    return df


def generalize_zip_to_safe_harbor(
    df: DataFrame, zip_col: str, restricted_zip3_prefixes: list[str], output_col: str | None = None
) -> DataFrame:
    """Safe Harbor allows the first 3 digits of ZIP, *except* for the initial
    3-digit prefixes HHS publishes as covering <20,000 people combined - those
    must be zeroed out (e.g. "000") to avoid re-identifying a sparse area.
    """
    output_col = output_col or zip_col
    zip3 = F.substring(F.col(zip_col), 1, 3)
    return df.withColumn(
        output_col,
        F.when(zip3.isin(restricted_zip3_prefixes), F.lit("000")).otherwise(zip3),
    )


def truncate_age_over_90(df: DataFrame, age_col: str) -> DataFrame:
    """Safe Harbor requires ages over 89 to be aggregated into a single '90+' bucket."""
    return df.withColumn(
        age_col,
        F.when(F.col(age_col) > 89, F.lit(90)).otherwise(F.col(age_col)),
    )
