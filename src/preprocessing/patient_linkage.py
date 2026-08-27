"""Master Patient Index: link the same patient across sources whose native IDs
don't share a key (hospital MRN vs SEER case ID vs GDC case_id vs cBioPortal
patientId).

Runs on quasi-identifiers *before* anonymization strips them - deterministic
exact-match blocking on (sex, birth_year, birth_state, zip3). This is the
cheap, high-precision tier. Cases that don't match exactly still need a
probabilistic fuzzy pass (Fellegi-Sunter, e.g. the `splink` library) for
typos/transcription drift - that is a deliberate v2 scope cut, not an
oversight, since the deterministic tier already covers the majority of clean
public-dataset joins this pipeline targets.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_blocking_key(df: DataFrame, sex_col: str, birth_year_col: str, zip_col: str) -> DataFrame:
    """Generates a composite '_blocking_key' (Sex|BirthYear|ZIP3) to optimize
    patient record linkage.

    Purpose:

    
    Group matching candidates into smaller buckets (strata) before fuzzy
    name-matching.
    This drastically reduces total pairwise comparisons from O(N*M) to O(k),
    preventing memory overflow and reducing execution time on large datasets.
    """
    zip3 = F.substring(F.col(zip_col), 1, 3)
    return df.withColumn(
        "_blocking_key",
        F.concat_ws("|", F.col(sex_col), F.col(birth_year_col).cast("string"), zip3),
    )


def assign_master_patient_id(*dfs_with_source_id: tuple[DataFrame, str], salt: str) -> DataFrame:
    """Union quasi-identifier + native-id pairs from every source and assign
    one master_patient_id per blocking key. The id is a salted hash of the
    blocking key itself, not a randomly generated one - a rerun (or next
    day's incremental load) must produce the *same* master_patient_id for the
    same patient, otherwise every downstream join against previously written
    silver/gold data breaks. A random id would silently fork every patient's
    history on each run.

    Returns a lookup table: (source_name, source_patient_id, master_patient_id).
    """
    union_df = None
    for df, source_name in dfs_with_source_id:
        tagged = df.select("_blocking_key", "_source_patient_id").withColumn("_source_name", F.lit(source_name))
        union_df = tagged if union_df is None else union_df.unionByName(tagged)

    return union_df.withColumn(
        "master_patient_id", F.sha2(F.concat(F.col("_blocking_key"), F.lit(salt)), 256)
    ).select("_source_name", "_source_patient_id", "master_patient_id", "_blocking_key")
