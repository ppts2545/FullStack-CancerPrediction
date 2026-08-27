"""Late fusion: join each modality's per-patient feature block on patient_key,
keeping an explicit `has_<modality>` flag rather than silently zero-filling.

Chosen over early fusion because genomic sequencing coverage in this cohort
is expected to be far from universal - concatenating raw vectors and zero-
filling absent genomic data would let the model implicitly read "zero" as a
real biological value instead of "untested". A missing-modality indicator
lets tree models (XGBoost/LightGBM) split on data availability itself, which
is often informative (e.g. sequencing tends to be ordered for advanced-stage
disease).
"""
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def fuse_modalities(modality_dfs: dict[str, DataFrame], id_col: str = "patient_key") -> DataFrame:
    """modality_dfs: {"clinical": df, "genomic": df, "environmental": df, "lab": df}
    Each df must have exactly one row per id_col. Outer-joins so a patient
    missing one modality still appears, with that modality's columns null.
    """
    joined = reduce(
        lambda left, right: left.join(right, on=id_col, how="outer"),
        modality_dfs.values(),
    )

    for name, df in modality_dfs.items():
        non_id_cols = [c for c in df.columns if c != id_col]
        has_data = reduce(lambda a, b: a | b, [F.col(c).isNotNull() for c in non_id_cols])
        joined = joined.withColumn(f"has_{name}_data", has_data)

    return joined
