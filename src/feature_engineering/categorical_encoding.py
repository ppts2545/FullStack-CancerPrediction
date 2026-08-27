"""Categorical/ordinal encoders that respect each field's actual structure -
one-hotting everything blindly throws away information TNM/ICD-10 already encode.
"""
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# AJCC TNM overall stage, ordered least -> most advanced.
TNM_STAGE_ORDER = ["0", "I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IIIC", "IV"]
TNM_STAGE_RANK = {stage: rank for rank, stage in enumerate(TNM_STAGE_ORDER)}


def encode_icd10_category(df: DataFrame, icd_col: str) -> DataFrame:
    """ICD-10 full codes have huge cardinality; the 3-character category
    (e.g. "C50" from "C50.911") is the clinically meaningful grouping and
    keeps one-hot dimensionality sane.
    """
    df = df.withColumn(f"{icd_col}_category", F.substring(F.col(icd_col), 1, 3))
    indexer = StringIndexer(inputCol=f"{icd_col}_category", outputCol=f"{icd_col}_category_idx", handleInvalid="keep")
    encoder = OneHotEncoder(inputCol=f"{icd_col}_category_idx", outputCol=f"{icd_col}_category_vec")
    return Pipeline(stages=[indexer, encoder]).fit(df).transform(df)


def encode_tnm_stage_ordinal(df: DataFrame, stage_col: str, output_col: str | None = None) -> DataFrame:
    """Stage has a real biological order (I < II < III < IV) - ordinal rank
    preserves that; one-hot would let the model treat Stage I and Stage IV
    as equally "different" from Stage II, which is not clinically true.
    """
    output_col = output_col or f"{stage_col}_rank"
    mapping = F.create_map([F.lit(x) for pair in TNM_STAGE_RANK.items() for x in pair])
    # cBioPortal/TCGA values look like "STAGE IIIA" / "STAGE X"; strip the prefix
    # and normalize case so the roman-numeral keys match. Unknown ("X", null)
    # falls through the map to null, which is the correct "no rank" outcome.
    normalized = F.upper(F.trim(F.regexp_replace(F.col(stage_col), r"(?i)^\s*stage\s+", "")))
    return df.withColumn(output_col, mapping[normalized])


def encode_days_since_treatment(df: DataFrame, treatment_date_col: str, index_date_col: str, output_col: str) -> DataFrame:
    """Flattening treatment history into one-hot "had chemo: y/n" throws away
    timing; days-since-treatment lets a survival model use recency directly.
    """
    return df.withColumn(output_col, F.datediff(F.col(index_date_col), F.col(treatment_date_col)))
