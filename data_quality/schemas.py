"""Pandera schemas gating each extractor's raw output before it lands in bronze.

Fail fast here - a schema drift caught at ingestion is cheap; the same drift
caught three stages downstream in feature engineering is not.
"""
import pandera as pa
from pandera import Column, Check

cbioportal_clinical_schema = pa.DataFrameSchema(
    {
        "sampleId": Column(str),
        "patientId": Column(str),
        "clinicalAttributeId": Column(str),
        "value": Column(str, nullable=True),
    },
    strict=False,
)

cbioportal_mutation_schema = pa.DataFrameSchema(
    {
        "sampleId": Column(str),
        "gene": Column(object),
        "proteinChange": Column(str, nullable=True),
        "mutationType": Column(str, nullable=True),
    },
    strict=False,
)

gdc_file_metadata_schema = pa.DataFrameSchema(
    {
        "file_id": Column(str),
        "file_name": Column(str),
        "data_type": Column(str, nullable=True),
    },
    strict=False,
)

gdc_methylation_schema = pa.DataFrameSchema(
    {
        "file_id": Column(str),
        "file_name": Column(str),
        "case_id": Column(object, nullable=True),
    },
    strict=False,
)

environmental_pm25_schema = pa.DataFrameSchema(
    {
        "state_code": Column(str),
        "county_code": Column(str),
        "date_local": Column(str),
        "arithmetic_mean": Column(float, Check.ge(0), nullable=True),
    },
    strict=False,
)

seer_extract_schema = pa.DataFrameSchema(
    {
        "patient_id_number": Column(str),
        "sex": Column(str, Check.isin(["1", "2"]), nullable=True),
        "age_at_diagnosis": Column(str, nullable=True),
        "vital_status": Column(str, Check.isin(["0", "1", "4"]), nullable=True),
    },
    strict=False,
)


def validate(df, schema):
    """Raises pandera.errors.SchemaError on violation - let Airflow mark the task failed."""
    return schema.validate(df, lazy=True)
