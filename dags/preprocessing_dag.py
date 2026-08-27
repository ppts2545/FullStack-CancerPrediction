"""Bronze -> silver: reshape, clean, and de-identify each domain.

Note on patient linkage scope: cBioPortal/GDC here both come from the same
TCGA cohort and already share a case/sample barcode, and SEER is a separate
population-registry cohort with no shared key to TCGA at all - joining them
via the Master Patient Index would silently fabricate matches between
unrelated patients. The MPI linkage step (`preprocessing.patient_linkage`)
is for the real deployment scenario this pipeline is templated for: matching
a hospital's own EHR records against that same hospital's in-house genomic
sequencing and lab orders, where both sides describe the same patient under
different native IDs. Each public source below is pseudonymized under its
own native ID instead.
"""
import datetime as dt

from airflow.decorators import dag, task

from common.alerts import on_failure_alert

default_args = {
    "owner": "data-eng",
    "retries": 2,
    "retry_delay": dt.timedelta(minutes=5),
    "on_failure_callback": on_failure_alert,
}


@dag(
    dag_id="2_preprocessing",
    schedule="@monthly",
    start_date=dt.datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["preprocessing", "silver"],
    # Each task spins its own local[*] Spark JVM (~3g). The Docker VM only has
    # ~7.4g total, so run them one at a time rather than OOM-racing.
    max_active_tasks=1,
)
def preprocessing():
    @task
    def clean_clinical():
        from common.config import bronze_path, env, silver_path
        from common.delta_io import overwrite_delta, read_delta
        from common.spark_session import get_spark
        from preprocessing.anonymize import (
            drop_direct_identifiers,
            pseudonymize_patient_id,
            truncate_age_over_90,
        )
        from preprocessing.missing_values import fill_categorical_unknown, impute_numeric_stratified
        from preprocessing.outliers import cap_iqr
        from preprocessing.shaping import pivot_long_to_wide

        spark = get_spark("preprocessing-clinical")
        try:
            raw = read_delta(spark, bronze_path("cbioportal_clinical"))
            wide = pivot_long_to_wide(raw, id_cols=["patientId", "sampleId"], attribute_col="clinicalAttributeId", value_col="value")

            numeric_cols = [c for c in ("AGE", "MUTATION_COUNT") if c in wide.columns]
            categorical_cols = [c for c in ("SEX", "CANCER_TYPE", "AJCC_PATHOLOGIC_TUMOR_STAGE") if c in wide.columns]

            for c in numeric_cols:
                wide = wide.withColumn(c, wide[c].cast("double"))

            cleaned = impute_numeric_stratified(wide, numeric_cols, strata_cols=["CANCER_TYPE"]) if numeric_cols and "CANCER_TYPE" in wide.columns else wide
            cleaned = cap_iqr(cleaned, numeric_cols, group_cols=["CANCER_TYPE"] if "CANCER_TYPE" in wide.columns else None) if numeric_cols else cleaned
            cleaned = fill_categorical_unknown(cleaned, categorical_cols) if categorical_cols else cleaned

            if "AGE" in cleaned.columns:
                cleaned = truncate_age_over_90(cleaned, "AGE")

            cleaned = pseudonymize_patient_id(cleaned, "patientId", salt=env("PHI_SALT", required=True))
            cleaned = drop_direct_identifiers(cleaned)

            overwrite_delta(cleaned, silver_path("clinical"))
        finally:
            spark.stop()

    @task
    def clean_mutations():
        from common.config import bronze_path, env, silver_path
        from common.delta_io import overwrite_delta, read_delta
        from common.spark_session import get_spark
        from preprocessing.anonymize import pseudonymize_patient_id
        from preprocessing.missing_values import flag_genomic_not_tested

        spark = get_spark("preprocessing-mutations")
        try:
            raw = read_delta(spark, bronze_path("cbioportal_mutations"))
            cleaned = flag_genomic_not_tested(raw, "mutationType") if "mutationType" in raw.columns else raw
            # Key on patientId (TCGA barcode), same as clinical, so the modalities fuse.
            cleaned = pseudonymize_patient_id(cleaned, "patientId", salt=env("PHI_SALT", required=True), output_col="patient_key")
            overwrite_delta(cleaned, silver_path("mutations"))
        finally:
            spark.stop()

    @task
    def clean_rna_seq():
        from common.config import bronze_path, env, silver_path
        from common.delta_io import overwrite_delta, read_delta
        from common.spark_session import get_spark
        from preprocessing.anonymize import pseudonymize_patient_id
        from preprocessing.outliers import flag_robust_zscore_outliers

        spark = get_spark("preprocessing-rna-seq")
        try:
            raw = read_delta(spark, bronze_path("gdc_rna_seq_counts"))
            numeric_cols = [f.name for f in raw.schema.fields if f.dataType.typeName() in ("double", "integer", "long")]
            cleaned = flag_robust_zscore_outliers(raw, numeric_cols) if numeric_cols else raw
            # Key on the TCGA patient barcode (submitter_id), not the GDC case UUID,
            # so RNA-seq lines up with clinical/mutations on patient_key.
            cleaned = pseudonymize_patient_id(cleaned, "submitter_id", salt=env("PHI_SALT", required=True), output_col="patient_key")
            overwrite_delta(cleaned, silver_path("rna_seq"))
        finally:
            spark.stop()

    @task
    def clean_environmental():
        from delta.tables import DeltaTable
        from airflow.exceptions import AirflowSkipException

        from common.config import bronze_path, silver_path
        from common.delta_io import overwrite_delta, read_delta
        from common.spark_session import get_spark
        from preprocessing.missing_values import fill_environmental_nearest

        spark = get_spark("preprocessing-environmental")
        try:
            path = bronze_path("environmental_pm25")
            if not DeltaTable.isDeltaTable(spark, path):
                raise AirflowSkipException(
                    f"no bronze data yet at {path} - run 1_ingest_environmental (needs EPA_AQS_API_KEY) first"
                )
            raw = read_delta(spark, path)
            raw = raw.withColumn("arithmetic_mean", raw["arithmetic_mean"].cast("double"))
            cleaned = fill_environmental_nearest(raw, "arithmetic_mean", "county_code", "date_local")
            overwrite_delta(cleaned, silver_path("environmental_pm25"))
        finally:
            spark.stop()

    @task
    def clean_seer():
        from delta.tables import DeltaTable
        from airflow.exceptions import AirflowSkipException

        from common.config import bronze_path, env, silver_path
        from common.delta_io import overwrite_delta, read_delta
        from common.spark_session import get_spark
        from preprocessing.anonymize import (
            generalize_zip_to_safe_harbor,
            pseudonymize_patient_id,
            shift_dates,
            truncate_age_over_90,
        )
        from common.config import load_yaml_config

        spark = get_spark("preprocessing-seer")
        try:
            path = bronze_path("seer_extract")
            if not DeltaTable.isDeltaTable(spark, path):
                raise AirflowSkipException(
                    f"no bronze data yet at {path} - needs a SEER*Stat extract dropped under an approved DUA"
                )
            raw = read_delta(spark, path)
            cleaned = raw.withColumn("age_at_diagnosis", raw["age_at_diagnosis"].cast("int"))
            cleaned = truncate_age_over_90(cleaned, "age_at_diagnosis")
            cleaned = pseudonymize_patient_id(
                cleaned, "patient_id_number", salt=env("PHI_SALT", required=True), output_col="patient_key"
            )
            if "zip_code" in cleaned.columns:
                restricted = load_yaml_config("hipaa_restricted_zip3.yaml")["restricted_zip3"]
                cleaned = generalize_zip_to_safe_harbor(cleaned, "zip_code", restricted)
            date_cols = [c for c in ("diagnosis_date", "last_contact_date") if c in cleaned.columns]
            if date_cols:
                cleaned = shift_dates(cleaned, date_cols, "patient_key")

            overwrite_delta(cleaned, silver_path("seer_extract"))
        finally:
            spark.stop()

    clean_clinical()
    clean_mutations()
    clean_rna_seq()
    clean_environmental()
    clean_seer()


preprocessing()
