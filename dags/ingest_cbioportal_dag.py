"""Ingest clinical + somatic mutation data from cBioPortal into the bronze layer.

Public API updates per-study data infrequently - monthly pull is enough to
stay current without hammering the endpoint.
"""
import datetime as dt

from airflow.decorators import dag, task

from common.alerts import on_failure_alert
from common.config import load_yaml_config

default_args = {
    "owner": "data-eng",
    "retries": 3,
    "retry_delay": dt.timedelta(minutes=5),
    "on_failure_callback": on_failure_alert,
}


@dag(
    dag_id="1_ingest_cbioportal",
    schedule="@monthly",
    start_date=dt.datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ingestion", "genomic"],
)
def ingest_cbioportal():
    @task
    def extract_and_validate_clinical() -> str:
        from ingestion.cbioportal_extractor import fetch_clinical_data
        from data_quality.schemas import cbioportal_clinical_schema, validate

        cfg = load_yaml_config("sources.yaml")["cbioportal"]
        df = fetch_clinical_data(cfg["study_id"])
        validate(df, cbioportal_clinical_schema)

        tmp_path = "/tmp/cbioportal_clinical.parquet"
        df.to_parquet(tmp_path)
        return tmp_path

    @task
    def extract_and_validate_mutations() -> str:
        from ingestion.cbioportal_extractor import fetch_mutations
        from data_quality.schemas import cbioportal_mutation_schema, validate

        cfg = load_yaml_config("sources.yaml")["cbioportal"]
        df = fetch_mutations(cfg["study_id"], cfg["molecular_profile_id"], cfg["sample_list_id"])
        validate(df, cbioportal_mutation_schema)

        tmp_path = "/tmp/cbioportal_mutations.parquet"
        df.to_parquet(tmp_path)
        return tmp_path

    @task
    def write_bronze_clinical(tmp_path: str):
        import pandas as pd
        from common.spark_session import get_spark
        from ingestion.bronze_writer import write_bronze

        spark = get_spark("ingest-cbioportal-clinical")
        try:
            write_bronze(spark, pd.read_parquet(tmp_path), domain="cbioportal_clinical")
        finally:
            spark.stop()

    @task
    def write_bronze_mutations(tmp_path: str):
        import pandas as pd
        from common.spark_session import get_spark
        from ingestion.bronze_writer import write_bronze

        spark = get_spark("ingest-cbioportal-mutations")
        try:
            write_bronze(spark, pd.read_parquet(tmp_path), domain="cbioportal_mutations")
        finally:
            spark.stop()

    write_bronze_clinical(extract_and_validate_clinical())
    write_bronze_mutations(extract_and_validate_mutations())


ingest_cbioportal()
