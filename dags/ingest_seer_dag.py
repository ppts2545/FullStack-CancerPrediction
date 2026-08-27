"""Ingest a SEER Research Data extract into bronze.

No `schedule` - SEER has no API, a new extract only exists when someone runs
SEER*Stat under an approved DUA and drops the file on the mount. Trigger this
DAG manually (or from a file-arrival sensor) after that happens.
"""
import datetime as dt

from airflow.decorators import dag, task

from common.alerts import on_failure_alert
from common.config import load_yaml_config

default_args = {
    "owner": "data-eng",
    "retries": 1,
    "on_failure_callback": on_failure_alert,
}


@dag(
    dag_id="1_ingest_seer",
    schedule=None,
    start_date=dt.datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ingestion", "clinical"],
)
def ingest_seer():
    @task
    def extract_and_validate_seer() -> str:
        from ingestion.seer_extractor import fetch_seer_extract
        from data_quality.schemas import seer_extract_schema, validate

        cfg = load_yaml_config("sources.yaml")["seer"]
        df = fetch_seer_extract(cfg["extract_mount_path"], cfg["layout_config"])
        validate(df, seer_extract_schema)

        tmp_path = "/tmp/seer_extract.parquet"
        df.to_parquet(tmp_path)
        return tmp_path

    @task
    def write_bronze_seer(tmp_path: str):
        import pandas as pd
        from common.spark_session import get_spark
        from ingestion.bronze_writer import write_bronze

        spark = get_spark("ingest-seer")
        try:
            write_bronze(spark, pd.read_parquet(tmp_path), domain="seer_extract")
        finally:
            spark.stop()

    write_bronze_seer(extract_and_validate_seer())


ingest_seer()
