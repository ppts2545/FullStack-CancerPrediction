"""Ingest PM2.5 exposure data (EPA AQS) into bronze, one pull per configured county.

Daily schedule - AQS publishes new daily monitor readings continuously and
this is a cheap API call, unlike the genomic sources.
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
    dag_id="1_ingest_environmental",
    schedule="@daily",
    start_date=dt.datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ingestion", "environmental"],
)
def ingest_environmental():
    @task
    def extract_and_validate_pm25(data_interval_end=None) -> str:
        import pandas as pd
        from ingestion.environmental_extractor import fetch_pm25_by_county
        from data_quality.schemas import environmental_pm25_schema, validate

        cfg = load_yaml_config("sources.yaml")["environmental"]
        edate = data_interval_end.date()
        bdate = edate - dt.timedelta(days=cfg["lookback_days"])

        frames = [
            fetch_pm25_by_county(c["state_code"], c["county_code"], bdate.strftime("%Y%m%d"), edate.strftime("%Y%m%d"))
            for c in cfg["counties"]
        ]
        df = pd.concat(frames, ignore_index=True)
        df["arithmetic_mean"] = df["arithmetic_mean"].astype(float)
        validate(df, environmental_pm25_schema)

        tmp_path = "/tmp/environmental_pm25.parquet"
        df.to_parquet(tmp_path)
        return tmp_path

    @task
    def write_bronze_pm25(tmp_path: str):
        import pandas as pd
        from common.spark_session import get_spark
        from ingestion.bronze_writer import write_bronze

        spark = get_spark("ingest-environmental")
        try:
            write_bronze(spark, pd.read_parquet(tmp_path), domain="environmental_pm25")
        finally:
            spark.stop()

    write_bronze_pm25(extract_and_validate_pm25())


ingest_environmental()
