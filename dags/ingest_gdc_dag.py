"""Ingest mutation (MAF) file metadata + RNA-Seq counts from GDC into bronze.

Downloads a bounded sample of files per run (SAMPLE_FILE_LIMIT) - GDC files
are large and controlled-access downloads are rate-limited, so a full-cohort
pull belongs in a dedicated backfill DAG, not the recurring one.
"""
import datetime as dt

from airflow.decorators import dag, task

from common.alerts import on_failure_alert
from common.config import load_yaml_config

SAMPLE_FILE_LIMIT = 25
METHYLATION_FILE_LIMIT = 5  # ~485k probe rows per file; keep the demo pull bounded

default_args = {
    "owner": "data-eng",
    "retries": 3,
    "retry_delay": dt.timedelta(minutes=10),
    "on_failure_callback": on_failure_alert,
}


@dag(
    dag_id="1_ingest_gdc",
    schedule="@monthly",
    start_date=dt.datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ingestion", "genomic"],
)
def ingest_gdc():
    @task
    def extract_mutation_file_metadata() -> str:
        from ingestion.gdc_extractor import search_files
        from data_quality.schemas import gdc_file_metadata_schema, validate
        cfg = load_yaml_config("sources.yaml")["gdc"]
        mut_cfg = next(item for item in cfg["data_files"] if item["name"] == "mutations")

        df = search_files(
            cfg["project_id"],
            mut_cfg["category"],
            mut_cfg["format"]
        )

        validate(df, gdc_file_metadata_schema)

        tmp_path = "/tmp/gdc_mutation_files.parquet"
        df.to_parquet(tmp_path)
        return tmp_path

    @task
    def extract_rna_seq_counts(metadata_path: str) -> str:
        import pandas as pd
        from ingestion.gdc_extractor import search_files, fetch_rna_seq_counts

        cfg = load_yaml_config("sources.yaml")["gdc"]

        # 1. ดึง config ของ gene_expression ออกมาจาก data_files
        expr_cfg = next(item for item in cfg["data_files"] if item["name"] == "gene_expression")

        # 2. ค้นหาไฟล์โดยระบุ category และ format ให้ถูกชั้น
        files_df = search_files(
            cfg["project_id"],
            expr_cfg["category"],
            expr_cfg["format"]
        )

        # 3. เช็กว่าคอลัมน์ cases.case_id มีอยู่จริงไหม (เผื่อกรณี json_normalize)
        if "cases.case_id" not in files_df.columns and "cases" in files_df.columns:
            files_df["cases.case_id"] = files_df["cases"].apply(
                lambda x: x[0]["case_id"] if isinstance(x, list) and len(x) > 0 and "case_id" in x[0] else None
            )

        frames, failed = [], []
        id_cols = ["file_id", "cases.case_id", "cases.submitter_id"]
        for file_id, case_id, submitter_id in files_df[id_cols].head(SAMPLE_FILE_LIMIT).values:
            try:
                counts = fetch_rna_seq_counts(file_id)
            except RuntimeError as exc:  # GDC dropped every retry for this file
                failed.append((file_id, str(exc)))
                continue
            counts["case_id"] = case_id
            counts["submitter_id"] = submitter_id  # TCGA patient barcode - the cross-modality join key
            counts["file_id"] = file_id
            frames.append(counts)

        if len(frames) < 5:
            raise ValueError(
                f"only {len(frames)} GDC RNA-Seq files downloaded, {len(failed)} failed - too few to proceed"
            )

        tmp_path = "/tmp/gdc_rna_seq_counts.parquet"
        pd.concat(frames, ignore_index=True).to_parquet(tmp_path)
        return tmp_path

    @task
    def extract_methylation_beta_values() -> str:
        import pandas as pd
        from ingestion.gdc_extractor import search_files, fetch_methylation_beta_values
        from data_quality.schemas import gdc_file_metadata_schema, validate

        cfg = load_yaml_config("sources.yaml")["gdc"]
        meth_cfg = next(item for item in cfg["data_files"] if item["name"] == "dna_methylation")

        files_df = search_files(
            cfg["project_id"],
            meth_cfg["category"],
            meth_cfg["format"]
        )

        validate(files_df, gdc_file_metadata_schema)

        frames = []
        meth_cols = ["file_id", "cases.case_id", "cases.submitter_id", "file_name"]
        for file_id, case_id, submitter_id, file_name in files_df[meth_cols].head(METHYLATION_FILE_LIMIT).values:
            beta_df = fetch_methylation_beta_values(file_id)
            beta_df["case_id"] = case_id
            beta_df["submitter_id"] = submitter_id
            beta_df["file_id"] = file_id
            beta_df["file_name"] = file_name
            frames.append(beta_df)

        if not frames:
            raise ValueError("no GDC methylation files downloaded - check project_id/data_format in sources.yaml")

        tmp_path = "/tmp/gdc_methylation_beta_values.parquet"
        pd.concat(frames, ignore_index=True).to_parquet(tmp_path)
        return tmp_path

    @task
    def write_bronze_mutation_metadata(tmp_path: str):
        import pandas as pd
        from common.spark_session import get_spark
        from ingestion.bronze_writer import write_bronze

        spark = get_spark("ingest-gdc-mutation-metadata")
        try:
            write_bronze(spark, pd.read_parquet(tmp_path), domain="gdc_mutation_file_metadata")
        finally:
            spark.stop()

    @task
    def write_bronze_rna_seq(tmp_path: str):
        import pandas as pd
        from common.spark_session import get_spark
        from ingestion.bronze_writer import write_bronze

        spark = get_spark("ingest-gdc-rna-seq")
        try:
            write_bronze(spark, pd.read_parquet(tmp_path), domain="gdc_rna_seq_counts")
        finally:
            spark.stop()

    @task
    def write_bronze_methylation(tmp_path: str):
        import pandas as pd
        from common.spark_session import get_spark
        from ingestion.bronze_writer import write_bronze

        spark = get_spark("ingest-gdc-methylation")
        try:
            write_bronze(spark, pd.read_parquet(tmp_path), domain="gdc_methylation_beta_values")
        finally:
            spark.stop()

    metadata_path = extract_mutation_file_metadata()
    write_bronze_mutation_metadata(metadata_path)
    write_bronze_rna_seq(extract_rna_seq_counts(metadata_path))
    write_bronze_methylation(extract_methylation_beta_values())


ingest_gdc()
