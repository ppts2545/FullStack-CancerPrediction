"""Direct GDC RNA-seq bronze ingest - same code path as the DAG task, run
outside Airflow to sidestep the 10-min retry_delay while GDC's data endpoint
is flaky. Writes s3a://cancer-lake/bronze/gdc_rna_seq_counts."""
import sys

sys.path.insert(0, "/opt/airflow/src")
import pandas as pd  # noqa: E402

from common.config import load_yaml_config  # noqa: E402
from common.spark_session import get_spark  # noqa: E402
from ingestion.bronze_writer import write_bronze  # noqa: E402
from ingestion.gdc_extractor import fetch_rna_seq_counts, search_files  # noqa: E402

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 25

cfg = load_yaml_config("sources.yaml")["gdc"]
expr = next(i for i in cfg["data_files"] if i["name"] == "gene_expression")
files_df = search_files(cfg["project_id"], expr["category"], expr["format"])
print(f"found {len(files_df)} gene-expression files; taking {LIMIT}")

frames, failed = [], []
rows = files_df[["file_id", "cases.case_id", "cases.submitter_id"]].head(LIMIT).values
for n, (file_id, case_id, submitter_id) in enumerate(rows, 1):
    try:
        counts = fetch_rna_seq_counts(file_id)
    except RuntimeError as exc:
        failed.append(file_id)
        print(f"  [{n}/{LIMIT}] FAIL {file_id[:8]} {exc}")
        continue
    counts["case_id"] = case_id
    counts["submitter_id"] = submitter_id
    counts["file_id"] = file_id
    frames.append(counts)
    print(f"  [{n}/{LIMIT}] ok {submitter_id} rows={len(counts)}")

print(f"downloaded {len(frames)} ok, {len(failed)} failed")
if len(frames) < 5:
    sys.exit(f"too few files: {len(frames)}")

pdf = pd.concat(frames, ignore_index=True)
print(f"concat: {len(pdf)} rows, {pdf['submitter_id'].nunique()} patients")

spark = get_spark("ingest-rna-seq-direct")
spark.sparkContext.setLogLevel("ERROR")
try:
    path = write_bronze(spark, pdf, domain="gdc_rna_seq_counts")
    print("wrote", path)
    print("bronze count:", spark.read.format("delta").load(path).count())
finally:
    spark.stop()
