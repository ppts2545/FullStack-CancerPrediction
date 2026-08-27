"""Silver -> gold: encode, reduce, and fuse into model-ready feature tables.

Two separate gold outputs, not one - the sources don't share a patient
cohort:

- `patient_features_tcga`: cBioPortal clinical + GDC expression/mutation,
  joined on patient_key. Same TCGA cohort end-to-end, so per-patient fusion
  is valid here. No environmental exposure - public TCGA clinical records
  withhold patient geography.
- `patient_features_seer_environmental`: SEER clinical joined to PM2.5
  exposure by (county, year). SEER is a separate population-registry cohort
  with no shared key to TCGA, so this table has no genomic columns - fusing
  it with the TCGA table would silently pair up unrelated patients.

Getting genomic + environmental + clinical + lab onto the *same* patient
row (true 4-modality fusion) requires one institution's own EHR/genomic/lab
records linked through `preprocessing.patient_linkage`'s MPI step - that is
the production path this scaffold is templated for, not something the public
sources here can provide.
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
    dag_id="3_feature_engineering",
    schedule="@monthly",
    start_date=dt.datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["feature-engineering", "gold"],
    # One local[*] Spark JVM per task; the Docker VM has ~7.4g, so serialize.
    max_active_tasks=1,
)
def feature_engineering():
    @task
    def encode_clinical() -> None:
        from common.config import gold_path, silver_path
        from common.delta_io import overwrite_delta, read_delta
        from common.spark_session import get_spark
        from feature_engineering.categorical_encoding import encode_tnm_stage_ordinal

        spark = get_spark("feature-clinical")
        try:
            df = read_delta(spark, silver_path("clinical"))
            if "AJCC_PATHOLOGIC_TUMOR_STAGE" in df.columns:
                df = encode_tnm_stage_ordinal(df, "AJCC_PATHOLOGIC_TUMOR_STAGE", "stage_rank")
            overwrite_delta(df, gold_path("clinical_features"))
        finally:
            spark.stop()

    @task
    def reduce_genomic() -> None:
        import re

        from common.config import gold_path, silver_path
        from common.delta_io import overwrite_delta, read_delta
        from common.spark_session import get_spark
        from feature_engineering.genomic_dim_reduction import reduce_expression_pca, top_variable_genes
        from preprocessing.shaping import pivot_long_to_wide

        spark = get_spark("feature-genomic")
        try:
            long_df = read_delta(spark, silver_path("rna_seq"))
            long_df = top_variable_genes(long_df, gene_col="gene_name", value_col="unstranded", n=2000)
            wide = pivot_long_to_wide(long_df, id_cols=["patient_key"], attribute_col="gene_name", value_col="unstranded")
            # Gene symbols like "AC011503.1" / "HLA-DRB1" become struct-access or
            # bad identifiers as Spark column names - sanitize before Spark ML touches them.
            wide = wide.toDF(*[c if c == "patient_key" else re.sub(r"\W", "_", c) for c in wide.columns])
            gene_cols = [c for c in wide.columns if c != "patient_key"]
            wide = wide.na.fill(0.0, subset=gene_cols)

            if len(gene_cols) < 2:
                raise ValueError("rna_seq silver table has too few gene columns to run PCA - check pivot upstream")

            # PCA rank is bounded by both gene count and (centered) sample count.
            n_patients = wide.count()
            k = max(1, min(50, len(gene_cols) - 1, n_patients - 1))
            reduced = reduce_expression_pca(wide, gene_cols, k=k)
            overwrite_delta(reduced, gold_path("expression_features"))
        finally:
            spark.stop()

    @task
    def aggregate_mutation_burden() -> None:
        from common.config import gold_path, load_yaml_config, silver_path
        from common.delta_io import overwrite_delta, read_delta
        from common.spark_session import get_spark
        from feature_engineering.genomic_dim_reduction import aggregate_mutation_burden as aggregate

        spark = get_spark("feature-mutation-burden")
        try:
            mutations = read_delta(spark, silver_path("mutations"))
            mapping_rows = load_yaml_config("gene_pathway_map.yaml")["mappings"]
            gene_pathway_map = spark.createDataFrame(mapping_rows)

            burden = aggregate(mutations, gene_pathway_map, id_col="patient_key")
            overwrite_delta(burden, gold_path("mutation_burden_features"))
        finally:
            spark.stop()

    @task
    def fuse_tcga_features() -> None:
        """Clinical + expression + mutation-burden, all keyed by the same
        TCGA patient_key - the one case in this pipeline where per-patient
        multi-modal fusion is actually valid against public data."""
        from common.config import gold_path
        from common.delta_io import overwrite_delta, read_delta
        from common.spark_session import get_spark
        from feature_engineering.multimodal_fusion import fuse_modalities

        spark = get_spark("feature-fusion-tcga")
        try:
            modality_dfs = {
                "clinical": read_delta(spark, gold_path("clinical_features")),
                "genomic_expression": read_delta(spark, gold_path("expression_features")),
                "genomic_mutation_burden": read_delta(spark, gold_path("mutation_burden_features")),
            }
            fused = fuse_modalities(modality_dfs, id_col="patient_key")
            overwrite_delta(fused, gold_path("patient_features_tcga"))
        finally:
            spark.stop()

    @task
    def build_seer_environmental_features() -> None:
        """SEER clinical + PM2.5 exposure, joined at (county, year) grain -
        a population-level environmental association table, not per-patient
        genomic fusion. Kept as its own gold output rather than forced into
        patient_features_tcga."""
        from delta.tables import DeltaTable
        from airflow.exceptions import AirflowSkipException

        from common.config import gold_path, silver_path
        from common.delta_io import overwrite_delta, read_delta
        from common.spark_session import get_spark
        from pyspark.sql import functions as F

        spark = get_spark("feature-seer-environmental")
        try:
            for dom in ("environmental_pm25", "seer_extract"):
                if not DeltaTable.isDeltaTable(spark, silver_path(dom)):
                    raise AirflowSkipException(
                        f"no silver {dom} - run 1_ingest_environmental / 1_ingest_seer + 2_preprocessing first"
                    )

            environmental_yearly = (
                read_delta(spark, silver_path("environmental_pm25"))
                .withColumn("year", F.substring("date_local", 1, 4))
                .groupBy("county_code", "year")
                .agg(F.avg("arithmetic_mean").alias("pm25_annual_mean"))
            )

            seer = read_delta(spark, silver_path("seer_extract")).withColumnRenamed(
                "county_of_residence", "county_code"
            )
            if "year_of_diagnosis" in seer.columns:
                seer = seer.withColumn("year", F.col("year_of_diagnosis").cast("string"))

            fused = seer.join(environmental_yearly, on=["county_code", "year"], how="left")
            overwrite_delta(fused, gold_path("patient_features_seer_environmental"))
        finally:
            spark.stop()

    clinical_done = encode_clinical()
    expression_done = reduce_genomic()
    mutation_done = aggregate_mutation_burden()

    [clinical_done, expression_done, mutation_done] >> fuse_tcga_features()
    build_seer_environmental_features()


feature_engineering()
