"""Gold -> trained models: patient-level split, then classification + survival.

Reads `patient_features_tcga` (the one gold table with genuine per-patient
multi-modal fusion - see feature_engineering_dag.py for why the SEER/
environmental table stays separate). The binary classification target here
(vital status at last follow-up) ignores censoring time and is included as
an illustrative XGBoost example only - the survival models below are the
methodologically correct way to use this outcome; don't promote the
classifier to production for a time-to-event question.
"""
import datetime as dt

from airflow.decorators import dag, task

from common.alerts import on_failure_alert

default_args = {
    "owner": "ml-eng",
    "retries": 1,
    "retry_delay": dt.timedelta(minutes=5),
    "on_failure_callback": on_failure_alert,
}

FEATURE_COLS = [
    "stage_rank",
    *[f"expression_pca_{i}" for i in range(50)],
    "p53_signaling", "dna_repair", "pi3k_akt", "ras_mapk", "cell_cycle", "unmapped",
]


@dag(
    dag_id="4_model_training",
    schedule="@monthly",
    start_date=dt.datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["training", "gold"],
)
def model_training():
    @task
    def prepare_train_test_split() -> dict:
        from common.config import gold_path
        from common.delta_io import read_delta
        from common.spark_session import get_spark
        from pyspark.sql import functions as F
        from training.patient_split import patient_level_split

        spark = get_spark("training-data-prep")
        try:
            df = read_delta(spark, gold_path("patient_features_tcga"))
            df = df.withColumn("OS_MONTHS", F.col("OS_MONTHS").cast("double"))
            df = df.withColumn("event", F.when(F.col("OS_STATUS").startswith("1"), 1).otherwise(0))
            df = df.dropna(subset=["OS_MONTHS", "patient_key"])

            pandas_df = df.toPandas()
        finally:
            spark.stop()

        available_features = [c for c in FEATURE_COLS if c in pandas_df.columns]
        if not available_features:
            raise ValueError("none of the expected feature columns are present in patient_features_tcga")

        # XGBoost handles NaN natively, but lifelines CoxPH / scikit-survival RSF
        # reject it. Genomic features are null for patients with no RNA-seq;
        # fill 0 (paired with the has_*_data flags the fused table already carries).
        pandas_df[available_features] = pandas_df[available_features].apply(
            lambda s: s.astype("float64")
        ).fillna(0.0)

        train_df, test_df = patient_level_split(pandas_df, id_col="patient_key", stratify_col="event")

        train_path, test_path = "/tmp/train_patients.parquet", "/tmp/test_patients.parquet"
        train_df.to_parquet(train_path)
        test_df.to_parquet(test_path)
        return {"train_path": train_path, "test_path": test_path, "feature_cols": available_features}

    @task
    def train_classification(split: dict) -> str:
        import pandas as pd
        from training.train_classification import train_xgboost_classifier

        train_df, test_df = pd.read_parquet(split["train_path"]), pd.read_parquet(split["test_path"])
        return train_xgboost_classifier(train_df, test_df, split["feature_cols"], target_col="event")

    @task
    def train_survival_models(split: dict) -> dict:
        import pandas as pd
        from training.train_survival import train_coxph, train_random_survival_forest

        train_df, test_df = pd.read_parquet(split["train_path"]), pd.read_parquet(split["test_path"])
        coxph_run_id = train_coxph(train_df, test_df, "OS_MONTHS", "event", split["feature_cols"])
        rsf_run_id = train_random_survival_forest(train_df, test_df, "OS_MONTHS", "event", split["feature_cols"])
        return {"coxph_run_id": coxph_run_id, "rsf_run_id": rsf_run_id}

    split = prepare_train_test_split()
    train_classification(split)
    train_survival_models(split)


model_training()
