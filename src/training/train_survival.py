"""Time-to-event models for right-censored survival outcomes (OS_MONTHS/OS_STATUS,
DFS_MONTHS/DFS_STATUS, ...). A classifier trained on "did the event happen"
throws away every censored patient's partial follow-up time - these models
don't.

CoxPH is the interpretable baseline (hazard ratios per covariate, easy to
defend to a clinical reviewer). Random Survival Forest is offered for when
non-linear covariate interaction matters more than interpretability.
Evaluated with concordance index, not accuracy/AUC - those aren't defined
for censored time-to-event data.
"""
import logging
import tempfile
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

logger = logging.getLogger(__name__)


def _log_model_artifact(model, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.joblib"
        joblib.dump(model, path)
        mlflow.log_artifact(str(path), artifact_path="model")


def train_coxph(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str],
    experiment_name: str = "cancer_survival",
) -> str:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="coxph") as run:
        mlflow.log_param("covariate_cols", covariate_cols)
        mlflow.log_param("n_train_patients", len(train_df))

        # Ridge penalty: genomic PCA covariates are ~constant for patients with
        # no RNA-seq, so the unpenalized design matrix is singular.
        penalizer = 0.1
        mlflow.log_param("penalizer", penalizer)
        model = CoxPHFitter(penalizer=penalizer)
        model.fit(train_df[covariate_cols + [duration_col, event_col]], duration_col=duration_col, event_col=event_col)

        test_risk_scores = model.predict_partial_hazard(test_df[covariate_cols])
        c_index = concordance_index(test_df[duration_col], -test_risk_scores, test_df[event_col])

        mlflow.log_metric("test_concordance_index", c_index)
        logger.info("CoxPH: test_concordance_index=%.4f", c_index)

        _log_model_artifact(model, "coxph")
        return run.info.run_id


def train_random_survival_forest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str],
    experiment_name: str = "cancer_survival",
    n_estimators: int = 300,
) -> str:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="random_survival_forest") as run:
        mlflow.log_param("covariate_cols", covariate_cols)
        mlflow.log_param("n_estimators", n_estimators)

        y_train = Surv.from_dataframe(event_col, duration_col, train_df)
        y_test = Surv.from_dataframe(event_col, duration_col, test_df)

        model = RandomSurvivalForest(n_estimators=n_estimators, min_samples_leaf=10, random_state=42)
        model.fit(train_df[covariate_cols], y_train)

        c_index = model.score(test_df[covariate_cols], y_test)
        mlflow.log_metric("test_concordance_index", c_index)
        logger.info("RandomSurvivalForest: test_concordance_index=%.4f", c_index)

        _log_model_artifact(model, "random_survival_forest")
        return run.info.run_id
