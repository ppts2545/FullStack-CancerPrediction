"""XGBoost classifier over the fused gold feature table.

XGBoost over a deep-learning classifier here because: it handles the missing
values that `has_<modality>_data`-flagged rows still carry natively (no
separate imputation-at-inference step), and SHAP-based feature importance is
close to a requirement for clinical stakeholder review - a black-box model
that can't explain "why" is a hard sell to an oncologist.
"""
import logging

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


def train_xgboost_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    experiment_name: str = "cancer_classification",
    params: dict | None = None,
) -> str:
    params = params or {
        "objective": "binary:logistic",
        "max_depth": 5,
        "eta": 0.05,
        "n_estimators": 300,
        "eval_metric": "auc",
    }

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_param("n_train_patients", len(train_df))
        mlflow.log_param("n_test_patients", len(test_df))
        mlflow.log_param("feature_cols", feature_cols)

        model = xgb.XGBClassifier(**params, missing=float("nan"))
        model.fit(train_df[feature_cols], train_df[target_col])

        test_pred_proba = model.predict_proba(test_df[feature_cols])[:, 1]
        auc = roc_auc_score(test_df[target_col], test_pred_proba)
        pr_auc = average_precision_score(test_df[target_col], test_pred_proba)

        mlflow.log_metric("test_roc_auc", auc)
        mlflow.log_metric("test_pr_auc", pr_auc)
        logger.info("XGBoost classifier: test_roc_auc=%.4f test_pr_auc=%.4f", auc, pr_auc)

        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        mlflow.log_dict(importances.to_dict(), "feature_importance.json")

        # No registered_model_name: the MLflow model registry needs a DB-backed
        # tracking store, and this scaffold logs to a local ./mlruns file store.
        mlflow.sklearn.log_model(model, artifact_path="model")

        return run.info.run_id
