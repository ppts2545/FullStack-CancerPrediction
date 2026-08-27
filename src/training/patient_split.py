"""Patient-level train/test split.

Splitting by row instead of by patient is the single most common way to fake
good metrics on clinical data: if a patient contributes more than one row
(multiple samples, multiple visits), a row-level random split lets the same
patient land in both train and test, and the model partly "memorizes" the
patient rather than generalizing. Every split in this pipeline goes through
this function - never `sklearn.train_test_split` directly on the row-level
dataframe.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def patient_level_split(
    df: pd.DataFrame,
    id_col: str,
    stratify_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patients = df[[id_col, stratify_col]].drop_duplicates(subset=id_col)

    train_ids, test_ids = train_test_split(
        patients[id_col],
        test_size=test_size,
        random_state=random_state,
        stratify=patients[stratify_col],
    )

    train_df = df[df[id_col].isin(train_ids)]
    test_df = df[df[id_col].isin(test_ids)]
    return train_df, test_df
