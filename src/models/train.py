from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_logistic_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "label",
    test_size: float = 0.3,
    random_state: int = 42,
):
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    groups = df["gender"].copy()

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, groups, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "g_train": g_train,
        "g_test": g_test,
    }
