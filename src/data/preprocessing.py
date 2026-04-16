from __future__ import annotations

import pandas as pd


def get_feature_target_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "label",
):
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def get_group_series(df: pd.DataFrame, group_col: str = "gender"):
    return df[group_col].copy()
