from __future__ import annotations

import numpy as np
import pandas as pd


def demographic_parity_difference(y_pred, group) -> float:
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)

    rate_0 = y_pred[group == 0].mean() if np.any(group == 0) else 0.0
    rate_1 = y_pred[group == 1].mean() if np.any(group == 1) else 0.0
    return float(rate_1 - rate_0)


def equal_opportunity_difference(y_true, y_pred, group) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)

    mask0 = (group == 0) & (y_true == 1)
    mask1 = (group == 1) & (y_true == 1)

    tpr0 = y_pred[mask0].mean() if np.any(mask0) else 0.0
    tpr1 = y_pred[mask1].mean() if np.any(mask1) else 0.0
    return float(tpr1 - tpr0)


def compute_group_summary(y_true, y_pred, group) -> pd.DataFrame:
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    group = pd.Series(group).reset_index(drop=True)

    rows = []

    for g in sorted(group.unique()):
        mask = group == g
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]

        positive_rate = float(y_pred_g.mean()) if len(y_pred_g) else 0.0

        positive_true_mask = y_true_g == 1
        if positive_true_mask.sum() > 0:
            true_positive_rate = float(y_pred_g[positive_true_mask].mean())
        else:
            true_positive_rate = 0.0

        rows.append(
            {
                "group": int(g),
                "n": int(mask.sum()),
                "positive_prediction_rate": positive_rate,
                "true_positive_rate": true_positive_rate,
            }
        )

    return pd.DataFrame(rows)


def compute_fairness_metrics(y_true, y_pred, group) -> dict:
    return {
        "demographic_parity_difference": demographic_parity_difference(y_pred, group),
        "equal_opportunity_difference": equal_opportunity_difference(y_true, y_pred, group),
    }