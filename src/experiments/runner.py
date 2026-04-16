from __future__ import annotations

import pandas as pd

from src.data.synthetic import generate_synthetic_data
from src.bias.injection import inject_group_imbalance, inject_label_bias
from src.models.train import train_logistic_regression
from src.models.predict import predict_with_threshold
from src.metrics.performance import compute_performance_metrics
from src.metrics.fairness import compute_fairness_metrics, compute_group_summary


def run_single_experiment(
    n_samples: int = 500,
    group_proportion: float = 0.5,
    imbalance_target: float | None = None,
    label_bias_strength: float = 0.0,
    threshold: float = 0.5,
    random_state: int = 42,
    include_sensitive_feature: bool = False,
) -> dict:
    df = generate_synthetic_data(
        n_samples=n_samples,
        group_proportion=group_proportion,
        random_state=random_state,
    )

    if imbalance_target is not None:
        df = inject_group_imbalance(
            df,
            target_group=1,
            target_proportion=imbalance_target,
            random_state=random_state,
        )

    if label_bias_strength > 0:
        df = inject_label_bias(
            df,
            favored_group=1,
            bias_strength=label_bias_strength,
            random_state=random_state,
        )

    feature_cols = ["qualification", "experience"]
    if include_sensitive_feature:
        feature_cols.append("gender")

    trained = train_logistic_regression(
        df=df,
        feature_cols=feature_cols,
        target_col="label",
        random_state=random_state,
    )

    model = trained["model"]
    y_test = trained["y_test"]
    g_test = trained["g_test"]
    X_test = trained["X_test"]

    probabilities, predictions = predict_with_threshold(model, X_test, threshold=threshold)

    performance = compute_performance_metrics(y_test, predictions)
    fairness = compute_fairness_metrics(y_test, predictions, g_test)
    group_summary = compute_group_summary(y_test, predictions, g_test)

    return {
        "df": df,
        "trained": trained,
        "probabilities": probabilities,
        "predictions": predictions,
        "performance": performance,
        "fairness": fairness,
        "group_summary": group_summary,
    }


def run_scenario_grid(configs: list[dict]) -> pd.DataFrame:
    rows = []
    for config in configs:
        result = run_single_experiment(**config)
        rows.append({
            **config,
            **result["performance"],
            **result["fairness"],
        })
    return pd.DataFrame(rows)
