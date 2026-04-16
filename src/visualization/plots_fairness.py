from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_group_metric_bars(group_summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(group_summary))
    width = 0.35

    ax.bar([i - width / 2 for i in x], group_summary["positive_prediction_rate"], width=width, label="Positive Rate")
    ax.bar([i + width / 2 for i in x], group_summary["true_positive_rate"], width=width, label="TPR")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Group {g}" for g in group_summary["group"]])
    ax.set_ylim(0, 1)
    ax.set_title("Fairness-Kennzahlen nach Gruppe")
    ax.set_ylabel("Rate")
    ax.legend()
    fig.tight_layout()
    return fig
