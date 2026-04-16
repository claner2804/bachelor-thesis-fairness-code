from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_scatter_predictions(df: pd.DataFrame, y_pred=None):
    plot_df = df.copy()
    if y_pred is not None:
        plot_df["prediction"] = y_pred
        color_col = "prediction"
        title = "Qualification vs Experience (nach Prediction)"
    else:
        color_col = "label"
        title = "Qualification vs Experience (nach Label)"

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        plot_df["qualification"],
        plot_df["experience"],
        c=plot_df[color_col],
        alpha=0.7,
    )
    ax.set_xlabel("qualification")
    ax.set_ylabel("experience")
    ax.set_title(title)
    fig.tight_layout()
    return fig
