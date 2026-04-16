from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_group_distribution(df: pd.DataFrame):
    counts = df["gender"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Gruppenverteilung")
    ax.set_xlabel("gender")
    ax.set_ylabel("Anzahl")
    fig.tight_layout()
    return fig


def plot_label_distribution(df: pd.DataFrame):
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Label-Verteilung")
    ax.set_xlabel("label")
    ax.set_ylabel("Anzahl")
    fig.tight_layout()
    return fig
