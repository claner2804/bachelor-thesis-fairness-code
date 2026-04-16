from __future__ import annotations

import numpy as np
import pandas as pd


def inject_group_imbalance(
    df: pd.DataFrame,
    target_group: int = 1,
    target_proportion: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Verändert die Gruppenverteilung durch Downsampling der jeweils anderen Gruppe.
    target_group=1 bedeutet: diese Gruppe soll target_proportion im Enddatensatz haben.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    group_a = df[df["gender"] == target_group]
    group_b = df[df["gender"] != target_group]

    total_n = len(df)
    desired_a = int(total_n * target_proportion)
    desired_b = total_n - desired_a

    if len(group_a) == 0 or len(group_b) == 0:
        return df

    sample_a = group_a.sample(n=min(desired_a, len(group_a)), random_state=random_state)
    sample_b = group_b.sample(n=min(desired_b, len(group_b)), random_state=random_state + 1)

    combined = pd.concat([sample_a, sample_b], axis=0).sample(frac=1.0, random_state=random_state + 2)
    combined = combined.reset_index(drop=True)
    return combined


def inject_label_bias(
    df: pd.DataFrame,
    favored_group: int = 1,
    bias_strength: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simuliert Label-Bias:
    Für die favoured group wird bei vorhandener positiver Eignung
    die Wahrscheinlichkeit eines positiven Labels erhöht.

    bias_strength sollte typischerweise zwischen 0.0 und 0.4 liegen.
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()

    mask = out["gender"] == favored_group
    qualification = out.loc[mask, "qualification"].to_numpy()
    experience = out.loc[mask, "experience"].to_numpy()

    latent = 1.7 * qualification + 1.1 * experience - 1.2 + bias_strength
    prob = 1 / (1 + np.exp(-latent))
    out.loc[mask, "label"] = rng.binomial(1, prob).astype(int)
    return out


def remove_sensitive_attribute(df: pd.DataFrame, sensitive_col: str = "gender") -> pd.DataFrame:
    out = df.copy()
    return out.drop(columns=[sensitive_col], errors="ignore")
