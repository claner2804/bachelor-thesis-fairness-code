from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_synthetic_data(
    n_samples: int = 500,
    group_proportion: float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Erzeugt einen synthetischen Datensatz fuer ein binäres Klassifikationsproblem.

    Spalten:
    - gender: sensibles Attribut (0/1)
    - qualification: Hauptmerkmal
    - experience: Zusatzmerkmal
    - score: kombinierter latent score
    - label: Zielvariable (0/1)
    """
    rng = np.random.default_rng(random_state)

    gender = rng.binomial(1, group_proportion, size=n_samples)

    qualification = np.clip(
        rng.normal(loc=0.55 + 0.05 * gender, scale=0.18, size=n_samples),
        0.0,
        1.0,
    )
    experience = np.clip(
        rng.normal(loc=0.50 + 0.03 * gender, scale=0.20, size=n_samples),
        0.0,
        1.0,
    )

    score = 1.7 * qualification + 1.1 * experience - 1.2
    prob = _sigmoid(score)
    label = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "gender": gender.astype(int),
            "qualification": qualification,
            "experience": experience,
            "score": score,
            "label": label.astype(int),
        }
    )
