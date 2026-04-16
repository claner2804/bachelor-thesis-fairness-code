from __future__ import annotations

import numpy as np


def predict_with_threshold(model, X, threshold: float = 0.5):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return probabilities, predictions
