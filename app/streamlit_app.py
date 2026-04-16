from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.experiments.runner import run_single_experiment
from src.visualization.plots_data import plot_group_distribution, plot_label_distribution
from src.visualization.plots_model import plot_scatter_predictions
from src.visualization.plots_fairness import plot_group_metric_bars


st.set_page_config(page_title="Fairness Bias Simulation", layout="wide")
st.title("Fairness in Machine Learning")
st.subheader("Visuelle Simulation von Bias-Effekten")

st.sidebar.header("Parameter")
n_samples = st.sidebar.slider("Anzahl Datenpunkte", 100, 3000, 500, 100)
group_proportion = st.sidebar.slider("Anteil Gruppe 1", 0.1, 0.9, 0.5, 0.05)
imbalance_enabled = st.sidebar.checkbox("Gruppenungleichgewicht aktivieren", value=False)
imbalance_target = st.sidebar.slider("Zielanteil Gruppe 1", 0.1, 0.9, 0.3, 0.05) if imbalance_enabled else None
label_bias_strength = st.sidebar.slider("Label-Bias Stärke", 0.0, 0.5, 0.0, 0.05)
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)
include_sensitive_feature = st.sidebar.checkbox("Gender als Feature verwenden", value=False)
random_state = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)

result = run_single_experiment(
    n_samples=n_samples,
    group_proportion=group_proportion,
    imbalance_target=imbalance_target,
    label_bias_strength=label_bias_strength,
    threshold=threshold,
    random_state=int(random_state),
    include_sensitive_feature=include_sensitive_feature,
)

df = result["df"]
performance = result["performance"]
fairness = result["fairness"]
group_summary = result["group_summary"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{performance['accuracy']:.3f}")
col2.metric("Precision", f"{performance['precision']:.3f}")
col3.metric("Demographic Parity Diff", f"{fairness['demographic_parity_difference']:.3f}")
col4.metric("Equal Opportunity Diff", f"{fairness['equal_opportunity_difference']:.3f}")

st.markdown("---")
left, right = st.columns(2)

with left:
    st.pyplot(plot_group_distribution(df))
    st.pyplot(plot_label_distribution(df))

with right:
    st.pyplot(plot_scatter_predictions(df))
    st.pyplot(plot_group_metric_bars(group_summary))

st.markdown("### Gruppenübersicht")
st.dataframe(group_summary, use_container_width=True)

st.markdown("""
**Didaktische Interpretation**
- Demographic Parity Difference misst Unterschiede in den positiven Vorhersageraten.
- Equal Opportunity Difference misst Unterschiede in den True-Positive-Raten.
- Werte nahe **0** deuten auf ähnliche Behandlung der Gruppen innerhalb der jeweiligen Metrik hin.
""")
