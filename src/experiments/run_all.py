from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.bias.scenarios import DEFAULT_SCENARIOS
from src.experiments.runner import run_single_experiment
from src.experiments.export_results import export_dataframe


SCENARIO_IDS = ["A", "B", "C", "D", "E", "F"]


def _get_group_value(group_summary: pd.DataFrame, group: int, column: str) -> float:
    values = group_summary.loc[group_summary["group"] == group, column]
    return float(values.iloc[0]) if len(values) else 0.0


def build_results_dataframe() -> pd.DataFrame:
    rows = []

    for scenario_id, scenario in zip(SCENARIO_IDS, DEFAULT_SCENARIOS):
        result = run_single_experiment(
            n_samples=500,
            group_proportion=scenario.group_proportion,
            imbalance_target=scenario.imbalance_target,
            label_bias_strength=scenario.label_bias_strength,
            threshold=0.5,
            random_state=42,
            include_sensitive_feature=False,
        )

        trained = result["trained"]

        y_test = trained["y_test"].reset_index(drop=True)
        g_test = trained["g_test"].reset_index(drop=True)

        group_summary = result["group_summary"]

        basis_a = float(y_test[g_test == 0].mean()) if (g_test == 0).any() else 0.0
        basis_b = float(y_test[g_test == 1].mean()) if (g_test == 1).any() else 0.0

        rows.append(
            {
                "scenario_id": scenario_id,
                "scenario": scenario.name,
                "n_total_after_bias": len(result["df"]),
                "n_test": len(y_test),
                "imbalance_target": scenario.imbalance_target,
                "label_bias_strength": scenario.label_bias_strength,
                "basis_a": basis_a,
                "basis_b": basis_b,
                "basis_diff": basis_b - basis_a,
                "pr_a": _get_group_value(
                    group_summary, 0, "positive_prediction_rate"
                ),
                "pr_b": _get_group_value(
                    group_summary, 1, "positive_prediction_rate"
                ),
                "tpr_a": _get_group_value(
                    group_summary, 0, "true_positive_rate"
                ),
                "tpr_b": _get_group_value(
                    group_summary, 1, "true_positive_rate"
                ),
                **result["performance"],
                **result["fairness"],
            }
        )

    return pd.DataFrame(rows)


def export_figures(results: pd.DataFrame) -> None:
    Path("figures").mkdir(exist_ok=True)

    labels = results["scenario_id"].tolist()

    # 1) Basisraten
    plt.figure()
    plt.plot(labels, results["basis_a"], marker="o", label="Gruppe A")
    plt.plot(labels, results["basis_b"], marker="o", label="Gruppe B")
    plt.xlabel("Szenario")
    plt.ylabel("Basisrate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/fig_basisraten.png", dpi=300)
    plt.close()

    # 2) DPD und EOD
    plt.figure()
    x = range(len(results))
    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        results["demographic_parity_difference"],
        width=width,
        label="DPD",
    )
    plt.bar(
        [i + width / 2 for i in x],
        results["equal_opportunity_difference"],
        width=width,
        label="EOD",
    )
    plt.axhline(0.10, linestyle="--")
    plt.axhline(-0.10, linestyle="--")
    plt.xticks(list(x), labels)
    plt.xlabel("Szenario")
    plt.ylabel("Metrikwert")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/fig_dpd_eod.png", dpi=300)
    plt.close()

    # 3) PR und TPR
    plt.figure()
    plt.plot(labels, results["pr_a"], marker="o", label="PR Gruppe A")
    plt.plot(labels, results["pr_b"], marker="o", label="PR Gruppe B")
    plt.plot(labels, results["tpr_a"], marker="s", label="TPR Gruppe A")
    plt.plot(labels, results["tpr_b"], marker="s", label="TPR Gruppe B")
    plt.xlabel("Szenario")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/fig_pr_tpr.png", dpi=300)
    plt.close()

    # 4) Accuracy
    plt.figure()
    plt.bar(labels, results["accuracy"])
    plt.axhline(0.5, linestyle="--")
    plt.xlabel("Szenario")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("figures/fig_accuracy.png", dpi=300)
    plt.close()

    # 5) DPD-EOD-Raum
    plt.figure()
    plt.scatter(
        results["demographic_parity_difference"],
        results["equal_opportunity_difference"],
    )

    for _, row in results.iterrows():
        plt.annotate(
            row["scenario_id"],
            (
                row["demographic_parity_difference"],
                row["equal_opportunity_difference"],
            ),
        )

    plt.scatter([0], [0], marker="*", s=150)
    plt.xlabel("Demographic Parity Difference")
    plt.ylabel("Equal Opportunity Difference")
    plt.tight_layout()
    plt.savefig("figures/fig_impossibility.png", dpi=300)
    plt.close()


def main() -> None:
    results = build_results_dataframe()

    export_dataframe(results, "outputs/results.csv")
    export_figures(results)

    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()