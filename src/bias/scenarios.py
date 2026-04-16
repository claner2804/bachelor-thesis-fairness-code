from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScenarioConfig:
    name: str
    group_proportion: float = 0.5
    imbalance_target: float | None = None
    label_bias_strength: float = 0.0
    favored_group: int = 1


DEFAULT_SCENARIOS = [
    ScenarioConfig(name="baseline", group_proportion=0.5, imbalance_target=None, label_bias_strength=0.0),
    ScenarioConfig(name="imbalance_low", group_proportion=0.5, imbalance_target=0.4, label_bias_strength=0.0),
    ScenarioConfig(name="imbalance_high", group_proportion=0.5, imbalance_target=0.2, label_bias_strength=0.0),
    ScenarioConfig(name="label_bias_low", group_proportion=0.5, imbalance_target=None, label_bias_strength=0.1),
    ScenarioConfig(name="label_bias_high", group_proportion=0.5, imbalance_target=None, label_bias_strength=0.25),
    ScenarioConfig(name="combined", group_proportion=0.5, imbalance_target=0.25, label_bias_strength=0.25),
]
