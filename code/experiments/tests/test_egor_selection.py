from __future__ import annotations

import json
from itertools import product
from math import isclose

from openagora_experiments.cago.linear import RidgeModel
from openagora_experiments.egor.protocol import PilotTask, parse_baseline_output
from openagora_experiments.egor.selection import (
    REPAIR_FEATURE_NAMES,
    CounterfactualRepairSelector,
    RepairSelectionConfig,
    RepairTrainingExample,
    evaluate_selector,
    select_config_by_source_cv,
    visible_repair_feature_vector,
)


def _task(*, source_id: str = "source-a", gold: str = "hidden-a") -> PilotTask:
    return PilotTask(
        task_id="task-1",
        source_id=source_id,
        question="What percentage value is shown in the chart?",
        image_path="/tmp/image.png",
        mime_type="image/png",
        metric="accuracy",
        answer=gold,
        reference_answers=(gold,),
        options=(),
        metadata={"primary_type": source_id},
    )


def _baseline() -> dict:
    return {
        "answer": "17%",
        "confidence": 0.4,
        "demands": [
            {
                "demand_id": "d1",
                "type": "exact_span",
                "requirement": "read the chart label",
                "weight": 1.0,
            }
        ],
        "evidence": [],
        "open_gap_types": ["exact_span"],
    }


def _features(signal: float) -> tuple[float, ...]:
    values = [0.0] * len(REPAIR_FEATURE_NAMES)
    values[0] = signal
    values[REPAIR_FEATURE_NAMES.index("gap_max")] = 1.0
    values[REPAIR_FEATURE_NAMES.index("gap_sum")] = 1.0
    values[REPAIR_FEATURE_NAMES.index("gap_count")] = 1.0
    return tuple(values)


def _examples() -> list[RepairTrainingExample]:
    examples = []
    for source_index, source in enumerate(("s1", "s2", "s3")):
        for offset, signal in enumerate((-1.0, -0.8, 0.8, 1.0)):
            examples.append(
                RepairTrainingExample(
                    source_id=source,
                    task_id=f"{source}-{offset}",
                    features=_features(signal),
                    gap_max=1.0,
                    strict_advantages={
                        "exact": 0.3 if signal > 0 else -0.1,
                        "visual": 0.3 if signal < 0 else -0.1,
                    },
                    incremental_calls={"exact": 2.0, "visual": 2.0},
                )
            )
    return examples


def test_visible_repair_features_exclude_gold_source_and_metadata() -> None:
    first = visible_repair_feature_vector(
        _task(source_id="private-source-a", gold="answer-a"), _baseline()
    )
    second = visible_repair_feature_vector(
        _task(source_id="private-source-b", gold="answer-b"), _baseline()
    )

    assert len(REPAIR_FEATURE_NAMES) == 40
    assert len(first) == len(REPAIR_FEATURE_NAMES)
    assert first == second
    assert first[REPAIR_FEATURE_NAMES.index("gap::exact_span")] == 1.0


def test_feature_counts_receive_only_parser_validated_evidence() -> None:
    payload = {
        "answer": "17%",
        "confidence": 1.0,
        "demands": [
            {
                "demand_id": "d1",
                "type": "exact_span",
                "requirement": "read the chart label",
                "weight": 1.0,
            }
        ],
        "evidence": [
            {
                "evidence_id": "candidate-card",
                "type": "exact_span",
                "claim": "17%",
                "support": "",
                "provenance": "chart-region",
                "confidence": 1.0,
            }
        ],
        "open_gap_types": [],
    }
    parsed_invalid, issues = parse_baseline_output(json.dumps(payload))
    invalid = visible_repair_feature_vector(_task(), parsed_invalid)
    assert issues
    assert invalid[REPAIR_FEATURE_NAMES.index("evidence_count")] == 0.0
    assert invalid[REPAIR_FEATURE_NAMES.index("evidence::exact_span")] == 0.0
    assert invalid[REPAIR_FEATURE_NAMES.index("gap::exact_span")] == 1.0

    payload["evidence"][0]["support"] = "The chart label reads 17%."
    parsed_valid, issues = parse_baseline_output(json.dumps(payload))
    valid = visible_repair_feature_vector(_task(), parsed_valid)
    assert not issues
    assert valid[REPAIR_FEATURE_NAMES.index("evidence_count")] == 0.2
    assert valid[REPAIR_FEATURE_NAMES.index("evidence::exact_span")] == 0.2
    assert valid[REPAIR_FEATURE_NAMES.index("gap::exact_span")] == 0.0


class _ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features: tuple[float, ...]) -> float:
        del features
        return self.value


def test_selector_implements_finite_catalog_stability_bound() -> None:
    epsilon_delta = 0.03
    epsilon_cost = 0.2
    cost_weight = 0.1
    epsilon = epsilon_delta + cost_weight * epsilon_cost
    true_costs = {"exact": 1.0, "visual": 2.0}

    for threshold in (0.0, 0.05, 0.2):
        for true_values in product((-0.2, 0.0, 0.2), repeat=2):
            for delta_errors in product((-epsilon_delta, 0.0, epsilon_delta), repeat=2):
                for cost_errors in product((-epsilon_cost, 0.0, epsilon_cost), repeat=2):
                    true_advantages = dict(zip(true_costs, true_values, strict=True))
                    predicted_advantages = {
                        action: true_advantages[action] + error
                        for action, error in zip(true_costs, delta_errors, strict=True)
                    }
                    estimated_costs = {
                        action: true_costs[action] + error
                        for action, error in zip(true_costs, cost_errors, strict=True)
                    }
                    selector = CounterfactualRepairSelector(
                        config=RepairSelectionConfig(
                            l2=1.0,
                            threshold=threshold,
                            cost_weight=cost_weight,
                            minimum_gap=0.0,
                        ),
                        models={
                            action: _ConstantModel(value)  # type: ignore[arg-type]
                            for action, value in predicted_advantages.items()
                        },
                        estimated_incremental_calls=estimated_costs,
                    )

                    decision = selector.select(_features(0.0), gap_max=1.0)
                    true_utilities = {
                        action: true_advantages[action] - cost_weight * true_costs[action]
                        for action in true_costs
                    }
                    true_utilities["baseline"] = 0.0
                    regret = max(true_utilities.values()) - true_utilities[decision.selected_id]
                    bound = max(2.0 * epsilon, threshold + epsilon)

                    assert regret <= bound + 1e-12


def test_selector_routes_complementary_repairs_and_abstains_without_gap() -> None:
    config = RepairSelectionConfig(l2=0.1, threshold=0.01, cost_weight=0.0)
    selector = CounterfactualRepairSelector.fit(_examples(), config=config)

    assert selector.select(_features(1.0), gap_max=1.0).selected_id == "exact"
    assert selector.select(_features(-1.0), gap_max=1.0).selected_id == "visual"
    no_gap = selector.select(_features(1.0), gap_max=0.0)
    assert no_gap.selected_id == "baseline"
    assert no_gap.reason == "no_residual_gap"


def test_selector_serialization_round_trip_preserves_decision() -> None:
    selector = CounterfactualRepairSelector.fit(
        _examples(),
        config=RepairSelectionConfig(l2=1.0, threshold=0.01, cost_weight=0.005),
    )
    restored = CounterfactualRepairSelector.from_dict(selector.to_dict())

    assert restored.to_dict() == selector.to_dict()
    assert restored.select(_features(-0.9), gap_max=1.0) == selector.select(
        _features(-0.9), gap_max=1.0
    )


def test_standalone_quality_control_predicts_absolute_candidate_value() -> None:
    examples = [
        RepairTrainingExample(
            source_id=f"s{index}",
            task_id=f"t{index}",
            features=_features(0.0),
            gap_max=1.0,
            strict_advantages={"exact": 0.2, "visual": -0.1},
            incremental_calls={"exact": 2.0, "visual": 2.0},
            baseline_quality=0.6,
        )
        for index in range(3)
    ]
    selector = CounterfactualRepairSelector.fit(
        examples,
        config=RepairSelectionConfig(l2=1.0, threshold=0.7, cost_weight=0.0),
        target="standalone_quality",
    )
    decision = selector.select(_features(0.0), gap_max=1.0)
    restored = CounterfactualRepairSelector.from_dict(selector.to_dict())

    assert selector.target == "standalone_quality"
    assert decision.selected_id == "exact"
    assert abs(decision.predicted_advantages["exact"] - 0.8) < 1e-9
    assert restored.to_dict() == selector.to_dict()


def test_same_design_ridge_subtraction_recovers_direct_paired_head() -> None:
    signals = (-1.5, -0.5, 0.5, 1.5)
    baseline_values = (0.2, 0.8, 0.4, 0.7)
    advantages = (0.4, -0.2, 0.1, -0.1)
    feature_rows = [_features(signal) for signal in signals]
    examples = [
        RepairTrainingExample(
            source_id=f"s{index}",
            task_id=f"t{index}",
            features=features,
            gap_max=1.0,
            strict_advantages={"exact": advantage},
            incremental_calls={"exact": 2.0},
            baseline_quality=baseline,
        )
        for index, (features, baseline, advantage) in enumerate(
            zip(feature_rows, baseline_values, advantages, strict=True)
        )
    ]
    config = RepairSelectionConfig(l2=1.0, threshold=0.0, cost_weight=0.0)
    paired = CounterfactualRepairSelector.fit(examples, config=config)
    repaired = CounterfactualRepairSelector.fit(
        examples,
        config=config,
        target="standalone_quality",
    )
    baseline = RidgeModel.fit(feature_rows, baseline_values, l2=config.l2)
    paired_head = paired.models["exact"]
    repaired_head = repaired.models["exact"]

    assert repaired_head.feature_means == baseline.feature_means
    assert repaired_head.feature_scales == baseline.feature_scales
    for repaired_weight, baseline_weight, paired_weight in zip(
        repaired_head.weights,
        baseline.weights,
        paired_head.weights,
        strict=True,
    ):
        assert isclose(
            repaired_weight - baseline_weight,
            paired_weight,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    for features in feature_rows:
        assert isclose(
            repaired_head.predict(features) - baseline.predict(features),
            paired_head.predict(features),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )


def test_selector_breaks_equal_advantage_ties_by_minimal_call_cost() -> None:
    examples = [
        RepairTrainingExample(
            source_id="s1",
            task_id=f"t{index}",
            features=_features(0.0),
            gap_max=1.0,
            strict_advantages={"large_team": 0.2, "small_team": 0.2},
            incremental_calls={"large_team": 3.0, "small_team": 2.0},
        )
        for index in range(3)
    ]
    selector = CounterfactualRepairSelector.fit(
        examples,
        config=RepairSelectionConfig(l2=1.0, threshold=0.0, cost_weight=0.0),
    )

    assert selector.select(_features(0.0), gap_max=1.0).selected_id == "small_team"


def test_forced_repair_ablation_bypasses_gap_and_abstention_only() -> None:
    selector = CounterfactualRepairSelector.fit(
        _examples(),
        config=RepairSelectionConfig(l2=0.1, threshold=10.0, cost_weight=0.0),
    )

    ordinary = selector.select(_features(1.0), gap_max=0.0)
    forced = selector.select(_features(1.0), gap_max=0.0, force_repair=True)

    assert ordinary.selected_id == "baseline"
    assert forced.selected_id == "exact"
    assert forced.reason == "forced_best_repair"
    assert forced.predicted_advantages == ordinary.predicted_advantages
    assert forced.net_advantage_scores == ordinary.net_advantage_scores


def test_source_grouped_grid_selection_is_positive_and_safe() -> None:
    selected, choices = select_config_by_source_cv(
        _examples(),
        l2_values=(0.1, 1.0),
        thresholds=(0.0, 0.2, 0.5),
        cost_weights=(0.0, 0.01),
        evaluation_call_price=0.01,
        maximum_harm_rate=0.05,
        minimum_repair_rate=0.05,
    )
    selector = CounterfactualRepairSelector.fit(_examples(), config=selected.config)
    summary = evaluate_selector(selector, _examples())

    assert choices
    assert selected.eligible
    assert selected.mean_quality_delta > 0.0
    assert selected.strict_harm_rate == 0.0
    assert summary.selection_counts == {"exact": 6, "visual": 6}

    standalone, _ = select_config_by_source_cv(
        _examples(),
        l2_values=(0.1, 1.0),
        thresholds=(0.0, 0.01),
        cost_weights=(0.0, 0.01),
        evaluation_call_price=0.01,
        maximum_harm_rate=0.05,
        minimum_repair_rate=0.05,
        target="standalone_quality",
    )
    assert standalone.eligible

    forced, _ = select_config_by_source_cv(
        _examples(),
        l2_values=(0.1, 1.0),
        thresholds=(0.0, 10.0),
        cost_weights=(0.0,),
        evaluation_call_price=0.01,
        maximum_harm_rate=0.50,
        minimum_repair_rate=1.0,
        force_repair=True,
    )
    forced_summary = evaluate_selector(
        CounterfactualRepairSelector.fit(_examples(), config=forced.config),
        _examples(),
        force_repair=True,
    )
    assert forced_summary.repair_rate == 1.0
