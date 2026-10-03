"""Fit, calibrate, and evaluate CAGO on a complete paired outcome matrix."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import mean

from openagora.mechanisms.conservative_adaptation import (
    CalibrationExample,
    ConservativeOrganizationSelector,
    OrganizationOption,
    SimultaneousLowerBoundCalibrator,
)

from openagora_experiments.cago.data import (
    CounterfactualMatrix,
    MatrixValidationError,
    OutcomeRecord,
    TaskKey,
)
from openagora_experiments.cago.features import FEATURE_NAMES, visible_feature_vector
from openagora_experiments.cago.linear import AdvantagePredictor, RidgeModel
from openagora_experiments.cago.splits import SplitAssignment


@dataclass(frozen=True)
class UtilitySpec:
    """Preregistered conversion from raw quality/resources to net utility."""

    cost_weight: float = 0.0
    token_weight: float = 0.0
    latency_weight: float = 0.0
    token_scale: float = 10_000.0
    latency_scale: float = 60.0

    def __post_init__(self) -> None:
        if min(self.cost_weight, self.token_weight, self.latency_weight) < 0.0:
            raise MatrixValidationError("utility weights must be non-negative")
        if self.token_scale <= 0.0 or self.latency_scale <= 0.0:
            raise MatrixValidationError("utility scales must be positive")

    def score(self, outcome: OutcomeRecord) -> float:
        return (
            outcome.quality
            - self.cost_weight * outcome.cost_usd
            - self.token_weight * outcome.tokens / self.token_scale
            - self.latency_weight * outcome.latency_seconds / self.latency_scale
        )


@dataclass(frozen=True)
class PolicySummary:
    policy: str
    count: int
    mean_quality: float
    mean_utility: float
    mean_cost_usd: float
    mean_tokens: float
    mean_latency_seconds: float
    quality_gain_from_baseline: float
    utility_gain_from_baseline: float
    switch_rate: float
    harmful_switch_rate_all_tasks: float
    harmful_switch_rate_given_switch: float
    strict_negative_switch_rate_all_tasks: float
    strict_negative_switch_rate_given_switch: float
    mean_downside: float
    mean_upside: float


@dataclass(frozen=True)
class DecisionRecord:
    source_id: str
    task_id: str
    baseline_id: str
    selected_id: str
    switched: bool
    predicted_advantages: Mapping[str, float]
    lower_bounds: Mapping[str, float]
    true_advantages: Mapping[str, float]
    selected_lower_bound: float
    realized_advantage: float
    simultaneous_covered: bool
    reason: str

    def to_csv_row(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "task_id": self.task_id,
            "baseline_id": self.baseline_id,
            "selected_id": self.selected_id,
            "switched": int(self.switched),
            "selected_lower_bound": self.selected_lower_bound,
            "realized_advantage": self.realized_advantage,
            "simultaneous_covered": int(self.simultaneous_covered),
            "reason": self.reason,
            "predicted_advantages_json": json.dumps(self.predicted_advantages, sort_keys=True),
            "lower_bounds_json": json.dumps(self.lower_bounds, sort_keys=True),
            "true_advantages_json": json.dumps(self.true_advantages, sort_keys=True),
        }


@dataclass(frozen=True)
class EvaluationResult:
    baseline_id: str
    best_static_id: str
    best_static_by_source: Mapping[str, str]
    candidate_ids: tuple[str, ...]
    alpha: float
    margin: float
    calibration_size: int
    calibration_correction: float
    empirical_simultaneous_coverage: float
    simultaneous_coverage_ci_low: float
    simultaneous_coverage_ci_high: float
    summaries: tuple[PolicySummary, ...]
    cago_decisions: tuple[DecisionRecord, ...]
    model_artifact: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_id": self.baseline_id,
            "best_static_id": self.best_static_id,
            "best_static_by_source": dict(self.best_static_by_source),
            "candidate_ids": list(self.candidate_ids),
            "alpha": self.alpha,
            "margin": self.margin,
            "calibration_size": self.calibration_size,
            "calibration_correction": self.calibration_correction,
            "empirical_simultaneous_coverage": self.empirical_simultaneous_coverage,
            "simultaneous_coverage_ci_low": self.simultaneous_coverage_ci_low,
            "simultaneous_coverage_ci_high": self.simultaneous_coverage_ci_high,
            "summaries": [asdict(summary) for summary in self.summaries],
            "model_artifact": dict(self.model_artifact),
        }


def run_cago_evaluation(
    *,
    matrix: CounterfactualMatrix,
    split: SplitAssignment,
    baseline_id: str,
    utility: UtilitySpec,
    alpha: float = 0.1,
    margin: float = 0.0,
    ridge_l2: float = 1.0,
) -> EvaluationResult:
    """Run the frozen library-only CAGO algorithm and required diagnostic policies."""

    candidate_ids = _validate_cago_inputs(matrix, split, baseline_id)

    predictor = _fit_predictor(
        matrix=matrix,
        keys=split.train,
        baseline_id=baseline_id,
        candidate_ids=candidate_ids,
        utility=utility,
        ridge_l2=ridge_l2,
    )
    calibration_examples = [
        CalibrationExample(
            predicted_advantages=predictor.predict(visible_feature_vector(matrix.tasks[key])),
            observed_advantages=_true_advantages(
                matrix=matrix,
                key=key,
                baseline_id=baseline_id,
                candidate_ids=candidate_ids,
                utility=utility,
            ),
        )
        for key in split.calibration
    ]
    calibrator = SimultaneousLowerBoundCalibrator.fit(calibration_examples, alpha=alpha)
    estimated_costs = {
        candidate_id: mean(matrix.outcome(key, candidate_id).cost_usd for key in split.train)
        for candidate_id in candidate_ids
    }
    options = [
        OrganizationOption(
            candidate_id=candidate_id,
            estimated_cost=estimated_costs[candidate_id],
            edit_distance=1,
        )
        for candidate_id in candidate_ids
    ]
    selector = ConservativeOrganizationSelector(min_advantage=margin)
    best_static_id = _best_static(
        matrix=matrix,
        keys=split.train,
        utility=utility,
    )
    best_static_by_source = _best_static_by_source(
        matrix=matrix,
        keys=split.train,
        utility=utility,
    )

    selected_by_policy, decisions = _evaluate_policies(
        matrix=matrix,
        split=split,
        predictor=predictor,
        calibrator=calibrator,
        selector=selector,
        options=options,
        baseline_id=baseline_id,
        candidate_ids=candidate_ids,
        utility=utility,
        margin=margin,
        best_static_id=best_static_id,
        best_static_by_source=best_static_by_source,
    )

    summaries = tuple(
        _summarize_policy(
            policy=policy,
            selections=selections,
            matrix=matrix,
            keys=split.test,
            baseline_id=baseline_id,
            utility=utility,
            margin=margin,
        )
        for policy, selections in selected_by_policy.items()
    )
    coverage = mean(float(decision.simultaneous_covered) for decision in decisions)
    coverage_low, coverage_high = _wilson_interval(
        successes=sum(decision.simultaneous_covered for decision in decisions),
        count=len(decisions),
    )
    model_artifact = {
        "feature_names": list(FEATURE_NAMES),
        "ridge_l2": ridge_l2,
        "models": {candidate_id: predictor.models[candidate_id].to_dict() for candidate_id in predictor.candidate_ids},
    }
    return EvaluationResult(
        baseline_id=baseline_id,
        best_static_id=best_static_id,
        best_static_by_source=best_static_by_source,
        candidate_ids=candidate_ids,
        alpha=alpha,
        margin=margin,
        calibration_size=calibrator.calibration_size,
        calibration_correction=calibrator.correction,
        empirical_simultaneous_coverage=coverage,
        simultaneous_coverage_ci_low=coverage_low,
        simultaneous_coverage_ci_high=coverage_high,
        summaries=summaries,
        cago_decisions=tuple(decisions),
        model_artifact=model_artifact,
    )


def _fit_predictor(
    *,
    matrix: CounterfactualMatrix,
    keys: Sequence[TaskKey],
    baseline_id: str,
    candidate_ids: Sequence[str],
    utility: UtilitySpec,
    ridge_l2: float,
) -> AdvantagePredictor:
    features = [visible_feature_vector(matrix.tasks[key]) for key in keys]
    models: dict[str, RidgeModel] = {}
    for candidate_id in candidate_ids:
        targets = [
            utility.score(matrix.outcome(key, candidate_id)) - utility.score(matrix.outcome(key, baseline_id))
            for key in keys
        ]
        models[candidate_id] = RidgeModel.fit(features, targets, l2=ridge_l2)
    return AdvantagePredictor(baseline_id=baseline_id, models=models)


def _true_advantages(
    *,
    matrix: CounterfactualMatrix,
    key: TaskKey,
    baseline_id: str,
    candidate_ids: Sequence[str],
    utility: UtilitySpec,
) -> dict[str, float]:
    baseline_utility = utility.score(matrix.outcome(key, baseline_id))
    return {
        candidate_id: utility.score(matrix.outcome(key, candidate_id)) - baseline_utility
        for candidate_id in candidate_ids
    }


def _best_static(
    *,
    matrix: CounterfactualMatrix,
    keys: Sequence[TaskKey],
    utility: UtilitySpec,
) -> str:
    return min(
        matrix.organization_ids,
        key=lambda organization_id: (
            -mean(utility.score(matrix.outcome(key, organization_id)) for key in keys),
            organization_id,
        ),
    )


def _best_static_by_source(
    *,
    matrix: CounterfactualMatrix,
    keys: Sequence[TaskKey],
    utility: UtilitySpec,
) -> dict[str, str]:
    by_source = {key.source_id for key in keys}
    return {
        source_id: _best_static(
            matrix=matrix,
            keys=tuple(key for key in keys if key.source_id == source_id),
            utility=utility,
        )
        for source_id in sorted(by_source)
    }


def _select_predicted(
    *,
    predicted: Mapping[str, float],
    baseline_id: str,
    margin: float,
) -> str:
    candidate_id = min(predicted, key=lambda item: (-predicted[item], item))
    return candidate_id if predicted[candidate_id] > margin else baseline_id


def _select_oracle(
    *,
    matrix: CounterfactualMatrix,
    key: TaskKey,
    utility: UtilitySpec,
) -> str:
    return min(
        matrix.organization_ids,
        key=lambda organization_id: (
            -utility.score(matrix.outcome(key, organization_id)),
            matrix.outcome(key, organization_id).cost_usd,
            organization_id,
        ),
    )


def _summarize_policy(
    *,
    policy: str,
    selections: Mapping[TaskKey, str],
    matrix: CounterfactualMatrix,
    keys: Sequence[TaskKey],
    baseline_id: str,
    utility: UtilitySpec,
    margin: float,
) -> PolicySummary:
    selected = [matrix.outcome(key, selections[key]) for key in keys]
    baselines = [matrix.outcome(key, baseline_id) for key in keys]
    switches = [selections[key] != baseline_id for key in keys]
    realized_advantages = [
        utility.score(matrix.outcome(key, selections[key])) - utility.score(matrix.outcome(key, baseline_id))
        for key in keys
    ]
    harmful, strict_negative = _switch_effects(switches, realized_advantages, margin)
    switch_count = sum(switches)
    return PolicySummary(
        policy=policy,
        count=len(keys),
        mean_quality=_mean_quality(selected),
        mean_utility=mean(utility.score(outcome) for outcome in selected),
        mean_cost_usd=mean(outcome.cost_usd for outcome in selected),
        mean_tokens=mean(outcome.tokens for outcome in selected),
        mean_latency_seconds=mean(outcome.latency_seconds for outcome in selected),
        quality_gain_from_baseline=_mean_quality(selected) - _mean_quality(baselines),
        utility_gain_from_baseline=mean(realized_advantages),
        switch_rate=switch_count / len(keys),
        harmful_switch_rate_all_tasks=sum(harmful) / len(keys),
        harmful_switch_rate_given_switch=sum(harmful) / switch_count if switch_count else 0.0,
        strict_negative_switch_rate_all_tasks=sum(strict_negative) / len(keys),
        strict_negative_switch_rate_given_switch=sum(strict_negative) / switch_count if switch_count else 0.0,
        mean_downside=mean(max(0.0, -advantage) for advantage in realized_advantages),
        mean_upside=mean(max(0.0, advantage) for advantage in realized_advantages),
    )


def _wilson_interval(*, successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 0.0
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    radius = z * (proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)) ** 0.5 / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _evaluate_policies(
    *,
    matrix: CounterfactualMatrix,
    split: SplitAssignment,
    predictor: AdvantagePredictor,
    calibrator: SimultaneousLowerBoundCalibrator,
    selector: ConservativeOrganizationSelector,
    options: Sequence[OrganizationOption],
    baseline_id: str,
    candidate_ids: Sequence[str],
    utility: UtilitySpec,
    margin: float,
    best_static_id: str,
    best_static_by_source: Mapping[str, str],
) -> tuple[dict[str, dict[TaskKey, str]], list[DecisionRecord]]:
    selected_by_policy: dict[str, dict[TaskKey, str]] = {
        "baseline": {},
        "best_static": {},
        "best_static_by_source": {},
        "uncalibrated_advantage": {},
        "cago": {},
        "oracle": {},
    }
    decisions: list[DecisionRecord] = []
    for key in split.test:
        features = visible_feature_vector(matrix.tasks[key])
        predicted = predictor.predict(features)
        lower_bounds = calibrator.lower_bounds(predicted)
        true_advantages = _true_advantages(
            matrix=matrix,
            key=key,
            baseline_id=baseline_id,
            candidate_ids=candidate_ids,
            utility=utility,
        )
        decision = selector.select(
            baseline_id=baseline_id,
            options=options,
            predicted_advantages=predicted,
            calibrator=calibrator,
        )
        uncalibrated_id = _select_predicted(
            predicted=predicted,
            baseline_id=baseline_id,
            margin=margin,
        )
        oracle_id = _select_oracle(matrix=matrix, key=key, utility=utility)
        selected_by_policy["baseline"][key] = baseline_id
        selected_by_policy["best_static"][key] = best_static_id
        selected_by_policy["best_static_by_source"][key] = best_static_by_source[key.source_id]
        selected_by_policy["uncalibrated_advantage"][key] = uncalibrated_id
        selected_by_policy["cago"][key] = decision.selected_id
        selected_by_policy["oracle"][key] = oracle_id
        covered = all(true_advantages[candidate_id] >= lower_bounds[candidate_id] for candidate_id in candidate_ids)
        realized_advantage = true_advantages[decision.selected_id] if decision.switched else 0.0
        decisions.append(
            DecisionRecord(
                source_id=key.source_id,
                task_id=key.task_id,
                baseline_id=baseline_id,
                selected_id=decision.selected_id,
                switched=decision.switched,
                predicted_advantages=dict(predicted),
                lower_bounds=dict(lower_bounds),
                true_advantages=dict(true_advantages),
                selected_lower_bound=decision.selected_lower_bound,
                realized_advantage=realized_advantage,
                simultaneous_covered=covered,
                reason=decision.reason,
            )
        )

    return selected_by_policy, decisions


def _switch_effects(
    switches: Sequence[bool], realized_advantages: Sequence[float], margin: float
) -> tuple[list[bool], list[bool]]:
    harmful = [
        switched and advantage <= margin for switched, advantage in zip(switches, realized_advantages, strict=True)
    ]
    strict_negative = [
        switched and advantage < 0.0 for switched, advantage in zip(switches, realized_advantages, strict=True)
    ]
    return harmful, strict_negative


def _validate_cago_inputs(matrix: CounterfactualMatrix, split: SplitAssignment, baseline_id: str) -> tuple[str, ...]:
    if baseline_id not in matrix.organization_ids:
        raise MatrixValidationError(f"unknown baseline organization: {baseline_id}")
    candidate_ids = tuple(item for item in matrix.organization_ids if item != baseline_id)
    if not candidate_ids:
        raise MatrixValidationError("CAGO requires at least one alternative organization")
    all_split_keys = set((*split.train, *split.calibration, *split.test))
    unknown_keys = all_split_keys.difference(matrix.tasks)
    if unknown_keys:
        raise MatrixValidationError(f"split contains {len(unknown_keys)} tasks absent from matrix")

    return candidate_ids


def _mean_quality(outcomes: Sequence[OutcomeRecord]) -> float:
    return mean(outcome.quality for outcome in outcomes)
