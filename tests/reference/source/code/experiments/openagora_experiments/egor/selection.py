"""Source-blind counterfactual selection of minimal EGOR repair organizations."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Literal, cast

from openagora.mechanisms.evidence_gap_repair import EvidenceType
from openagora_experiments.cago.linear import RidgeModel
from openagora_experiments.egor.evaluation import residual_gap_vector
from openagora_experiments.egor.protocol import PilotTask


_EVIDENCE_TYPES = tuple(EvidenceType)
_OCR_TERMS = ("text", "word", "name", "written", "read", "label", "title", "number")
_CHART_TERMS = ("chart", "graph", "plot", "trend", "axis", "table", "figure", "value")
_MATH_TERMS = (
    "calculate",
    "equation",
    "sum",
    "difference",
    "ratio",
    "probability",
    "volume",
    "mass",
    "rate",
    "average",
)
_SPATIAL_TERMS = (
    "above",
    "below",
    "left",
    "right",
    "behind",
    "front",
    "near",
    "between",
)
_UNIT_PATTERN = re.compile(
    r"(?:\bkg\b|\bkm\b|\bcm\b|\bmm\b|\bmeters?\b|\bseconds?\b|\bdollars?\b|\$)",
    flags=re.IGNORECASE,
)


REPAIR_FEATURE_NAMES = (
    "question_chars",
    "question_words",
    "digit_tokens",
    "math_symbols",
    "option_count",
    "option_text_chars",
    "is_multiple_choice",
    "has_percent",
    "has_units",
    "ocr_term_hits",
    "chart_term_hits",
    "math_term_hits",
    "spatial_term_hits",
    "yes_no_form",
    "baseline_confidence",
    "baseline_answer_chars",
    "baseline_answer_numeric",
    "demand_count",
    "evidence_count",
    "gap_max",
    "gap_sum",
    "gap_count",
    *tuple(f"demand::{item.value}" for item in _EVIDENCE_TYPES),
    *tuple(f"evidence::{item.value}" for item in _EVIDENCE_TYPES),
    *tuple(f"gap::{item.value}" for item in _EVIDENCE_TYPES),
)

SelectorTarget = Literal["paired_advantage", "standalone_quality"]
_SELECTOR_TARGETS = {"paired_advantage", "standalone_quality"}


@dataclass(frozen=True)
class RepairTrainingExample:
    """One complete counterfactual development row group.

    Implements the task--repair rows under paper label
    ``sec:development-counterfactuals`` and stores the baseline-relative target
    defined at ``eq:paired-delta``.
    ``source_id`` is used only as a cross-validation group and never enters the
    feature vector or a fitted model.
    """

    source_id: str
    task_id: str
    features: tuple[float, ...]
    gap_max: float
    strict_advantages: Mapping[str, float]
    incremental_calls: Mapping[str, float]
    baseline_quality: float = 0.0

    def __post_init__(self) -> None:
        if not self.source_id or not self.task_id:
            raise ValueError("repair examples require source_id and task_id")
        if len(self.features) != len(REPAIR_FEATURE_NAMES):
            raise ValueError("repair feature width does not match the frozen schema")
        if set(self.strict_advantages) != set(self.incremental_calls):
            raise ValueError(
                "repair advantages and call costs must share candidate IDs"
            )
        if not self.strict_advantages:
            raise ValueError("repair examples require at least one candidate")
        if min(self.incremental_calls.values()) < 0.0:
            raise ValueError("incremental calls must be non-negative")


@dataclass(frozen=True)
class RepairSelectionConfig:
    l2: float
    threshold: float
    cost_weight: float
    minimum_gap: float = 0.05

    def __post_init__(self) -> None:
        if self.l2 <= 0.0:
            raise ValueError("l2 must be positive")
        if self.threshold < 0.0 or self.cost_weight < 0.0 or self.minimum_gap < 0.0:
            raise ValueError("selector thresholds and costs must be non-negative")


@dataclass(frozen=True)
class RepairDecision:
    selected_id: str
    predicted_advantages: Mapping[str, float]
    net_advantage_scores: Mapping[str, float]
    reason: str

    @property
    def repaired(self) -> bool:
        return self.selected_id != "baseline"


@dataclass(frozen=True)
class RepairPolicySummary:
    tasks: int
    mean_quality_delta: float
    mean_incremental_calls: float
    repair_rate: float
    strict_harm_rate: float
    useful_repair_rate: float
    selection_counts: Mapping[str, int]


@dataclass(frozen=True)
class CrossValidationChoice:
    config: RepairSelectionConfig
    objective: float
    mean_quality_delta: float
    mean_incremental_calls: float
    repair_rate: float
    strict_harm_rate: float
    useful_repair_rate: float
    folds: int
    eligible: bool


@dataclass(frozen=True)
class CounterfactualRepairSelector:
    """One ridge score head per repair, with an explicit serialized target.

    This class supplies the learned component of the method-v3 repairability
    gate in ``sec:repairability-gate`` and ``eq:repairability-gate``. The
    downstream staged policy materializes only whether ``select`` returns a
    nonbaseline action; the predicted action identity is discarded.
    """

    config: RepairSelectionConfig
    models: Mapping[str, RidgeModel]
    estimated_incremental_calls: Mapping[str, float]
    target: SelectorTarget = "paired_advantage"

    @classmethod
    def fit(
        cls,
        examples: Sequence[RepairTrainingExample],
        *,
        config: RepairSelectionConfig,
        target: SelectorTarget = "paired_advantage",
    ) -> CounterfactualRepairSelector:
        """Fit the action-value heads used by ``sec:repairability-gate``.

        Their thresholded maximum implements the learned term in
        ``eq:repairability-gate`` before method v3 discards action identity.
        """
        _validate_target(target)
        candidate_ids = _validate_examples(examples)
        features = [item.features for item in examples]
        models = {
            candidate_id: RidgeModel.fit(
                features,
                [
                    (
                        item.strict_advantages[candidate_id]
                        if target == "paired_advantage"
                        else item.baseline_quality
                        + item.strict_advantages[candidate_id]
                    )
                    for item in examples
                ],
                l2=config.l2,
            )
            for candidate_id in candidate_ids
        }
        costs = {
            candidate_id: fmean(
                item.incremental_calls[candidate_id] for item in examples
            )
            for candidate_id in candidate_ids
        }
        return cls(
            config=config,
            models=models,
            estimated_incremental_calls=costs,
            target=target,
        )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.models))

    def select(
        self,
        features: Sequence[float],
        *,
        gap_max: float,
        force_repair: bool = False,
    ) -> RepairDecision:
        """Return the latent action underlying ``eq:repairability-gate``.

        The method-v3 caller converts this to a Boolean gate as described in
        ``sec:repairability-gate``. Source ID, references, and candidate
        outcomes are not accepted by this interface.
        """
        if len(features) != len(REPAIR_FEATURE_NAMES):
            raise ValueError("selection feature width does not match the frozen schema")
        predicted = {
            candidate_id: self.models[candidate_id].predict(features)
            for candidate_id in self.candidate_ids
        }
        adjusted = {
            candidate_id: predicted[candidate_id]
            - self.config.cost_weight * self.estimated_incremental_calls[candidate_id]
            for candidate_id in self.candidate_ids
        }
        best = min(
            adjusted,
            key=lambda item: (
                -adjusted[item],
                self.estimated_incremental_calls[item],
                item,
            ),
        )
        if force_repair:
            return RepairDecision(
                selected_id=best,
                predicted_advantages=predicted,
                net_advantage_scores=adjusted,
                reason="forced_best_repair",
            )
        if gap_max <= self.config.minimum_gap:
            return RepairDecision(
                selected_id="baseline",
                predicted_advantages=predicted,
                net_advantage_scores=adjusted,
                reason="no_residual_gap",
            )
        if adjusted[best] <= self.config.threshold:
            return RepairDecision(
                selected_id="baseline",
                predicted_advantages=predicted,
                net_advantage_scores=adjusted,
                reason="nonpositive_net_advantage",
            )
        return RepairDecision(
            selected_id=best,
            predicted_advantages=predicted,
            net_advantage_scores=adjusted,
            reason="positive_net_advantage",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "target": self.target,
            "feature_names": list(REPAIR_FEATURE_NAMES),
            "config": asdict(self.config),
            "estimated_incremental_calls": dict(self.estimated_incremental_calls),
            "models": {
                candidate_id: self.models[candidate_id].to_dict()
                for candidate_id in self.candidate_ids
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CounterfactualRepairSelector:
        if tuple(payload.get("feature_names", [])) != REPAIR_FEATURE_NAMES:
            raise ValueError("serialized selector feature schema mismatch")
        raw_config = payload.get("config", {})
        raw_models = payload.get("models", {})
        raw_costs = payload.get("estimated_incremental_calls", {})
        if not isinstance(raw_config, Mapping) or not isinstance(raw_models, Mapping):
            raise ValueError("serialized selector is malformed")
        target = _validate_target(str(payload.get("target", "paired_advantage")))
        models: dict[str, RidgeModel] = {}
        for candidate_id, raw in raw_models.items():
            if not isinstance(raw, Mapping):
                raise ValueError("serialized ridge model is malformed")
            models[str(candidate_id)] = _ridge_from_payload(raw)
        return cls(
            config=RepairSelectionConfig(**dict(raw_config)),
            models=models,
            estimated_incremental_calls={
                str(key): float(value) for key, value in dict(raw_costs).items()
            },
            target=target,
        )


def visible_repair_feature_vector(
    task: PilotTask,
    baseline: Mapping[str, Any],
) -> tuple[float, ...]:
    """Extract features for ``sec:repairability-gate`` without outcome labels.

    The exact 40 coordinates are listed in the
    technical supplement's deployed-feature table. Evidence-count coordinates
    consume parser-accepted records; the formal parser discards structurally
    invalid raw cards before this function, and the residual-gap computation
    validates accepted cards again.
    """

    question = task.visible_question()
    lowered = question.lower()
    words = re.findall(r"[a-z0-9]+", lowered)
    option_text_chars = sum(len(text) for _, text in task.options)
    raw_demands = baseline.get("demands", [])
    demands = (
        [item for item in raw_demands if isinstance(item, Mapping)]
        if isinstance(raw_demands, list)
        else []
    )
    raw_evidence = baseline.get("evidence", [])
    evidence_records = (
        [item for item in raw_evidence if isinstance(item, Mapping)]
        if isinstance(raw_evidence, list)
        else []
    )
    demand_counts = Counter(str(item.get("type", "")) for item in demands)
    evidence_counts = Counter(
        str(item.get("type", "")) for item in evidence_records
    )
    gaps = residual_gap_vector(baseline)
    gap_values = [float(gaps.get(item, 0.0)) for item in _EVIDENCE_TYPES]
    answer = str(baseline.get("answer", "")).strip()
    values = (
        min(len(question), 2000) / 500.0,
        min(len(words), 300) / 100.0,
        min(len(re.findall(r"\d+(?:\.\d+)?", question)), 20) / 10.0,
        min(sum(question.count(symbol) for symbol in "+-*/=<>^"), 20) / 10.0,
        min(len(task.options), 20) / 10.0,
        min(option_text_chars, 3000) / 1000.0,
        float(bool(task.options)),
        float("%" in question or "percent" in lowered or "percentage" in lowered),
        float(bool(_UNIT_PATTERN.search(question))),
        _term_density(words, _OCR_TERMS),
        _term_density(words, _CHART_TERMS),
        _term_density(words, _MATH_TERMS),
        _term_density(words, _SPATIAL_TERMS),
        float(
            lowered.startswith(("is ", "are ", "does ", "do ", "can ", "was ", "were "))
        ),
        _probability(baseline.get("confidence", 0.0)),
        min(len(answer), 100) / 50.0,
        float(bool(re.fullmatch(r"[-+]?[$]?\d[\d,.%\s-]*", answer))),
        min(len(demands), 10) / 5.0,
        min(len(evidence_records), 10) / 5.0,
        max(gap_values, default=0.0),
        sum(gap_values),
        float(sum(value > 0.05 for value in gap_values)),
        *(min(demand_counts[item.value], 5) / 5.0 for item in _EVIDENCE_TYPES),
        *(min(evidence_counts[item.value], 5) / 5.0 for item in _EVIDENCE_TYPES),
        *gap_values,
    )
    if len(values) != len(REPAIR_FEATURE_NAMES):
        raise AssertionError("repair feature schema and vector length diverged")
    return tuple(float(value) for value in values)


def evaluate_selector(
    selector: CounterfactualRepairSelector,
    examples: Sequence[RepairTrainingExample],
    *,
    useful_margin: float = 0.05,
    harm_tolerance: float = 0.05,
    force_repair: bool = False,
) -> RepairPolicySummary:
    if not examples:
        raise ValueError("selector evaluation requires examples")
    deltas: list[float] = []
    calls: list[float] = []
    selections: list[str] = []
    for item in examples:
        decision = selector.select(
            item.features,
            gap_max=item.gap_max,
            force_repair=force_repair,
        )
        selections.append(decision.selected_id)
        if decision.repaired:
            deltas.append(float(item.strict_advantages[decision.selected_id]))
            calls.append(float(item.incremental_calls[decision.selected_id]))
        else:
            deltas.append(0.0)
            calls.append(0.0)
    return RepairPolicySummary(
        tasks=len(examples),
        mean_quality_delta=fmean(deltas),
        mean_incremental_calls=fmean(calls),
        repair_rate=sum(item != "baseline" for item in selections) / len(examples),
        strict_harm_rate=sum(item < -harm_tolerance for item in deltas) / len(examples),
        useful_repair_rate=sum(item > useful_margin for item in deltas) / len(examples),
        selection_counts=dict(sorted(Counter(selections).items())),
    )


def select_config_by_source_cv(
    examples: Sequence[RepairTrainingExample],
    *,
    l2_values: Sequence[float],
    thresholds: Sequence[float],
    cost_weights: Sequence[float],
    evaluation_call_price: float,
    maximum_harm_rate: float = 0.05,
    minimum_repair_rate: float = 0.05,
    minimum_gap: float = 0.05,
    target: SelectorTarget = "paired_advantage",
    force_repair: bool = False,
) -> tuple[CrossValidationChoice, tuple[CrossValidationChoice, ...]]:
    """Freeze ``eq:repairability-gate`` with leave-one-source-out folds.

    This is the grouped development protocol in ``sec:repairability-gate``;
    every candidate operating point is evaluated on source-held-out rows
    before deterministic feasibility and tie rules apply.
    """

    _validate_target(target)
    candidate_ids = _validate_examples(examples)
    del candidate_ids
    sources = tuple(sorted({item.source_id for item in examples}))
    if len(sources) < 2:
        raise ValueError("source-grouped selection requires at least two sources")
    if evaluation_call_price < 0.0:
        raise ValueError("evaluation call price must be non-negative")
    choices: list[CrossValidationChoice] = []
    for l2 in l2_values:
        for threshold in thresholds:
            for cost_weight in cost_weights:
                config = RepairSelectionConfig(
                    l2=float(l2),
                    threshold=float(threshold),
                    cost_weight=float(cost_weight),
                    minimum_gap=minimum_gap,
                )
                decisions: list[tuple[RepairTrainingExample, RepairDecision]] = []
                for held_source in sources:
                    train = [item for item in examples if item.source_id != held_source]
                    held = [item for item in examples if item.source_id == held_source]
                    selector = CounterfactualRepairSelector.fit(
                        train,
                        config=config,
                        target=target,
                    )
                    decisions.extend(
                        (
                            item,
                            selector.select(
                                item.features,
                                gap_max=item.gap_max,
                                force_repair=force_repair,
                            ),
                        )
                        for item in held
                    )
                deltas = [
                    0.0
                    if not decision.repaired
                    else float(item.strict_advantages[decision.selected_id])
                    for item, decision in decisions
                ]
                calls = [
                    0.0
                    if not decision.repaired
                    else float(item.incremental_calls[decision.selected_id])
                    for item, decision in decisions
                ]
                repair_rate = sum(decision.repaired for _, decision in decisions) / len(
                    decisions
                )
                harm_rate = sum(value < -0.05 for value in deltas) / len(deltas)
                useful_rate = sum(value > 0.05 for value in deltas) / len(deltas)
                mean_delta = fmean(deltas)
                mean_calls = fmean(calls)
                eligible = (
                    harm_rate <= maximum_harm_rate
                    and repair_rate >= minimum_repair_rate
                )
                choices.append(
                    CrossValidationChoice(
                        config=config,
                        objective=(
                            mean_delta - evaluation_call_price * mean_calls
                            if eligible
                            else float("-inf")
                        ),
                        mean_quality_delta=mean_delta,
                        mean_incremental_calls=mean_calls,
                        repair_rate=repair_rate,
                        strict_harm_rate=harm_rate,
                        useful_repair_rate=useful_rate,
                        folds=len(sources),
                        eligible=eligible,
                    )
                )
    if not choices or not any(item.eligible for item in choices):
        raise ValueError(
            "no finite-grid selector satisfies the preregistered constraints"
        )
    selected = min(
        (item for item in choices if item.eligible),
        key=lambda item: (
            -item.objective,
            -item.mean_quality_delta,
            item.mean_incremental_calls,
            item.config.l2,
            -item.config.threshold,
            item.config.cost_weight,
        ),
    )
    return selected, tuple(choices)


def _validate_examples(
    examples: Sequence[RepairTrainingExample],
) -> tuple[str, ...]:
    if not examples:
        raise ValueError("repair selector fitting requires examples")
    candidate_ids = tuple(sorted(examples[0].strict_advantages))
    if any(tuple(sorted(item.strict_advantages)) != candidate_ids for item in examples):
        raise ValueError("all repair examples must share a fixed candidate set")
    return candidate_ids


def _validate_target(target: str) -> SelectorTarget:
    if target not in _SELECTOR_TARGETS:
        raise ValueError(f"unknown selector target: {target}")
    return cast(SelectorTarget, target)


def _ridge_from_payload(payload: Mapping[str, Any]) -> RidgeModel:
    return RidgeModel(
        feature_means=tuple(float(value) for value in payload.get("feature_means", [])),
        feature_scales=tuple(
            float(value) for value in payload.get("feature_scales", [])
        ),
        weights=tuple(float(value) for value in payload.get("weights", [])),
        l2=float(payload.get("l2", 0.0)),
    )


def _term_density(words: Sequence[str], terms: Sequence[str]) -> float:
    hits = sum(1 for word in words if word in terms)
    return min(hits, 10) / 5.0


def _probability(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
