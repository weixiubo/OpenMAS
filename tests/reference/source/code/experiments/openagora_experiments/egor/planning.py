"""Capability fitting and outcome-blind dynamic repair planning for EGOR."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any

from openagora.mechanisms.evidence_gap_repair import (
    CapabilityCard,
    EvidenceGapRepairPlanner,
    EvidenceType,
)
from openagora_experiments.egor.evaluation import (
    PolicyRow,
    evaluate_candidate_rows,
    residual_gap_vector,
)
from openagora_experiments.egor.protocol import PilotTask


@dataclass(frozen=True)
class CapabilitySupplyEstimate:
    action_id: str
    candidate_id: str
    gap_type: EvidenceType
    eligible_tasks: int
    successful_closures: int
    posterior_mean: float
    wilson_low: float
    wilson_high: float
    used_for_planning: bool


@dataclass(frozen=True)
class PlannedPolicyResult:
    policy_row: PolicyRow
    selected_action_ids: tuple[str, ...]
    combination_id: str


def fit_capability_cards(
    rows: Sequence[Mapping[str, str]],
    *,
    action_candidates: Mapping[str, str],
    admissible_gap_types: Mapping[str, Sequence[EvidenceType]],
    minimum_gap: float = 0.05,
    minimum_eligible: int = 5,
    useful_margin: float = 0.05,
    correct_threshold: float = 0.95,
    noninferiority_tolerance: float = 0.05,
) -> tuple[list[CapabilityCard], list[CapabilitySupplyEstimate]]:
    """Fit cross-type supply probabilities from singleton counterfactual rows."""

    if minimum_gap < 0.0 or minimum_eligible <= 0:
        raise ValueError(
            "minimum gap must be non-negative and minimum eligible positive"
        )
    cards: list[CapabilityCard] = []
    estimates: list[CapabilitySupplyEstimate] = []
    for action_id, candidate_id in sorted(action_candidates.items()):
        if action_id not in admissible_gap_types or not admissible_gap_types[action_id]:
            raise ValueError(f"action {action_id} requires an admissible gap type")
        admitted = set(admissible_gap_types[action_id])
        candidate_rows = [
            row for row in rows if row.get("candidate_id") == candidate_id
        ]
        if not candidate_rows:
            raise ValueError(f"no matrix rows for singleton candidate {candidate_id}")
        incremental_calls = [
            max(0.0, _float(row.get("calls")) - _float(row.get("baseline_calls")))
            for row in candidate_rows
        ]
        supply_by_type: dict[EvidenceType, float] = {}
        for gap_type in EvidenceType:
            eligible = [
                row
                for row in candidate_rows
                if _residual_gaps(row).get(gap_type, 0.0) > minimum_gap
            ]
            successes = sum(
                _successful_closure(
                    row,
                    useful_margin=useful_margin,
                    correct_threshold=correct_threshold,
                    noninferiority_tolerance=noninferiority_tolerance,
                )
                for row in eligible
            )
            count = len(eligible)
            posterior_mean = (successes + 1.0) / (count + 2.0)
            low, high = _wilson_interval(successes, count)
            used = count >= minimum_eligible and gap_type in admitted
            supply_by_type[gap_type] = posterior_mean if used else 0.0
            estimates.append(
                CapabilitySupplyEstimate(
                    action_id=action_id,
                    candidate_id=candidate_id,
                    gap_type=gap_type,
                    eligible_tasks=count,
                    successful_closures=successes,
                    posterior_mean=posterior_mean,
                    wilson_low=low,
                    wilson_high=high,
                    used_for_planning=used,
                )
            )
        cards.append(
            CapabilityCard(
                action_id=action_id,
                supply_by_type=supply_by_type,
                estimated_cost=fmean(incremental_calls),
            )
        )
    return cards, estimates


def build_repair_plans(
    baseline_records: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    capabilities: Sequence[CapabilityCard],
    max_workers: int,
    minimum_gap: float,
    minimum_marginal_gain: float,
) -> list[dict[str, Any]]:
    """Create deterministic per-task action plans from baseline-visible state."""

    planner = EvidenceGapRepairPlanner(
        max_workers=max_workers,
        min_residual_gap=minimum_gap,
        min_marginal_gain=minimum_marginal_gain,
    )
    plans: list[dict[str, Any]] = []
    for (source_id, task_id), record in sorted(baseline_records.items()):
        if str(record.get("status", "")) != "ok":
            plans.append(
                {
                    "schema_version": 1,
                    "source_id": source_id,
                    "task_id": task_id,
                    "baseline_status": str(record.get("status", "missing")),
                    "residual_gaps": {},
                    "selected_action_ids": [],
                    "combination_id": "baseline",
                    "reason": "baseline_stage_failed",
                    "steps": [],
                }
            )
            continue
        parsed = record.get("parsed", {})
        baseline = parsed if isinstance(parsed, Mapping) else {}
        gaps = residual_gap_vector(baseline)
        plan = planner.plan_from_gaps(
            residual_gaps=gaps,
            capabilities=capabilities,
        )
        selected = list(plan.selected_action_ids)
        plans.append(
            {
                "schema_version": 1,
                "source_id": source_id,
                "task_id": task_id,
                "baseline_status": "ok",
                "residual_gaps": {
                    key.value: value for key, value in sorted(gaps.items())
                },
                "selected_action_ids": selected,
                "combination_id": combination_id(selected),
                "reason": plan.reason,
                "baseline_coverage": plan.baseline_coverage,
                "repaired_coverage": plan.repaired_coverage,
                "total_estimated_cost": plan.total_estimated_cost,
                "steps": [asdict(step) for step in plan.steps],
            }
        )
    return plans


def evaluate_planned_policy(
    tasks: Sequence[PilotTask],
    *,
    baseline_records: Mapping[tuple[str, str], Mapping[str, Any]],
    plan_records: Mapping[tuple[str, str], Mapping[str, Any]],
    action_record_sets: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
    synthesis_record_sets: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
    policy_id: str,
    decision_records: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    require_evidence_for_change: bool = True,
    useful_margin: float = 0.05,
    harm_tolerance: float = 0.05,
) -> list[PlannedPolicyResult]:
    """Score the stages actually selected by each outcome-blind repair plan."""

    results: list[PlannedPolicyResult] = []
    for task in tasks:
        key = (task.source_id, task.task_id)
        baseline = baseline_records.get(key)
        plan = plan_records.get(key)
        if baseline is None or plan is None:
            raise ValueError(
                f"missing baseline/plan for {task.source_id}::{task.task_id}"
            )
        selected = tuple(str(value) for value in plan.get("selected_action_ids", []))
        selected_combination = combination_id(selected)
        if str(plan.get("combination_id", "")) != selected_combination:
            raise ValueError(
                f"plan combination mismatch for {task.source_id}::{task.task_id}"
            )
        if selected:
            missing_actions = [
                action_id
                for action_id in selected
                if action_id not in action_record_sets
            ]
            if missing_actions or selected_combination not in synthesis_record_sets:
                raise ValueError(
                    f"plan references unavailable stages for {selected_combination}: "
                    f"{missing_actions}"
                )
            specialist_sets = [action_record_sets[action_id] for action_id in selected]
            synthesis_records = synthesis_record_sets[selected_combination]
        else:
            specialist_sets = []
            synthesis_records = {key: baseline}
        candidate = evaluate_candidate_rows(
            tasks=[task],
            baseline_records={key: baseline},
            specialist_record_sets=specialist_sets,
            synthesis_records=synthesis_records,
            candidate_id=f"planned::{selected_combination}",
        )[0]
        provenance_rejected = (
            bool(selected)
            and require_evidence_for_change
            and candidate.synthesis_audit.actual_changed
            and not candidate.synthesis_audit.evidence_supported_change
        )
        policy_quality = (
            candidate.candidate_quality
            if selected and not provenance_rejected
            else candidate.baseline_quality
        )
        delta = policy_quality - candidate.baseline_quality
        failures = (
            candidate.failed_stages
            if selected
            else tuple(
                stage for stage in candidate.failed_stages if stage == "baseline"
            )
        )
        decision_record = (
            decision_records.get(key) if decision_records is not None else None
        )
        decision_failures = (
            ("decision",)
            if decision_records is not None
            and (
                decision_record is None
                or str(decision_record.get("status", "")) != "ok"
            )
            else ()
        )
        base_calls = candidate.calls if selected else candidate.baseline_calls
        base_tokens = (
            candidate.total_tokens if selected else candidate.baseline_total_tokens
        )
        base_latency = (
            candidate.latency_seconds
            if selected
            else candidate.baseline_latency_seconds
        )
        results.append(
            PlannedPolicyResult(
                policy_row=PolicyRow(
                    source_id=task.source_id,
                    task_id=task.task_id,
                    policy_id=policy_id,
                    candidate_id=f"planned::{selected_combination}",
                    triggered=bool(selected),
                    baseline_quality=candidate.baseline_quality,
                    policy_quality=policy_quality,
                    quality_delta=delta,
                    calls=base_calls + _record_usage(decision_record, "attempts"),
                    total_tokens=base_tokens
                    + _record_usage(decision_record, "total_tokens"),
                    latency_seconds=base_latency
                    + _record_usage(decision_record, "latency_seconds"),
                    strict_harm=delta < -harm_tolerance,
                    useful_repair=delta > useful_margin,
                    failed_stages=(*failures, *decision_failures),
                    provenance_rejected=provenance_rejected,
                ),
                selected_action_ids=selected,
                combination_id=selected_combination,
            )
        )
    return results


def capability_cards_payload(
    cards: Sequence[CapabilityCard],
    estimates: Sequence[CapabilitySupplyEstimate],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "metadata": dict(metadata),
        "actions": [
            {
                "action_id": card.action_id,
                "supply_by_type": {
                    key.value: value
                    for key, value in sorted(card.supply_by_type.items())
                },
                "estimated_cost": card.estimated_cost,
                "heterogeneous": card.heterogeneous,
                "admissible": card.admissible,
            }
            for card in cards
        ],
        "estimates": [
            {
                **asdict(estimate),
                "gap_type": estimate.gap_type.value,
            }
            for estimate in estimates
        ],
    }


def capability_cards_from_payload(payload: Mapping[str, Any]) -> list[CapabilityCard]:
    raw_actions = payload.get("actions", [])
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("capability payload requires a non-empty actions list")
    cards: list[CapabilityCard] = []
    for raw in raw_actions:
        if not isinstance(raw, Mapping):
            raise ValueError("each capability action must be an object")
        raw_supply = raw.get("supply_by_type", {})
        if not isinstance(raw_supply, Mapping):
            raise ValueError("supply_by_type must be an object")
        cards.append(
            CapabilityCard(
                action_id=str(raw.get("action_id", "")),
                supply_by_type={
                    EvidenceType(str(key)): float(value)
                    for key, value in raw_supply.items()
                },
                estimated_cost=float(raw.get("estimated_cost", 0.0)),
                heterogeneous=bool(raw.get("heterogeneous", True)),
                admissible=bool(raw.get("admissible", True)),
            )
        )
    return cards


def combination_id(action_ids: Sequence[str]) -> str:
    return "__".join(sorted(set(action_ids))) if action_ids else "baseline"


def _successful_closure(
    row: Mapping[str, str],
    *,
    useful_margin: float,
    correct_threshold: float,
    noninferiority_tolerance: float,
) -> int:
    if str(row.get("failed_stages", "")).strip():
        return 0
    cited = bool(str(row.get("used_valid_specialist_evidence_ids", "")).strip())
    baseline = _float(row.get("baseline_quality"))
    candidate = _float(row.get("candidate_quality"))
    changed = _bool(row.get("actual_changed"))
    supported = _bool(row.get("evidence_supported_change"))
    strict_candidate = baseline if changed and not supported else candidate
    useful = strict_candidate - baseline > useful_margin
    preserves_correct = (
        baseline >= correct_threshold
        and strict_candidate >= baseline - noninferiority_tolerance
    )
    return int(cited and (useful or preserves_correct))


def _residual_gaps(row: Mapping[str, str]) -> dict[EvidenceType, float]:
    raw = json.loads(str(row.get("residual_gaps_json", "{}")))
    if not isinstance(raw, Mapping):
        raise ValueError("residual_gaps_json must encode an object")
    return {EvidenceType(str(key)): float(value) for key, value in raw.items()}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _record_usage(record: Mapping[str, Any] | None, field: str) -> float:
    if record is None:
        return 0.0
    usage = record.get("usage", {})
    if not isinstance(usage, Mapping):
        return 0.0
    return _float(usage.get(field, 0.0))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _wilson_interval(
    successes: int,
    count: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 1.0
    probability = successes / count
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / count + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)
