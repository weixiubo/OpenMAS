"""Uniform scoring, gap reconstruction, and evidence audits for EGOR matrices."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from openagora.mechanisms.evidence_gap_repair import EvidenceType
from openagora_experiments.cago.scoring import score_prediction
from openagora_experiments.egor.protocol import PilotTask


RecordKey = tuple[str, str]


@dataclass(frozen=True)
class EvidenceObservation:
    evidence_id: str
    evidence_type: EvidenceType
    confidence: float
    producer_id: str


@dataclass(frozen=True)
class SynthesisAudit:
    valid_baseline_ids: tuple[str, ...]
    valid_specialist_ids: tuple[str, ...]
    used_valid_specialist_ids: tuple[str, ...]
    unknown_used_ids: tuple[str, ...]
    actual_changed: bool
    reported_changed: bool
    change_flag_agrees: bool
    evidence_supported_change: bool


@dataclass(frozen=True)
class CandidateRow:
    source_id: str
    task_id: str
    candidate_id: str
    baseline_answer: str
    candidate_answer: str
    baseline_quality: float
    candidate_quality: float
    quality_delta: float
    residual_gaps: Mapping[EvidenceType, float]
    specialist_cards: tuple[EvidenceObservation, ...]
    synthesis_audit: SynthesisAudit
    baseline_calls: float
    baseline_total_tokens: float
    baseline_latency_seconds: float
    calls: float
    total_tokens: float
    latency_seconds: float
    failed_stages: tuple[str, ...]


@dataclass(frozen=True)
class PolicyRow:
    source_id: str
    task_id: str
    policy_id: str
    candidate_id: str
    triggered: bool
    baseline_quality: float
    policy_quality: float
    quality_delta: float
    calls: float
    total_tokens: float
    latency_seconds: float
    strict_harm: bool
    useful_repair: bool
    failed_stages: tuple[str, ...]
    provenance_rejected: bool = False


@dataclass(frozen=True)
class CapabilityEstimate:
    action_id: str
    evidence_type: EvidenceType
    eligible_tasks: int
    successful_closures: int
    posterior_mean: float
    wilson_low: float
    wilson_high: float


def residual_gap_vector(baseline: Mapping[str, Any]) -> dict[EvidenceType, float]:
    """Reconstruct typed residuals from validated baseline cards and confidence.

    Paper mapping: ``sec:typed-evidence-state`` and ``eq:gap``. Invalid
    evidence is filtered before per-type coverage can reduce a residual.
    """

    valid_cards = _valid_evidence(baseline.get("evidence", []))
    max_confidence: dict[EvidenceType, float] = {}
    for card in valid_cards:
        max_confidence[card.evidence_type] = max(
            max_confidence.get(card.evidence_type, 0.0),
            card.confidence,
        )
    gaps: dict[EvidenceType, float] = {}
    for item in _objects(baseline.get("demands", [])):
        evidence_type = _evidence_type(item.get("type"))
        if evidence_type is None:
            continue
        weight = _clip_probability(item.get("weight", 1.0))
        residual = weight * (1.0 - max_confidence.get(evidence_type, 0.0))
        gaps[evidence_type] = gaps.get(evidence_type, 0.0) + residual
    for value in _strings(baseline.get("open_gap_types", [])):
        evidence_type = _evidence_type(value)
        if evidence_type is not None:
            gaps[evidence_type] = max(1.0, gaps.get(evidence_type, 0.0))
    baseline_confidence = _clip_probability(baseline.get("confidence", 0.0))
    verification_gap = 1.0 - baseline_confidence
    if verification_gap > 0.0:
        gaps[EvidenceType.INDEPENDENT_VERIFICATION] = max(
            verification_gap,
            gaps.get(EvidenceType.INDEPENDENT_VERIFICATION, 0.0),
        )
    return {key: value for key, value in gaps.items() if value > 0.0}


def evaluate_candidate_rows(
    *,
    tasks: Sequence[PilotTask],
    baseline_records: Mapping[RecordKey, Mapping[str, Any]],
    specialist_record_sets: Sequence[Mapping[RecordKey, Mapping[str, Any]]],
    synthesis_records: Mapping[RecordKey, Mapping[str, Any]],
    candidate_id: str,
) -> list[CandidateRow]:
    """Materialize one complete task--repair counterfactual catalog slice.

    Paper mapping: ``sec:development-counterfactuals`` and
    ``eq:paired-delta``. Missing/failed stages retain zero candidate quality
    and remain in the returned denominator.
    """
    rows: list[CandidateRow] = []
    for task in tasks:
        key = (task.source_id, task.task_id)
        baseline_record = baseline_records.get(key)
        if baseline_record is None:
            raise ValueError(
                f"missing baseline record for {task.source_id}::{task.task_id}"
            )
        specialist_records = [records.get(key) for records in specialist_record_sets]
        synthesis_record = synthesis_records.get(key)
        baseline_parsed = _parsed(baseline_record)
        synthesis_parsed = _parsed(synthesis_record or {})
        baseline_answer = str(baseline_parsed.get("answer", "")).strip()
        candidate_answer = str(synthesis_parsed.get("answer", "")).strip()
        baseline_quality = (
            _score(task, baseline_answer) if _ok(baseline_record) else 0.0
        )
        candidate_quality = (
            _score(task, candidate_answer)
            if synthesis_record is not None and _ok(synthesis_record)
            else 0.0
        )
        specialist_parsed = [_parsed(item or {}) for item in specialist_records]
        cards = tuple(
            card
            for parsed in specialist_parsed
            for card in _valid_evidence(parsed.get("evidence", []))
        )
        audit = audit_synthesis(
            baseline=baseline_parsed,
            specialists=specialist_parsed,
            synthesis=synthesis_parsed,
        )
        stage_records = [baseline_record, *specialist_records, synthesis_record]
        failed_stages: list[str] = []
        for index, record in enumerate(stage_records):
            if record is None or not _ok(record):
                if index == 0:
                    failed_stages.append("baseline")
                elif index == len(stage_records) - 1:
                    failed_stages.append("synthesis")
                else:
                    failed_stages.append(f"specialist_{index}")
        rows.append(
            CandidateRow(
                source_id=task.source_id,
                task_id=task.task_id,
                candidate_id=candidate_id,
                baseline_answer=baseline_answer,
                candidate_answer=candidate_answer,
                baseline_quality=baseline_quality,
                candidate_quality=candidate_quality,
                quality_delta=candidate_quality - baseline_quality,
                residual_gaps=residual_gap_vector(baseline_parsed),
                specialist_cards=cards,
                synthesis_audit=audit,
                baseline_calls=_usage(baseline_record, "attempts"),
                baseline_total_tokens=_usage(baseline_record, "total_tokens"),
                baseline_latency_seconds=_usage(baseline_record, "latency_seconds"),
                calls=sum(_usage(record, "attempts") for record in stage_records),
                total_tokens=sum(
                    _usage(record, "total_tokens") for record in stage_records
                ),
                latency_seconds=sum(
                    _usage(record, "latency_seconds") for record in stage_records
                ),
                failed_stages=tuple(failed_stages),
            )
        )
    return rows


def derive_gap_policy(
    rows: Sequence[CandidateRow],
    *,
    policy_id: str,
    threshold: float,
    useful_margin: float = 0.05,
    harm_tolerance: float = 0.05,
    require_evidence_for_change: bool = False,
) -> list[PolicyRow]:
    if threshold < 0.0:
        raise ValueError("gap threshold must be non-negative")
    return _derive_policy(
        rows,
        policy_id=policy_id,
        should_trigger=lambda row: any(
            value > threshold for value in row.residual_gaps.values()
        ),
        useful_margin=useful_margin,
        harm_tolerance=harm_tolerance,
        require_evidence_for_change=require_evidence_for_change,
    )


def derive_all_policy(
    rows: Sequence[CandidateRow],
    *,
    policy_id: str,
    useful_margin: float = 0.05,
    harm_tolerance: float = 0.05,
    require_evidence_for_change: bool = False,
) -> list[PolicyRow]:
    """Evaluate an explicit all-task control without overloading a threshold."""

    return _derive_policy(
        rows,
        policy_id=policy_id,
        should_trigger=lambda _row: True,
        useful_margin=useful_margin,
        harm_tolerance=harm_tolerance,
        require_evidence_for_change=require_evidence_for_change,
    )


def _derive_policy(
    rows: Sequence[CandidateRow],
    *,
    policy_id: str,
    should_trigger: Callable[[CandidateRow], bool],
    useful_margin: float,
    harm_tolerance: float,
    require_evidence_for_change: bool,
) -> list[PolicyRow]:
    if not policy_id.strip():
        raise ValueError("policy_id is required")
    policy_rows: list[PolicyRow] = []
    for row in rows:
        triggered = bool(should_trigger(row))
        provenance_rejected = (
            triggered
            and require_evidence_for_change
            and row.synthesis_audit.actual_changed
            and not row.synthesis_audit.evidence_supported_change
        )
        policy_quality = (
            row.candidate_quality
            if triggered and not provenance_rejected
            else row.baseline_quality
        )
        delta = policy_quality - row.baseline_quality
        if triggered:
            calls = row.calls
            tokens = row.total_tokens
            latency = row.latency_seconds
            failures = row.failed_stages
        else:
            calls = row.baseline_calls
            tokens = row.baseline_total_tokens
            latency = row.baseline_latency_seconds
            failures = tuple(
                stage for stage in row.failed_stages if stage == "baseline"
            )
        policy_rows.append(
            PolicyRow(
                source_id=row.source_id,
                task_id=row.task_id,
                policy_id=policy_id,
                candidate_id=row.candidate_id,
                triggered=triggered,
                baseline_quality=row.baseline_quality,
                policy_quality=policy_quality,
                quality_delta=delta,
                calls=calls,
                total_tokens=tokens,
                latency_seconds=latency,
                strict_harm=delta < -harm_tolerance,
                useful_repair=delta > useful_margin,
                failed_stages=failures,
                provenance_rejected=provenance_rejected,
            )
        )
    return policy_rows


def summarize_policy(rows: Sequence[PolicyRow]) -> dict[str, float]:
    if not rows:
        raise ValueError("at least one policy row is required")
    return {
        "tasks": float(len(rows)),
        "trigger_rate": fmean(row.triggered for row in rows),
        "baseline_quality": fmean(row.baseline_quality for row in rows),
        "policy_quality": fmean(row.policy_quality for row in rows),
        "quality_delta": fmean(row.quality_delta for row in rows),
        "useful_rate": fmean(row.useful_repair for row in rows),
        "strict_harm_rate": fmean(row.strict_harm for row in rows),
        "provenance_rejection_rate": fmean(row.provenance_rejected for row in rows),
        "complete_rate": fmean(not row.failed_stages for row in rows),
        "calls": fmean(row.calls for row in rows),
        "total_tokens": fmean(row.total_tokens for row in rows),
        "latency_seconds": fmean(row.latency_seconds for row in rows),
    }


def audit_synthesis(
    *,
    baseline: Mapping[str, Any],
    specialists: Sequence[Mapping[str, Any]],
    synthesis: Mapping[str, Any],
) -> SynthesisAudit:
    """Audit whether a changed synthesis cites valid recruited evidence.

    Paper mapping: ``sec:evidence-certificate`` and ``eq:certificate``.
    This is a provenance check only; it never reads a reference answer or
    predicts correctness.
    """
    baseline_ids = {
        card.evidence_id for card in _valid_evidence(baseline.get("evidence", []))
    }
    specialist_ids = {
        card.evidence_id
        for specialist in specialists
        for card in _valid_evidence(specialist.get("evidence", []))
    }
    used = set(_strings(synthesis.get("used_evidence_ids", [])))
    known = baseline_ids | specialist_ids
    actual_changed = _canonical_text(synthesis.get("answer", "")) != _canonical_text(
        baseline.get("answer", "")
    )
    reported_changed = bool(synthesis.get("changed_baseline", False))
    used_specialist = used & specialist_ids
    return SynthesisAudit(
        valid_baseline_ids=tuple(sorted(baseline_ids)),
        valid_specialist_ids=tuple(sorted(specialist_ids)),
        used_valid_specialist_ids=tuple(sorted(used_specialist)),
        unknown_used_ids=tuple(sorted(used - known)),
        actual_changed=actual_changed,
        reported_changed=reported_changed,
        change_flag_agrees=actual_changed == reported_changed,
        evidence_supported_change=actual_changed
        and bool(used_specialist)
        and not (used - known),
    )


def estimate_capability(
    rows: Sequence[CandidateRow],
    *,
    action_id: str,
    evidence_type: EvidenceType,
    minimum_gap: float = 0.05,
    minimum_card_confidence: float = 0.5,
    useful_margin: float = 0.05,
    correct_threshold: float = 0.95,
    noninferiority_tolerance: float = 0.05,
) -> CapabilityEstimate:
    eligible = [
        row for row in rows if row.residual_gaps.get(evidence_type, 0.0) > minimum_gap
    ]
    successes = 0
    for row in eligible:
        qualifying_ids = {
            card.evidence_id
            for card in row.specialist_cards
            if card.evidence_type is evidence_type
            and card.confidence >= minimum_card_confidence
        }
        cited = qualifying_ids.intersection(
            row.synthesis_audit.used_valid_specialist_ids
        )
        useful = row.quality_delta > useful_margin
        preserves_correct = (
            row.baseline_quality >= correct_threshold
            and row.candidate_quality >= row.baseline_quality - noninferiority_tolerance
        )
        successes += int(
            bool(cited) and (useful or preserves_correct) and not row.failed_stages
        )
    count = len(eligible)
    posterior_mean = (successes + 1.0) / (count + 2.0)
    low, high = _wilson_interval(successes, count)
    return CapabilityEstimate(
        action_id=action_id,
        evidence_type=evidence_type,
        eligible_tasks=count,
        successful_closures=successes,
        posterior_mean=posterior_mean,
        wilson_low=low,
        wilson_high=high,
    )


def summarize_candidate(
    rows: Sequence[CandidateRow],
    *,
    useful_margin: float = 0.05,
    harm_tolerance: float = 0.05,
) -> dict[str, float]:
    if not rows:
        raise ValueError("at least one candidate row is required")
    return {
        "tasks": float(len(rows)),
        "baseline_quality": fmean(row.baseline_quality for row in rows),
        "candidate_quality": fmean(row.candidate_quality for row in rows),
        "quality_delta": fmean(row.quality_delta for row in rows),
        "useful_rate": fmean(row.quality_delta > useful_margin for row in rows),
        "strict_harm_rate": fmean(row.quality_delta < -harm_tolerance for row in rows),
        "complete_rate": fmean(not row.failed_stages for row in rows),
        "evidence_supported_change_rate": fmean(
            row.synthesis_audit.evidence_supported_change for row in rows
        ),
        "calls": fmean(row.calls for row in rows),
        "total_tokens": fmean(row.total_tokens for row in rows),
        "latency_seconds": fmean(row.latency_seconds for row in rows),
    }


def _valid_evidence(value: Any) -> list[EvidenceObservation]:
    cards: list[EvidenceObservation] = []
    for item in _objects(value):
        evidence_type = _evidence_type(item.get("type"))
        evidence_id = str(item.get("evidence_id", "")).strip()
        producer_id = str(item.get("producer_id", "")).strip()
        required = (
            str(item.get("claim", "")).strip(),
            str(item.get("support", "")).strip(),
            str(item.get("provenance", "")).strip(),
        )
        if (
            evidence_type is None
            or not evidence_id
            or not producer_id
            or any(not item for item in required)
        ):
            continue
        cards.append(
            EvidenceObservation(
                evidence_id=evidence_id,
                evidence_type=evidence_type,
                confidence=_clip_probability(item.get("confidence", 0.0)),
                producer_id=producer_id,
            )
        )
    return cards


def _score(task: PilotTask, answer: str) -> float:
    return score_prediction(
        metric=task.metric,
        predicted=answer,
        answer=task.answer,
        reference_answers=task.reference_answers,
        options=[{"id": option_id, "text": text} for option_id, text in task.options],
    )


def _ok(record: Mapping[str, Any]) -> bool:
    return str(record.get("status", "")) == "ok"


def _parsed(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("parsed", {})
    return value if isinstance(value, Mapping) else {}


def _usage(record: Mapping[str, Any] | None, field: str) -> float:
    if record is None:
        return 0.0
    usage = record.get("usage", {})
    if not isinstance(usage, Mapping):
        return 0.0
    try:
        return float(usage.get(field, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _evidence_type(value: Any) -> EvidenceType | None:
    try:
        return EvidenceType(str(value).strip())
    except ValueError:
        return None


def _objects(value: Any) -> list[Mapping[str, Any]]:
    return (
        [item for item in value if isinstance(item, Mapping)]
        if isinstance(value, list)
        else []
    )


def _strings(value: Any) -> list[str]:
    return (
        [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, list)
        else []
    )


def _clip_probability(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, result))


def _canonical_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _wilson_interval(
    successes: int, count: int, *, z: float = 1.959963984540054
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
