from __future__ import annotations

import pytest

from openagora.mechanisms.evidence_gap_repair import EvidenceType
from openagora_experiments.egor.evaluation import (
    audit_synthesis,
    derive_all_policy,
    derive_gap_policy,
    estimate_capability,
    evaluate_candidate_rows,
    residual_gap_vector,
    summarize_candidate,
    summarize_policy,
)
from openagora_experiments.egor.protocol import PilotTask


def test_residual_gaps_use_validated_card_confidence_and_baseline_confidence() -> None:
    baseline = {
        "confidence": 0.25,
        "demands": [
            {"type": "visual_relation", "weight": 1.0},
            {"type": "exact_span", "weight": 0.5},
        ],
        "evidence": [
            _card("b1", "visual_relation", confidence=0.4, producer="baseline"),
            {
                **_card("b2", "exact_span", confidence=1.0, producer="baseline"),
                "provenance": "",
            },
        ],
        "open_gap_types": [],
    }

    gaps = residual_gap_vector(baseline)

    assert gaps[EvidenceType.VISUAL_RELATION] == 0.6
    assert gaps[EvidenceType.EXACT_SPAN] == 0.5
    assert gaps[EvidenceType.INDEPENDENT_VERIFICATION] == 0.75


def test_residual_gaps_accumulate_same_type_demands_and_honor_open_gap() -> None:
    baseline = {
        "confidence": 1.0,
        "demands": [
            {"type": "visual_relation", "weight": 0.4},
            {"type": "visual_relation", "weight": 0.6},
        ],
        "evidence": [
            _card("b1", "visual_relation", confidence=0.5, producer="baseline")
        ],
        "open_gap_types": [],
    }

    gaps = residual_gap_vector(baseline)
    baseline["open_gap_types"] = ["visual_relation"]
    open_gaps = residual_gap_vector(baseline)

    assert gaps[EvidenceType.VISUAL_RELATION] == 0.5
    assert open_gaps[EvidenceType.VISUAL_RELATION] == 1.0
    assert EvidenceType.INDEPENDENT_VERIFICATION not in gaps


def test_synthesis_audit_rejects_unknown_citations() -> None:
    audit = audit_synthesis(
        baseline={
            "answer": "wrong",
            "evidence": [_card("b1", "exact_span", producer="baseline")],
        },
        specialists=[
            {"evidence": [_card("s1", "visual_relation", producer="visual-worker")]}
        ],
        synthesis={
            "answer": "right",
            "used_evidence_ids": ["s1", "invented"],
            "changed_baseline": True,
        },
    )

    assert audit.used_valid_specialist_ids == ("s1",)
    assert audit.unknown_used_ids == ("invented",)
    assert audit.actual_changed is True
    assert audit.change_flag_agrees is True
    assert audit.evidence_supported_change is False


def test_synthesis_audit_canonicalizes_only_change_status() -> None:
    audit = audit_synthesis(
        baseline={"answer": "  Forty TWO ", "evidence": []},
        specialists=[],
        synthesis={
            "answer": "forty   two",
            "used_evidence_ids": [],
            "changed_baseline": False,
        },
    )

    assert audit.actual_changed is False
    assert audit.reported_changed is False
    assert audit.change_flag_agrees is True
    assert audit.evidence_supported_change is False


def test_candidate_matrix_scores_all_rows_and_estimates_capability() -> None:
    task = _task()
    key = (task.source_id, task.task_id)
    baseline = _record(
        parsed={
            "answer": "wrong",
            "confidence": 0.0,
            "demands": [{"type": "visual_relation", "weight": 1.0}],
            "evidence": [],
            "open_gap_types": ["visual_relation"],
        },
        tokens=100,
    )
    specialist = _record(
        parsed={
            "candidate_answer": "right",
            "evidence": [
                _card("s1", "visual_relation", confidence=0.9, producer="visual-worker")
            ],
        },
        tokens=50,
    )
    synthesis = _record(
        parsed={
            "answer": "right",
            "used_evidence_ids": ["s1"],
            "rejected_evidence_ids": [],
            "changed_baseline": True,
        },
        tokens=75,
    )

    rows = evaluate_candidate_rows(
        tasks=[task],
        baseline_records={key: baseline},
        specialist_record_sets=[{key: specialist}],
        synthesis_records={key: synthesis},
        candidate_id="visual-repair",
    )
    estimate = estimate_capability(
        rows,
        action_id="visual-worker",
        evidence_type=EvidenceType.VISUAL_RELATION,
    )
    summary = summarize_candidate(rows)

    assert rows[0].baseline_quality == 0.0
    assert rows[0].candidate_quality == 1.0
    assert rows[0].quality_delta == 1.0
    assert rows[0].calls == 3.0
    assert rows[0].baseline_calls == 1.0
    assert rows[0].baseline_total_tokens == 100.0
    assert rows[0].total_tokens == 225.0
    assert rows[0].synthesis_audit.evidence_supported_change is True
    assert estimate.eligible_tasks == 1
    assert estimate.successful_closures == 1
    assert estimate.posterior_mean == 2 / 3
    assert summary["quality_delta"] == 1.0
    assert summary["useful_rate"] == 1.0
    assert summary["complete_rate"] == 1.0

    policy_rows = derive_gap_policy(rows, policy_id="egor-b1", threshold=0.05)
    policy_summary = summarize_policy(policy_rows)
    assert policy_rows[0].triggered is True
    assert policy_rows[0].provenance_rejected is False
    assert policy_summary["policy_quality"] == 1.0
    assert policy_summary["calls"] == 3.0
    assert policy_summary["provenance_rejection_rate"] == 0.0


def test_explicit_all_policy_triggers_an_empty_gap_row_without_relaxing_thresholds() -> None:
    task = _task()
    key = (task.source_id, task.task_id)
    baseline = _record(
        parsed={
            "answer": "wrong",
            "confidence": 1.0,
            "demands": [],
            "evidence": [],
            "open_gap_types": [],
        }
    )
    synthesis = _record(
        parsed={
            "answer": "right",
            "used_evidence_ids": [],
            "changed_baseline": True,
        }
    )
    row = evaluate_candidate_rows(
        tasks=[task],
        baseline_records={key: baseline},
        specialist_record_sets=[],
        synthesis_records={key: synthesis},
        candidate_id="baseline-resynthesis",
    )[0]

    assert row.residual_gaps == {}
    with pytest.raises(ValueError, match="non-negative"):
        derive_gap_policy([row], policy_id="invalid", threshold=-1.0)

    policy = derive_all_policy([row], policy_id="resynthesis-all")[0]
    assert policy.triggered is True
    assert policy.policy_quality == 1.0
    assert policy.calls == row.calls


def test_strict_provenance_gate_reverts_an_unsupported_changed_answer() -> None:
    task = _task()
    key = (task.source_id, task.task_id)
    baseline = _record(
        parsed={
            "answer": "wrong",
            "confidence": 0.0,
            "demands": [{"type": "visual_relation", "weight": 1.0}],
            "evidence": [],
            "open_gap_types": ["visual_relation"],
        },
        tokens=100,
    )
    specialist = _record(
        parsed={
            "candidate_answer": "right",
            "evidence": [_card("s1", "visual_relation", producer="visual-worker")],
        },
        tokens=50,
    )
    unsupported_synthesis = _record(
        parsed={
            "answer": "right",
            "used_evidence_ids": [],
            "changed_baseline": True,
        },
        tokens=75,
    )
    row = evaluate_candidate_rows(
        tasks=[task],
        baseline_records={key: baseline},
        specialist_record_sets=[{key: specialist}],
        synthesis_records={key: unsupported_synthesis},
        candidate_id="visual-repair",
    )[0]

    ungated = derive_gap_policy([row], policy_id="ungated", threshold=0.05)[0]
    strict = derive_gap_policy(
        [row],
        policy_id="strict",
        threshold=0.05,
        require_evidence_for_change=True,
    )[0]
    all_strict = derive_all_policy(
        [row],
        policy_id="all-strict",
        require_evidence_for_change=True,
    )[0]

    assert ungated.policy_quality == 1.0
    assert ungated.provenance_rejected is False
    assert strict.policy_quality == 0.0
    assert strict.quality_delta == 0.0
    assert strict.calls == 3.0
    assert strict.provenance_rejected is True
    assert all_strict.triggered is True
    assert all_strict.provenance_rejected is True
    assert all_strict.policy_quality == row.baseline_quality


def test_failed_synthesis_is_scored_as_an_all_row_failure() -> None:
    task = _task()
    key = (task.source_id, task.task_id)
    baseline = _record(
        parsed={
            "answer": "right",
            "confidence": 1.0,
            "demands": [],
            "evidence": [],
            "open_gap_types": [],
        }
    )
    failed = {"status": "error", "usage": {"attempts": 1, "total_tokens": 0}}

    row = evaluate_candidate_rows(
        tasks=[task],
        baseline_records={key: baseline},
        specialist_record_sets=[],
        synthesis_records={key: failed},
        candidate_id="failed",
    )[0]

    assert row.baseline_quality == 1.0
    assert row.candidate_quality == 0.0
    assert row.quality_delta == -1.0
    assert row.failed_stages == ("synthesis",)
    assert summarize_candidate([row])["strict_harm_rate"] == 1.0


def _task() -> PilotTask:
    return PilotTask(
        task_id="t1",
        source_id="fixture",
        question="What is shown?",
        image_path="/tmp/image.png",
        mime_type="image/png",
        metric="accuracy",
        answer="right",
        reference_answers=("right",),
        options=(),
        metadata={},
    )


def _card(
    evidence_id: str,
    evidence_type: str,
    *,
    confidence: float = 1.0,
    producer: str,
) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "type": evidence_type,
        "claim": "supported claim",
        "support": "visible support",
        "provenance": "image region",
        "producer_id": producer,
        "confidence": confidence,
    }


def _record(*, parsed: dict[str, object], tokens: int = 10) -> dict[str, object]:
    return {
        "status": "ok",
        "parsed": parsed,
        "usage": {
            "attempts": 1,
            "total_tokens": tokens,
            "latency_seconds": 0.5,
        },
    }
