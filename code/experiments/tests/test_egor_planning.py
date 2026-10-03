from __future__ import annotations

import json

from openagora.mechanisms.evidence_gap_repair import EvidenceType
from openagora_experiments.egor.planning import (
    build_repair_plans,
    capability_cards_from_payload,
    evaluate_planned_policy,
    fit_capability_cards,
)
from openagora_experiments.egor.task_model import PilotTask


def test_cross_type_capability_fit_uses_strict_provenance_closures() -> None:
    rows = [
        _matrix_row(
            task_id="t1",
            gaps={"exact_span": 1.0, "numerical_derivation": 0.8},
            baseline=0.0,
            candidate=1.0,
            cited=True,
            supported=True,
        ),
        _matrix_row(
            task_id="t2",
            gaps={"exact_span": 1.0},
            baseline=0.0,
            candidate=1.0,
            cited=False,
            supported=False,
        ),
    ]

    cards, estimates = fit_capability_cards(
        rows,
        action_candidates={"ocr": "exact_singleton"},
        admissible_gap_types={"ocr": [EvidenceType.EXACT_SPAN, EvidenceType.NUMERICAL_DERIVATION]},
        minimum_eligible=1,
    )

    assert cards[0].estimated_cost == 2.0
    assert cards[0].supply(EvidenceType.EXACT_SPAN) == 0.5
    assert cards[0].supply(EvidenceType.NUMERICAL_DERIVATION) == 2 / 3
    exact = next(item for item in estimates if item.gap_type is EvidenceType.EXACT_SPAN)
    assert (exact.eligible_tasks, exact.successful_closures) == (2, 1)


def test_dynamic_plans_use_only_baseline_gaps_and_frozen_cards() -> None:
    capabilities = capability_cards_from_payload(
        {
            "actions": [
                {
                    "action_id": "ocr",
                    "supply_by_type": {"exact_span": 0.9},
                    "estimated_cost": 2.0,
                },
                {
                    "action_id": "visual",
                    "supply_by_type": {"visual_relation": 0.8},
                    "estimated_cost": 2.0,
                },
            ]
        }
    )
    baseline_records = {
        ("fixture", "t1"): {
            "status": "ok",
            "parsed": {
                "confidence": 1.0,
                "demands": [{"type": "exact_span", "weight": 1.0}],
                "evidence": [],
                "open_gap_types": [],
            },
        }
    }

    plans = build_repair_plans(
        baseline_records,
        capabilities=capabilities,
        max_workers=1,
        minimum_gap=0.05,
        minimum_marginal_gain=0.01,
    )

    assert plans[0]["selected_action_ids"] == ["ocr"]
    assert plans[0]["combination_id"] == "ocr"
    assert plans[0]["residual_gaps"] == {"exact_span": 1.0}


def test_planned_policy_scores_only_selected_stages_with_strict_gate() -> None:
    task = PilotTask(
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
    key = (task.source_id, task.task_id)
    baseline = _stage(
        {
            "answer": "wrong",
            "confidence": 0.0,
            "demands": [{"type": "exact_span", "weight": 1.0}],
            "evidence": [],
            "open_gap_types": ["exact_span"],
        }
    )
    specialist = _stage(
        {
            "candidate_answer": "right",
            "evidence": [
                {
                    "evidence_id": "ocr::e1",
                    "type": "exact_span",
                    "claim": "right",
                    "support": "right",
                    "provenance": "image region",
                    "producer_id": "ocr",
                    "confidence": 1.0,
                }
            ],
        }
    )
    synthesis = _stage(
        {
            "answer": "right",
            "used_evidence_ids": ["ocr::e1"],
            "changed_baseline": True,
        }
    )
    plan = {
        "source_id": "fixture",
        "task_id": "t1",
        "selected_action_ids": ["ocr"],
        "combination_id": "ocr",
    }

    result = evaluate_planned_policy(
        [task],
        baseline_records={key: baseline},
        plan_records={key: plan},
        action_record_sets={"ocr": {key: specialist}},
        synthesis_record_sets={"ocr": {key: synthesis}},
        policy_id="dynamic",
    )[0]

    assert result.policy_row.policy_quality == 1.0
    assert result.policy_row.calls == 3.0
    assert result.policy_row.provenance_rejected is False
    assert result.selected_action_ids == ("ocr",)

    paid_router_result = evaluate_planned_policy(
        [task],
        baseline_records={key: baseline},
        plan_records={key: plan},
        action_record_sets={"ocr": {key: specialist}},
        synthesis_record_sets={"ocr": {key: synthesis}},
        decision_records={key: _stage({"selected_template_id": "ocr"})},
        policy_id="llm_scheduler",
    )[0]
    assert paid_router_result.policy_row.calls == 4.0
    assert paid_router_result.policy_row.total_tokens == 40.0
    assert paid_router_result.policy_row.latency_seconds == 2.0


def _matrix_row(
    *,
    task_id: str,
    gaps: dict[str, float],
    baseline: float,
    candidate: float,
    cited: bool,
    supported: bool,
) -> dict[str, str]:
    return {
        "source_id": "fixture",
        "task_id": task_id,
        "candidate_id": "exact_singleton",
        "baseline_quality": str(baseline),
        "candidate_quality": str(candidate),
        "quality_delta": str(candidate - baseline),
        "residual_gaps_json": json.dumps(gaps),
        "used_valid_specialist_evidence_ids": "ocr::e1" if cited else "",
        "actual_changed": "1",
        "evidence_supported_change": "1" if supported else "0",
        "calls": "3",
        "baseline_calls": "1",
        "failed_stages": "",
    }


def _stage(parsed: dict[str, object]) -> dict[str, object]:
    return {
        "status": "ok",
        "parsed": parsed,
        "usage": {"attempts": 1, "total_tokens": 10, "latency_seconds": 0.5},
    }
