from __future__ import annotations

import pytest

from openagora.mechanisms.evidence_gap_repair import (
    CapabilityCard,
    EvidenceCard,
    EvidenceDemand,
    EvidenceGapRepairPlanner,
    EvidenceRepairError,
    EvidenceType,
)


def test_no_repair_when_existing_evidence_closes_every_gap() -> None:
    demand = EvidenceDemand("span", EvidenceType.EXACT_SPAN, "Find the printed value")
    evidence = EvidenceCard(
        "card",
        EvidenceType.EXACT_SPAN,
        claim="The value is 42",
        support="42",
        provenance="image:0#bbox=1,2,3,4",
        producer_id="baseline",
        confidence=0.9,
    )

    plan = EvidenceGapRepairPlanner(max_workers=1).plan(
        demands=[demand],
        evidence=[evidence],
        validation_scores={("span", "card"): 1.0},
        capabilities=[_capability("ocr", EvidenceType.EXACT_SPAN, 0.9)],
    )

    assert plan.repaired is False
    assert plan.reason == "no_residual_evidence_gap"
    assert plan.baseline_coverage == pytest.approx(1.0)


def test_planner_selects_capability_matching_largest_gap() -> None:
    demands = [
        EvidenceDemand("span", EvidenceType.EXACT_SPAN, "Read label", weight=1.0),
        EvidenceDemand("math", EvidenceType.NUMERICAL_DERIVATION, "Compute value", weight=0.4),
    ]

    plan = EvidenceGapRepairPlanner(max_workers=1).plan(
        demands=demands,
        evidence=[],
        validation_scores={},
        capabilities=[
            _capability("ocr", EvidenceType.EXACT_SPAN, 0.8, cost=0.2),
            _capability("calculator", EvidenceType.NUMERICAL_DERIVATION, 1.0, cost=0.1),
        ],
    )

    assert plan.selected_action_ids == ("ocr",)
    assert plan.steps[0].marginal_coverage_gain == pytest.approx(0.8)


def test_coverage_has_diminishing_returns_for_redundant_capabilities() -> None:
    demand = EvidenceDemand("span", EvidenceType.EXACT_SPAN, "Read label")

    plan = EvidenceGapRepairPlanner(max_workers=2).plan(
        demands=[demand],
        evidence=[],
        validation_scores={},
        capabilities=[
            _capability("ocr_a", EvidenceType.EXACT_SPAN, 0.6),
            _capability("ocr_b", EvidenceType.EXACT_SPAN, 0.5),
        ],
    )

    assert plan.selected_action_ids == ("ocr_a", "ocr_b")
    assert plan.steps[0].marginal_coverage_gain == pytest.approx(0.6)
    assert plan.steps[1].marginal_coverage_gain == pytest.approx(0.2)
    assert plan.repaired_coverage == pytest.approx(0.8)


def test_minimum_gain_and_worker_budget_bound_repair() -> None:
    demand = EvidenceDemand("visual", EvidenceType.VISUAL_RELATION, "Resolve relation")
    capabilities = [
        _capability("weak", EvidenceType.VISUAL_RELATION, 0.1),
        _capability("strong", EvidenceType.VISUAL_RELATION, 0.9),
    ]

    no_budget = EvidenceGapRepairPlanner(max_workers=0).plan(
        demands=[demand],
        evidence=[],
        validation_scores={},
        capabilities=capabilities,
    )
    high_threshold = EvidenceGapRepairPlanner(max_workers=2, min_marginal_gain=0.95).plan(
        demands=[demand],
        evidence=[],
        validation_scores={},
        capabilities=capabilities,
    )

    assert no_budget.reason == "worker_budget_is_zero"
    assert high_threshold.reason == "no_capability_with_sufficient_marginal_gain"


def test_ties_prefer_lower_cost_then_action_id() -> None:
    demand = EvidenceDemand("check", EvidenceType.CONSTRAINT_CHECK, "Validate option")

    plan = EvidenceGapRepairPlanner(max_workers=1).plan(
        demands=[demand],
        evidence=[],
        validation_scores={},
        capabilities=[
            _capability("z", EvidenceType.CONSTRAINT_CHECK, 0.5, cost=0.3),
            _capability("b", EvidenceType.CONSTRAINT_CHECK, 0.5, cost=0.1),
            _capability("a", EvidenceType.CONSTRAINT_CHECK, 0.5, cost=0.1),
        ],
    )

    assert plan.selected_action_ids == ("a",)


def test_planner_accepts_a_frozen_residual_gap_vector() -> None:
    plan = EvidenceGapRepairPlanner(max_workers=2, min_residual_gap=0.05).plan_from_gaps(
        residual_gaps={
            EvidenceType.EXACT_SPAN: 0.9,
            EvidenceType.NUMERICAL_DERIVATION: 0.4,
        },
        capabilities=[
            CapabilityCard(
                action_id="cross-type-worker",
                supply_by_type={
                    EvidenceType.EXACT_SPAN: 0.6,
                    EvidenceType.NUMERICAL_DERIVATION: 0.8,
                },
                estimated_cost=1.0,
            ),
            _capability("ocr", EvidenceType.EXACT_SPAN, 0.7),
        ],
    )

    assert plan.selected_action_ids == ("cross-type-worker", "ocr")
    assert plan.steps[0].marginal_coverage_gain == pytest.approx(0.86)
    assert {gap.evidence_type for gap in plan.gaps} == {
        EvidenceType.EXACT_SPAN,
        EvidenceType.NUMERICAL_DERIVATION,
    }


def test_planner_rejects_invalid_external_gap_values() -> None:
    with pytest.raises(EvidenceRepairError, match="finite and non-negative"):
        EvidenceGapRepairPlanner(max_workers=1).plan_from_gaps(
            residual_gaps={EvidenceType.EXACT_SPAN: -0.1},
            capabilities=[_capability("ocr", EvidenceType.EXACT_SPAN, 0.8)],
        )


def test_evidence_requires_support_and_provenance() -> None:
    with pytest.raises(EvidenceRepairError, match="provenance"):
        EvidenceCard(
            "card",
            EvidenceType.EXTERNAL_FACT,
            claim="claim",
            support="support",
            provenance="",
            producer_id="worker",
            confidence=0.5,
        )


def test_validation_scores_must_reference_known_objects() -> None:
    with pytest.raises(EvidenceRepairError, match="unknown"):
        EvidenceGapRepairPlanner(max_workers=1).plan(
            demands=[EvidenceDemand("d", EvidenceType.EXACT_SPAN, "read")],
            evidence=[],
            validation_scores={("d", "missing"): 0.5},
            capabilities=[_capability("ocr", EvidenceType.EXACT_SPAN, 0.8)],
        )


def _capability(
    action_id: str,
    evidence_type: EvidenceType,
    supply: float,
    *,
    cost: float = 0.1,
) -> CapabilityCard:
    return CapabilityCard(
        action_id=action_id,
        supply_by_type={evidence_type: supply},
        estimated_cost=cost,
    )
