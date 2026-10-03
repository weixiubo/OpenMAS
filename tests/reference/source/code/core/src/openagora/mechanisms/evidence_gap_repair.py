"""Typed evidence-gap assessment and minimal capability recruitment for EGOR."""

from __future__ import annotations

from math import isfinite, prod
from typing import Mapping, Sequence

from openagora.mechanisms.evidence_types import (
    CapabilityCard as CapabilityCard,
)
from openagora.mechanisms.evidence_types import (
    EvidenceCard as EvidenceCard,
)
from openagora.mechanisms.evidence_types import (
    EvidenceDemand as EvidenceDemand,
)
from openagora.mechanisms.evidence_types import (
    EvidenceRepairError as EvidenceRepairError,
)
from openagora.mechanisms.evidence_types import (
    EvidenceRepairPlan as EvidenceRepairPlan,
)
from openagora.mechanisms.evidence_types import (
    EvidenceType as EvidenceType,
)
from openagora.mechanisms.evidence_types import (
    GapAssessment as GapAssessment,
)
from openagora.mechanisms.evidence_types import (
    RepairStep as RepairStep,
)


class EvidenceGapRepairPlanner:
    """Recruit the smallest high-marginal-gain capability set under a hard cap."""

    def __init__(
        self,
        *,
        max_workers: int,
        min_residual_gap: float = 0.0,
        min_marginal_gain: float = 0.0,
    ) -> None:
        if max_workers < 0:
            raise EvidenceRepairError("max_workers must be non-negative")
        if min_residual_gap < 0.0 or min_marginal_gain < 0.0:
            raise EvidenceRepairError("gap and gain thresholds must be non-negative")
        self._max_workers = max_workers
        self._min_residual_gap = min_residual_gap
        self._min_marginal_gain = min_marginal_gain

    def plan(
        self,
        *,
        demands: Sequence[EvidenceDemand],
        evidence: Sequence[EvidenceCard],
        validation_scores: Mapping[tuple[str, str], float],
        capabilities: Sequence[CapabilityCard],
    ) -> EvidenceRepairPlan:
        """Assess residual gaps and greedily maximize the submodular coverage surrogate."""

        self._validate_unique_ids(demands=demands, evidence=evidence, capabilities=capabilities)
        gaps = self._assess_gaps(
            demands=demands,
            evidence=evidence,
            validation_scores=validation_scores,
        )
        baseline_coverage = sum(demand.weight for demand in demands) - sum(gap.residual_gap for gap in gaps)
        return self._plan_assessed_gaps(
            gaps=gaps,
            baseline_coverage=baseline_coverage,
            capabilities=capabilities,
        )

    def plan_from_gaps(
        self,
        *,
        residual_gaps: Mapping[EvidenceType, float],
        capabilities: Sequence[CapabilityCard],
    ) -> EvidenceRepairPlan:
        """Plan from a frozen external gap assessment without reconstructing evidence."""

        self._validate_unique_ids(demands=(), evidence=(), capabilities=capabilities)
        invalid = [
            (evidence_type, value)
            for evidence_type, value in residual_gaps.items()
            if not isfinite(float(value)) or float(value) < 0.0
        ]
        if invalid:
            raise EvidenceRepairError("residual gaps must be finite and non-negative")
        gaps = tuple(
            GapAssessment(
                demand_id=f"residual::{evidence_type.value}",
                evidence_type=evidence_type,
                validated_coverage=0.0,
                residual_gap=float(value),
            )
            for evidence_type, value in sorted(residual_gaps.items(), key=lambda item: item[0].value)
        )
        return self._plan_assessed_gaps(
            gaps=gaps,
            baseline_coverage=0.0,
            capabilities=capabilities,
        )

    def _plan_assessed_gaps(
        self,
        *,
        gaps: tuple[GapAssessment, ...],
        baseline_coverage: float,
        capabilities: Sequence[CapabilityCard],
    ) -> EvidenceRepairPlan:
        active_gaps = tuple(gap for gap in gaps if gap.residual_gap > self._min_residual_gap)
        if not active_gaps:
            return EvidenceRepairPlan(
                selected_action_ids=(),
                gaps=gaps,
                steps=(),
                baseline_coverage=baseline_coverage,
                repaired_coverage=baseline_coverage,
                total_estimated_cost=0.0,
                reason="no_residual_evidence_gap",
            )
        if self._max_workers == 0:
            return EvidenceRepairPlan(
                selected_action_ids=(),
                gaps=gaps,
                steps=(),
                baseline_coverage=baseline_coverage,
                repaired_coverage=baseline_coverage,
                total_estimated_cost=0.0,
                reason="worker_budget_is_zero",
            )

        remaining = {card.action_id: card for card in capabilities if card.admissible}
        selected: list[CapabilityCard] = []
        steps: list[RepairStep] = []
        current_gain = 0.0
        for _ in range(self._max_workers):
            candidates = [
                (
                    self._coverage_gain(active_gaps, [*selected, card]) - current_gain,
                    card,
                )
                for card in remaining.values()
            ]
            if not candidates:
                break
            marginal_gain, chosen = min(
                candidates,
                key=lambda item: (
                    -item[0],
                    item[1].estimated_cost,
                    item[1].action_id,
                ),
            )
            if marginal_gain <= self._min_marginal_gain:
                break
            selected.append(chosen)
            remaining.pop(chosen.action_id)
            current_gain += marginal_gain
            steps.append(
                RepairStep(
                    action_id=chosen.action_id,
                    marginal_coverage_gain=marginal_gain,
                    cumulative_coverage=baseline_coverage + current_gain,
                    estimated_cost=chosen.estimated_cost,
                )
            )

        selected_ids = tuple(card.action_id for card in selected)
        reason = "positive_gap_coverage_repair" if selected else "no_capability_with_sufficient_marginal_gain"
        return EvidenceRepairPlan(
            selected_action_ids=selected_ids,
            gaps=gaps,
            steps=tuple(steps),
            baseline_coverage=baseline_coverage,
            repaired_coverage=baseline_coverage + current_gain,
            total_estimated_cost=sum(card.estimated_cost for card in selected),
            reason=reason,
        )

    @staticmethod
    def _validate_unique_ids(
        *,
        demands: Sequence[EvidenceDemand],
        evidence: Sequence[EvidenceCard],
        capabilities: Sequence[CapabilityCard],
    ) -> None:
        for label, values in (
            ("demand", [item.demand_id for item in demands]),
            ("evidence", [item.evidence_id for item in evidence]),
            ("capability", [item.action_id for item in capabilities]),
        ):
            if len(values) != len(set(values)):
                raise EvidenceRepairError(f"{label} ids must be unique")

    @staticmethod
    def _assess_gaps(
        *,
        demands: Sequence[EvidenceDemand],
        evidence: Sequence[EvidenceCard],
        validation_scores: Mapping[tuple[str, str], float],
    ) -> tuple[GapAssessment, ...]:
        demand_ids = {demand.demand_id for demand in demands}
        evidence_ids = {card.evidence_id for card in evidence}
        for (demand_id, evidence_id), score in validation_scores.items():
            if demand_id not in demand_ids or evidence_id not in evidence_ids:
                raise EvidenceRepairError("validation score references an unknown demand or evidence card")
            if not 0.0 <= float(score) <= 1.0:
                raise EvidenceRepairError("validation scores must lie in [0, 1]")
        assessments: list[GapAssessment] = []
        for demand in demands:
            compatible = [
                float(validation_scores.get((demand.demand_id, card.evidence_id), 0.0))
                for card in evidence
                if card.evidence_type == demand.evidence_type
            ]
            coverage = max(compatible, default=0.0)
            assessments.append(
                GapAssessment(
                    demand_id=demand.demand_id,
                    evidence_type=demand.evidence_type,
                    validated_coverage=coverage,
                    residual_gap=demand.weight * (1.0 - coverage),
                )
            )
        return tuple(assessments)

    @staticmethod
    def _coverage_gain(
        gaps: Sequence[GapAssessment],
        selected: Sequence[CapabilityCard],
    ) -> float:
        return sum(
            gap.residual_gap * (1.0 - prod(1.0 - card.supply(gap.evidence_type) for card in selected)) for gap in gaps
        )
