"""Typed evidence and capability contracts shared by EGOR planners."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


class EvidenceRepairError(ValueError):
    """Raised when an evidence-repair object violates the frozen grammar."""

class EvidenceType(StrEnum):
    """Finite evidence grammar used by the first EGOR implementation."""

    EXACT_SPAN = "exact_span"
    VISUAL_RELATION = "visual_relation"
    NUMERICAL_DERIVATION = "numerical_derivation"
    EXTERNAL_FACT = "external_fact"
    CONSTRAINT_CHECK = "constraint_check"
    INDEPENDENT_VERIFICATION = "independent_verification"

@dataclass(frozen=True)
class EvidenceDemand:
    """One verifiable requirement that the current organization should satisfy."""

    demand_id: str
    evidence_type: EvidenceType
    requirement: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.demand_id.strip() or not self.requirement.strip():
            raise EvidenceRepairError("demand id and requirement must be non-empty")
        if not 0.0 <= self.weight <= 1.0:
            raise EvidenceRepairError("demand weight must lie in [0, 1]")

@dataclass(frozen=True)
class EvidenceCard:
    """Provenance-bearing evidence produced by the baseline or a recruited worker."""

    evidence_id: str
    evidence_type: EvidenceType
    claim: str
    support: str
    provenance: str
    producer_id: str
    confidence: float

    def __post_init__(self) -> None:
        required = (
            self.evidence_id,
            self.claim,
            self.support,
            self.provenance,
            self.producer_id,
        )
        if any(not value.strip() for value in required):
            raise EvidenceRepairError("evidence id, claim, support, provenance, and producer are required")
        if not 0.0 <= self.confidence <= 1.0:
            raise EvidenceRepairError("evidence confidence must lie in [0, 1]")

@dataclass(frozen=True)
class CapabilityCard:
    """Empirical probability that a repair action closes each typed gap."""

    action_id: str
    supply_by_type: Mapping[EvidenceType, float]
    estimated_cost: float
    heterogeneous: bool = True
    admissible: bool = True

    def __post_init__(self) -> None:
        if not self.action_id.strip():
            raise EvidenceRepairError("repair action id must be non-empty")
        if self.estimated_cost < 0.0:
            raise EvidenceRepairError("estimated cost must be non-negative")
        if not self.supply_by_type:
            raise EvidenceRepairError("a capability card must supply at least one evidence type")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.supply_by_type.values()):
            raise EvidenceRepairError("capability supply probabilities must lie in [0, 1]")

    def supply(self, evidence_type: EvidenceType) -> float:
        return float(self.supply_by_type.get(evidence_type, 0.0))

@dataclass(frozen=True)
class GapAssessment:
    demand_id: str
    evidence_type: EvidenceType
    validated_coverage: float
    residual_gap: float

@dataclass(frozen=True)
class RepairStep:
    action_id: str
    marginal_coverage_gain: float
    cumulative_coverage: float
    estimated_cost: float

@dataclass(frozen=True)
class EvidenceRepairPlan:
    """Auditable greedy plan over one finite repair grammar."""

    selected_action_ids: tuple[str, ...]
    gaps: tuple[GapAssessment, ...]
    steps: tuple[RepairStep, ...]
    baseline_coverage: float
    repaired_coverage: float
    total_estimated_cost: float
    reason: str

    @property
    def repaired(self) -> bool:
        return bool(self.selected_action_ids)
