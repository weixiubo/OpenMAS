"""Outcome-blind evidence signals for staged EGOR team escalation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from openagora_experiments.egor.evaluation import SynthesisAudit, audit_synthesis


EscalationTier = Literal["baseline", "core_pair", "full_team"]


@dataclass(frozen=True)
class EvidenceEscalationDecision:
    """One outcome-blind decision after the core pair has executed."""

    tier: EscalationTier
    reason: str
    core_audit: SynthesisAudit | None
    boolean_disagreement: bool = False

    @property
    def triggered(self) -> bool:
        return self.tier != "baseline"

    @property
    def escalated(self) -> bool:
        return self.tier == "full_team"


def decide_evidence_escalation(
    *,
    repairability_triggered: bool,
    baseline: Mapping[str, Any],
    exact_specialist: Mapping[str, Any],
    visual_specialist: Mapping[str, Any],
    core_synthesis: Mapping[str, Any],
    core_complete: bool = True,
) -> EvidenceEscalationDecision:
    """Implement ``sec:breadth-rule`` and ``eq:breadth-rule`` outcome-blindly.

    The second-stage decision consumes only records that would exist after the
    exact--visual core pair has run.  It escalates an incomplete core stage, an
    unresolved typed gap, an uncertified changed answer, or an unchanged answer
    under an exact/visual Boolean contradiction.  Gold answers, task scores,
    numerical/verifier records, and full-team records are deliberately absent
    from the interface. Its four cases are ``eq:escalation-predicates``.
    """

    if not repairability_triggered:
        return EvidenceEscalationDecision(
            tier="baseline",
            reason="repairability_gate_abstained",
            core_audit=None,
        )

    audit = audit_synthesis(
        baseline=baseline,
        specialists=(exact_specialist, visual_specialist),
        synthesis=core_synthesis,
    )
    exact_boolean = canonical_boolean(exact_specialist.get("candidate_answer", ""))
    visual_boolean = canonical_boolean(
        visual_specialist.get("candidate_answer", "")
    )
    boolean_disagreement = bool(
        exact_boolean and visual_boolean and exact_boolean != visual_boolean
    )

    if not core_complete:
        reason = "core_stage_incomplete"
    elif _strings(core_synthesis.get("remaining_gap_types", [])):
        reason = "core_gap_remains"
    elif audit.actual_changed and not audit.evidence_supported_change:
        reason = "core_change_uncertified"
    elif not audit.actual_changed and boolean_disagreement:
        reason = "unchanged_under_boolean_disagreement"
    else:
        return EvidenceEscalationDecision(
            tier="core_pair",
            reason="core_certificate_satisfied",
            core_audit=audit,
            boolean_disagreement=boolean_disagreement,
        )

    return EvidenceEscalationDecision(
        tier="full_team",
        reason=reason,
        core_audit=audit,
        boolean_disagreement=boolean_disagreement,
    )


def canonical_boolean(value: Any) -> str:
    """Canonicalize the narrow conflict in ``eq:escalation-predicates``.

    This helper is part of the breadth rule in ``sec:breadth-rule`` and returns
    ``yes``/``no`` only for an explicit leading Boolean answer.
    """

    tokens = re.findall(r"[a-z]+", str(value).strip().lower())
    if not tokens:
        return ""
    if tokens[0] in {"yes", "true"}:
        return "yes"
    if tokens[0] in {"no", "false"}:
        return "no"
    return ""


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
