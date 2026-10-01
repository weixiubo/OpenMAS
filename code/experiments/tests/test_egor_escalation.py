from __future__ import annotations

from openagora_experiments.egor.escalation import (
    canonical_boolean,
    decide_evidence_escalation,
)


def _baseline(answer: str = "no") -> dict:
    return {
        "answer": answer,
        "evidence": [
            {
                "evidence_id": "baseline::b1",
                "producer_id": "baseline",
                "type": "exact_span",
                "claim": "baseline claim",
                "support": "baseline support",
                "provenance": "image region",
                "confidence": 0.9,
            }
        ],
    }


def _specialist(producer: str, answer: str) -> dict:
    return {
        "candidate_answer": answer,
        "evidence": [
            {
                "evidence_id": f"{producer}::s1",
                "producer_id": producer,
                "type": "exact_span",
                "claim": f"answer is {answer}",
                "support": "specialist support",
                "provenance": "image region",
                "confidence": 1.0,
            }
        ],
    }


def _synthesis(
    answer: str = "yes",
    *,
    used: list[str] | None = None,
    remaining: list[str] | None = None,
) -> dict:
    return {
        "answer": answer,
        "changed_baseline": answer != "no",
        "used_evidence_ids": used or [],
        "remaining_gap_types": remaining or [],
    }


def test_gate_abstention_does_not_consume_core_records() -> None:
    decision = decide_evidence_escalation(
        repairability_triggered=False,
        baseline={},
        exact_specialist={},
        visual_specialist={},
        core_synthesis={},
    )

    assert decision.tier == "baseline"
    assert decision.reason == "repairability_gate_abstained"
    assert decision.core_audit is None


def test_certified_core_change_stops_at_pair() -> None:
    decision = decide_evidence_escalation(
        repairability_triggered=True,
        baseline=_baseline(),
        exact_specialist=_specialist("exact", "yes"),
        visual_specialist=_specialist("visual", "yes"),
        core_synthesis=_synthesis("yes", used=["exact::s1"]),
    )

    assert decision.tier == "core_pair"
    assert decision.reason == "core_certificate_satisfied"
    assert decision.core_audit is not None
    assert decision.core_audit.evidence_supported_change


def test_remaining_gap_escalates_even_when_core_change_is_certified() -> None:
    decision = decide_evidence_escalation(
        repairability_triggered=True,
        baseline=_baseline(),
        exact_specialist=_specialist("exact", "yes"),
        visual_specialist=_specialist("visual", "yes"),
        core_synthesis=_synthesis(
            "yes",
            used=["exact::s1"],
            remaining=["independent_verification"],
        ),
    )

    assert decision.tier == "full_team"
    assert decision.reason == "core_gap_remains"


def test_uncertified_changed_answer_escalates() -> None:
    decision = decide_evidence_escalation(
        repairability_triggered=True,
        baseline=_baseline(),
        exact_specialist=_specialist("exact", "yes"),
        visual_specialist=_specialist("visual", "yes"),
        core_synthesis=_synthesis("yes", used=["baseline::b1"]),
    )

    assert decision.tier == "full_team"
    assert decision.reason == "core_change_uncertified"


def test_unchanged_boolean_contradiction_escalates() -> None:
    decision = decide_evidence_escalation(
        repairability_triggered=True,
        baseline=_baseline(),
        exact_specialist=_specialist("exact", "No."),
        visual_specialist=_specialist("visual", "YES -- the bars cross."),
        core_synthesis=_synthesis("no", used=["exact::s1"]),
    )

    assert decision.tier == "full_team"
    assert decision.reason == "unchanged_under_boolean_disagreement"
    assert decision.boolean_disagreement


def test_nonboolean_disagreement_does_not_force_escalation() -> None:
    decision = decide_evidence_escalation(
        repairability_triggered=True,
        baseline=_baseline(answer="17"),
        exact_specialist=_specialist("exact", "17.8"),
        visual_specialist=_specialist("visual", "17.9"),
        core_synthesis={
            "answer": "17",
            "changed_baseline": False,
            "used_evidence_ids": ["baseline::b1"],
            "remaining_gap_types": [],
        },
    )

    assert decision.tier == "core_pair"
    assert not decision.boolean_disagreement


def test_canonical_boolean_requires_an_explicit_leading_answer() -> None:
    assert canonical_boolean(" YES, because the bars cross.") == "yes"
    assert canonical_boolean("false") == "no"
    assert canonical_boolean("There is no visible label") == ""
    assert canonical_boolean("17.8%") == ""
