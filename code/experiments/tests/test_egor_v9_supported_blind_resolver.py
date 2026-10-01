from __future__ import annotations

import inspect

from openagora_experiments.egor.verification_v9 import (
    decide_supported_blind_resolver_overwrite,
)


def resolver(answer: str, *, operational: bool = True) -> dict:
    return {
        "answer": answer,
        "transport": {"operational_complete": operational},
    }


def specialist(answer: str, *, evidence: bool = True) -> dict:
    return {
        "candidate_answer": answer,
        "evidence": (
            [
                {
                    "evidence_id": "e1",
                    "claim": "visible value",
                    "support": answer,
                    "provenance": "image center",
                }
            ]
            if evidence
            else []
        ),
    }


def test_v9_accepts_blind_answer_with_one_evidence_bearing_specialist() -> None:
    decision = decide_supported_blind_resolver_overwrite(
        baseline_answer="A",
        resolver=resolver("B"),
        specialists={
            "exact": specialist("B"),
            "visual": specialist("C"),
        },
        metric="exact_match",
    )
    assert decision.accept_overwrite
    assert decision.reason == "supported_blind_resolver_overwrite"
    assert decision.corroborating_specialists == ("exact",)


def test_v9_falls_back_for_no_support_same_answer_or_transport_failure() -> None:
    no_support = decide_supported_blind_resolver_overwrite(
        baseline_answer="A",
        resolver=resolver("B"),
        specialists={"exact": specialist("B", evidence=False)},
        metric="exact_match",
    )
    assert not no_support.accept_overwrite
    assert no_support.reason == "no_heterogeneous_corroborator"

    same = decide_supported_blind_resolver_overwrite(
        baseline_answer="B",
        resolver=resolver("B"),
        specialists={"exact": specialist("B")},
        metric="exact_match",
    )
    assert not same.accept_overwrite
    assert same.reason == "no_baseline_overwrite"

    failed = decide_supported_blind_resolver_overwrite(
        baseline_answer="A",
        resolver=resolver("B", operational=False),
        specialists={"exact": specialist("B")},
        metric="exact_match",
    )
    assert not failed.accept_overwrite
    assert failed.reason == "blind_resolver_incomplete"


def test_v9_two_specialist_control_is_stricter_and_has_no_outcome_inputs() -> None:
    decision = decide_supported_blind_resolver_overwrite(
        baseline_answer="A",
        resolver=resolver("B"),
        specialists={
            "exact": specialist("B"),
            "visual": specialist("C"),
        },
        metric="exact_match",
        minimum_specialist_corroborators=2,
    )
    assert not decision.accept_overwrite
    forbidden = {"source_id", "reference", "score", "realized_outcome"}
    assert forbidden.isdisjoint(
        inspect.signature(
            decide_supported_blind_resolver_overwrite
        ).parameters
    )
