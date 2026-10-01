from __future__ import annotations

import inspect

from openagora_experiments.egor.verification import (
    answers_agree,
    canonical_answer,
    decide_verified_overwrite,
)


def _record(answer: str, *, supported: bool = True) -> dict:
    return {
        "candidate_answer": answer,
        "evidence": (
            [
                {
                    "evidence_id": "s1",
                    "claim": "independent result",
                    "support": "visible or derived support",
                    "provenance": "independent image inspection",
                }
            ]
            if supported
            else []
        ),
    }


def test_verified_overwrite_requires_verifier_and_heterogeneous_support() -> None:
    decision = decide_verified_overwrite(
        baseline_answer="B",
        core_answer="H",
        verifier=_record("H"),
        specialists={"exact": _record("H"), "visual": _record("B")},
        metric="choice_accuracy",
        options=[{"id": "B", "text": "3.5"}, {"id": "H", "text": "36"}],
    )
    assert decision.accept_overwrite
    assert decision.reason == "independently_verified_overwrite"
    assert decision.corroborating_specialists == ("exact",)


def test_verified_overwrite_rejects_missing_or_disagreeing_verifier() -> None:
    missing = decide_verified_overwrite(
        baseline_answer="deer",
        core_answer="camel",
        verifier=_record("", supported=False),
        specialists={"visual": _record("camel")},
        metric="accuracy",
    )
    disagreeing = decide_verified_overwrite(
        baseline_answer="deer",
        core_answer="camel",
        verifier=_record("deer"),
        specialists={"visual": _record("camel")},
        metric="accuracy",
    )
    assert not missing.accept_overwrite
    assert missing.reason == "verifier_incomplete"
    assert not disagreeing.accept_overwrite
    assert disagreeing.reason == "verifier_disagrees"


def test_verified_overwrite_rejects_uncorroborated_or_unchanged_answer() -> None:
    uncorroborated = decide_verified_overwrite(
        baseline_answer="old",
        core_answer="new",
        verifier=_record("new"),
        specialists={"visual": _record("other")},
        metric="accuracy",
    )
    unchanged = decide_verified_overwrite(
        baseline_answer="The output is [1, 2]",
        core_answer="[1, 2]",
        verifier=_record("[1, 2]"),
        specialists={"visual": _record("[1, 2]")},
        metric="accuracy",
    )
    assert not uncorroborated.accept_overwrite
    assert uncorroborated.reason == "no_heterogeneous_corroborator"
    assert not unchanged.accept_overwrite
    assert unchanged.reason == "no_baseline_overwrite"


def test_answer_agreement_handles_json_wrappers_and_output_prefixes() -> None:
    assert canonical_answer('{"answer": "H"}') == "h"
    assert canonical_answer('{"answer": "", "open_gap_types": ["exact_span"]}') == ""
    assert answers_agree(
        "The output is [[], [1], [2], [1, 2]]",
        "[[], [1], [2], [1, 2]]",
        metric="accuracy",
    )


def test_verification_interface_has_no_outcome_or_source_inputs() -> None:
    forbidden = {"source_id", "benchmark", "answer", "reference_answers", "score"}
    assert forbidden.isdisjoint(inspect.signature(decide_verified_overwrite).parameters)
