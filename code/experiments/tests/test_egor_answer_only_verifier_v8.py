from __future__ import annotations

import inspect
from typing import Any

from openagora_experiments.egor.protocol import (
    PilotTask,
    parse_answer_only_verifier_output,
)
from openagora_experiments.egor.runner import run_verifier_v8_stage
from openagora_experiments.egor.runtime_types import ArtifactSpec
from openagora_experiments.egor.verification import (
    decide_answer_only_verified_overwrite,
)


class Provider:
    model_name = "fake"

    def __init__(self, response: str = "B") -> None:
        self.response = response
        self.request_id = ""

    def complete_multimodal(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        artifacts: list[ArtifactSpec],
        temperature: float,
        seed: int,
        request_id: str | None = None,
    ) -> str:
        assert "blind" in system_prompt.lower()
        assert "only the final answer" in system_prompt.lower()
        assert "SECRET" not in user_prompt
        self.request_id = str(request_id)
        return self.response

    def pop_audit_events(self) -> list[dict[str, Any]]:
        return []


def task() -> PilotTask:
    return PilotTask(
        task_id="t",
        source_id="s",
        question="Which option is correct?",
        image_path="/tmp/x.png",
        mime_type="image/png",
        metric="exact_match",
        answer="SECRET",
        reference_answers=("SECRET",),
        options=(("A", "one"), ("B", "two")),
        metadata={},
    )


def specialist(answer: str = "B") -> dict[str, Any]:
    return {
        "candidate_answer": answer,
        "evidence": [
            {
                "evidence_id": "e1",
                "claim": "option B is visible",
                "support": "the marked option is B",
                "provenance": "center of image",
            }
        ],
    }


def test_v8_parser_accepts_one_line_and_explicit_abstention() -> None:
    parsed, issues = parse_answer_only_verifier_output("B")
    assert not issues
    assert parsed["answer"] == "B"
    assert parsed["evidence"] == []
    assert parsed["transport"] == {
        "mode": "plain_single_line",
        "operational_complete": True,
        "sanitized_abstention": False,
        "decisive": True,
    }

    abstained, issues = parse_answer_only_verifier_output("<ABSTAIN>")
    assert not issues
    assert abstained["answer"] == ""
    assert abstained["transport"]["operational_complete"]
    assert abstained["transport"]["sanitized_abstention"]


def test_v8_parser_safely_abstains_on_nonconforming_transport() -> None:
    for raw in ("B\nbecause it is visible", "```B```", "x" * 257, ""):
        parsed, issues = parse_answer_only_verifier_output(raw)
        assert issues
        assert parsed["transport"]["operational_complete"]
        assert parsed["transport"]["sanitized_abstention"]
        assert not parsed["transport"]["decisive"]
        assert parsed["transport"]["mode"] == "sanitized_transport_abstention"
        assert parsed["answer"] == ""


def test_v8_runner_binds_blind_prompt_parser_and_token_contract() -> None:
    provider = Provider()
    record = list(
        run_verifier_v8_stage(provider=provider, tasks=[task()], seed=3)
    )[0]
    assert record["status"] == "ok"
    assert record["stage"] == "verifier_v8"
    assert record["information_boundary"] == "question_options_image_only"
    assert record["parser_contract"] == (
        "v10_answer_only_with_fail_closed_abstention"
    )
    assert record["max_new_tokens_contract"] == 64
    assert record["parsed"]["answer"] == "B"
    assert "answer_only_verifier_v8" in provider.request_id


def test_v8_authorization_separates_verifier_answer_from_specialist_evidence() -> None:
    verifier, _ = parse_answer_only_verifier_output("B")
    decision = decide_answer_only_verified_overwrite(
        baseline_answer="A",
        core_answer="B",
        verifier=verifier,
        specialists={"exact": specialist()},
        metric="exact_match",
    )
    assert decision.accept_overwrite
    assert decision.reason == "answer_only_independently_verified_overwrite"
    assert decision.corroborating_specialists == ("exact",)
    assert decision.verifier_evidence_count == 0

    no_evidence = decide_answer_only_verified_overwrite(
        baseline_answer="A",
        core_answer="B",
        verifier=verifier,
        specialists={"exact": {"candidate_answer": "B", "evidence": []}},
        metric="exact_match",
    )
    assert not no_evidence.accept_overwrite
    assert no_evidence.reason == "no_heterogeneous_corroborator"

    forbidden = {"source_id", "reference", "score", "realized_outcome"}
    assert forbidden.isdisjoint(
        inspect.signature(decide_answer_only_verified_overwrite).parameters
    )
