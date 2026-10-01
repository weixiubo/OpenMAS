from __future__ import annotations

import json
from typing import Any

from openagora_experiments.egor.protocol import (
    PilotTask,
    parse_verifier_output_v6,
)
from openagora_experiments.egor.runner import run_verifier_v6_stage
from openagora_experiments.egor.runtime_types import ArtifactSpec


def valid_payload() -> dict[str, Any]:
    return {
        "answer": "H",
        "evidence": [
            {
                "evidence_id": "v1",
                "type": "visual_relation",
                "claim": "the second option matches",
                "support": "the marked object aligns with the second label",
                "provenance": "center image region",
                "confidence": 0.9,
            }
        ],
    }


def task() -> PilotTask:
    return PilotTask(
        task_id="t1",
        source_id="fixture",
        question="Which option is supported?",
        image_path="/tmp/fixture.png",
        mime_type="image/png",
        metric="choice_accuracy",
        answer="SECRET_GOLD",
        reference_answers=("SECRET_GOLD",),
        options=(("A", "first"), ("H", "second")),
        metadata={},
    )


class FakeProvider:
    model_name = "fake-v6-verifier"

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "artifacts": artifacts,
                "temperature": temperature,
                "seed": seed,
                "request_id": request_id,
            }
        )
        return self.output

    def pop_audit_events(self) -> list[dict[str, Any]]:
        return []


def test_v6_accepts_exact_fenced_and_one_missing_outer_brace() -> None:
    raw = json.dumps(valid_payload())
    for text, expected_mode in (
        (raw, "exact_json"),
        (f"```json\n{raw}\n```", "single_json_fence"),
        (raw[:-1], "appended_missing_outer_brace"),
    ):
        parsed, issues = parse_verifier_output_v6(text)
        assert parsed["answer"] == "H"
        assert parsed["transport"]["operational_complete"]
        assert parsed["transport"]["decisive"]
        assert parsed["transport"]["mode"] == expected_mode
        assert not issues


def test_v6_rejects_nonlocal_truncation_nested_recovery_and_extra_keys() -> None:
    payload = json.dumps(valid_payload())
    malformed = (
        payload[:-8],
        'prefix {"answer":"H","evidence":[]}',
        '{"answer":"H","evidence":[],"extra":1}',
    )
    for text in malformed:
        parsed, issues = parse_verifier_output_v6(text)
        assert parsed["answer"] == ""
        assert not parsed["transport"]["operational_complete"]
        assert issues


def test_v6_counts_explicit_or_sanitized_abstention_as_operational() -> None:
    explicit, issues = parse_verifier_output_v6('{"answer":"","evidence":[]}')
    assert explicit["transport"]["operational_complete"]
    assert explicit["transport"]["sanitized_abstention"]
    assert not issues

    invalid = valid_payload()
    invalid["evidence"][0]["support"] = ""
    sanitized, issues = parse_verifier_output_v6(json.dumps(invalid))
    assert sanitized["answer"] == ""
    assert sanitized["evidence"] == []
    assert sanitized["transport"]["operational_complete"]
    assert sanitized["transport"]["sanitized_abstention"]
    assert issues


def test_v6_runner_is_answer_blind_and_records_parser_contract() -> None:
    provider = FakeProvider(json.dumps(valid_payload()))
    record = list(run_verifier_v6_stage(provider=provider, tasks=[task()], seed=9))[0]
    assert record["status"] == "ok"
    assert record["stage"] == "verifier_v6"
    assert record["information_boundary"] == "question_options_image_only"
    assert record["parser_contract"] == "v6_bounded_transport_and_safe_abstention"
    assert record["parsed"]["transport"]["operational_complete"]
    assert "SECRET_GOLD" not in provider.calls[0]["user_prompt"]
    assert "independent_verifier_v6" in provider.calls[0]["request_id"]
