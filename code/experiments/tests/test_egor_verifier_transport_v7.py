from __future__ import annotations

import json
from typing import Any

from openagora_experiments.egor.protocol import (
    PilotTask,
    parse_verifier_output_v6,
    parse_verifier_output_v7,
)
from openagora_experiments.egor.runner import run_verifier_v7_stage
from openagora_experiments.egor.runtime_types import ArtifactSpec


def payload() -> str:
    return json.dumps(
        {
            "answer": "7",
            "evidence": [{
                "evidence_id": "v1", "type": "exact_span", "claim": "value is seven",
                "support": "the visible value is 7", "provenance": "center label",
                "confidence": 1.0,
            }],
        }
    )


class Provider:
    model_name = "fake"
    def complete_multimodal(self, *, system_prompt: str, user_prompt: str,
                            artifacts: list[ArtifactSpec], temperature: float,
                            seed: int, request_id: str | None = None) -> str:
        return payload() + "`"
    def pop_audit_events(self) -> list[dict[str, Any]]:
        return []


def task() -> PilotTask:
    return PilotTask(
        task_id="t", source_id="s", question="Read the value.",
        image_path="/tmp/x.png", mime_type="image/png", metric="exact_match",
        answer="SECRET", reference_answers=("SECRET",), options=(), metadata={},
    )


def test_v7_accepts_only_one_proven_trailing_backtick() -> None:
    raw = payload()
    old, _ = parse_verifier_output_v6(raw + "`")
    assert not old["transport"]["operational_complete"]

    parsed, issues = parse_verifier_output_v7(raw + "`")
    assert parsed["answer"] == "7"
    assert parsed["transport"]["operational_complete"]
    assert parsed["transport"]["mode"] == "stripped_single_trailing_backtick"
    assert not issues

    for invalid in (raw + "``", raw + "x", raw[:-8] + "`"):
        parsed, issues = parse_verifier_output_v7(invalid)
        assert not parsed["transport"]["operational_complete"]
        assert issues


def test_v7_runner_binds_stage_and_parser_contract() -> None:
    record = list(run_verifier_v7_stage(provider=Provider(), tasks=[task()], seed=1))[0]
    assert record["status"] == "ok"
    assert record["stage"] == "verifier_v7"
    assert record["parser_contract"] == (
        "v7_bounded_trailing_delimiter_and_safe_abstention"
    )
    assert record["parsed"]["answer"] == "7"
