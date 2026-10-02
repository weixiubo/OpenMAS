from __future__ import annotations

import json
from typing import Any

from openagora_experiments.egor.parsing import parse_verifier_output
from openagora_experiments.egor.prompts import build_verifier_prompt
from openagora_experiments.egor.runner import run_verifier_stage
from openagora_experiments.egor.runtime_types import ArtifactSpec
from openagora_experiments.egor.task_model import PilotTask


class FakeProvider:
    model_name = "fake-verifier"

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


def task(*, answer: str = "SECRET_GOLD") -> PilotTask:
    return PilotTask(
        task_id="t1",
        source_id="fixture",
        question="Which option is supported by the image?",
        image_path="/tmp/fixture.png",
        mime_type="image/png",
        metric="choice_accuracy",
        answer=answer,
        reference_answers=(answer,),
        options=(("A", "first"), ("H", "second")),
        metadata={},
    )


def valid_output() -> str:
    return json.dumps(
        {
            "answer": "H",
            "evidence": [
                {
                    "evidence_id": "v1",
                    "type": "visual_relation",
                    "claim": "the second option matches",
                    "support": "the marked object is aligned with the second label",
                    "provenance": "center image region",
                    "confidence": 0.9,
                }
            ],
        }
    )


def test_compact_verifier_prompt_and_stage_are_answer_blind() -> None:
    first = task(answer="SECRET_A")
    second = task(answer="SECRET_B")
    assert build_verifier_prompt(first) == build_verifier_prompt(second)
    system, user = build_verifier_prompt(first)
    assert "blind" in system
    assert "SECRET_A" not in user
    assert "demands" in user
    assert "Do not output demands" in user

    provider = FakeProvider(valid_output())
    record = list(run_verifier_stage(provider=provider, tasks=[first], seed=17))[0]
    assert record["status"] == "ok"
    assert record["stage"] == "verifier"
    assert record["information_boundary"] == "question_options_image_only"
    assert record["parsed"]["answer"] == "H"
    assert record["parsed"]["evidence"][0]["producer_id"] == "independent_verifier"
    assert provider.calls[0]["seed"] == 17
    assert "proposal::independent_verifier" in provider.calls[0]["request_id"]
    assert "SECRET_A" not in provider.calls[0]["user_prompt"]


def test_compact_verifier_parser_rejects_truncated_outer_json() -> None:
    truncated = (
        '{"answer":"H","evidence":['
        '{"evidence_id":"v1","type":"visual_relation","claim":"c",'
        '"support":"s","provenance":"p","confidence":1.0}]'
    )
    parsed, issues = parse_verifier_output(truncated)
    assert parsed == {"answer": "", "evidence": []}
    assert any("no valid JSON object" in issue for issue in issues)


def test_compact_verifier_keeps_only_one_valid_card() -> None:
    payload = json.loads(valid_output())
    payload["evidence"].append(
        {
            "evidence_id": "v2",
            "type": "exact_span",
            "claim": "another claim",
            "support": "another support",
            "provenance": "another region",
            "confidence": 0.8,
        }
    )
    parsed, issues = parse_verifier_output(json.dumps(payload))
    assert len(parsed["evidence"]) == 1
    assert any("beyond the first" in issue for issue in issues)

    payload = json.loads(valid_output())
    payload["demands"] = []
    parsed, issues = parse_verifier_output(json.dumps(payload))
    assert parsed["answer"] == "H"
    assert any("unexpected verifier keys" in issue for issue in issues)
