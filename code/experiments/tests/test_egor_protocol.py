from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openagora.mechanisms.evidence_gap_repair import EvidenceType
from openagora_experiments.egor.protocol import (
    PilotTask,
    build_baseline_prompt,
    build_scheduler_prompt,
    build_specialist_prompt,
    build_synthesis_prompt,
    parse_baseline_output,
    parse_scheduler_output,
    parse_specialist_output,
)
from openagora_experiments.egor.runner import (
    load_latest_stage_records,
    load_successful_stage_records,
    run_baseline_stage,
    run_scheduler_stage,
    run_specialist_stage,
    run_synthesis_stage,
)
from openagora_experiments.egor.runtime_types import ArtifactSpec


class FakeProvider:
    model_name = "fake-vlm"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []

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
        self._events = [
            {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "latency_ms": 250,
                "attempts": 1,
            }
        ]
        return self.outputs.pop(0)

    def pop_audit_events(self) -> list[dict[str, Any]]:
        events = list(self._events)
        self._events.clear()
        return events


def test_pilot_task_infers_real_image_mime_types(tmp_path: Path) -> None:
    required = {
        "task_id": "mime-fixture",
        "benchmark": "fixture",
        "question": "What is shown?",
    }
    expected_by_suffix = {
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }
    for suffix, expected in expected_by_suffix.items():
        task = PilotTask.from_json(
            {**required, "image_path": str(tmp_path / f"fixture{suffix}")}
        )
        assert task.mime_type == expected

    disguised_webp = tmp_path / "fixture.img"
    disguised_webp.write_bytes(b"RIFF\x04\x00\x00\x00WEBP")
    task = PilotTask.from_json({**required, "image_path": str(disguised_webp)})
    assert task.mime_type == "image/webp"

    unknown = PilotTask.from_json(
        {**required, "image_path": str(tmp_path / "missing.img")}
    )
    assert unknown.mime_type == "application/octet-stream"


def test_all_generation_prompts_are_answer_blind() -> None:
    first = _task(answer="SECRET_A", references=("REFERENCE_A",))
    second = _task(answer="SECRET_B", references=("REFERENCE_B",))
    baseline = {
        "answer": "HIDDEN_BASELINE_DRAFT_9X",
        "demands": [],
        "evidence": [{"support": "HIDDEN_BASELINE_EVIDENCE_9X"}],
        "open_gap_types": ["exact_span"],
    }
    specialists = [
        {
            "specialist_type": "exact_span",
            "candidate_answer": "candidate",
            "evidence": [],
        }
    ]

    assert build_baseline_prompt(first) == build_baseline_prompt(second)
    assert build_specialist_prompt(
        first,
        specialist_type=EvidenceType.EXACT_SPAN,
        baseline=baseline,
    ) == build_specialist_prompt(
        second,
        specialist_type=EvidenceType.EXACT_SPAN,
        baseline=baseline,
    )
    specialist_system, specialist_user = build_specialist_prompt(
        first,
        specialist_type=EvidenceType.EXACT_SPAN,
        baseline=baseline,
    )
    assert "intentionally blind" in specialist_system
    assert "HIDDEN_BASELINE_DRAFT_9X" not in specialist_user
    assert "HIDDEN_BASELINE_EVIDENCE_9X" not in specialist_user
    assert build_synthesis_prompt(
        first,
        baseline=baseline,
        specialist_records=specialists,
    ) == build_synthesis_prompt(
        second,
        baseline=baseline,
        specialist_records=specialists,
    )


def test_scheduler_prompt_is_answer_blind_and_parser_is_catalog_bound() -> None:
    first = _task(answer="SECRET_A", references=("REFERENCE_A",))
    second = _task(answer="SECRET_B", references=("REFERENCE_B",))
    baseline = {
        "answer": "draft",
        "confidence": 0.4,
        "demands": [],
        "evidence": [],
        "open_gap_types": ["exact_span"],
    }
    catalog = [
        {
            "template_id": "exact",
            "action_ids": ["exact"],
            "evidence_types": ["exact_span"],
        }
    ]

    assert build_scheduler_prompt(
        first, baseline=baseline, template_catalog=catalog
    ) == build_scheduler_prompt(second, baseline=baseline, template_catalog=catalog)
    parsed, issues = parse_scheduler_output(
        json.dumps(
            {
                "selected_template_id": "invented",
                "target_gap_types": ["exact_span"],
                "confidence": 0.8,
                "rationale": "try an unknown worker",
            }
        ),
        allowed_template_ids=("exact",),
    )
    assert parsed["selected_template_id"] == "baseline"
    assert any("unknown scheduler template" in issue for issue in issues)


def test_scheduler_stage_is_a_paid_outcome_blind_router_call() -> None:
    task = _task()
    key = (task.source_id, task.task_id)
    provider = FakeProvider(
        [
            json.dumps(
                {
                    "selected_template_id": "exact",
                    "target_gap_types": ["exact_span"],
                    "confidence": 0.7,
                    "rationale": "the visible span remains open",
                }
            )
        ]
    )
    baseline = {
        "status": "ok",
        "parsed": {
            "answer": "draft",
            "confidence": 0.4,
            "demands": [],
            "evidence": [],
            "open_gap_types": ["exact_span"],
        },
    }

    record = list(
        run_scheduler_stage(
            provider=provider,
            tasks=[task],
            baseline_records={key: baseline},
            template_catalog=[
                {
                    "template_id": "exact",
                    "action_ids": ["exact"],
                    "evidence_types": ["exact_span"],
                }
            ],
            seed=19,
        )
    )[0]

    assert record["status"] == "ok"
    assert record["parsed"]["selected_template_id"] == "exact"
    assert record["usage"]["attempts"] == 1.0
    assert "route::llm_scheduler" in provider.calls[0]["request_id"]
    assert task.answer not in provider.calls[0]["user_prompt"]


def test_baseline_parser_discards_unknown_demands_and_unprovenanced_cards() -> None:
    raw = """```json
    {
      "answer": "42",
      "confidence": 1.2,
      "demands": [
        {"demand_id": "d1", "type": "numerical_derivation", "requirement": "derive it", "weight": 0.8},
        {"demand_id": "d2", "type": "mystery", "requirement": "unknown", "weight": 1.0}
      ],
      "evidence": [
        {"evidence_id": "b1", "type": "numerical_derivation", "claim": "result", "support": "6*7=42", "provenance": "calculation trace", "confidence": 0.9},
        {"evidence_id": "b2", "type": "exact_span", "claim": "text", "support": "42", "provenance": "", "confidence": 0.7}
      ],
      "open_gap_types": ["exact_span", "invented_type"]
    }
    ```"""

    parsed, issues = parse_baseline_output(raw)

    assert parsed["answer"] == "42"
    assert parsed["confidence"] == 1.0
    assert [item["demand_id"] for item in parsed["demands"]] == ["d1"]
    assert [item["evidence_id"] for item in parsed["evidence"]] == ["baseline::b1"]
    assert parsed["open_gap_types"] == ["exact_span"]
    assert any("unknown evidence type" in issue for issue in issues)
    assert any("provenance" in issue for issue in issues)


def test_specialist_parser_enforces_assigned_type() -> None:
    raw = json.dumps(
        {
            "candidate_answer": "A",
            "evidence": [
                {
                    "evidence_id": "s1",
                    "type": "visual_relation",
                    "claim": "wrong specialist type",
                    "support": "object left of label",
                    "provenance": "upper-left image region",
                    "confidence": 0.8,
                }
            ],
            "remaining_gap_types": [],
        }
    )

    parsed, issues = parse_specialist_output(
        raw,
        specialist_type=EvidenceType.EXACT_SPAN,
        producer_id="ocr-worker",
    )

    assert parsed["specialist_type"] == "exact_span"
    assert parsed["evidence"] == []
    assert parsed["remaining_gap_types"] == ["exact_span"]
    assert any("mismatched specialist type" in issue for issue in issues)


def test_specialist_evidence_ids_are_namespaced_by_producer() -> None:
    raw = json.dumps(
        {
            "candidate_answer": "answer",
            "evidence": [
                {
                    "evidence_id": "s1",
                    "type": "exact_span",
                    "claim": "claim",
                    "support": "visible text",
                    "provenance": "center label",
                    "confidence": 1.0,
                }
            ],
            "remaining_gap_types": [],
        }
    )

    first, _ = parse_specialist_output(
        raw,
        specialist_type=EvidenceType.EXACT_SPAN,
        producer_id="worker-a",
    )
    second, _ = parse_specialist_output(
        raw,
        specialist_type=EvidenceType.EXACT_SPAN,
        producer_id="worker-b",
    )

    assert first["evidence"][0]["evidence_id"] == "worker-a::s1"
    assert second["evidence"][0]["evidence_id"] == "worker-b::s1"


def test_specialist_parser_owns_namespace_not_model_output() -> None:
    parsed, _ = parse_specialist_output(
        json.dumps(
            {
                "candidate_answer": "answer",
                "evidence": [
                    {
                        "evidence_id": "foreign-worker::s1",
                        "type": "exact_span",
                        "claim": "claim",
                        "support": "visible text",
                        "provenance": "center label",
                        "producer_id": "foreign-worker",
                        "confidence": 1.0,
                    }
                ],
                "remaining_gap_types": [],
            }
        ),
        specialist_type=EvidenceType.EXACT_SPAN,
        producer_id="assigned-worker",
    )

    card = parsed["evidence"][0]
    assert card["producer_id"] == "assigned-worker"
    assert card["evidence_id"] == "assigned-worker::foreign-worker::s1"


def test_specialist_abstention_keeps_the_assigned_gap_open() -> None:
    parsed, issues = parse_specialist_output(
        json.dumps(
            {
                "candidate_answer": "",
                "evidence": [],
                "remaining_gap_types": [],
            }
        ),
        specialist_type=EvidenceType.INDEPENDENT_VERIFICATION,
        producer_id="verifier",
    )

    assert parsed["remaining_gap_types"] == ["independent_verification"]
    assert any("remains open" in issue for issue in issues)


def test_staged_runner_logs_usage_and_fuses_dependencies() -> None:
    task = _task()
    baseline_provider = FakeProvider(
        [
            json.dumps(
                {
                    "answer": "draft",
                    "confidence": 0.4,
                    "demands": [
                        {
                            "demand_id": "d1",
                            "type": "exact_span",
                            "requirement": "read label",
                            "weight": 1.0,
                        }
                    ],
                    "evidence": [],
                    "open_gap_types": ["exact_span"],
                }
            )
        ]
    )
    baseline_record = list(
        run_baseline_stage(provider=baseline_provider, tasks=[task], seed=7)
    )[0]
    key = (task.source_id, task.task_id)

    assert baseline_record["status"] == "ok"
    assert baseline_record["usage"] == {
        "prompt_tokens": 10.0,
        "completion_tokens": 4.0,
        "total_tokens": 14.0,
        "latency_seconds": 0.25,
        "attempts": 1.0,
    }
    assert baseline_provider.calls[0]["artifacts"][0].path == task.image_path
    assert task.answer not in baseline_provider.calls[0]["user_prompt"]
    assert "proposal" in baseline_provider.calls[0]["request_id"]

    specialist_provider = FakeProvider(
        [
            json.dumps(
                {
                    "candidate_answer": "repaired",
                    "evidence": [
                        {
                            "evidence_id": "s1",
                            "type": "exact_span",
                            "claim": "the label says repaired",
                            "support": "repaired",
                            "provenance": "center label",
                            "confidence": 0.9,
                        }
                    ],
                    "remaining_gap_types": [],
                }
            )
        ]
    )
    specialist_record = list(
        run_specialist_stage(
            provider=specialist_provider,
            tasks=[task],
            baseline_records={key: baseline_record},
            specialist_type=EvidenceType.EXACT_SPAN,
            specialist_id="ocr-worker",
            seed=11,
        )
    )[0]

    synthesis_provider = FakeProvider(
        [
            json.dumps(
                {
                    "answer": "repaired",
                    "used_evidence_ids": ["ocr-worker::s1"],
                    "rejected_evidence_ids": [],
                    "remaining_gap_types": [],
                    "changed_baseline": True,
                    "confidence": 0.9,
                }
            )
        ]
    )
    synthesis_record = list(
        run_synthesis_stage(
            provider=synthesis_provider,
            tasks=[task],
            baseline_records={key: baseline_record},
            specialist_record_sets=[{key: specialist_record}],
            candidate_id="ocr-repair",
            seed=13,
        )
    )[0]

    assert specialist_record["parsed"]["evidence"][0]["producer_id"] == "ocr-worker"
    assert specialist_record["parsed"]["evidence"][0]["evidence_id"] == "ocr-worker::s1"
    assert synthesis_record["parsed"]["answer"] == "repaired"
    assert synthesis_record["parsed"]["used_evidence_ids"] == ["ocr-worker::s1"]
    assert "proposal" in specialist_provider.calls[0]["request_id"]
    assert "finalize" in synthesis_provider.calls[0]["request_id"]


def test_missing_dependency_is_recorded_without_invoking_provider() -> None:
    provider = FakeProvider([])

    record = list(
        run_specialist_stage(
            provider=provider,
            tasks=[_task()],
            baseline_records={},
            specialist_type=EvidenceType.EXACT_SPAN,
            specialist_id="ocr-worker",
            seed=3,
        )
    )[0]

    assert record["status"] == "error"
    assert record["error_type"] == "MissingDependency"
    assert provider.calls == []


def test_baseline_only_resynthesis_needs_no_specialist_dependency() -> None:
    task = _task()
    key = (task.source_id, task.task_id)
    baseline = {
        "status": "ok",
        "parsed": {
            "answer": "draft",
            "confidence": 0.5,
            "demands": [],
            "evidence": [],
            "open_gap_types": [],
        },
    }
    provider = FakeProvider(
        [
            json.dumps(
                {
                    "answer": "draft",
                    "used_evidence_ids": [],
                    "rejected_evidence_ids": [],
                    "remaining_gap_types": [],
                    "changed_baseline": False,
                    "confidence": 0.5,
                }
            )
        ]
    )

    record = list(
        run_synthesis_stage(
            provider=provider,
            tasks=[task],
            baseline_records={key: baseline},
            specialist_record_sets=[],
            candidate_id="baseline-resynthesis",
            seed=17,
        )
    )[0]

    assert record["status"] == "ok"
    assert record["parsed"]["answer"] == "draft"
    assert "Specialist records:\n[]" in provider.calls[0]["user_prompt"]


def test_empty_provider_completion_is_a_resumable_error() -> None:
    provider = FakeProvider(["local:mm_empty_completion"])

    record = list(run_baseline_stage(provider=provider, tasks=[_task()], seed=5))[0]

    assert record["status"] == "error"
    assert record["error_type"] == "RuntimeError"
    assert "empty completion" in record["error"]
    assert record["usage"]["attempts"] == 1.0


def test_resume_subset_keeps_seed_from_original_task_order() -> None:
    task = _task()
    provider = FakeProvider(
        [
            json.dumps(
                {
                    "answer": "draft",
                    "confidence": 0.5,
                    "demands": [],
                    "evidence": [],
                    "open_gap_types": [],
                }
            )
        ]
    )

    record = list(
        run_baseline_stage(
            provider=provider,
            tasks=[task],
            seed=100,
            task_indices={(task.source_id, task.task_id): 17},
        )
    )[0]

    assert provider.calls[0]["seed"] == 117
    assert record["seed"] == 117


def test_stage_loader_keeps_only_successful_records(tmp_path: Path) -> None:
    path = tmp_path / "stage.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"source_id": "s", "task_id": "1", "status": "error"}),
                json.dumps(
                    {
                        "source_id": "s",
                        "task_id": "2",
                        "status": "ok",
                        "parsed": {"answer": "kept"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_successful_stage_records(path)

    assert set(records) == {("s", "2")}
    assert records[("s", "2")]["parsed"]["answer"] == "kept"


def test_latest_stage_loader_preserves_final_failed_attempt(tmp_path: Path) -> None:
    path = tmp_path / "stage.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_id": "s",
                        "task_id": "1",
                        "status": "error",
                        "error": "first",
                    }
                ),
                json.dumps(
                    {
                        "source_id": "s",
                        "task_id": "1",
                        "status": "error",
                        "error": "second",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_latest_stage_records(path)

    assert records[("s", "1")]["error"] == "second"


def _task(
    *,
    answer: str = "SECRET_ANSWER",
    references: tuple[str, ...] = ("SECRET_REFERENCE",),
) -> PilotTask:
    return PilotTask(
        task_id="t1",
        source_id="fixture",
        question="What does the visible label say?",
        image_path="/tmp/fixture.png",
        mime_type="image/png",
        metric="accuracy",
        answer=answer,
        reference_answers=references,
        options=(),
        metadata={},
    )
