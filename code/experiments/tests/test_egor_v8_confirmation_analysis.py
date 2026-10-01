from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from openagora_experiments.egor.protocol import PilotTask


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_egor_v7_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_egor_confirmation_shared_v8", SCRIPT
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def evidence(answer: str) -> dict[str, Any]:
    return {
        "candidate_answer": answer,
        "evidence": [
            {
                "evidence_id": "e1",
                "claim": "visible value",
                "support": answer,
                "provenance": "image center",
            }
        ],
    }


def stage(source: str, task_id: str, parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": source,
        "task_id": task_id,
        "status": "ok",
        "parsed": parsed,
        "usage": {
            "attempts": 1,
            "total_tokens": 10,
            "latency_seconds": 1,
        },
    }


def test_v8_confirmation_uses_answer_only_verifier_and_specialist_evidence(
    tmp_path: Path,
) -> None:
    source = "Source"
    tasks = [
        PilotTask(
            "win",
            source,
            "q",
            "/tmp/x.png",
            "image/png",
            "accuracy",
            "new",
            (),
            (),
            {},
        ),
        PilotTask(
            "reject",
            source,
            "q",
            "/tmp/y.png",
            "image/png",
            "accuracy",
            "old",
            (),
            (),
            {},
        ),
    ]
    write_jsonl(
        tmp_path / "repairability.jsonl",
        [
            {
                "source_id": source,
                "task_id": task.task_id,
                "repairability_triggered": True,
            }
            for task in tasks
        ],
    )
    write_jsonl(
        tmp_path / "overwrite_plans.jsonl",
        [
            {
                "source_id": source,
                "task_id": task.task_id,
                "verifier_triggered": True,
            }
            for task in tasks
        ],
    )
    write_jsonl(
        tmp_path / "baseline_qwen3.jsonl",
        [stage(source, task.task_id, {"answer": "old"}) for task in tasks],
    )
    for filename in (
        "specialist_exact_qwen25.jsonl",
        "specialist_visual_internvl3.jsonl",
    ):
        write_jsonl(
            tmp_path / filename,
            [stage(source, task.task_id, evidence("new")) for task in tasks],
        )
    write_jsonl(
        tmp_path / "core_pair.jsonl",
        [stage(source, task.task_id, {"answer": "new"}) for task in tasks],
    )
    write_jsonl(
        tmp_path / "verifier_v8_qwen35.jsonl",
        [
            stage(
                source,
                "win",
                {
                    "answer": "new",
                    "evidence": [],
                    "transport": {
                        "operational_complete": True,
                        "mode": "plain_single_line",
                    },
                },
            ),
            stage(
                source,
                "reject",
                {
                    "answer": "different",
                    "evidence": [],
                    "transport": {
                        "operational_complete": True,
                        "mode": "plain_single_line",
                    },
                },
            ),
        ],
    )

    rows = module.analyze_source(
        source="fixture",
        tasks=tasks,
        root=tmp_path,
        verifier_filename="verifier_v8_qwen35.jsonl",
        answer_only=True,
        policy_key="v8",
    )

    assert rows[0]["accept_overwrite"] is True
    assert rows[0]["v8_delta"] == 1.0
    assert rows[1]["accept_overwrite"] is False
    assert rows[1]["decision_reason"] == "answer_only_verifier_disagrees"
    summary = module.summarize_source(
        rows,
        bootstrap_samples=100,
        seed=1,
        policy_key="v8",
    )
    assert summary["v8"]["quality_delta"] == 0.5
    assert "v7" not in summary
