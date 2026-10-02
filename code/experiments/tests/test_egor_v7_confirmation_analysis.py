from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from openagora_experiments.egor.protocol import PilotTask
from openmas_test_history import historical_root

SCRIPT = (historical_root() / "code/experiments") / "scripts" / "analyze_egor_v7_confirmation.py"
SPEC = importlib.util.spec_from_file_location("analyze_egor_v7_confirmation", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def evidence(answer: str, prefix: str) -> dict[str, Any]:
    return {
        "candidate_answer": answer,
        "evidence": [
            {
                "evidence_id": f"{prefix}1",
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
        "usage": {"attempts": 1, "total_tokens": 10, "latency_seconds": 1},
    }


def test_confirmation_analysis_accepts_only_independently_verified_change(
    tmp_path: Path,
) -> None:
    source = "Source"
    tasks = [
        PilotTask("win", source, "q", "/tmp/x.png", "image/png", "accuracy", "new", (), (), {}),
        PilotTask("reject", source, "q", "/tmp/y.png", "image/png", "accuracy", "old", (), (), {}),
    ]
    write_jsonl(
        tmp_path / "repairability.jsonl",
        [{"source_id": source, "task_id": task.task_id, "repairability_triggered": True} for task in tasks],
    )
    write_jsonl(
        tmp_path / "overwrite_plans.jsonl",
        [{"source_id": source, "task_id": task.task_id, "verifier_triggered": True} for task in tasks],
    )
    write_jsonl(
        tmp_path / "baseline_qwen3.jsonl",
        [stage(source, "win", {"answer": "old"}), stage(source, "reject", {"answer": "old"})],
    )
    write_jsonl(
        tmp_path / "specialist_exact_qwen25.jsonl",
        [stage(source, task.task_id, evidence("new", "e")) for task in tasks],
    )
    write_jsonl(
        tmp_path / "specialist_visual_internvl3.jsonl",
        [stage(source, task.task_id, evidence("new", "v")) for task in tasks],
    )
    write_jsonl(
        tmp_path / "core_pair.jsonl",
        [stage(source, task.task_id, {"answer": "new"}) for task in tasks],
    )
    verifier_rows = [
        stage(
            source,
            "win",
            {
                "answer": "new",
                "evidence": evidence("new", "z")["evidence"],
                "transport": {"operational_complete": True, "mode": "exact_json"},
            },
        ),
        stage(
            source,
            "reject",
            {
                "answer": "different",
                "evidence": evidence("different", "z")["evidence"],
                "transport": {"operational_complete": True, "mode": "exact_json"},
            },
        ),
    ]
    write_jsonl(tmp_path / "verifier_v7_qwen35.jsonl", verifier_rows)

    rows = module.analyze_source(source="fixture", tasks=tasks, root=tmp_path)

    assert rows[0]["accept_overwrite"] is True
    assert rows[0]["v7_delta"] == 1.0
    assert rows[1]["accept_overwrite"] is False
    assert rows[1]["decision_reason"] == "verifier_disagrees"
    assert rows[1]["v7_delta"] == 0.0
    summary = module.summarize_source(rows, bootstrap_samples=100, seed=1)
    assert summary["accepted_overwrites"] == 1
    assert summary["v7"]["quality_delta"] == 0.5
    assert summary["v7"]["strict_harm_rate"] == 0.0


def test_exact_sign_test_handles_ties_and_one_sided_counts() -> None:
    assert module._exact_sign_p(0, 0) == 1.0
    assert module._exact_sign_p(5, 0) == 0.0625
