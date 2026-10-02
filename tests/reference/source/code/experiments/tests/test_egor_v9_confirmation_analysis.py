from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from openagora_experiments.egor.protocol import PilotTask


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "analyze_egor_v9_confirmation.py"
SPEC = importlib.util.spec_from_file_location("analyze_egor_v9_test", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


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
        "usage": {"attempts": 1, "total_tokens": 10, "latency_seconds": 1},
    }


def test_v9_keeps_supported_win_and_blocks_unsupported_direct_harm(tmp_path: Path) -> None:
    source = "Fixture"
    tasks = [
        PilotTask("win", source, "q", "/tmp/a.png", "image/png", "accuracy", "new", (), (), {}),
        PilotTask("blocked-harm", source, "q", "/tmp/b.png", "image/png", "accuracy", "old", (), (), {}),
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
        tmp_path / "baseline_qwen3.jsonl",
        [stage(source, task.task_id, {"answer": "old"}) for task in tasks],
    )
    write_jsonl(
        tmp_path / "specialist_exact_qwen25.jsonl",
        [stage(source, "win", evidence("new")), stage(source, "blocked-harm", evidence("old"))],
    )
    write_jsonl(
        tmp_path / "specialist_visual_internvl3.jsonl",
        [stage(source, task.task_id, evidence("other")) for task in tasks],
    )
    write_jsonl(
        tmp_path / "core_pair_ablation.jsonl",
        [stage(source, task.task_id, {"answer": "other"}) for task in tasks],
    )
    write_jsonl(
        tmp_path / "resolver_v9_qwen35.jsonl",
        [
            stage(
                source,
                task.task_id,
                {
                    "answer": "new",
                    "transport": {
                        "operational_complete": True,
                        "mode": "plain_single_line",
                    },
                },
            )
            for task in tasks
        ],
    )

    rows = module.analyze_source(source="fixture", tasks=tasks, root=tmp_path)
    assert rows[0]["accept_overwrite"] is True
    assert rows[0]["v9_delta"] == 1.0
    assert rows[1]["direct_resolver_delta"] == -1.0
    assert rows[1]["v9_delta"] == 0.0
    assert rows[1]["decision_reason"] == "no_heterogeneous_corroborator"

    summary = module.summarize_source(rows, bootstrap_samples=100, seed=1)
    assert summary["v9"]["quality_delta"] == 0.5
    assert summary["direct_resolver"]["quality_delta"] == 0.0
    assert summary["accepted_overwrites"] == 1
