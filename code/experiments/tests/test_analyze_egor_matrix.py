from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_analysis_script_writes_scored_matrix_and_capability_cards(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "benchmark": "fixture",
                "question": "What is shown?",
                "image_path": "/tmp/image.png",
                "metric": "accuracy",
                "answer": "right",
                "reference_answers": ["right"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.jsonl"
    _write_stage(
        baseline,
        {
            "answer": "wrong",
            "confidence": 0.0,
            "demands": [{"type": "visual_relation", "weight": 1.0}],
            "evidence": [],
            "open_gap_types": ["visual_relation"],
        },
    )
    specialist = tmp_path / "specialist.jsonl"
    _write_stage(
        specialist,
        {
            "candidate_answer": "right",
            "evidence": [
                {
                    "evidence_id": "s1",
                    "type": "visual_relation",
                    "claim": "claim",
                    "support": "visible support",
                    "provenance": "image region",
                    "producer_id": "visual-worker",
                    "confidence": 1.0,
                }
            ],
        },
    )
    synthesis = tmp_path / "synthesis.jsonl"
    _write_stage(
        synthesis,
        {
            "answer": "right",
            "used_evidence_ids": ["s1"],
            "changed_baseline": True,
        },
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "baseline": baseline.name,
                "candidates": [
                    {
                        "candidate_id": "visual-repair",
                        "synthesis": synthesis.name,
                        "specialists": [
                            {
                                "path": specialist.name,
                                "action_id": "visual-worker",
                                "evidence_type": "visual_relation",
                            }
                        ],
                    }
                ],
                "policies": [
                    {
                        "policy_id": "egor-gap",
                        "candidate_id": "visual-repair",
                        "threshold": 0.05,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_egor_matrix.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--tasks",
            str(tasks),
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ],
        check=True,
    )

    summary = json.loads(
        (output / "candidate_summary.json").read_text(encoding="utf-8")
    )
    capabilities = json.loads(
        (output / "capability_cards.json").read_text(encoding="utf-8")
    )
    policies = json.loads((output / "policy_summary.json").read_text(encoding="utf-8"))
    with (output / "matrix_rows.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["visual-repair"]["quality_delta"] == 1.0
    assert capabilities[0]["successful_closures"] == 1
    assert rows[0]["evidence_supported_change"] == "1"
    assert rows[0]["quality_delta"] == "1.0"
    assert policies["egor-gap"]["policy_quality"] == 1.0
    assert policies["egor-gap"]["provenance_rejection_rate"] == 0.0
    assert (output / "policy_rows.csv").is_file()
    assert (
        json.loads((output / "manifest.json").read_text(encoding="utf-8"))["tasks"] == 1
    )


def test_frozen_run_all_sentinel_is_narrowly_translated(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "benchmark": "fixture",
                "question": "What is shown?",
                "image_path": "/tmp/image.png",
                "metric": "accuracy",
                "answer": "right",
                "reference_answers": ["right"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.jsonl"
    _write_stage(
        baseline,
        {
            "answer": "wrong",
            "confidence": 1.0,
            "demands": [],
            "evidence": [],
            "open_gap_types": [],
        },
    )
    synthesis = tmp_path / "synthesis.jsonl"
    _write_stage(
        synthesis,
        {
            "answer": "right",
            "used_evidence_ids": [],
            "changed_baseline": True,
        },
    )
    config = tmp_path / "config.json"
    payload = {
        "baseline": baseline.name,
        "candidates": [
            {
                "candidate_id": "baseline_resynthesis",
                "synthesis": synthesis.name,
                "specialists": [],
            }
        ],
        "policies": [
            {
                "policy_id": "baseline_resynthesis_all",
                "candidate_id": "baseline_resynthesis",
                "threshold": -1.0,
            }
        ],
    }
    config.write_text(json.dumps(payload), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_egor_matrix.py"
    output = tmp_path / "analysis"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--tasks",
            str(tasks),
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ],
        check=True,
    )

    with (output / "policy_rows.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selection = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )["policy_selection"]
    assert rows[0]["triggered"] == "1"
    assert selection == [
        {
            "candidate_id": "baseline_resynthesis",
            "policy_id": "baseline_resynthesis_all",
            "threshold": None,
            "translated_legacy_run_all_sentinel": True,
            "trigger_mode": "all",
        }
    ]

    payload["policies"][0] = {
        "policy_id": "explicit_all",
        "candidate_id": "baseline_resynthesis",
        "trigger_mode": "all",
    }
    config.write_text(json.dumps(payload), encoding="utf-8")
    explicit = tmp_path / "explicit"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--tasks",
            str(tasks),
            "--config",
            str(config),
            "--output-dir",
            str(explicit),
        ],
        check=True,
    )
    explicit_selection = json.loads(
        (explicit / "manifest.json").read_text(encoding="utf-8")
    )["policy_selection"]
    assert explicit_selection[0]["trigger_mode"] == "all"
    assert explicit_selection[0]["translated_legacy_run_all_sentinel"] is False

    payload["policies"][0] = {
        "policy_id": "unregistered_all",
        "candidate_id": "baseline_resynthesis",
        "threshold": -1.0,
    }
    config.write_text(json.dumps(payload), encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--tasks",
            str(tasks),
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "rejected"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "legacy run-all sentinel is restricted" in rejected.stderr


def _write_stage(path: Path, parsed: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            {
                "source_id": "fixture",
                "task_id": "t1",
                "status": "ok",
                "parsed": parsed,
                "usage": {
                    "attempts": 1,
                    "total_tokens": 10,
                    "latency_seconds": 0.5,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
