from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_fit_selector_script_writes_source_cross_validated_audit(
    tmp_path: Path,
) -> None:
    sources = []
    for index in (1, 2, 3):
        source = f"fixture-{index}"
        task_id = f"t{index}"
        tasks = tmp_path / f"tasks-{index}.jsonl"
        tasks.write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "benchmark": source,
                    "question": "What exact text is shown?",
                    "image_path": "/tmp/image.png",
                    "metric": "accuracy",
                    "answer": "right",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        baseline = tmp_path / f"baseline-{index}.jsonl"
        baseline.write_text(
            json.dumps(
                {
                    "source_id": source,
                    "task_id": task_id,
                    "status": "ok",
                    "parsed": {
                        "answer": "wrong",
                        "confidence": 0.0,
                        "demands": [{"type": "exact_span", "weight": 1.0}],
                        "evidence": [],
                        "open_gap_types": ["exact_span"],
                    },
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
        matrix = tmp_path / f"matrix-{index}.csv"
        with matrix.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "source_id",
                "task_id",
                "candidate_id",
                "baseline_quality",
                "candidate_quality",
                "actual_changed",
                "evidence_supported_change",
                "calls",
                "baseline_calls",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "source_id": source,
                    "task_id": task_id,
                    "candidate_id": "exact_raw",
                    "baseline_quality": 0.0,
                    "candidate_quality": 1.0,
                    "actual_changed": 1,
                    "evidence_supported_change": 1,
                    "calls": 3.0,
                    "baseline_calls": 1.0,
                }
            )
            writer.writerow(
                {
                    "source_id": source,
                    "task_id": task_id,
                    "candidate_id": "visual_raw",
                    "baseline_quality": 0.0,
                    "candidate_quality": 0.0,
                    "actual_changed": 0,
                    "evidence_supported_change": 0,
                    "calls": 3.0,
                    "baseline_calls": 1.0,
                }
            )
        sources.append(
            {
                "source_id": f"development-{index}",
                "tasks": str(tasks),
                "baseline": str(baseline),
                "matrices": [
                    {
                        "path": str(matrix),
                        "candidate_mapping": {
                            "exact_raw": "exact",
                            "visual_raw": "visual",
                        },
                    }
                ],
            }
        )

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "candidate_ids": ["exact", "visual"],
                "grid": {
                    "l2_values": [1.0],
                    "thresholds": [0.0],
                    "standalone_value_thresholds": [0.0, 0.5],
                    "cost_weights": [0.0],
                    "evaluation_call_price": 0.01,
                    "maximum_harm_rate": 0.05,
                    "minimum_repair_rate": 0.0,
                    "minimum_gap": 0.05,
                },
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "selector"
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "fit_egor_repair_selector.py"
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ],
        check=True,
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    selector = json.loads((output / "selector.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    standalone = json.loads(
        (output / "selector_standalone_value.json").read_text(encoding="utf-8")
    )
    with (output / "cross_validated_rows.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert summary["source_cross_validated_selector"]["mean_quality_delta"] == 1.0
    assert summary["source_cross_validated_selector"]["strict_harm_rate"] == 0.0
    assert summary["cross_validation_protocol"] == "nested_loso"
    assert summary["standalone_cross_validation_protocol"] == "nested_loso"
    assert (
        summary["source_cross_validated_standalone_value"]["mean_quality_delta"] == 1.0
    )
    assert selector["config"]["l2"] == 1.0
    assert selector["target"] == "paired_advantage"
    assert selector["models"].keys() == {"exact", "visual"}
    assert standalone["target"] == "standalone_quality"
    assert len(rows) == 3
    assert {row["selected_id"] for row in rows} == {"exact"}
    assert {row["selection_protocol"] for row in rows} == {"nested_loso"}
    assert all(json.loads(row["fold_config_json"])["l2"] == 1.0 for row in rows)
    assert manifest["tasks"] == 3
    assert manifest["cross_validation_protocol"] == "nested_loso"
    assert manifest["analysis_controls"] == ["standalone_quality_selector"]
    assert (output / "cv_grid_standalone_value.json").is_file()
    assert (output / "cross_validated_rows_standalone_value.csv").is_file()
