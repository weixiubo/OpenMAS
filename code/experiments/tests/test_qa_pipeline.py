from __future__ import annotations

import json

import yaml

from openagora_experiments.suites.qa_mini import run_qa_mini_ablation


def test_qa_mini_ablation_outputs(tmp_path) -> None:
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "task_id": "qa-1",
                "prompt": "Use tools to answer: what is 2 + 3?",
                "metadata": {
                    "reference_answer": "5",
                    "primary_type": "multi_step_qa",
                    "uncertainty": "medium",
                    "complexity": "medium",
                    "interdependence": "reciprocal",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cfg_path = tmp_path / "cfg.yaml"
    cfg = {
        "provider": {"name": "mock", "provider_mode": "mock", "seed": 42, "model": "mock-leader-v1"},
        "paths": {
            "db_path": str(tmp_path / "openagora.db"),
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
        "task": {
            "task_id": "placeholder",
            "prompt": "placeholder",
            "metadata": {},
        },
        "eval": {
            "dataset_path": str(dataset),
            "output_csv": str(tmp_path / "rows.csv"),
            "output_summary_csv": str(tmp_path / "summary.csv"),
            "output_failure_csv": str(tmp_path / "failures.csv"),
            "output_manifest_json": str(tmp_path / "manifest.json"),
            "output_hashes_json": str(tmp_path / "hashes.json"),
        },
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    rows = run_qa_mini_ablation(
        str(cfg_path),
        str(tmp_path / "rows.csv"),
        str(tmp_path / "summary.csv"),
        str(tmp_path / "failures.csv"),
    )

    assert rows
    assert "OpenAGORA_AlwaysOnRT" in {row["variant"] for row in rows}
    assert "OpenAGORA_NoOT" in {row["variant"] for row in rows}
    assert (tmp_path / "rows.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "failures.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "hashes.json").exists()
