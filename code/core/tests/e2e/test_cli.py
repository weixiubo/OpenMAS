from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from openagora.cli.app import app


def test_cli_run_replay_eval_template_doctor_manifest_export(tmp_path) -> None:
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "task_id": "qa-e2e-1",
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
    cfg_payload = {
        "provider": {
            "name": "mock",
            "model": "mock-leader-v1",
            "provider_mode": "mock",
            "temperature": 0.0,
            "seed": 42,
        },
        "paths": {"db_path": str(tmp_path / "openagora.db"), "artifacts_dir": str(tmp_path / "artifacts")},
        "task": {
            "task_id": "qa-e2e-1",
            "prompt": "Use tools to answer: what is 2 + 3?",
            "metadata": {
                "reference_answer": "5",
                "primary_type": "multi_step_qa",
                "uncertainty": "medium",
                "complexity": "medium",
                "interdependence": "reciprocal",
            },
        },
        "eval": {
            "dataset_path": str(dataset),
            "output_csv": str(tmp_path / "eval_rows.csv"),
            "output_summary_csv": str(tmp_path / "eval_summary.csv"),
            "output_failure_csv": str(tmp_path / "eval_failures.csv"),
            "output_manifest_json": str(tmp_path / "eval_manifest.json"),
            "output_hashes_json": str(tmp_path / "eval_hashes.json"),
        },
    }
    cfg_path.write_text(yaml.safe_dump(cfg_payload), encoding="utf-8")

    runner = CliRunner()

    validate_result = runner.invoke(app, ["template", "validate", "--config", str(cfg_path)])
    assert validate_result.exit_code == 0

    doctor_result = runner.invoke(app, ["doctor", "--config", str(cfg_path)])
    assert doctor_result.exit_code == 0
    doctor_payload = json.loads(doctor_result.stdout)
    assert doctor_payload["status"] in {"ok", "degraded"}

    run_result = runner.invoke(app, ["run", "--config", str(cfg_path), "--provider-mode", "mock"])
    assert run_result.exit_code == 0
    run_json = json.loads(run_result.stdout)
    run_id = run_json["run_id"]

    replay_result = runner.invoke(app, ["replay", "--run-id", run_id, "--config", str(cfg_path)])
    assert replay_result.exit_code == 0
    replay_json = json.loads(replay_result.stdout)
    assert replay_json["run_id"] == run_id

    verify_result = runner.invoke(
        app,
        [
            "manifest",
            "verify",
            "--run-id",
            run_id,
            "--compare-run-id",
            run_id,
            "--config",
            str(cfg_path),
        ],
    )
    assert verify_result.exit_code == 0
    verify_payload = json.loads(verify_result.stdout)
    assert verify_payload["config_hash_match"]
    assert verify_payload["fingerprint_match"]

    export_path = tmp_path / "bundle.zip"
    export_result = runner.invoke(
        app,
        ["export", "--run-id", run_id, "--config", str(cfg_path), "--output", str(export_path)],
    )
    assert export_result.exit_code == 0
    assert export_path.exists()

    eval_result = runner.invoke(app, ["eval", "--suite", "qa_mini", "--config", str(cfg_path)])
    assert eval_result.exit_code == 0
    assert Path(tmp_path / "eval_rows.csv").exists()
    assert Path(tmp_path / "eval_summary.csv").exists()
    assert Path(tmp_path / "eval_failures.csv").exists()
    assert Path(tmp_path / "eval_manifest.json").exists()
    assert Path(tmp_path / "eval_hashes.json").exists()

    gaia_eval_result = runner.invoke(app, ["eval", "--suite", "gaia_mini", "--config", str(cfg_path)])
    assert gaia_eval_result.exit_code == 0
