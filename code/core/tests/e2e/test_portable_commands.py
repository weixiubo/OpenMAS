import json
from importlib import import_module
from pathlib import Path

import yaml
from typer.testing import CliRunner

from openagora.cli.app import app
from openagora.config.loader import load_run_config
from openagora.resources import resolve_paths


def test_experiment_failed_records_signal_failure_to_shell(tmp_path, monkeypatch):
    cli = import_module("openagora.cli.app")
    monkeypatch.setattr(cli, "run_experiment", lambda *args, **kwargs: {"attempted": 2, "successful": 1})
    config = tmp_path / "stage.yaml"
    config.write_text("workflow: egor-stage\n")
    result = CliRunner().invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == 4
    assert json.loads(result.stdout) == {"attempted": 2, "successful": 1}


def test_doctor_is_read_only_and_reports_missing_requirements(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENMAS_RUNS_DIR", str(tmp_path / "uncreated-runs"))
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump({"task": {"task_id": "x", "prompt": "2 + 3"}}))
    runner = CliRunner()
    result = runner.invoke(app, ["doctor", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "ok"
    assert not (tmp_path / "uncreated-runs").exists()
    result = runner.invoke(app, ["doctor", "--models-dir", str(tmp_path), "--require", "qwen3-vl-8b"])
    assert result.exit_code == 3
    assert "missing" in result.stdout and "how_to_get" in result.stdout


def test_run_preflight_before_database_creation(tmp_path):
    config = tmp_path / "local.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "task": {"task_id": "x", "prompt": "x"},
                "provider": {"provider_mode": "local", "local_model_path": "no-such-model"},
            }
        )
    )
    result = CliRunner().invoke(app, ["run", "--config", str(config), "--runs-dir", str(tmp_path / "runs")])
    assert result.exit_code == 3
    assert "Local model directory missing" in result.output
    assert not (tmp_path / "runs").exists()


def test_installed_commands_from_unrelated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    demo = runner.invoke(app, ["demo"])
    assert demo.exit_code == 0, demo.output
    result = json.loads(demo.stdout)
    assert result["result"]["is_correct"]
    assert result["second_run_cache_hit"]
    reproduction = runner.invoke(app, ["reproduce"])
    assert reproduction.exit_code == 0, reproduction.output
    payload = json.loads(reproduction.stdout)
    assert payload["tasks"] == 800
    assert payload["evidence_verified"] and not payload["scientific_gate_passed"]
    assert list(tmp_path.iterdir()) == []


def test_config_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    config = cfg_dir / "tiny.yaml"
    config.write_text(
        yaml.safe_dump({"task": {"task_id": "tiny", "prompt": "tiny"}, "eval": {"dataset_path": "small.jsonl"}})
    )
    paths = resolve_paths(env={}, home=tmp_path)
    first = load_run_config(str(config), resources=paths)
    monkeypatch.chdir(cfg_dir)
    second = load_run_config(str(config), resources=paths)
    assert first == second
    assert Path(first.eval.dataset_path) == cfg_dir / "small.jsonl"
    assert Path(first.paths.artifacts_dir) == tmp_path / "research/runs/OpenMAS"


def test_experiment_doctor_keeps_catalog_and_run_fails_before_outputs(tmp_path):
    config = tmp_path / "stage.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "workflow": "egor-stage",
                "stage": "baseline",
                "tasks": "tasks.jsonl",
                "model_path": "model",
                "model_name": "fixture",
                "model_revision": "fixture-v1",
            }
        )
    )
    runner = CliRunner()
    doctor = runner.invoke(app, ["doctor", "--config", str(config)])
    assert doctor.exit_code == 3
    report = json.loads(doctor.stdout)
    assert report["workflow"] == "egor-stage" and report["resources"]
    assert any("Input missing" in message for message in report["problems"])
    run = runner.invoke(app, ["run", "--config", str(config), "--runs-dir", str(tmp_path / "runs")])
    assert run.exit_code == 3
    assert "resource_preflight_failed" in run.output
    assert not (tmp_path / "runs").exists()
