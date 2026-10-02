from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType


def _load_builder() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_egor_prospective_configs.py"
    )
    spec = importlib.util.spec_from_file_location("prospective_configs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configs_share_paths_and_freeze_visual_as_best_fixed(tmp_path: Path) -> None:
    module = _load_builder()
    execution = module.build_execution_config(tmp_path)
    matrix = module.build_matrix_config(tmp_path)

    assert execution["combinations"]["exact__visual"].endswith(
        "exact_visual_repair.jsonl"
    )
    candidates = {item["candidate_id"]: item for item in matrix["candidates"]}
    assert len(candidates["full_team"]["specialists"]) == 4
    policies = {item["policy_id"]: item for item in matrix["policies"]}
    assert policies["best_fixed_visual_gap"] == {
        "policy_id": "best_fixed_visual_gap",
        "candidate_id": "visual_repair",
        "trigger_mode": "gap",
        "threshold": 0.05,
        "require_evidence_for_change": True,
    }
    assert policies["baseline_resynthesis_all"] == {
        "policy_id": "baseline_resynthesis_all",
        "candidate_id": "baseline_resynthesis",
        "trigger_mode": "all",
        "require_evidence_for_change": False,
    }
    assert policies["full_team_all"]["trigger_mode"] == "all"
    assert "threshold" not in policies["full_team_all"]


def test_policy_combiner_rejects_task_mismatch_and_writes_manifest(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _policy_csv(first, "one", ("t1", "t2"))
    _policy_csv(second, "two", ("t1", "t2"))
    output = tmp_path / "bundle.csv"
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "combine_egor_policy_rows.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(first),
            "--input",
            str(second),
            "--output",
            str(output),
        ],
        check=True,
    )
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    manifest = json.loads(
        output.with_suffix(".csv.manifest.json").read_text(encoding="utf-8")
    )
    assert len(rows) == 4
    assert rows[0]["provenance_rejected"] == "0"
    assert rows[0]["failed_stages"] == ""
    assert manifest["policies"] == ["one", "two"]
    assert manifest["tasks_per_policy"] == 2


def _policy_csv(path: Path, policy: str, task_ids: tuple[str, ...]) -> None:
    fields = (
        "source_id",
        "task_id",
        "policy_id",
        "triggered",
        "baseline_quality",
        "policy_quality",
        "quality_delta",
        "calls",
        "total_tokens",
        "latency_seconds",
        "strict_harm",
        "useful_repair",
        "provenance_rejected",
        "failed_stages",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task_id in task_ids:
            writer.writerow(
                {
                    "source_id": "fixture",
                    "task_id": task_id,
                    "policy_id": policy,
                    "triggered": 0,
                    "baseline_quality": 1,
                    "policy_quality": 1,
                    "quality_delta": 0,
                    "calls": 1,
                    "total_tokens": 1,
                    "latency_seconds": 1,
                    "strict_harm": 0,
                    "useful_repair": 0,
                    "provenance_rejected": 0,
                    "failed_stages": "",
                }
            )
