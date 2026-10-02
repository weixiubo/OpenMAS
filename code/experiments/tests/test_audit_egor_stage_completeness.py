from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from openmas_test_history import historical_root


def _load_module() -> ModuleType:
    script = (historical_root() / "code/experiments") / "scripts" / "audit_egor_stage_completeness.py"
    spec = importlib.util.spec_from_file_location("audit_egor_stage_completeness", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_append_only_stage_is_outcome_blind_and_incomplete(
    tmp_path: Path,
) -> None:
    module = _load_module()
    tasks = _tasks(tmp_path)
    stage = tmp_path / "baseline.jsonl"
    stage.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_id": "fixture",
                        "task_id": "t1",
                        "status": "ok",
                        "seed": 100,
                        "raw_output": "must not appear in audit",
                    }
                ),
                json.dumps(
                    {
                        "source_id": "fixture",
                        "task_id": "t2",
                        "status": "error",
                        "seed": 101,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = module.audit_stage_matrix(
        tasks_path=tasks,
        stages={"baseline": stage},
        expected_seeds={"baseline": 100},
    )

    assert report["passed"] is False
    assert report["outcome_blind"] is True
    audited = report["stages"]["baseline"]
    assert audited["successful_tasks"] == 1
    assert audited["missing_successful"]["examples"] == ["fixture::t2"]
    assert "must not appear" not in json.dumps(report)


def test_completed_stage_requires_original_order_seed_and_compatible_manifest(
    tmp_path: Path,
) -> None:
    module = _load_module()
    tasks = _tasks(tmp_path)
    stage = tmp_path / "baseline.jsonl"
    rows = [
        {"source_id": "fixture", "task_id": "t1", "status": "ok", "seed": 100},
        {"source_id": "fixture", "task_id": "t2", "status": "ok", "seed": 101},
    ]
    stage.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest = {
        "tasks_sha256": module._sha256(tasks),
        "requested": 2,
        "successful_total": 2,
        "seed": 100,
        "git_commit": "abc123",
        "seed_schedule": "base_plus_original_requested_index",
    }
    stage.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = module.audit_stage_matrix(
        tasks_path=tasks,
        stages={"baseline": stage},
        expected_seeds={"baseline": 100},
        require_manifest=True,
    )
    assert report["passed"] is True
    assert report["stages"]["baseline"]["manifest"]["compatible"] is True

    rows[1]["seed"] = 100
    stage.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    mismatched = module.audit_stage_matrix(
        tasks_path=tasks,
        stages={"baseline": stage},
        expected_seeds={"baseline": 100},
        require_manifest=True,
    )
    assert mismatched["passed"] is False
    assert mismatched["stages"]["baseline"]["seed_mismatches"]["examples"] == ["fixture::t2"]


def test_task_id_subset_uses_original_task_file_seed_index(tmp_path: Path) -> None:
    module = _load_module()
    tasks = _tasks(tmp_path)
    stage = tmp_path / "baseline.jsonl"
    stage.write_text(
        json.dumps(
            {
                "source_id": "fixture",
                "task_id": "t2",
                "status": "ok",
                "seed": 701,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = module.audit_stage_matrix(
        tasks_path=tasks,
        stages={"baseline": stage},
        expected_seeds={"baseline": 700},
        allowed_task_ids={"t2"},
    )

    assert report["passed"] is True
    assert report["tasks"]["count"] == 1
    assert report["tasks"]["subset_filter"] is True

    stage.write_text(
        json.dumps(
            {
                "source_id": "fixture",
                "task_id": "t2",
                "status": "ok",
                "seed": 700,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    mismatched = module.audit_stage_matrix(
        tasks_path=tasks,
        stages={"baseline": stage},
        expected_seeds={"baseline": 700},
        allowed_task_ids={"t2"},
    )
    assert mismatched["stages"]["baseline"]["seed_mismatches"]["count"] == 1


def _tasks(tmp_path: Path) -> Path:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"benchmark": "fixture", "task_id": "t1"}),
                json.dumps({"benchmark": "fixture", "task_id": "t2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path
