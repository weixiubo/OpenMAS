from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from openmas_test_history import historical_root


def _load_module() -> ModuleType:
    script = (historical_root() / "code/experiments") / "scripts" / "run_egor_stage.py"
    spec = importlib.util.spec_from_file_location("run_egor_stage", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_uses_explicit_source_commit_outside_git(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    module = _load_module()
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "stage.jsonl"
    output.write_text(
        json.dumps({"source_id": "s", "task_id": "1", "status": "ok"}) + "\n",
        encoding="utf-8",
    )
    module.ROOT = tmp_path
    monkeypatch.setenv("EGOR_SOURCE_COMMIT", "abc123")  # type: ignore[attr-defined]
    args = argparse.Namespace(
        stage="baseline",
        model_name="fixture",
        model_path="/models/fixture",
        model_profile="medium",
        gpu_devices="1",
        max_gpu_mem_gb=20.0,
        seed=7,
    )

    module._write_manifest(
        output_path=output,
        task_path=tasks,
        args=args,
        requested=1,
        attempted=0,
        successful=0,
    )

    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8"))
    assert manifest["git_commit"] == "abc123"
    assert manifest["git_worktree_dirty"] is False
    assert manifest["successful_total"] == 1
    assert manifest["cublas_workspace_config"] == ":4096:8"
    assert manifest["seed_schedule"] == "base_plus_original_requested_index"
    assert manifest["resume_policy"].startswith("skip successful keys")
    assert manifest["schema_version"] == 2
    assert "transformers" in manifest["runtime"]["packages"]
    assert len(manifest["runtime"]["source_files"]) == 2
    assert manifest["runtime"]["model_weight_layout"] == []


def test_scheduler_catalog_is_derived_from_frozen_action_spec() -> None:
    module = _load_module()
    config = (historical_root() / "code/experiments") / "configs" / "egor_repair_execution_v1.json"

    catalog = module._scheduler_catalog(config)

    pair = next(item for item in catalog if item["template_id"] == "exact_visual")
    assert pair["action_ids"] == ["exact", "visual"]
    assert pair["evidence_types"] == ["exact_span", "visual_relation"]


def test_task_id_filter_preserves_original_seed_indices(tmp_path: Path) -> None:
    module = _load_module()
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "source_id": "fixture",
                    "question": f"question {task_id}",
                    "image_path": f"/{task_id}.png",
                }
            )
            + "\n"
            for task_id in ("t0", "t1", "t2", "t3")
        ),
        encoding="utf-8",
    )

    tasks, indices = module._load_requested_tasks(
        tasks_path,
        allowed_task_ids={"t1", "t3"},
        limit=0,
    )

    assert [task.task_id for task in tasks] == ["t1", "t3"]
    assert indices[("fixture", "t1")] == 1
    assert indices[("fixture", "t3")] == 3
