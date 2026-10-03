from __future__ import annotations

import copy
import importlib.util
from types import ModuleType

import pytest
from openmas_test_history import historical_root

HASH = "a" * 64
COMMIT = "b" * 40


def test_runtime_equivalence_allows_only_declared_placement_fields() -> None:
    module = _load_module()
    reference_rows = [_row("t1", "answer one"), _row("t2", "answer two")]
    candidate_rows = copy.deepcopy(reference_rows)
    for index, row in enumerate(candidate_rows, start=1):
        row["usage"]["latency_seconds"] = 1000.0 + index
        row["audit_events"][0]["latency_ms"] = 1_000_000.0 + index
        row["audit_events"][0]["gpu_devices"] = "1,0"
        row["audit_events"][0]["max_gpu_mem_gb"] = 11

    report = _compare(
        module,
        reference_rows=reference_rows,
        candidate_rows=candidate_rows,
    )

    assert report["passed"] is True
    assert report["mismatch_count"] == 0
    assert report["outcome_blind"] is True


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda row: row.__setitem__("raw_output", "different"), "raw_output"),
        (
            lambda row: row.__setitem__("system_prompt_sha256", "c" * 64),
            "system_prompt_sha256",
        ),
        (
            lambda row: row["usage"].__setitem__("total_tokens", 999),
            "usage.total_tokens",
        ),
        (
            lambda row: row["audit_events"][0].__setitem__("model", "/other"),
            "audit_events_without_runtime_fields",
        ),
    ],
)
def test_runtime_equivalence_fails_closed_on_semantic_or_token_drift(
    mutation: object,
    field: str,
) -> None:
    module = _load_module()
    reference_rows = [_row("t1", "answer")]
    candidate_rows = copy.deepcopy(reference_rows)
    mutation(candidate_rows[0])

    report = _compare(
        module,
        reference_rows=reference_rows,
        candidate_rows=candidate_rows,
        expected_tasks=1,
    )

    assert report["passed"] is False
    assert any(item["field"] == field for item in report["mismatches"])


def test_runtime_equivalence_rejects_duplicate_task_keys() -> None:
    module = _load_module()
    rows = [_row("t1", "one"), _row("t1", "two")]

    with pytest.raises(ValueError, match="duplicate task key"):
        _compare(
            module,
            reference_rows=rows,
            candidate_rows=copy.deepcopy(rows),
        )


def test_runtime_equivalence_enforces_candidate_runtime_contract() -> None:
    module = _load_module()
    candidate_manifest = _manifest()
    candidate_manifest["gpu_devices"] = "0,1"
    candidate_manifest["max_gpu_mem_gb"] = 12
    candidate_manifest["git_commit"] = "c" * 40

    report = _compare(
        module,
        reference_rows=[_row("t1", "one"), _row("t2", "two")],
        candidate_rows=[_row("t1", "one"), _row("t2", "two")],
        candidate_manifest=candidate_manifest,
    )

    fields = {item["field"] for item in report["mismatches"]}
    assert report["passed"] is False
    assert fields >= {
        "candidate_gpu_devices",
        "candidate_max_gpu_mem_gb",
        "candidate_git_commit",
    }


def _compare(
    module: ModuleType,
    *,
    reference_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    expected_tasks: int = 2,
    candidate_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    reference_manifest = _manifest()
    candidate = candidate_manifest or _manifest()
    reference_manifest["requested"] = expected_tasks
    candidate["requested"] = expected_tasks
    candidate["output"] = "/candidate.jsonl"
    candidate["created_at"] = "later"
    candidate["gpu_devices"] = candidate.get("gpu_devices", "1,0")
    candidate["max_gpu_mem_gb"] = candidate.get("max_gpu_mem_gb", 11)
    return module.compare_runtime_equivalence(
        reference_records=reference_rows,
        candidate_records=candidate_rows,
        reference_manifest=reference_manifest,
        candidate_manifest=candidate,
        reference_sha256=HASH,
        candidate_sha256="b" * 64,
        reference_manifest_sha256="c" * 64,
        candidate_manifest_sha256="d" * 64,
        expected_tasks=expected_tasks,
        expected_candidate_gpu_devices="1,0",
        expected_candidate_max_gpu_mem_gb=11,
        expected_candidate_commit=COMMIT,
    )


def _row(task_id: str, raw_output: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_id": "source",
        "task_id": task_id,
        "stage": "baseline",
        "model_name": "qwen3-vl-8b-instruct",
        "seed": 7,
        "request_id": f"request::{task_id}",
        "image_path": f"/images/{task_id}.png",
        "system_prompt_sha256": "1" * 64,
        "user_prompt_sha256": "2" * 64,
        "status": "ok",
        "raw_output": raw_output,
        "parsed": {"answer": raw_output},
        "parse_issues": [],
        "usage": {
            "attempts": 1,
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "total_tokens": 13,
            "latency_seconds": 1.0,
        },
        "audit_events": [
            {
                "status": "ok",
                "model": "/models/qwen3",
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
                "latency_ms": 1000.0,
                "gpu_devices": "0",
                "max_gpu_mem_gb": 20,
            }
        ],
    }


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 2,
        "created_at": "now",
        "stage": "baseline",
        "tasks": "/tasks.jsonl",
        "tasks_sha256": "e" * 64,
        "output": "/reference.jsonl",
        "model_name": "qwen3-vl-8b-instruct",
        "model_path": "/models/qwen3",
        "model_profile": "medium",
        "gpu_devices": "1,0",
        "max_gpu_mem_gb": 11,
        "seed": 7,
        "requested": 2,
        "git_commit": COMMIT,
        "cublas_workspace_config": ":4096:8",
        "answer_leakage_policy": "no outcomes",
    }


def _load_module() -> ModuleType:
    path = (historical_root() / "code/experiments") / "scripts" / "compare_egor_runtime_equivalence.py"
    spec = importlib.util.spec_from_file_location("runtime_equivalence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
