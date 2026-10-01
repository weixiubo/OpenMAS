#!/usr/bin/env python3
"""Audit one confirmation source without reading labels or scoring outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STAGE_SPECS = {
    "baseline": {
        "file": "baseline_qwen3.jsonl",
        "stage": "baseline",
        "model_name": "qwen3-vl-8b-instruct",
        "model_arg": "qwen3_model",
        "seed_offset": 10,
        "ids": "all",
    },
    "exact": {
        "file": "specialist_exact_qwen25.jsonl",
        "stage": "specialist",
        "model_name": "qwen2_5_vl_7b_instruct",
        "model_arg": "qwen25_model",
        "seed_offset": 30,
        "ids": "repairability",
    },
    "visual": {
        "file": "specialist_visual_internvl3.jsonl",
        "stage": "specialist",
        "model_name": "internvl3-8b-hf",
        "model_arg": "internvl_model",
        "seed_offset": 31,
        "ids": "repairability",
    },
    "core": {
        "file": "core_pair.jsonl",
        "stage": "synthesis",
        "model_name": "qwen3-vl-8b-instruct",
        "model_arg": "qwen3_model",
        "seed_offset": 40,
        "ids": "repairability",
    },
    "verifier": {
        "file": "verifier_v7_qwen35.jsonl",
        "stage": "verifier_v7",
        "model_name": "qwen3.5-9b",
        "model_arg": "qwen35_model",
        "seed_offset": 50,
        "ids": "overwrite",
    },
}


def main() -> int:
    args = _parse_args()
    report = audit_source(
        tasks_path=Path(args.tasks).resolve(),
        root=Path(args.root).resolve(),
        repairability_path=Path(args.repairability).resolve(),
        overwrite_plans_path=Path(args.overwrite_plans).resolve(),
        source_commit=args.source_commit,
        mode=args.mode,
        expected_tasks=args.expected_tasks,
        expected_tasks_sha256=args.expected_tasks_sha256,
        seed_root=args.seed_root,
        model_paths={
            "qwen3_model": str(Path(args.qwen3_model).resolve()),
            "qwen25_model": str(Path(args.qwen25_model).resolve()),
            "internvl_model": str(Path(args.internvl_model).resolve()),
            "qwen35_model": str(Path(args.qwen35_model).resolve()),
        },
        verifier_stage=args.verifier_stage,
        verifier_file=args.verifier_file,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 2


def audit_source(
    *,
    tasks_path: Path,
    root: Path,
    repairability_path: Path,
    overwrite_plans_path: Path,
    source_commit: str,
    mode: str,
    expected_tasks: int,
    expected_tasks_sha256: str,
    seed_root: int,
    model_paths: dict[str, str],
    verifier_stage: str = "verifier_v7",
    verifier_file: str = "verifier_v7_qwen35.jsonl",
) -> dict[str, Any]:
    task_keys, task_indices = _load_task_keys(tasks_path)
    actual_tasks_sha256 = _sha256(tasks_path)
    repairability = _load_plan(
        repairability_path, decision="repairability_triggered"
    )
    overwrite = _load_plan(overwrite_plans_path, decision="verifier_triggered")
    task_key_set = set(task_keys)
    plan_checks = {
        "repairability_keys": set(repairability) == task_key_set,
        "overwrite_keys": set(overwrite) == task_key_set,
        "overwrite_subset": all(
            not row["verifier_triggered"]
            or repairability[key]["repairability_triggered"]
            for key, row in overwrite.items()
        ),
    }
    id_sets = {
        "all": task_key_set,
        "repairability": {
            key
            for key, row in repairability.items()
            if row["repairability_triggered"]
        },
        "overwrite": {
            key for key, row in overwrite.items() if row["verifier_triggered"]
        },
    }
    stage_specs = {name: dict(spec) for name, spec in STAGE_SPECS.items()}
    stage_specs["verifier"]["stage"] = verifier_stage
    stage_specs["verifier"]["file"] = verifier_file
    stages = {
        name: _audit_stage(
            path=root / str(spec["file"]),
            tasks_path=tasks_path,
            expected_keys=id_sets[str(spec["ids"])],
            task_indices=task_indices,
            expected_stage=str(spec["stage"]),
            expected_model_name=str(spec["model_name"]),
            expected_model_path=model_paths[str(spec["model_arg"])],
            expected_seed=seed_root + int(spec["seed_offset"]),
            expected_commit=source_commit,
            expect_cpu_offload=name == "verifier",
        )
        for name, spec in stage_specs.items()
    }
    verifier_expected = len(id_sets["overwrite"])
    verifier_operational = stages["verifier"]["operational_records"]
    completion = (
        verifier_operational / verifier_expected if verifier_expected else 0.0
    )
    task_checks = {
        "count": len(task_keys) == expected_tasks,
        "sha256": actual_tasks_sha256 == expected_tasks_sha256,
        "single_source": len({source for source, _ in task_keys}) == 1,
        "unique_task_ids": len({task_id for _, task_id in task_keys})
        == len(task_keys),
    }
    integrity_passed = (
        all(task_checks.values())
        and all(plan_checks.values())
        and all(stage["integrity_complete"] for stage in stages.values())
    )
    format_gate = {
        "all_nonverifier_rows_successful": all(
            stages[name]["successful_records"] == stages[name]["expected_records"]
            for name in ("baseline", "exact", "visual", "core")
        ),
        "verifier_was_exercised": verifier_expected > 0,
        "verifier_operational_completion": completion >= 0.95,
    }
    passed = integrity_passed and (
        all(format_gate.values()) if mode == "format" else True
    )
    return {
        "schema_version": 1,
        "mode": mode,
        "passed": passed,
        "integrity_passed": integrity_passed,
        "outcome_blind": True,
        "allowed_task_fields_read": [
            "benchmark/source_id",
            "task_id",
            "image_path",
        ],
        "outcome_fields_read": False,
        "scores_computed": False,
        "verifier_contract": {
            "stage": verifier_stage,
            "file": verifier_file,
        },
        "source_commit": source_commit,
        "tasks": {
            "path": str(tasks_path),
            "sha256": actual_tasks_sha256,
            "count": len(task_keys),
            "checks": task_checks,
        },
        "plan_checks": plan_checks,
        "repairability_triggered": len(id_sets["repairability"]),
        "verifier_triggered": verifier_expected,
        "verifier_operational_completion": completion,
        "format_gate": format_gate,
        "stages": stages,
    }


def _audit_stage(
    *,
    path: Path,
    tasks_path: Path,
    expected_keys: set[tuple[str, str]],
    task_indices: dict[tuple[str, str], int],
    expected_stage: str,
    expected_model_name: str,
    expected_model_path: str,
    expected_seed: int,
    expected_commit: str,
    expect_cpu_offload: bool,
) -> dict[str, Any]:
    rows, parse_errors = _load_rows(path)
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    seen_keys: set[tuple[str, str]] = set()
    invalid_keys = 0
    for row in rows:
        key = (str(row.get("source_id", "")), str(row.get("task_id", "")))
        if not all(key):
            invalid_keys += 1
            continue
        seen_keys.add(key)
        latest[key] = row
    actual_keys = set(latest)
    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys
    successful = sum(
        str(latest[key].get("status", "")) == "ok"
        for key in expected_keys & actual_keys
    )
    seed_mismatches = 0
    metadata_mismatches = 0
    operational = 0
    native_transport = 0
    sanitized_transport = 0
    for key in expected_keys & actual_keys:
        row = latest[key]
        if "seed" in row and row.get("seed") != expected_seed + task_indices[key]:
            seed_mismatches += 1
        if row.get("stage") != expected_stage:
            metadata_mismatches += 1
        if row.get("model_name") not in {None, expected_model_name}:
            metadata_mismatches += 1
        if str(row.get("status", "")) == "ok":
            if not row.get("system_prompt_sha256") or not row.get("user_prompt_sha256"):
                metadata_mismatches += 1
            parsed = row.get("parsed", {})
            transport = parsed.get("transport", {}) if isinstance(parsed, dict) else {}
            if bool(transport.get("operational_complete")):
                operational += 1
                if transport.get("mode") == "sanitized_transport_abstention":
                    sanitized_transport += 1
                else:
                    native_transport += 1
    manifest = _audit_manifest(
        path=path.with_suffix(path.suffix + ".manifest.json"),
        tasks_sha256=_sha256(tasks_path),
        expected_records=len(expected_keys),
        successful_records=successful,
        expected_stage=expected_stage,
        expected_model_name=expected_model_name,
        expected_model_path=expected_model_path,
        expected_seed=expected_seed,
        expected_commit=expected_commit,
        expect_cpu_offload=expect_cpu_offload,
    )
    integrity = (
        not parse_errors
        and invalid_keys == 0
        and not missing
        and not unexpected
        and seed_mismatches == 0
        and metadata_mismatches == 0
        and manifest["compatible"]
    )
    return {
        "path": str(path),
        "sha256": _sha256(path) if path.exists() else "",
        "integrity_complete": integrity,
        "expected_records": len(expected_keys),
        "attempt_rows": len(rows),
        "latest_records": len(actual_keys),
        "successful_records": successful,
        "failed_records": len(expected_keys) - successful,
        "operational_records": (
            operational if expected_stage.startswith("verifier_") else 0
        ),
        "native_transport_records": (
            native_transport if expected_stage.startswith("verifier_") else 0
        ),
        "sanitized_transport_records": (
            sanitized_transport if expected_stage.startswith("verifier_") else 0
        ),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "parse_error_lines": parse_errors[:20],
        "invalid_key_rows": invalid_keys,
        "seed_mismatches": seed_mismatches,
        "metadata_mismatches": metadata_mismatches,
        "manifest": manifest,
    }


def _audit_manifest(
    *,
    path: Path,
    tasks_sha256: str,
    expected_records: int,
    successful_records: int,
    expected_stage: str,
    expected_model_name: str,
    expected_model_path: str,
    expected_seed: int,
    expected_commit: str,
    expect_cpu_offload: bool,
) -> dict[str, Any]:
    if expected_records == 0 and not path.exists():
        return {"exists": False, "compatible": True, "empty_stage": True}
    if not path.exists():
        return {"exists": False, "compatible": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"exists": True, "compatible": False, "error": type(exc).__name__}
    checks = {
        "tasks_sha256": payload.get("tasks_sha256") == tasks_sha256,
        "requested": payload.get("requested") == expected_records,
        "successful_total": payload.get("successful_total") == successful_records,
        "stage": payload.get("stage") == expected_stage,
        "model_name": payload.get("model_name") == expected_model_name,
        "model_path": str(Path(str(payload.get("model_path", ""))).resolve())
        == expected_model_path,
        "seed": payload.get("seed") == expected_seed,
        "seed_schedule": payload.get("seed_schedule")
        == "base_plus_original_requested_index",
        "source_commit": payload.get("git_commit") == expected_commit,
        "clean_worktree": payload.get("git_worktree_dirty") is False,
        "cpu_offload": bool(payload.get("allow_cpu_offload"))
        is expect_cpu_offload,
        "runtime_fingerprint": bool(payload.get("runtime")),
    }
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "exists": True,
        "compatible": all(checks.values()),
        "checks": checks,
    }


def _load_task_keys(
    path: Path,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], int]]:
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row.get("benchmark") or row.get("source_id") or ""),
                str(row.get("task_id", "")),
            )
            if not all(key) or key in seen:
                raise ValueError(f"invalid or duplicate task key at line {line_number}")
            seen.add(key)
            keys.append(key)
    return keys, {key: index for index, key in enumerate(keys)}


def _load_plan(path: Path, *, decision: str) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row.get("source_id", "")), str(row.get("task_id", "")))
        if not all(key) or key in output or not isinstance(row.get(decision), bool):
            raise ValueError(f"invalid plan row for {decision}: {key}")
        output[key] = row
    return output


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    if not path.exists():
        return [], []
    rows: list[dict[str, Any]] = []
    errors: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append(line_number)
            continue
        if not isinstance(row, dict):
            errors.append(line_number)
            continue
        rows.append(row)
    return rows, errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--repairability", required=True)
    parser.add_argument("--overwrite-plans", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--mode", choices=["format", "confirmation"], required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--expected-tasks-sha256", required=True)
    parser.add_argument("--seed-root", type=int, required=True)
    parser.add_argument("--qwen3-model", required=True)
    parser.add_argument("--qwen25-model", required=True)
    parser.add_argument("--internvl-model", required=True)
    parser.add_argument("--qwen35-model", required=True)
    parser.add_argument("--verifier-stage", default="verifier_v7")
    parser.add_argument("--verifier-file", default="verifier_v7_qwen35.jsonl")
    parser.add_argument("--output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
