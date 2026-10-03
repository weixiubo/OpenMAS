#!/usr/bin/env python3
"""Audit append-only EGOR stage coverage without exposing model outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> int:
    args = _parse_args()
    stages = _assignments(args.stage, value_type=Path)
    expected_seeds = _assignments(args.expected_seed, value_type=int)
    allowed_task_ids = (
        _load_task_ids(Path(args.task_ids).resolve()) if args.task_ids else None
    )
    unknown_seed_stages = sorted(set(expected_seeds) - set(stages))
    if unknown_seed_stages:
        raise SystemExit(
            "expected seeds reference unknown stages: " + ", ".join(unknown_seed_stages)
        )
    report = audit_stage_matrix(
        tasks_path=Path(args.tasks).resolve(),
        stages={name: path.resolve() for name, path in stages.items()},
        expected_seeds=expected_seeds,
        require_manifest=args.require_manifest,
        allowed_task_ids=allowed_task_ids,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if bool(report["passed"]) or args.allow_incomplete else 2


def audit_stage_matrix(
    *,
    tasks_path: Path,
    stages: dict[str, Path],
    expected_seeds: dict[str, int] | None = None,
    require_manifest: bool = False,
    allowed_task_ids: set[str] | None = None,
) -> dict[str, Any]:
    all_task_keys = _load_task_keys(tasks_path)
    task_keys = (
        _load_task_keys(tasks_path, allowed_task_ids=allowed_task_ids)
        if allowed_task_ids is not None
        else all_task_keys
    )
    task_index = {key: index for index, key in enumerate(all_task_keys)}
    seeds = expected_seeds or {}
    stage_reports = {
        name: _audit_stage(
            path=path,
            task_keys=task_keys,
            task_index=task_index,
            tasks_path=tasks_path,
            expected_seed=seeds.get(name),
            require_manifest=require_manifest,
        )
        for name, path in sorted(stages.items())
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": bool(stage_reports)
        and all(bool(item["complete"]) for item in stage_reports.values()),
        "outcome_blind": True,
        "tasks": {
            "path": str(tasks_path),
            "sha256": _sha256(tasks_path),
            "count": len(task_keys),
            "subset_filter": allowed_task_ids is not None,
        },
        "require_manifest": require_manifest,
        "stages": stage_reports,
    }


def _audit_stage(
    *,
    path: Path,
    task_keys: list[tuple[str, str]],
    task_index: dict[tuple[str, str], int],
    tasks_path: Path,
    expected_seed: int | None,
    require_manifest: bool,
) -> dict[str, Any]:
    expected = set(task_keys)
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "complete": False,
            "rows": 0,
            "successful_tasks": 0,
            "missing_successful": _key_summary(expected),
            "unexpected": _key_summary(set()),
            "parse_error_lines": [],
            "invalid_key_lines": [],
            "seed_mismatches": _key_summary(set()),
            "manifest": {"exists": False, "compatible": not require_manifest},
        }

    rows = 0
    status_counts: dict[str, int] = {}
    parse_error_lines: list[int] = []
    invalid_key_lines: list[int] = []
    seen_keys: set[tuple[str, str]] = set()
    successful: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                parse_error_lines.append(line_number)
                continue
            if not isinstance(row, dict):
                parse_error_lines.append(line_number)
                continue
            key = (str(row.get("source_id", "")), str(row.get("task_id", "")))
            if not all(key):
                invalid_key_lines.append(line_number)
                continue
            seen_keys.add(key)
            status = str(row.get("status", "missing"))
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "ok":
                successful[key] = row

    successful_keys = set(successful)
    missing = expected - successful_keys
    unexpected = seen_keys - expected
    seed_mismatch_keys: set[tuple[str, str]] = set()
    if expected_seed is not None:
        for key, row in successful.items():
            if key not in task_index:
                continue
            wanted = expected_seed + task_index[key]
            try:
                actual = int(row.get("seed"))
            except (TypeError, ValueError):
                seed_mismatch_keys.add(key)
                continue
            if actual != wanted:
                seed_mismatch_keys.add(key)

    manifest = _audit_manifest(
        path=path,
        tasks_path=tasks_path,
        expected_tasks=len(task_keys),
        successful_tasks=len(successful_keys & expected),
        expected_seed=expected_seed,
    )
    manifest_gate = bool(manifest["compatible"]) if require_manifest else True
    complete = (
        not any(
            (
                missing,
                unexpected,
                parse_error_lines,
                invalid_key_lines,
                seed_mismatch_keys,
            )
        )
        and manifest_gate
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "exists": True,
        "complete": complete,
        "rows": rows,
        "unique_tasks_with_attempts": len(seen_keys & expected),
        "duplicate_attempt_rows": max(0, rows - len(seen_keys)),
        "successful_tasks": len(successful_keys & expected),
        "status_counts": dict(sorted(status_counts.items())),
        "missing_successful": _key_summary(missing),
        "unexpected": _key_summary(unexpected),
        "parse_error_lines": parse_error_lines[:20],
        "invalid_key_lines": invalid_key_lines[:20],
        "expected_base_seed": expected_seed,
        "seed_mismatches": _key_summary(seed_mismatch_keys),
        "manifest": manifest,
    }


def _audit_manifest(
    *,
    path: Path,
    tasks_path: Path,
    expected_tasks: int,
    successful_tasks: int,
    expected_seed: int | None,
) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        return {"path": str(manifest_path), "exists": False, "compatible": False}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "path": str(manifest_path),
            "exists": True,
            "compatible": False,
            "error": type(exc).__name__,
        }
    checks = {
        "tasks_sha256": payload.get("tasks_sha256") == _sha256(tasks_path),
        "requested": int(payload.get("requested", -1)) == expected_tasks,
        "successful_total": int(payload.get("successful_total", -1))
        == successful_tasks,
    }
    if expected_seed is not None:
        checks["base_seed"] = int(payload.get("seed", -1)) == expected_seed
    return {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "exists": True,
        "compatible": all(checks.values()),
        "checks": checks,
        "git_commit": str(payload.get("git_commit", "")),
        "seed_schedule": str(payload.get("seed_schedule", "legacy-unrecorded")),
    }


def _load_task_keys(
    path: Path, *, allowed_task_ids: set[str] | None = None
) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"task row {line_number} is not an object: {path}")
            key = (
                str(row.get("source_id") or row.get("benchmark") or ""),
                str(row.get("task_id", "")),
            )
            if not all(key):
                raise ValueError(f"task row {line_number} lacks source/task ID: {path}")
            if key in seen:
                raise ValueError(f"duplicate task {key[0]}::{key[1]}: {path}")
            seen.add(key)
            if allowed_task_ids is not None and key[1] not in allowed_task_ids:
                continue
            keys.append(key)
    if not keys:
        raise ValueError(f"no tasks found: {path}")
    if allowed_task_ids is not None:
        found = {task_id for _, task_id in keys}
        missing = sorted(allowed_task_ids - found)
        if missing:
            raise ValueError(f"task-ID filter contains unknown IDs: {missing[:20]}")
    return keys


def _load_task_ids(path: Path) -> set[str]:
    task_ids = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not task_ids:
        raise ValueError(f"task-ID filter is empty: {path}")
    return task_ids


def _key_summary(keys: set[tuple[str, str]]) -> dict[str, Any]:
    rendered = sorted(f"{source}::{task}" for source, task in keys)
    return {
        "count": len(rendered),
        "sha256": hashlib.sha256("\n".join(rendered).encode()).hexdigest(),
        "examples": rendered[:20],
    }


def _assignments(values: list[str], *, value_type: type[Any]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw in values:
        name, separator, value = raw.partition("=")
        if not separator or not name.strip() or not value.strip():
            raise SystemExit(f"expected NAME=VALUE, received: {raw}")
        name = name.strip()
        if name in parsed:
            raise SystemExit(f"duplicate assignment: {name}")
        parsed[name] = value_type(value.strip())
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--task-ids", default="")
    parser.add_argument(
        "--stage", action="append", default=[], metavar="NAME=PATH", required=True
    )
    parser.add_argument(
        "--expected-seed", action="append", default=[], metavar="NAME=INTEGER"
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--require-manifest", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return zero for a live partial matrix while still reporting passed=false.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
