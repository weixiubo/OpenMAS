#!/usr/bin/env python3
"""Validate the latest records of one EGOR stage against a frozen task set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TaskKey = tuple[str, str]


def validate_stage_output(
    *,
    stage_output: Path,
    tasks_path: Path,
    task_ids_path: Path | None = None,
) -> dict[str, int]:
    """Validate keys and latest status using protocol-normalized source IDs."""

    tasks = _jsonl(tasks_path)
    task_keys: list[TaskKey] = []
    for row in tasks:
        source_id = str(row.get("source_id") or row.get("benchmark") or "").strip()
        task_id = str(row.get("task_id") or "").strip()
        key = (source_id, task_id)
        if not all(key):
            raise ValueError(
                f"task file lacks task_id or benchmark/source_id: {tasks_path}"
            )
        task_keys.append(key)
    if len(task_keys) != len(set(task_keys)):
        raise ValueError(f"duplicate source/task key in {tasks_path}")

    allowed_task_ids: set[str] | None = None
    if task_ids_path is not None:
        task_ids = [
            line.strip()
            for line in task_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"duplicate task IDs in {task_ids_path}")
        allowed_task_ids = set(task_ids)
        known_task_ids = {task_id for _, task_id in task_keys}
        unknown = sorted(allowed_task_ids - known_task_ids)
        if unknown:
            raise ValueError(
                f"unknown task IDs in {task_ids_path}: {unknown[:5]}"
            )

    expected = {
        key
        for key in task_keys
        if allowed_task_ids is None or key[1] in allowed_task_ids
    }
    latest: dict[TaskKey, dict[str, Any]] = {}
    history = 0
    for row in _jsonl(stage_output):
        key = (
            str(row.get("source_id") or row.get("benchmark") or "").strip(),
            str(row.get("task_id") or "").strip(),
        )
        if key not in expected:
            raise ValueError(f"unexpected stage key in {stage_output}: {key}")
        latest[key] = row
        history += 1

    missing = sorted(expected - set(latest))
    non_ok = sorted(
        key for key, row in latest.items() if str(row.get("status", "")) != "ok"
    )
    if missing or non_ok:
        raise ValueError(
            f"incomplete stage: {stage_output} expected={len(expected)} "
            f"history={history} latest={len(latest)} missing={missing[:5]} "
            f"non_ok={non_ok[:5]}"
        )
    return {
        "expected": len(expected),
        "history": history,
        "latest": len(latest),
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(row)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-output", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task-ids", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = validate_stage_output(
        stage_output=args.stage_output.resolve(),
        tasks_path=args.tasks.resolve(),
        task_ids_path=args.task_ids.resolve() if args.task_ids else None,
    )
    print(
        "validated stage output: "
        f"expected={summary['expected']} history={summary['history']} "
        f"latest={summary['latest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
