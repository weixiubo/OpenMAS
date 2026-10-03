#!/usr/bin/env python3
"""Freeze an outcome-blind verifier-call plan after core execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openagora_experiments.egor.runner import load_latest_stage_records
from openagora_experiments.egor.verification import answers_agree, canonical_answer


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class VisibleTask:
    source_id: str
    task_id: str
    metric: str
    options: tuple[tuple[str, str], ...]


def main() -> int:
    args = _parse_args()
    tasks_path = Path(args.tasks).resolve()
    baseline_path = Path(args.baseline).resolve()
    core_path = Path(args.core).resolve()
    repairability_path = Path(args.repairability).resolve()
    output_path = Path(args.output).resolve()
    task_ids_path = Path(args.task_ids_output).resolve()

    tasks = _load_visible_tasks(tasks_path)
    if args.expected_tasks and len(tasks) != args.expected_tasks:
        raise ValueError(
            f"visible task count {len(tasks)} != expected {args.expected_tasks}"
        )
    repairability = _load_repairability(repairability_path)
    task_keys = {(task.source_id, task.task_id) for task in tasks}
    if set(repairability) != task_keys:
        raise ValueError("repairability plan keys do not match the task set")
    baseline = load_latest_stage_records(baseline_path)
    core = load_latest_stage_records(core_path)

    plans = plan_verified_overwrites(
        tasks=tasks,
        repairability=repairability,
        baseline=baseline,
        core=core,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_ids_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in plans),
        encoding="utf-8",
    )
    task_ids_path.write_text(
        "".join(
            f"{row['task_id']}\n" for row in plans if row["verifier_triggered"]
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("EGOR_SOURCE_COMMIT", "").strip()
        or _git_output("rev-parse", "HEAD"),
        "policy": args.policy,
        "allowed_task_fields_read": [
            "benchmark/source_id",
            "task_id",
            "metric",
            "options",
        ],
        "outcome_fields_read": False,
        "tasks": len(plans),
        "repairability_triggered": sum(
            row["repairability_triggered"] for row in plans
        ),
        "verifier_triggered": sum(row["verifier_triggered"] for row in plans),
        "reason_counts": _counts(str(row["reason"]) for row in plans),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (
                tasks_path,
                baseline_path,
                core_path,
                repairability_path,
            )
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (output_path, task_ids_path)
        ],
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"planned verifier calls={manifest['verifier_triggered']} "
        f"of repairability triggers={manifest['repairability_triggered']}"
    )
    return 0


def plan_verified_overwrites(
    *,
    tasks: list[VisibleTask],
    repairability: dict[tuple[str, str], dict[str, Any]],
    baseline: dict[tuple[str, str], dict[str, Any]],
    core: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for task in tasks:
        key = (task.source_id, task.task_id)
        triggered = bool(repairability[key]["repairability_triggered"])
        baseline_record = baseline.get(key)
        core_record = core.get(key)
        verifier_triggered = False
        if not triggered:
            reason = "repairability_gate_abstained"
        elif not _ok(baseline_record):
            reason = "baseline_unavailable"
        elif not _ok(core_record):
            reason = "core_unavailable"
        else:
            baseline_answer = _answer(baseline_record)
            core_answer = _answer(core_record)
            if not canonical_answer(core_answer):
                reason = "empty_core_answer"
            elif answers_agree(
                baseline_answer,
                core_answer,
                metric=task.metric,
                options=[{"id": key, "text": text} for key, text in task.options],
            ):
                reason = "no_baseline_overwrite"
            else:
                reason = "core_proposes_overwrite"
                verifier_triggered = True
        plans.append(
            {
                "schema_version": 1,
                "source_id": task.source_id,
                "task_id": task.task_id,
                "repairability_triggered": triggered,
                "verifier_triggered": verifier_triggered,
                "reason": reason,
            }
        )
    return plans


def _load_visible_tasks(path: Path) -> list[VisibleTask]:
    tasks: list[VisibleTask] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            source_id = str(
                payload.get("benchmark") or payload.get("source_id") or ""
            ).strip()
            task_id = str(payload.get("task_id", "")).strip()
            metric = str(payload.get("metric", "accuracy")).strip()
            options = tuple(
                (
                    str(item.get("id", "")).strip().upper(),
                    str(item.get("text", "")).strip(),
                )
                for item in payload.get("options", [])
                if isinstance(item, dict)
                and str(item.get("id", "")).strip()
                and str(item.get("text", "")).strip()
            )
            key = (source_id, task_id)
            if not all(key) or key in seen:
                raise ValueError(f"invalid or duplicate visible task at line {line_number}")
            seen.add(key)
            tasks.append(VisibleTask(source_id, task_id, metric, options))
    if not tasks:
        raise ValueError("task set is empty")
    return tasks


def _load_repairability(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("source_id", "")), str(row.get("task_id", "")))
            if not all(key) or key in output:
                raise ValueError(f"invalid or duplicate repairability key: {key}")
            if not isinstance(row.get("repairability_triggered"), bool):
                raise ValueError(f"repairability decision is not boolean: {key}")
            output[key] = row
    return output


def _ok(record: dict[str, Any] | None) -> bool:
    return bool(record and str(record.get("status", "")) == "ok")


def _answer(record: dict[str, Any] | None) -> str:
    parsed = record.get("parsed", {}) if record else {}
    return str(parsed.get("answer", "")) if isinstance(parsed, dict) else ""


def _counts(values: Any) -> dict[str, int]:
    output: dict[str, int] = {}
    for value in values:
        output[value] = output.get(value, 0) + 1
    return dict(sorted(output.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--core", required=True)
    parser.add_argument("--repairability", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-ids-output", required=True)
    parser.add_argument("--expected-tasks", type=int, default=0)
    parser.add_argument(
        "--policy",
        default="call v7 only for a nonempty core answer that differs from baseline",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
