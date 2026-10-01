#!/usr/bin/env python3
"""Materialize answer-blind resolver-change decisions for EGOR-v10."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from openagora_experiments.egor.runner import load_latest_stage_records, load_tasks
from openagora_experiments.egor.verification import answers_agree, canonical_answer


def main() -> int:
    args = _parse_args()
    tasks_path = Path(args.tasks).resolve()
    baseline_path = Path(args.baseline).resolve()
    resolver_path = Path(args.resolver).resolve()
    repairability_path = Path(args.repairability).resolve()
    output_path = Path(args.output).resolve()
    ids_path = Path(args.task_ids_output).resolve()

    tasks = load_tasks(tasks_path)
    if args.expected_tasks and len(tasks) != args.expected_tasks:
        raise ValueError(f"task denominator is {len(tasks)}, expected {args.expected_tasks}")
    baseline = load_latest_stage_records(baseline_path)
    resolver = load_latest_stage_records(resolver_path)
    repairability = _load_repairability(repairability_path)

    rows: list[dict[str, Any]] = []
    changed_ids: list[str] = []
    for task in tasks:
        key = (task.source_id, task.task_id)
        if key not in repairability:
            raise ValueError(f"missing repairability decision: {key}")
        triggered = repairability[key]
        baseline_record = baseline.get(key)
        resolver_record = resolver.get(key)
        baseline_answer = str(_parsed(baseline_record).get("answer", ""))
        resolver_payload = _parsed(resolver_record)
        resolver_answer = str(resolver_payload.get("answer", ""))
        transport = resolver_payload.get("transport", {})
        decisive = bool(
            triggered
            and _ok(resolver_record)
            and isinstance(transport, Mapping)
            and transport.get("operational_complete")
            and transport.get("decisive")
            and not transport.get("sanitized_abstention")
            and canonical_answer(resolver_answer)
        )
        options = [{"id": option_id, "text": text} for option_id, text in task.options]
        changed = bool(
            decisive
            and _ok(baseline_record)
            and not answers_agree(
                baseline_answer,
                resolver_answer,
                metric=task.metric,
                options=options,
            )
        )
        reason = (
            "gate_not_triggered"
            if not triggered
            else "resolver_failed"
            if not _ok(resolver_record)
            else "resolver_incomplete_or_abstained"
            if not decisive
            else "resolver_matches_baseline"
            if not changed
            else "resolver_changed"
        )
        rows.append(
            {
                "schema_version": 1,
                "source_id": task.source_id,
                "task_id": task.task_id,
                "repairability_triggered": triggered,
                "resolver_changed": changed,
                "reason": reason,
                "information_boundary": (
                    "task metric/options plus generated baseline and candidate-blind "
                    "resolver answers; no reference or realized score"
                ),
            }
        )
        if changed:
            changed_ids.append(task.task_id)

    if len({(row["source_id"], row["task_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate output decision")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text("".join(f"{task_id}\n" for task_id in changed_ids), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("EGOR_SOURCE_COMMIT", "").strip(),
        "policy": "mechanical candidate-blind resolver-change gate",
        "allowed_fields_read": [
            "source_id",
            "task_id",
            "metric",
            "answer options",
            "repairability_triggered",
            "baseline generated answer",
            "resolver generated answer and transport",
        ],
        "outcome_fields_read": False,
        "scores_computed": False,
        "tasks": len(rows),
        "reason_counts": dict(sorted(Counter(row["reason"] for row in rows).items())),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (
                ("tasks", tasks_path),
                ("baseline", baseline_path),
                ("resolver", resolver_path),
                ("repairability", repairability_path),
            )
        },
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "task_ids_output": {"path": str(ids_path), "sha256": _sha256(ids_path)},
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["reason_counts"], sort_keys=True))
    return 0


def _load_repairability(path: Path) -> dict[tuple[str, str], bool]:
    rows: dict[tuple[str, str], bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["source_id"]), str(row["task_id"]))
        if key in rows:
            raise ValueError(f"duplicate repairability decision: {key}")
        rows[key] = bool(row["repairability_triggered"])
    return rows


def _parsed(record: Mapping[str, Any] | None) -> dict[str, Any]:
    value = record.get("parsed", {}) if record else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _ok(record: Mapping[str, Any] | None) -> bool:
    return bool(record and str(record.get("status", "")) == "ok")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--resolver", required=True)
    parser.add_argument("--repairability", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-ids-output", required=True)
    parser.add_argument("--expected-tasks", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
