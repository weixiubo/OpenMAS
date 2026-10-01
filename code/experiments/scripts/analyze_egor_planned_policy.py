#!/usr/bin/env python3
"""Score an outcome-blind dynamic EGOR policy using only its executed stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openagora_experiments.egor.evaluation import summarize_policy
from openagora_experiments.egor.planning import evaluate_planned_policy
from openagora_experiments.egor.runner import (
    load_latest_stage_records,
    load_tasks,
)


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    args = _parse_args()
    tasks_path = Path(args.tasks).resolve()
    baseline_path = Path(args.baseline).resolve()
    plans_path = Path(args.plans).resolve()
    execution_config_path = Path(args.execution_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_config = json.loads(execution_config_path.read_text(encoding="utf-8"))
    if not isinstance(execution_config, dict):
        raise ValueError("execution config must be an object")
    action_paths = _path_map(execution_config.get("actions", {}), execution_config_path)
    synthesis_paths = _path_map(
        execution_config.get("combinations", {}), execution_config_path
    )
    decision_path = (
        Path(args.decision_records).resolve() if args.decision_records else None
    )
    results = evaluate_planned_policy(
        load_tasks(tasks_path),
        baseline_records=load_latest_stage_records(baseline_path),
        plan_records=_load_jsonl_by_key(plans_path),
        action_record_sets={
            action_id: load_latest_stage_records(path)
            for action_id, path in action_paths.items()
        },
        synthesis_record_sets={
            combination: load_latest_stage_records(path)
            for combination, path in synthesis_paths.items()
        },
        policy_id=args.policy_id,
        decision_records=(
            load_latest_stage_records(decision_path)
            if decision_path is not None
            else None
        ),
        require_evidence_for_change=not args.audit_only,
    )
    rows_path = output_dir / "policy_rows.csv"
    summary_path = output_dir / "policy_summary.json"
    _write_rows(rows_path, results)
    summary = summarize_policy([result.policy_row for result in results])
    summary["mean_selected_workers"] = sum(
        len(result.selected_action_ids) for result in results
    ) / len(results)
    summary["distinct_combinations"] = float(
        len({result.combination_id for result in results})
    )
    summary_path.write_text(
        json.dumps({args.policy_id: summary}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    input_paths = [
        tasks_path,
        baseline_path,
        plans_path,
        execution_config_path,
        *action_paths.values(),
        *synthesis_paths.values(),
    ]
    if decision_path is not None:
        input_paths.append(decision_path)
    manifest = {
        "schema_version": 1,
        "source_commit": os.environ.get("EGOR_SOURCE_COMMIT", "").strip()
        or _git_output("rev-parse", "HEAD"),
        "policy_id": args.policy_id,
        "tasks": len(results),
        "strict_provenance_gate": not args.audit_only,
        "combination_counts": dict(
            Counter(result.combination_id for result in results)
        ),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in dict.fromkeys(input_paths)
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (rows_path, summary_path)
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"scored {len(results)} planned-policy tasks in {output_dir}")
    return 0


def _write_rows(path: Path, results: list[Any]) -> None:
    fieldnames = [
        "source_id",
        "task_id",
        "policy_id",
        "candidate_id",
        "selected_action_ids",
        "combination_id",
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
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            payload = asdict(result.policy_row)
            payload["selected_action_ids"] = "|".join(result.selected_action_ids)
            payload["combination_id"] = result.combination_id
            payload["triggered"] = int(result.policy_row.triggered)
            payload["strict_harm"] = int(result.policy_row.strict_harm)
            payload["useful_repair"] = int(result.policy_row.useful_repair)
            payload["provenance_rejected"] = int(result.policy_row.provenance_rejected)
            payload["failed_stages"] = "|".join(result.policy_row.failed_stages)
            writer.writerow(payload)


def _path_map(value: Any, config_path: Path) -> dict[str, Path]:
    if not isinstance(value, dict):
        raise ValueError("execution action/combination maps must be objects")
    paths: dict[str, Path] = {}
    for key, raw_path in value.items():
        path = Path(str(raw_path)).expanduser()
        paths[str(key)] = (
            path if path.is_absolute() else config_path.parent / path
        ).resolve()
    return paths


def _load_jsonl_by_key(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("source_id", "")), str(row.get("task_id", "")))
            if not all(key):
                raise ValueError("plan row lacks source_id/task_id")
            rows[key] = row
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--execution-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-id", default="egor_dynamic_b2")
    parser.add_argument(
        "--decision-records",
        default="",
        help="Optional paid LLM-router stage whose usage is added to every task",
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
