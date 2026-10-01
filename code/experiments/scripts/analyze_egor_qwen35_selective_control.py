#!/usr/bin/env python3
"""Post-hoc selective strong-solver controls for sealed EGOR-v8 tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import analyze_egor_qwen35_direct_control as direct
import analyze_egor_v7_confirmation as shared
from openagora_experiments.egor.runner import load_latest_stage_records, load_tasks


POLICIES = ("qwen35_direct", "gate_then_qwen35", "core_change_then_qwen35", "egor_v8")


def main() -> int:
    args = _parse_args()
    sources = {
        "chartqa": (
            Path(args.chartqa_tasks).resolve(),
            Path(args.chartqa_v8_root).resolve(),
            Path(args.chartqa_control_root).resolve(),
        ),
        "infographic": (
            Path(args.infographic_tasks).resolve(),
            Path(args.infographic_v8_root).resolve(),
            Path(args.infographic_control_root).resolve(),
        ),
    }
    preflight = shared.preflight_confirmation(
        sources={name: (tasks, v8_root) for name, (tasks, v8_root, _) in sources.items()},
        expected_commit=args.v8_source_commit,
        expected_tasks_per_source=args.tasks_per_source,
    )
    if not preflight["passed"]:
        raise ValueError("sealed v8 input preflight failed")
    primary_rows = direct._load_primary_rows(Path(args.primary_rows).resolve())
    rows = [
        row
        for name in ("chartqa", "infographic")
        for row in _load_source(
            tasks_path=sources[name][0],
            v8_root=sources[name][1],
            control_root=sources[name][2],
            primary_rows=primary_rows,
            expected_control_commit=args.control_source_commit,
            expected_tasks=args.tasks_per_source,
        )
    ]
    summaries = {
        policy: _summarize(rows, policy, args.bootstrap_samples, args.seed + i)
        for i, policy in enumerate(POLICIES)
    }
    direct_wins = {r["key"] for r in rows if r["direct_delta"] > 0.0}
    direct_losses = {r["key"] for r in rows if r["direct_delta"] < 0.0}
    accepted = {r["key"] for r in rows if r["egor_accept"]}
    report = {
        "schema_version": 1,
        "claim_boundary": "posthoc_matched_backbone_selectivity_diagnostic_not_confirmation",
        "v8_source_commit": args.v8_source_commit,
        "control_source_commit": args.control_source_commit,
        "tasks": len(rows),
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed_root": args.seed,
            "unit": "paired task with equal task weight",
        },
        "preflight_passed": preflight["passed"],
        "policies": summaries,
        "selection_overlap": {
            "qwen35_direct_wins": len(direct_wins),
            "qwen35_direct_losses": len(direct_losses),
            "egor_accepted": len(accepted),
            "egor_accepted_qwen35_wins": len(accepted & direct_wins),
            "egor_accepted_qwen35_losses": len(accepted & direct_losses),
            "egor_accepted_qwen35_metric_ties": len(accepted - direct_wins - direct_losses),
        },
        "paired_contrasts": {
            "egor_minus_direct": _contrast(
                rows, "egor_v8", "qwen35_direct", args.bootstrap_samples, args.seed + 101
            ),
            "egor_minus_gate_then_direct": _contrast(
                rows, "egor_v8", "gate_then_qwen35", args.bootstrap_samples, args.seed + 102
            ),
            "egor_minus_core_change_then_direct": _contrast(
                rows,
                "egor_v8",
                "core_change_then_qwen35",
                args.bootstrap_samples,
                args.seed + 103,
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _load_source(
    *,
    tasks_path: Path,
    v8_root: Path,
    control_root: Path,
    primary_rows: dict[tuple[str, str], dict[str, str]],
    expected_control_commit: str,
    expected_tasks: int,
) -> list[dict[str, Any]]:
    direct._validate_control_root(
        control_root, tasks_path=tasks_path, expected_commit=expected_control_commit
    )
    tasks = load_tasks(tasks_path)
    if len(tasks) != expected_tasks:
        raise ValueError(f"unexpected task count: {len(tasks)}")
    baseline = load_latest_stage_records(v8_root / "baseline_qwen3.jsonl")
    strong = load_latest_stage_records(control_root / "qwen35_direct.jsonl")
    rows: list[dict[str, Any]] = []
    for task in tasks:
        key = (task.source_id, task.task_id)
        baseline_record = baseline.get(key)
        strong_record = strong.get(key)
        if not direct._ok(baseline_record) or not direct._ok(strong_record):
            raise ValueError(f"missing successful row: {key}")
        primary = primary_rows[key]
        baseline_quality = shared._score(
            task, str(direct._parsed(baseline_record).get("answer", ""))
        )
        if abs(baseline_quality - float(primary["baseline_quality"])) > 1e-12:
            raise ValueError(f"baseline score drift: {key}")
        strong_quality = shared._score(
            task, str(direct._parsed(strong_record).get("answer", ""))
        )
        triggered = _as_bool(primary["repairability_triggered"])
        core_changed = _as_bool(primary["core_changed"])
        egor_quality = float(primary["v8_quality"])
        values = {
            "qwen35_direct": strong_quality,
            "gate_then_qwen35": strong_quality if triggered else baseline_quality,
            "core_change_then_qwen35": strong_quality if core_changed else baseline_quality,
            "egor_v8": egor_quality,
        }
        rows.append(
            {
                "key": f"{task.source_id}::{task.task_id}",
                "baseline_quality": baseline_quality,
                "direct_delta": strong_quality - baseline_quality,
                "egor_accept": _as_bool(primary["accept_overwrite"]),
                "policy_quality": values,
            }
        )
    return rows


def _summarize(
    rows: list[dict[str, Any]], policy: str, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    deltas = [r["policy_quality"][policy] - r["baseline_quality"] for r in rows]
    low, high = shared._bootstrap_interval(deltas, samples=bootstrap_samples, seed=seed)
    harms = sum(d < -0.05 for d in deltas)
    nonzero = sum(d != 0.0 for d in deltas)
    return {
        "quality_delta": fmean(deltas),
        "ci95_low": low,
        "ci95_high": high,
        "wins": sum(d > 0.0 for d in deltas),
        "losses": sum(d < 0.0 for d in deltas),
        "ties": sum(d == 0.0 for d in deltas),
        "strict_harms": harms,
        "strict_harm_rate": harms / len(deltas),
        "nonzero_metric_changes": nonzero,
        "conditional_loss_rate_on_nonzero_metric_changes": (
            sum(d < 0.0 for d in deltas) / nonzero if nonzero else 0.0
        ),
        "conditional_strict_harm_rate_on_nonzero_metric_changes": (
            harms / nonzero if nonzero else 0.0
        ),
    }


def _contrast(
    rows: list[dict[str, Any]], left: str, right: str, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    values = [r["policy_quality"][left] - r["policy_quality"][right] for r in rows]
    low, high = shared._bootstrap_interval(values, samples=bootstrap_samples, seed=seed)
    return {
        "left": left,
        "right": right,
        "mean_difference": fmean(values),
        "ci95_low": low,
        "ci95_high": high,
        "left_better_rows": sum(v > 0.0 for v in values),
        "left_worse_rows": sum(v < 0.0 for v in values),
    }


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chartqa-tasks", required=True)
    parser.add_argument("--chartqa-v8-root", required=True)
    parser.add_argument("--chartqa-control-root", required=True)
    parser.add_argument("--infographic-tasks", required=True)
    parser.add_argument("--infographic-v8-root", required=True)
    parser.add_argument("--infographic-control-root", required=True)
    parser.add_argument("--primary-rows", required=True)
    parser.add_argument("--v8-source-commit", required=True)
    parser.add_argument("--control-source-commit", required=True)
    parser.add_argument("--tasks-per-source", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260791)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
