#!/usr/bin/env python3
"""Score the post-hoc Qwen3.5 direct-solver control on sealed v8 tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

import analyze_egor_v7_confirmation as shared
from openagora_experiments.egor.runner import load_latest_stage_records, load_tasks


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
    primary_rows = _load_primary_rows(Path(args.primary_rows).resolve())
    rows_by_source = {
        name: analyze_source(
            tasks_path=tasks,
            v8_root=v8_root,
            control_root=control_root,
            primary_rows=primary_rows,
            expected_control_commit=args.control_source_commit,
            expected_tasks=args.tasks_per_source,
        )
        for name, (tasks, v8_root, control_root) in sources.items()
    }
    rows = [
        row
        for name in ("chartqa", "infographic")
        for row in rows_by_source[name]
    ]
    report = {
        "schema_version": 1,
        "claim_boundary": "posthoc_matched_backbone_control_not_confirmation",
        "v8_source_commit": args.v8_source_commit,
        "control_source_commit": args.control_source_commit,
        "tasks": len(rows),
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "unit": "paired task with equal task weight",
        },
        "preflight": preflight,
        "pooled": summarize(
            rows, bootstrap_samples=args.bootstrap_samples, seed=args.seed
        ),
        "sources": {
            name: summarize(
                source_rows,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + index + 1,
            )
            for index, (name, source_rows) in enumerate(rows_by_source.items())
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def analyze_source(
    *,
    tasks_path: Path,
    v8_root: Path,
    control_root: Path,
    primary_rows: dict[tuple[str, str], dict[str, str]],
    expected_control_commit: str,
    expected_tasks: int,
) -> list[dict[str, Any]]:
    _validate_control_root(
        control_root,
        tasks_path=tasks_path,
        expected_commit=expected_control_commit,
    )
    tasks = load_tasks(tasks_path)
    if len(tasks) != expected_tasks:
        raise ValueError(f"unexpected task count: {len(tasks)}")
    baseline = load_latest_stage_records(v8_root / "baseline_qwen3.jsonl")
    direct = load_latest_stage_records(control_root / "qwen35_direct.jsonl")
    rows: list[dict[str, Any]] = []
    for task in tasks:
        key = (task.source_id, task.task_id)
        baseline_record = baseline.get(key)
        direct_record = direct.get(key)
        if not _ok(baseline_record) or not _ok(direct_record):
            raise ValueError(f"missing successful control row: {key}")
        baseline_answer = str(_parsed(baseline_record).get("answer", ""))
        direct_answer = str(_parsed(direct_record).get("answer", ""))
        baseline_quality = shared._score(task, baseline_answer)
        direct_quality = shared._score(task, direct_answer)
        primary = primary_rows[key]
        sealed_baseline = float(primary["baseline_quality"])
        if abs(baseline_quality - sealed_baseline) > 1e-12:
            raise ValueError(f"baseline score drift: {key}")
        egor_quality = float(primary["v8_quality"])
        usage = direct_record.get("usage", {})
        rows.append(
            {
                "source_id": task.source_id,
                "task_id": task.task_id,
                "baseline_quality": baseline_quality,
                "direct_quality": direct_quality,
                "egor_quality": egor_quality,
                "direct_calls": float(usage.get("attempts", 0.0)),
                "direct_tokens": float(usage.get("total_tokens", 0.0)),
                "direct_latency_seconds": float(usage.get("latency_seconds", 0.0)),
            }
        )
    return rows


def summarize(
    rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    direct_delta = [
        row["direct_quality"] - row["baseline_quality"] for row in rows
    ]
    egor_minus_direct = [
        row["egor_quality"] - row["direct_quality"] for row in rows
    ]
    direct_low, direct_high = shared._bootstrap_interval(
        direct_delta, samples=bootstrap_samples, seed=seed
    )
    contrast_low, contrast_high = shared._bootstrap_interval(
        egor_minus_direct, samples=bootstrap_samples, seed=seed + 1000
    )
    return {
        "tasks": len(rows),
        "baseline_quality": fmean(row["baseline_quality"] for row in rows),
        "qwen35_direct_quality": fmean(row["direct_quality"] for row in rows),
        "qwen35_direct_delta": fmean(direct_delta),
        "qwen35_direct_ci95_low": direct_low,
        "qwen35_direct_ci95_high": direct_high,
        "qwen35_direct_wins": sum(value > 0.0 for value in direct_delta),
        "qwen35_direct_losses": sum(value < 0.0 for value in direct_delta),
        "qwen35_direct_ties": sum(value == 0.0 for value in direct_delta),
        "qwen35_direct_strict_harm_rate": fmean(
            value < -0.05 for value in direct_delta
        ),
        "egor_quality": fmean(row["egor_quality"] for row in rows),
        "egor_minus_qwen35_direct": fmean(egor_minus_direct),
        "egor_minus_qwen35_direct_ci95_low": contrast_low,
        "egor_minus_qwen35_direct_ci95_high": contrast_high,
        "mean_direct_calls": fmean(row["direct_calls"] for row in rows),
        "mean_direct_tokens": fmean(row["direct_tokens"] for row in rows),
        "mean_direct_latency_seconds": fmean(
            row["direct_latency_seconds"] for row in rows
        ),
    }


def _validate_control_root(
    root: Path, *, tasks_path: Path, expected_commit: str
) -> None:
    checks = {
        "complete": (root / "CONTROL_COMPLETE").is_file(),
        "checksums": shared._verify_sha256sums(root / "SHA256SUMS", root),
        "source_commit": (root / "SOURCE_COMMIT").read_text().strip()
        == expected_commit,
        "tasks_sha256": (root / "TASKS_SHA256").read_text().strip()
        == shared._sha256(tasks_path),
        "claim_boundary": (root / "CLAIM_BOUNDARY").read_text().strip()
        == "posthoc_matched_backbone_control_not_confirmation",
    }
    if not all(checks.values()):
        raise ValueError(f"invalid direct-control root {root}: {checks}")


def _load_primary_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        key = (row["source_id"], row["task_id"])
        if key in rows:
            raise ValueError(f"duplicate primary row: {key}")
        rows[key] = row
    return rows


def _parsed(record: Mapping[str, Any] | None) -> dict[str, Any]:
    value = record.get("parsed", {}) if record else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _ok(record: Mapping[str, Any] | None) -> bool:
    return bool(record and str(record.get("status", "")) == "ok")


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
    parser.add_argument("--seed", type=int, default=20260790)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
