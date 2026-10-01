#!/usr/bin/env python3
"""Post-hoc v8 development of specialist-supported blind-resolver policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

import analyze_egor_qwen35_direct_control as direct
import analyze_egor_v7_confirmation as shared
from openagora_experiments.egor.runner import load_latest_stage_records, load_tasks
from openagora_experiments.egor.verification import answers_agree, valid_evidence_cards


POLICIES = ("gate_direct", "one_specialist_support", "two_specialist_support")


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
        sources={name: (tasks, root) for name, (tasks, root, _) in sources.items()},
        expected_commit=args.v8_source_commit,
        expected_tasks_per_source=args.tasks_per_source,
    )
    if not preflight["passed"]:
        raise ValueError("sealed v8 input preflight failed")
    primary = direct._load_primary_rows(Path(args.primary_rows).resolve())
    rows = [
        row
        for source in ("chartqa", "infographic")
        for row in _load_source(
            tasks_path=sources[source][0],
            v8_root=sources[source][1],
            control_root=sources[source][2],
            primary=primary,
            expected_control_commit=args.control_source_commit,
            expected_tasks=args.tasks_per_source,
        )
    ]
    report = {
        "schema_version": 1,
        "claim_boundary": "posthoc_v10_policy_development_not_confirmation",
        "v8_source_commit": args.v8_source_commit,
        "control_source_commit": args.control_source_commit,
        "tasks": len(rows),
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed_root": args.seed,
            "unit": "paired task with equal task weight",
        },
        "preflight_passed": preflight["passed"],
        "policies": {
            policy: _summarize(rows, policy, args.bootstrap_samples, args.seed + i)
            for i, policy in enumerate(POLICIES)
        },
        "funnel": {
            "gate_triggered": sum(r["triggered"] for r in rows),
            "blind_resolver_changed": sum(r["resolver_changed"] for r in rows),
            "one_specialist_support": sum(r["one_support"] for r in rows),
            "two_specialist_support": sum(r["two_support"] for r in rows),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _load_source(
    *,
    tasks_path: Path,
    v8_root: Path,
    control_root: Path,
    primary: dict[tuple[str, str], dict[str, str]],
    expected_control_commit: str,
    expected_tasks: int,
) -> list[dict[str, Any]]:
    direct._validate_control_root(
        control_root, tasks_path=tasks_path, expected_commit=expected_control_commit
    )
    tasks = load_tasks(tasks_path)
    if len(tasks) != expected_tasks:
        raise ValueError(f"unexpected task count: {len(tasks)}")
    records = {
        "baseline": load_latest_stage_records(v8_root / "baseline_qwen3.jsonl"),
        "resolver": load_latest_stage_records(control_root / "qwen35_direct.jsonl"),
        "exact": load_latest_stage_records(v8_root / "specialist_exact_qwen25.jsonl"),
        "visual": load_latest_stage_records(v8_root / "specialist_visual_internvl3.jsonl"),
    }
    rows: list[dict[str, Any]] = []
    for task in tasks:
        key = (task.source_id, task.task_id)
        base_record = records["baseline"].get(key)
        resolver_record = records["resolver"].get(key)
        if not direct._ok(base_record) or not direct._ok(resolver_record):
            raise ValueError(f"missing successful baseline/resolver row: {key}")
        baseline_answer = str(direct._parsed(base_record).get("answer", ""))
        resolver = direct._parsed(resolver_record)
        resolver_answer = str(resolver.get("answer", ""))
        transport = resolver.get("transport", {})
        decisive = bool(
            isinstance(transport, Mapping)
            and transport.get("operational_complete")
            and transport.get("decisive")
            and not transport.get("sanitized_abstention")
        )
        options = [{"id": option_id, "text": text} for option_id, text in task.options]
        row = primary[key]
        triggered = _as_bool(row["repairability_triggered"])
        changed = bool(
            triggered
            and decisive
            and not answers_agree(
                baseline_answer, resolver_answer, metric=task.metric, options=options
            )
        )
        support: list[bool] = []
        for name in ("exact", "visual"):
            specialist = direct._parsed(records[name].get(key))
            support.append(
                bool(
                    changed
                    and valid_evidence_cards(specialist)
                    and answers_agree(
                        str(specialist.get("candidate_answer", "")),
                        resolver_answer,
                        metric=task.metric,
                        options=options,
                    )
                )
            )
        baseline_quality = shared._score(task, baseline_answer)
        if abs(baseline_quality - float(row["baseline_quality"])) > 1e-12:
            raise ValueError(f"baseline score drift: {key}")
        resolver_quality = shared._score(task, resolver_answer)
        one_support = changed and any(support)
        two_support = changed and all(support)
        rows.append(
            {
                "baseline_quality": baseline_quality,
                "triggered": triggered,
                "resolver_changed": changed,
                "one_support": one_support,
                "two_support": two_support,
                "policy_quality": {
                    "gate_direct": resolver_quality if triggered and decisive else baseline_quality,
                    "one_specialist_support": resolver_quality if one_support else baseline_quality,
                    "two_specialist_support": resolver_quality if two_support else baseline_quality,
                },
            }
        )
    return rows


def _summarize(
    rows: list[dict[str, Any]], policy: str, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    deltas = [r["policy_quality"][policy] - r["baseline_quality"] for r in rows]
    low, high = shared._bootstrap_interval(deltas, samples=bootstrap_samples, seed=seed)
    harms = sum(d < -0.05 for d in deltas)
    return {
        "quality_delta": fmean(deltas),
        "ci95_low": low,
        "ci95_high": high,
        "wins": sum(d > 0.0 for d in deltas),
        "losses": sum(d < 0.0 for d in deltas),
        "ties": sum(d == 0.0 for d in deltas),
        "strict_harms": harms,
        "strict_harm_rate": harms / len(deltas),
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
    parser.add_argument("--seed", type=int, default=20260792)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
