#!/usr/bin/env python3
"""Open and score both sealed v8 confirmation sources after integrity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import analyze_egor_v7_confirmation as shared
from openagora_experiments.egor.runner import load_tasks


POLICY_KEY = "v8"
VERIFIER_FILE = "verifier_v8_qwen35.jsonl"
METHOD_ID = "egor_v8_answer_only_verified_overwrite"


def analyze_confirmation(
    *,
    sources: dict[str, tuple[Path, Path]],
    expected_commit: str,
    tasks_per_source: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preflight = shared.preflight_confirmation(
        sources=sources,
        expected_commit=expected_commit,
        expected_tasks_per_source=tasks_per_source,
    )
    if not preflight["passed"]:
        raise ValueError("confirmation integrity preflight failed before scoring")

    rows_by_source = {
        name: shared.analyze_source(
            source=name,
            tasks=load_tasks(tasks_path),
            root=root,
            verifier_filename=VERIFIER_FILE,
            answer_only=True,
            policy_key=POLICY_KEY,
        )
        for name, (tasks_path, root) in sources.items()
    }
    rows = [
        row
        for name in ("chartqa", "infographic")
        for row in rows_by_source[name]
    ]
    expected_total = tasks_per_source * len(sources)
    if len(rows) != expected_total:
        raise ValueError(
            f"confirmation denominator is {len(rows)}, expected {expected_total}"
        )

    summaries = {
        name: shared.summarize_source(
            source_rows,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index + 1,
            policy_key=POLICY_KEY,
        )
        for index, (name, source_rows) in enumerate(rows_by_source.items())
    }
    pooled = shared.summarize_source(
        rows,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        policy_key=POLICY_KEY,
    )
    completion = {
        name: summary["verifier_operational_completion"]
        for name, summary in summaries.items()
    }
    source_deltas = {
        name: summary[POLICY_KEY]["quality_delta"]
        for name, summary in summaries.items()
    }
    source_harm = {
        name: summary[POLICY_KEY]["strict_harm_rate"]
        for name, summary in summaries.items()
    }
    accepted_by_source = {
        name: summary["accepted_overwrites"]
        for name, summary in summaries.items()
    }
    criteria = [
        shared._criterion(
            "pooled_positive_with_strict_ci",
            pooled[POLICY_KEY]["quality_delta"] > 0.0
            and pooled[POLICY_KEY]["ci95_low"] > 0.0,
            {
                "delta": pooled[POLICY_KEY]["quality_delta"],
                "ci95_low": pooled[POLICY_KEY]["ci95_low"],
            },
            "pooled delta > 0 and paired-bootstrap 95% lower bound > 0",
        ),
        shared._criterion(
            "nonnegative_each_source",
            min(source_deltas.values()) >= 0.0,
            source_deltas,
            "each source point delta >= 0",
        ),
        shared._criterion(
            "bounded_strict_harm",
            pooled[POLICY_KEY]["strict_harm_rate"] <= 0.01
            and max(source_harm.values()) <= 0.02,
            {
                "pooled": pooled[POLICY_KEY]["strict_harm_rate"],
                "sources": source_harm,
            },
            "pooled <= 0.01 and every source <= 0.02",
        ),
        shared._criterion(
            "verifier_operational_completion",
            min(completion.values()) >= 0.95,
            completion,
            "every source >= 0.95 on verifier-called rows",
        ),
        shared._criterion(
            "accepted_overwrite_breadth",
            sum(accepted_by_source.values()) >= 4
            and min(accepted_by_source.values()) >= 1,
            {
                "total": sum(accepted_by_source.values()),
                "sources": accepted_by_source,
            },
            "at least four total and at least one on each source",
        ),
        shared._criterion(
            "integrity_manifests",
            bool(preflight["passed"]),
            preflight["checks"],
            "all task, source, seed, model, prompt, code, and stage checks pass",
        ),
    ]
    report = {
        "schema_version": 1,
        "claim_boundary": "independent_sealed_confirmation",
        "method": METHOD_ID,
        "source_commit": expected_commit,
        "task_denominator": len(rows),
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "unit": "paired task with equal task weight",
        },
        "preflight": preflight,
        "pooled": pooled,
        "sources": summaries,
        "criteria": criteria,
        "passed": all(item["passed"] for item in criteria),
    }
    return report, rows


def main() -> int:
    args = _parse_args()
    sources = {
        "chartqa": (
            Path(args.chartqa_tasks).resolve(),
            Path(args.chartqa_root).resolve(),
        ),
        "infographic": (
            Path(args.infographic_tasks).resolve(),
            Path(args.infographic_root).resolve(),
        ),
    }
    report, rows = analyze_confirmation(
        sources=sources,
        expected_commit=args.source_commit,
        tasks_per_source=args.tasks_per_source,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    if args.rows_output:
        shared._write_rows(
            Path(args.rows_output).resolve(),
            rows,
            policy_key=POLICY_KEY,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chartqa-tasks", required=True)
    parser.add_argument("--chartqa-root", required=True)
    parser.add_argument("--infographic-tasks", required=True)
    parser.add_argument("--infographic-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--tasks-per-source", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260782)
    parser.add_argument("--rows-output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
