#!/usr/bin/env python3
"""Compute preregistered paired uncertainty estimates from EGOR policy rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openagora_experiments.egor.statistics import paired_bootstrap_contrast


TaskKey = tuple[str, str]


def main() -> int:
    args = _parse_args()
    policy_rows_path = Path(args.policy_rows).resolve()
    grouped = _load_policy_rows(policy_rows_path)
    primary = _quality_by_task(grouped, args.primary_policy, "policy_quality")
    ordered_keys = sorted(primary)
    contrasts: dict[str, dict[str, Any]] = {}
    for contrast_id in args.contrasts:
        if contrast_id == "baseline":
            comparison = _quality_by_task(
                grouped, args.primary_policy, "baseline_quality"
            )
        else:
            comparison = _quality_by_task(grouped, contrast_id, "policy_quality")
        if set(comparison) != set(primary):
            raise ValueError(
                f"task mismatch between {args.primary_policy} and {contrast_id}"
            )
        result = paired_bootstrap_contrast(
            [primary[key] for key in ordered_keys],
            [comparison[key] for key in ordered_keys],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        contrasts[contrast_id] = asdict(result)

    primary_rows = grouped[args.primary_policy]
    output = {
        "schema_version": 1,
        "policy_rows": str(policy_rows_path),
        "policy_rows_sha256": _sha256(policy_rows_path),
        "primary_policy": args.primary_policy,
        "tasks": len(ordered_keys),
        "triggered_tasks": sum(int(row["triggered"]) for row in primary_rows.values()),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "contrasts": contrasts,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(contrasts)} paired contrasts to {output_path}")
    return 0


def _load_policy_rows(path: Path) -> dict[str, dict[TaskKey, dict[str, str]]]:
    grouped: dict[str, dict[TaskKey, dict[str, str]]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            policy_id = str(row.get("policy_id", "")).strip()
            key = (str(row.get("source_id", "")), str(row.get("task_id", "")))
            if not policy_id or not all(key):
                raise ValueError("policy row lacks policy_id/source_id/task_id")
            if key in grouped[policy_id]:
                raise ValueError(f"duplicate task for {policy_id}: {key}")
            grouped[policy_id][key] = row
    if not grouped:
        raise ValueError("policy row file is empty")
    return dict(grouped)


def _quality_by_task(
    grouped: dict[str, dict[TaskKey, dict[str, str]]],
    policy_id: str,
    field: str,
) -> dict[TaskKey, float]:
    if policy_id not in grouped:
        raise ValueError(f"unknown policy: {policy_id}")
    return {key: float(row[field]) for key, row in grouped[policy_id].items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-rows", required=True)
    parser.add_argument("--primary-policy", default="egor_b2_gap")
    parser.add_argument(
        "--contrasts",
        nargs="+",
        default=[
            "baseline",
            "baseline_resynthesis_gap",
            "verifier_repair_gap",
            "full_team_gap",
        ],
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_261_010)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
