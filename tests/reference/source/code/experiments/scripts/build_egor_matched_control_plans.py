#!/usr/bin/env python3
"""Freeze outcome-blind controls matched to an EGOR selector plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    args = _parse_args()
    input_path = Path(args.selector_plans).resolve()
    output_path = Path(args.output).resolve()
    plans = _load_jsonl(input_path)
    controls = build_matched_control_plans(plans, seed=args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in controls),
        encoding="utf-8",
    )
    selected_counts = Counter(
        str(row["selector_template_id"])
        for row in controls
        if row["selector_template_id"] != "baseline"
    )
    random_counts = Counter(
        str(row["matched_random_template_id"])
        for row in controls
        if row["matched_random_template_id"] != "baseline"
    )
    if selected_counts != random_counts:
        raise AssertionError("matched random plan changed the template multiset")
    manifest = {
        "schema_version": 1,
        "source_commit": os.environ.get("EGOR_SOURCE_COMMIT", "").strip(),
        "outcome_blind": True,
        "seed": args.seed,
        "tasks": len(controls),
        "triggered_tasks": sum(
            row["selector_template_id"] != "baseline" for row in controls
        ),
        "template_counts": dict(sorted(selected_counts.items())),
        "input": {"path": str(input_path), "sha256": _sha256(input_path)},
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "matching_rule": (
            "preserve the selector's trigger set and exact template multiset, "
            "then seed-shuffle templates across triggered tasks without outcomes"
        ),
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"froze matched controls for {len(controls)} tasks; "
        f"triggered={manifest['triggered_tasks']}"
    )
    return 0


def build_matched_control_plans(
    selector_plans: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    triggered_indices: list[int] = []
    templates: list[str] = []
    seen: set[tuple[str, str]] = set()
    for index, plan in enumerate(selector_plans):
        key = (str(plan.get("source_id", "")), str(plan.get("task_id", "")))
        if not all(key) or key in seen:
            raise ValueError(f"invalid or duplicate selector plan key: {key}")
        seen.add(key)
        template = str(plan.get("selected_template_id", "baseline"))
        if template != "baseline":
            triggered_indices.append(index)
            templates.append(template)
    random.Random(seed).shuffle(templates)
    assignments = dict(zip(triggered_indices, templates, strict=True))
    controls: list[dict[str, Any]] = []
    for index, plan in enumerate(selector_plans):
        selector_template = str(plan.get("selected_template_id", "baseline"))
        controls.append(
            {
                "schema_version": 1,
                "source_id": str(plan["source_id"]),
                "task_id": str(plan["task_id"]),
                "selector_template_id": selector_template,
                "matched_random_template_id": assignments.get(index, "baseline"),
                "selector_triggered": selector_template != "baseline",
                "resynthesis_candidate_id": (
                    "baseline_resynthesis"
                    if selector_template != "baseline"
                    else "baseline"
                ),
            }
        )
    return controls


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"plan row is not an object: {path}")
                rows.append(row)
    if not rows:
        raise ValueError(f"selector plan is empty: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-plans", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20_260_712)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
