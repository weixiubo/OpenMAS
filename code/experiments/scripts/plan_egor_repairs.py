#!/usr/bin/env python3
"""Plan outcome-blind EGOR repair actions and materialize routing task lists."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from openagora_experiments.egor.planning import (
    build_repair_plans,
    capability_cards_from_payload,
)
from openagora_experiments.egor.runner import load_latest_stage_records


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    args = _parse_args()
    baseline_path = Path(args.baseline).resolve()
    capabilities_path = Path(args.capabilities).resolve()
    output_path = Path(args.output).resolve()
    routing_dir = Path(args.routing_dir).resolve()
    capabilities_payload = json.loads(capabilities_path.read_text(encoding="utf-8"))
    if not isinstance(capabilities_payload, dict):
        raise ValueError("capabilities file must contain a JSON object")
    cards = capability_cards_from_payload(capabilities_payload)
    plans = build_repair_plans(
        load_latest_stage_records(baseline_path),
        capabilities=cards,
        max_workers=args.max_workers,
        minimum_gap=args.minimum_gap,
        minimum_marginal_gain=args.minimum_marginal_gain,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in plans
        ),
        encoding="utf-8",
    )
    routing_dir.mkdir(parents=True, exist_ok=True)
    action_tasks: dict[str, list[str]] = defaultdict(list)
    combination_tasks: dict[str, list[str]] = defaultdict(list)
    seen_task_ids: set[str] = set()
    for row in plans:
        task_id = str(row["task_id"])
        if task_id in seen_task_ids:
            raise ValueError(f"routing task ids must be globally unique: {task_id}")
        seen_task_ids.add(task_id)
        selected = [str(value) for value in row["selected_action_ids"]]
        for action_id in selected:
            action_tasks[action_id].append(task_id)
        if selected:
            combination_tasks[str(row["combination_id"])].append(task_id)
    routing_files: list[Path] = []
    for action_id, task_ids in sorted(action_tasks.items()):
        path = routing_dir / f"action--{_safe(action_id)}.task_ids.txt"
        _write_task_ids(path, task_ids)
        routing_files.append(path)
    for combination, task_ids in sorted(combination_tasks.items()):
        path = routing_dir / f"combination--{_safe(combination)}.task_ids.txt"
        _write_task_ids(path, task_ids)
        routing_files.append(path)
    manifest = {
        "schema_version": 1,
        "source_commit": os.environ.get("EGOR_SOURCE_COMMIT", "").strip()
        or _git_output("rev-parse", "HEAD"),
        "baseline": str(baseline_path),
        "baseline_sha256": _sha256(baseline_path),
        "capabilities": str(capabilities_path),
        "capabilities_sha256": _sha256(capabilities_path),
        "max_workers": args.max_workers,
        "minimum_gap": args.minimum_gap,
        "minimum_marginal_gain": args.minimum_marginal_gain,
        "tasks": len(plans),
        "repair_rate": sum(bool(row["selected_action_ids"]) for row in plans)
        / len(plans),
        "action_counts": dict(
            Counter(action for row in plans for action in row["selected_action_ids"])
        ),
        "combination_counts": dict(
            Counter(str(row["combination_id"]) for row in plans)
        ),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "routing_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in routing_files
        ],
        "outcome_blind_policy": "plans use only parsed baseline state and frozen capability cards",
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"planned {len(plans)} tasks repair_rate={manifest['repair_rate']:.3f} "
        f"combinations={len(combination_tasks)}"
    )
    return 0


def _write_task_ids(path: Path, task_ids: list[str]) -> None:
    path.write_text("".join(f"{task_id}\n" for task_id in task_ids), encoding="utf-8")


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "baseline"


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
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--capabilities", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--routing-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--minimum-gap", type=float, default=0.05)
    parser.add_argument("--minimum-marginal-gain", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
