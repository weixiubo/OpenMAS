#!/usr/bin/env python3
"""Convert a frozen EGOR selector or LLM scheduler into executable task lists."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openagora_experiments.egor.execution import (
    build_scheduler_repair_plans,
    build_selector_repair_plans,
    task_lists_from_plans,
)
from openagora_experiments.egor.planning import combination_id
from openagora_experiments.egor.runner import load_latest_stage_records, load_tasks
from openagora_experiments.egor.selection import CounterfactualRepairSelector


ROOT = Path(__file__).resolve().parents[3]
_SAFE_ID = re.compile(r"^[a-zA-Z0-9_.-]+$")


def main() -> int:
    args = _parse_args()
    tasks_path = Path(args.tasks).resolve()
    baseline_path = Path(args.baseline).resolve()
    selector_path = Path(args.selector).resolve() if args.selector else None
    scheduler_path = (
        Path(args.scheduler_records).resolve() if args.scheduler_records else None
    )
    action_spec_path = Path(args.action_spec).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    action_spec = json.loads(action_spec_path.read_text(encoding="utf-8"))
    if not isinstance(action_spec, dict):
        raise ValueError("action spec must be a JSON object")
    template_actions, action_ids = _execution_mapping(action_spec)
    tasks = load_tasks(tasks_path)
    baseline_records = load_latest_stage_records(baseline_path)
    if selector_path is not None:
        selector_payload = json.loads(selector_path.read_text(encoding="utf-8"))
        if not isinstance(selector_payload, dict):
            raise ValueError("selector must be a JSON object")
        selector = CounterfactualRepairSelector.from_dict(selector_payload)
        plans = build_selector_repair_plans(
            tasks,
            baseline_records=baseline_records,
            selector=selector,
            template_actions=template_actions,
        )
        policy = "source-blind frozen counterfactual selector"
        decision_path = selector_path
    else:
        assert scheduler_path is not None
        plans = build_scheduler_repair_plans(
            tasks,
            baseline_records=baseline_records,
            scheduler_records=load_latest_stage_records(scheduler_path),
            template_actions=template_actions,
        )
        policy = "outcome-blind strong-LLM runtime scheduler baseline"
        decision_path = scheduler_path
    combinations = sorted(
        {combination_id(actions) for actions in template_actions.values()}
    )
    action_lists, combination_lists = task_lists_from_plans(
        plans,
        action_ids=action_ids,
        combination_ids=combinations,
    )

    plans_path = output_dir / "plans.jsonl"
    plans_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in plans
        ),
        encoding="utf-8",
    )
    list_paths: list[Path] = []
    for namespace, values in (
        ("actions", action_lists),
        ("combinations", combination_lists),
    ):
        directory = output_dir / f"{namespace}_task_ids"
        directory.mkdir(parents=True, exist_ok=True)
        for item_id, task_ids in values.items():
            if not _SAFE_ID.fullmatch(item_id):
                raise ValueError(f"unsafe execution id: {item_id}")
            path = directory / f"{item_id}.txt"
            path.write_text(
                "".join(f"{task_id}\n" for task_id in task_ids), encoding="utf-8"
            )
            list_paths.append(path)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("EGOR_SOURCE_COMMIT", "").strip()
        or _git_output("rev-parse", "HEAD"),
        "policy": policy,
        "tasks": len(plans),
        "selection_counts": dict(
            sorted(Counter(str(row["selected_template_id"]) for row in plans).items())
        ),
        "combination_counts": dict(
            sorted(Counter(str(row["combination_id"]) for row in plans).items())
        ),
        "source_id_feature_policy": (
            "source_id is written only as an audit key after feature extraction "
            "and selection; it is absent from the serialized feature schema"
        ),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (tasks_path, baseline_path, decision_path, action_spec_path)
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (plans_path, *list_paths)
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"planned tasks={len(plans)} selection_counts="
        f"{json.dumps(manifest['selection_counts'], sort_keys=True)}"
    )
    return 0


def _execution_mapping(
    action_spec: dict[str, Any],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    raw_actions = action_spec.get("actions", {})
    raw_templates = action_spec.get("templates", {})
    if not isinstance(raw_actions, dict) or not isinstance(raw_templates, dict):
        raise ValueError("action spec requires actions and templates objects")
    action_ids = tuple(sorted(str(value) for value in raw_actions))
    templates: dict[str, tuple[str, ...]] = {}
    for template_id, raw in raw_templates.items():
        if not isinstance(raw, dict) or not isinstance(raw.get("action_ids"), list):
            raise ValueError(f"template {template_id} requires action_ids")
        actions = tuple(str(value) for value in raw["action_ids"])
        if any(action_id not in raw_actions for action_id in actions):
            raise ValueError(f"template {template_id} references an unknown action")
        templates[str(template_id)] = actions
    return templates, action_ids


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
    decisions = parser.add_mutually_exclusive_group(required=True)
    decisions.add_argument("--selector", default="")
    decisions.add_argument("--scheduler-records", default="")
    parser.add_argument("--action-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
