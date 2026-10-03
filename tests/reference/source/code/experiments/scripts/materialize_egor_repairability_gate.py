#!/usr/bin/env python3
"""Materialize frozen out-of-fold repairability decisions without outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CV_ALLOWED_INPUT_FIELDS = (
    "source_id",
    "task_id",
    "selected_id",
    "reason",
    "selection_protocol",
    "selector_target",
    "fold_config_json",
)
SELECTOR_ALLOWED_INPUT_FIELDS = (
    "source_id",
    "task_id",
    "selected_template_id",
    "reason",
)


def main() -> int:
    args = _parse_args()
    input_path = Path(args.cross_validated_rows or args.selector_plans).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.cross_validated_rows:
        plans = materialize_repairability_gate(input_path)
        input_kind = "nested_cross_validation_rows"
        allowed_fields = CV_ALLOWED_INPUT_FIELDS
    else:
        plans = materialize_selector_repairability_gate(input_path)
        input_kind = "frozen_selector_plans"
        allowed_fields = SELECTOR_ALLOWED_INPUT_FIELDS
    if args.source_id:
        plans = [row for row in plans if row["source_id"] == args.source_id]
        if not plans:
            raise ValueError(f"source_id not present: {args.source_id}")
    if args.output_source_id:
        if not args.source_id:
            raise ValueError("--output-source-id requires --source-id")
        plans = [
            {**row, "source_id": args.output_source_id}
            for row in plans
        ]
    if args.expected_tasks and len(plans) != args.expected_tasks:
        raise ValueError(
            f"repairability gate has {len(plans)} tasks, "
            f"expected {args.expected_tasks}"
        )
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in plans
        ),
        encoding="utf-8",
    )
    task_ids_path = (
        Path(args.task_ids_output).resolve() if args.task_ids_output else None
    )
    if task_ids_path is not None:
        task_ids_path.parent.mkdir(parents=True, exist_ok=True)
        task_ids_path.write_text(
            "".join(
                f"{row['task_id']}\n"
                for row in plans
                if row["repairability_triggered"]
            ),
            encoding="utf-8",
        )
    policy = (
        "mechanical projection of nested out-of-fold selector decisions to a "
        "binary repairability gate"
        if input_kind == "nested_cross_validation_rows"
        else "mechanical projection of a frozen development-trained selector "
        "decision to a binary repairability gate"
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("EGOR_SOURCE_COMMIT", "").strip()
        or _git_output("rev-parse", "HEAD"),
        "policy": policy,
        "input_kind": input_kind,
        "allowed_input_fields_read": list(allowed_fields),
        "outcome_fields_read": False,
        "source_filter": args.source_id,
        "output_source_id": args.output_source_id,
        "tasks": len(plans),
        "source_counts": dict(
            sorted(Counter(row["source_id"] for row in plans).items())
        ),
        "trigger_counts": dict(
            sorted(
                Counter(
                    "trigger" if row["repairability_triggered"] else "abstain"
                    for row in plans
                ).items()
            )
        ),
        "input": {"path": str(input_path), "sha256": _sha256(input_path)},
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "task_ids_output": (
            {"path": str(task_ids_path), "sha256": _sha256(task_ids_path)}
            if task_ids_path is not None
            else None
        ),
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"materialized {len(plans)} repairability decisions: "
        f"{json.dumps(manifest['trigger_counts'], sort_keys=True)}"
    )
    return 0


def materialize_repairability_gate(path: Path) -> list[dict[str, Any]]:
    """Materialize cross-validated ``eq:repairability-gate`` decisions.

    The output is the frozen binary plan described in
    ``sec:repairability-gate``; repair action identity is not serialized.
    """
    plans: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(CV_ALLOWED_INPUT_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"cross-validated rows lack fields: {sorted(missing)}")
        for raw in reader:
            source_id = str(raw["source_id"]).strip()
            task_id = str(raw["task_id"]).strip()
            selected_id = str(raw["selected_id"]).strip()
            key = (source_id, task_id)
            if not all(key) or not selected_id or key in seen:
                raise ValueError(f"invalid or duplicate repairability row: {key}")
            seen.add(key)
            plans.append(
                {
                    "schema_version": 1,
                    "source_id": source_id,
                    "task_id": task_id,
                    "repairability_triggered": selected_id != "baseline",
                    "gate_reason": str(raw["reason"]).strip(),
                    "selection_protocol": str(raw["selection_protocol"]).strip(),
                    "selector_target": str(raw["selector_target"]).strip(),
                    "fold_config_json": str(raw["fold_config_json"]).strip(),
                }
            )
    if not plans:
        raise ValueError("cross-validated rows are empty")
    return sorted(plans, key=lambda row: (row["source_id"], row["task_id"]))


def materialize_selector_repairability_gate(path: Path) -> list[dict[str, Any]]:
    """Compress selector plans into ``eq:repairability-gate`` decisions.

    As required by ``sec:repairability-gate``, this adapter keeps only whether
    the latent selected action is baseline or nonbaseline.
    """
    plans: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("selector plan rows must be objects")
            missing = set(SELECTOR_ALLOWED_INPUT_FIELDS) - set(raw)
            if missing:
                raise ValueError(f"selector plan row lacks fields: {sorted(missing)}")
            source_id = str(raw["source_id"]).strip()
            task_id = str(raw["task_id"]).strip()
            selected_id = str(raw["selected_template_id"]).strip()
            key = (source_id, task_id)
            if not all(key) or not selected_id or key in seen:
                raise ValueError(f"invalid or duplicate selector plan: {key}")
            seen.add(key)
            plans.append(
                {
                    "schema_version": 1,
                    "source_id": source_id,
                    "task_id": task_id,
                    "repairability_triggered": selected_id != "baseline",
                    "gate_reason": str(raw["reason"]).strip(),
                    "selection_protocol": "frozen_selector_inference",
                    "selector_target": "paired_advantage",
                    "fold_config_json": "",
                }
            )
    if not plans:
        raise ValueError("selector plans are empty")
    return sorted(plans, key=lambda row: (row["source_id"], row["task_id"]))


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
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--cross-validated-rows")
    inputs.add_argument("--selector-plans")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-ids-output", default="")
    parser.add_argument("--source-id", default="")
    parser.add_argument("--output-source-id", default="")
    parser.add_argument("--expected-tasks", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
