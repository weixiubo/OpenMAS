#!/usr/bin/env python3
"""Plan staged EGOR breadth using only baseline and executed core records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openagora_experiments.egor.escalation import decide_evidence_escalation
from openagora_experiments.egor.runner import load_latest_stage_records


ROOT = Path(__file__).resolve().parents[3]
TaskKey = tuple[str, str]


def main() -> int:
    args = _parse_args()
    paths = {
        "repairability_plans": Path(args.repairability_plans).resolve(),
        "baseline": Path(args.baseline).resolve(),
        "exact_specialist": Path(args.exact_specialist).resolve(),
        "visual_specialist": Path(args.visual_specialist).resolve(),
        "core_synthesis": Path(args.core_synthesis).resolve(),
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repairability = _load_repairability_plans(paths["repairability_plans"])
    if args.expected_tasks and len(repairability) != args.expected_tasks:
        raise ValueError(
            f"repairability plan has {len(repairability)} tasks, "
            f"expected {args.expected_tasks}"
        )
    baseline = load_latest_stage_records(paths["baseline"])
    exact = load_latest_stage_records(paths["exact_specialist"])
    visual = load_latest_stage_records(paths["visual_specialist"])
    core = load_latest_stage_records(paths["core_synthesis"])

    rows: list[dict[str, Any]] = []
    for key, source_plan in sorted(repairability.items()):
        triggered = _repairability_triggered(source_plan)
        base_record = baseline.get(key)
        if base_record is None:
            raise ValueError(f"missing baseline record for {key}")
        stage_records = (exact.get(key), visual.get(key), core.get(key))
        core_complete = all(_ok(record) for record in stage_records)
        decision = decide_evidence_escalation(
            repairability_triggered=triggered,
            baseline=_parsed(base_record),
            exact_specialist=_parsed(stage_records[0]),
            visual_specialist=_parsed(stage_records[1]),
            core_synthesis=_parsed(stage_records[2]),
            core_complete=core_complete,
        )
        audit = decision.core_audit
        rows.append(
            {
                "schema_version": 1,
                "source_id": key[0],
                "task_id": key[1],
                "repairability_triggered": triggered,
                "repairability_gate_reason": str(
                    source_plan.get("gate_reason", "")
                ),
                "repairability_protocol": str(
                    source_plan.get("selection_protocol", "")
                ),
                "tier": decision.tier,
                "reason": decision.reason,
                "boolean_disagreement": decision.boolean_disagreement,
                "core_complete": core_complete,
                "core_actual_changed": (
                    audit.actual_changed if audit is not None else False
                ),
                "core_evidence_supported_change": (
                    audit.evidence_supported_change if audit is not None else False
                ),
                "core_unknown_used_ids": (
                    list(audit.unknown_used_ids) if audit is not None else []
                ),
            }
        )

    plan_path = output_dir / "plans.jsonl"
    plan_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    core_control_path = output_dir / "core_pair_control_plans.jsonl"
    core_control_path.write_text(
        "".join(
            json.dumps(
                {
                    **row,
                    "tier": (
                        "baseline" if row["tier"] == "baseline" else "core_pair"
                    ),
                    "reason": (
                        row["reason"]
                        if row["tier"] == "baseline"
                        else "repairability_trigger_core_control"
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    task_dir = output_dir / "task_ids"
    task_dir.mkdir(parents=True, exist_ok=True)
    core_path = task_dir / "core_pair.txt"
    full_path = task_dir / "full_team.txt"
    core_path.write_text(
        "".join(f"{row['task_id']}\n" for row in rows if row["tier"] != "baseline"),
        encoding="utf-8",
    )
    full_path.write_text(
        "".join(f"{row['task_id']}\n" for row in rows if row["tier"] == "full_team"),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("EGOR_SOURCE_COMMIT", "").strip()
        or _git_output("rev-parse", "HEAD"),
        "policy": (
            "frozen repairability gate; exact--visual core; deterministic "
            "evidence-conditioned full-team escalation"
        ),
        "outcome_fields_read": False,
        "forbidden_inputs": [
            "reference answers",
            "task scores",
            "numerical specialist records",
            "verifier specialist records",
            "full-team synthesis records",
        ],
        "tasks": len(rows),
        "tier_counts": dict(sorted(Counter(row["tier"] for row in rows).items())),
        "reason_counts": dict(
            sorted(Counter(row["reason"] for row in rows).items())
        ),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in paths.values()
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (plan_path, core_control_path, core_path, full_path)
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"planned staged escalation for {len(rows)} tasks: "
        f"{json.dumps(manifest['tier_counts'], sort_keys=True)}"
    )
    return 0


def _load_repairability_plans(path: Path) -> dict[TaskKey, dict[str, Any]]:
    plans: dict[TaskKey, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("repairability plan rows must be objects")
            key = (str(raw.get("source_id", "")), str(raw.get("task_id", "")))
            if not all(key) or key in plans:
                raise ValueError(f"invalid or duplicate repairability key: {key}")
            plans[key] = raw
    if not plans:
        raise ValueError("repairability plans are empty")
    return plans


def _parsed(record: Mapping[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {}
    parsed = record.get("parsed", {})
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _repairability_triggered(plan: Mapping[str, Any]) -> bool:
    if "repairability_triggered" in plan:
        value = plan["repairability_triggered"]
        if not isinstance(value, bool):
            raise ValueError("repairability_triggered must be Boolean")
        return value
    selected = str(plan.get("selected_template_id", "")).strip()
    if not selected:
        raise ValueError("repairability plan lacks a trigger decision")
    return selected != "baseline"


def _ok(record: Mapping[str, Any] | None) -> bool:
    return record is not None and str(record.get("status", "")) == "ok"


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
    parser.add_argument("--repairability-plans", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--exact-specialist", required=True)
    parser.add_argument("--visual-specialist", required=True)
    parser.add_argument("--core-synthesis", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-tasks", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
