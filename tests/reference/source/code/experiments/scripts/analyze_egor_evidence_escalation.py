#!/usr/bin/env python3
"""Score a frozen staged EGOR escalation plan against a sparse matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from openagora_experiments.egor.evaluation import PolicyRow, summarize_policy
from openagora_experiments.egor.runner import load_latest_stage_records


TaskKey = tuple[str, str]


def main() -> int:
    args = _parse_args()
    plan_path = Path(args.plans).resolve()
    matrix_path = Path(args.matrix_rows).resolve()
    core_synthesis_path = Path(args.core_synthesis).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plans = _load_plans(plan_path)
    matrix = _load_matrix(matrix_path)
    core_records = load_latest_stage_records(core_synthesis_path)
    rows, audit_rows = analyze_escalation_policy(
        plans=plans,
        matrix=matrix,
        core_synthesis_records=core_records,
        policy_id=args.policy_id,
        core_candidate_id=args.core_candidate_id,
        full_candidate_id=args.full_candidate_id,
        require_evidence_for_change=not args.allow_uncertified_changes,
    )
    policy_path = output_dir / "policy_rows.csv"
    audit_path = output_dir / "execution_audit_rows.csv"
    summary_path = output_dir / "policy_summary.json"
    _write_policy_rows(policy_path, rows)
    _write_audit_rows(audit_path, audit_rows)
    summary = summarize_policy(rows)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "source_commit": os.environ.get("EGOR_SOURCE_COMMIT", "").strip(),
        "policy_id": args.policy_id,
        "core_candidate_id": args.core_candidate_id,
        "full_candidate_id": args.full_candidate_id,
        "require_evidence_for_change": not args.allow_uncertified_changes,
        "tasks": len(rows),
        "tier_counts": dict(
            sorted(Counter(row["tier"] for row in audit_rows).items())
        ),
        "sequential_accounting": (
            "full-team rows add the already executed core-synthesis call; "
            "baseline and core specialist calls are not double counted"
        ),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (plan_path, matrix_path, core_synthesis_path)
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (policy_path, audit_path, summary_path)
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"scored staged policy over {len(rows)} tasks: "
        f"{json.dumps(manifest['tier_counts'], sort_keys=True)}"
    )
    return 0


def analyze_escalation_policy(
    *,
    plans: dict[TaskKey, dict[str, Any]],
    matrix: dict[tuple[str, str, str], dict[str, str]],
    core_synthesis_records: dict[TaskKey, dict[str, Any]],
    policy_id: str,
    core_candidate_id: str,
    full_candidate_id: str,
    require_evidence_for_change: bool = True,
) -> tuple[list[PolicyRow], list[dict[str, Any]]]:
    """Score ``eq:staged-policy`` with ``eq:certificate`` accounting.

    Sequential core-call accounting follows ``sec:execution``; policy outcomes
    are read only here, after the outcome-blind breadth plan is frozen.
    """
    if not policy_id.strip():
        raise ValueError("policy_id is required")
    rows: list[PolicyRow] = []
    audit_rows: list[dict[str, Any]] = []
    for key, plan in sorted(plans.items()):
        tier = str(plan.get("tier", ""))
        if tier not in {"baseline", "core_pair", "full_team"}:
            raise ValueError(f"invalid escalation tier for {key}: {tier}")
        core_row = matrix.get((*key, core_candidate_id))
        if core_row is None:
            raise ValueError(f"matrix lacks core candidate for {key}")
        if tier == "full_team":
            full_row = matrix.get((*key, full_candidate_id))
            if full_row is None:
                raise ValueError(f"matrix lacks escalated candidate for {key}")
            _validate_shared_baseline(core_row, full_row, key=key)
            chosen = full_row
        else:
            chosen = core_row
        baseline_quality = float(chosen["baseline_quality"])
        provenance_rejected = (
            require_evidence_for_change
            and tier != "baseline"
            and _provenance_rejected(chosen)
        )
        if tier == "baseline" or provenance_rejected:
            policy_quality = baseline_quality
        else:
            policy_quality = float(chosen["candidate_quality"])
        delta = policy_quality - baseline_quality
        if tier == "baseline":
            calls = float(chosen["baseline_calls"])
            tokens = float(chosen["baseline_total_tokens"])
            latency = float(chosen["baseline_latency_seconds"])
            failures = tuple(
                value
                for value in _failures(chosen)
                if value == "baseline"
            )
            candidate_id = "baseline"
            extra_usage = {"attempts": 0.0, "total_tokens": 0.0, "latency_seconds": 0.0}
        else:
            calls = float(chosen["calls"])
            tokens = float(chosen["total_tokens"])
            latency = float(chosen["latency_seconds"])
            failures = _failures(chosen)
            candidate_id = (
                core_candidate_id if tier == "core_pair" else full_candidate_id
            )
            extra_usage = {
                "attempts": 0.0,
                "total_tokens": 0.0,
                "latency_seconds": 0.0,
            }
            if tier == "full_team":
                core_record = core_synthesis_records.get(key)
                if core_record is None:
                    raise ValueError(f"missing core synthesis usage for {key}")
                extra_usage = _usage(core_record)
                calls += extra_usage["attempts"]
                tokens += extra_usage["total_tokens"]
                latency += extra_usage["latency_seconds"]
                if str(core_record.get("status", "")) != "ok":
                    failures = (*failures, "core_synthesis")
        rows.append(
            PolicyRow(
                source_id=key[0],
                task_id=key[1],
                policy_id=policy_id,
                candidate_id=candidate_id,
                triggered=tier != "baseline",
                baseline_quality=baseline_quality,
                policy_quality=policy_quality,
                quality_delta=delta,
                calls=calls,
                total_tokens=tokens,
                latency_seconds=latency,
                strict_harm=delta < -0.05,
                useful_repair=delta > 0.05,
                failed_stages=failures,
                provenance_rejected=provenance_rejected,
            )
        )
        audit_rows.append(
            {
                "source_id": key[0],
                "task_id": key[1],
                "tier": tier,
                "reason": str(plan.get("reason", "")),
                "candidate_id": candidate_id,
                "provenance_rejected": int(provenance_rejected),
                "extra_core_synthesis_calls": extra_usage["attempts"],
                "extra_core_synthesis_tokens": extra_usage["total_tokens"],
                "extra_core_synthesis_latency_seconds": extra_usage[
                    "latency_seconds"
                ],
            }
        )
    return rows, audit_rows


def _load_plans(path: Path) -> dict[TaskKey, dict[str, Any]]:
    rows: dict[TaskKey, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("source_id", "")), str(row.get("task_id", "")))
            if not all(key) or key in rows:
                raise ValueError(f"invalid or duplicate escalation plan: {key}")
            rows[key] = row
    if not rows:
        raise ValueError("escalation plans are empty")
    return rows


def _load_matrix(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["source_id"], row["task_id"], row["candidate_id"])
            if key in rows:
                raise ValueError(f"duplicate matrix row: {key}")
            rows[key] = row
    if not rows:
        raise ValueError("matrix is empty")
    return rows


def _validate_shared_baseline(
    core: dict[str, str],
    full: dict[str, str],
    *,
    key: TaskKey,
) -> None:
    fields = (
        "baseline_answer",
        "baseline_quality",
        "baseline_calls",
        "baseline_total_tokens",
        "baseline_latency_seconds",
    )
    if any(core.get(field) != full.get(field) for field in fields):
        raise ValueError(f"staged candidates disagree on baseline for {key}")


def _provenance_rejected(row: dict[str, str]) -> bool:
    return _bool(row.get("actual_changed")) and not _bool(
        row.get("evidence_supported_change")
    )


def _failures(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(value for value in row.get("failed_stages", "").split("|") if value)


def _usage(record: dict[str, Any]) -> dict[str, float]:
    usage = record.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    return {
        key: _float(usage.get(key, 0.0))
        for key in ("attempts", "total_tokens", "latency_seconds")
    }


def _write_policy_rows(path: Path, rows: list[PolicyRow]) -> None:
    fields = [
        "source_id",
        "task_id",
        "policy_id",
        "candidate_id",
        "triggered",
        "baseline_quality",
        "policy_quality",
        "quality_delta",
        "calls",
        "total_tokens",
        "latency_seconds",
        "strict_harm",
        "useful_repair",
        "provenance_rejected",
        "failed_stages",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_id": row.source_id,
                    "task_id": row.task_id,
                    "policy_id": row.policy_id,
                    "candidate_id": row.candidate_id,
                    "triggered": int(row.triggered),
                    "baseline_quality": row.baseline_quality,
                    "policy_quality": row.policy_quality,
                    "quality_delta": row.quality_delta,
                    "calls": row.calls,
                    "total_tokens": row.total_tokens,
                    "latency_seconds": row.latency_seconds,
                    "strict_harm": int(row.strict_harm),
                    "useful_repair": int(row.useful_repair),
                    "provenance_rejected": int(row.provenance_rejected),
                    "failed_stages": "|".join(row.failed_stages),
                }
            )


def _write_audit_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("execution audit rows are empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--matrix-rows", required=True)
    parser.add_argument("--core-synthesis", required=True)
    parser.add_argument("--policy-id", default="evidence_escalation")
    parser.add_argument("--core-candidate-id", required=True)
    parser.add_argument("--full-candidate-id", required=True)
    parser.add_argument("--allow-uncertified-changes", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
