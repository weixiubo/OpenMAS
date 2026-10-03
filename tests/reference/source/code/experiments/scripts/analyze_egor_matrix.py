#!/usr/bin/env python3
"""Uniformly score a complete EGOR candidate matrix and audit its evidence use."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openagora.mechanisms.evidence_gap_repair import EvidenceType
from openagora_experiments.egor.evaluation import (
    CandidateRow,
    PolicyRow,
    derive_all_policy,
    derive_gap_policy,
    estimate_capability,
    evaluate_candidate_rows,
    summarize_candidate,
    summarize_policy,
)
from openagora_experiments.egor.runner import (
    load_latest_stage_records,
    load_tasks,
)


ROOT = Path(__file__).resolve().parents[3]
LEGACY_RUN_ALL_THRESHOLD = -1.0
LEGACY_RUN_ALL_POLICIES = {
    "baseline_resynthesis_all": "baseline_resynthesis",
    "full_team_all": "full_team",
}


def main() -> int:
    args = _parse_args()
    task_path = Path(args.tasks).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tasks = load_tasks(task_path, limit=args.limit)
    baseline_path = _resolve_path(config.get("baseline", ""), config_path.parent)
    baseline_records = load_latest_stage_records(baseline_path)
    candidates = config.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("config.candidates must be a non-empty list")

    all_rows: list[CandidateRow] = []
    rows_by_candidate: dict[str, list[CandidateRow]] = {}
    summaries: dict[str, dict[str, float]] = {}
    capability_cards: list[dict[str, Any]] = []
    input_paths = [task_path, config_path, baseline_path]
    seen_candidate_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("each candidate config must be an object")
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        if not candidate_id or candidate_id in seen_candidate_ids:
            raise ValueError(
                f"candidate_id must be non-empty and unique: {candidate_id}"
            )
        seen_candidate_ids.add(candidate_id)
        synthesis_path = _resolve_path(
            candidate.get("synthesis", ""), config_path.parent
        )
        raw_specialists = candidate.get("specialists", [])
        if not isinstance(raw_specialists, list):
            raise ValueError(f"candidate {candidate_id} specialists must be a list")
        specialist_paths = [
            _resolve_path(item.get("path", ""), config_path.parent)
            for item in raw_specialists
            if isinstance(item, dict)
        ]
        if len(specialist_paths) != len(raw_specialists):
            raise ValueError(f"candidate {candidate_id} has a non-object specialist")
        rows = evaluate_candidate_rows(
            tasks=tasks,
            baseline_records=baseline_records,
            specialist_record_sets=[
                load_latest_stage_records(path) for path in specialist_paths
            ],
            synthesis_records=load_latest_stage_records(synthesis_path),
            candidate_id=candidate_id,
        )
        all_rows.extend(rows)
        rows_by_candidate[candidate_id] = rows
        summaries[candidate_id] = summarize_candidate(rows)
        for item in raw_specialists:
            action_id = str(item.get("action_id", "")).strip()
            evidence_type = EvidenceType(str(item.get("evidence_type", "")).strip())
            estimate = estimate_capability(
                rows,
                action_id=action_id,
                evidence_type=evidence_type,
            )
            payload = asdict(estimate)
            payload["evidence_type"] = evidence_type.value
            payload["candidate_id"] = candidate_id
            capability_cards.append(payload)
        input_paths.extend([synthesis_path, *specialist_paths])

    matrix_path = output_dir / "matrix_rows.csv"
    summary_path = output_dir / "candidate_summary.json"
    capability_path = output_dir / "capability_cards.json"
    policy_rows_path = output_dir / "policy_rows.csv"
    policy_summary_path = output_dir / "policy_summary.json"
    _write_rows(matrix_path, all_rows)
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    capability_path.write_text(
        json.dumps(capability_cards, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    policies = config.get("policies", [])
    if not isinstance(policies, list):
        raise ValueError("config.policies must be a list")
    all_policy_rows: list[PolicyRow] = []
    policy_summaries: dict[str, dict[str, float]] = {}
    policy_selection: list[dict[str, Any]] = []
    strict_provenance_policy_ids: list[str] = []
    for policy in policies:
        if not isinstance(policy, dict):
            raise ValueError("each policy config must be an object")
        policy_id = str(policy.get("policy_id", "")).strip()
        candidate_id = str(policy.get("candidate_id", "")).strip()
        if policy_id in policy_summaries or candidate_id not in rows_by_candidate:
            raise ValueError(f"invalid or duplicate policy: {policy_id}")
        require_evidence_for_change = args.require_evidence_for_change or bool(
            policy.get("require_evidence_for_change", False)
        )
        if require_evidence_for_change:
            strict_provenance_policy_ids.append(policy_id)
        trigger_mode, threshold, legacy_sentinel = _resolve_policy_selection(
            policy,
            policy_id=policy_id,
            candidate_id=candidate_id,
        )
        if trigger_mode == "all":
            policy_rows = derive_all_policy(
                rows_by_candidate[candidate_id],
                policy_id=policy_id,
                require_evidence_for_change=require_evidence_for_change,
            )
        else:
            if threshold is None:
                raise ValueError(f"gap policy {policy_id} has no threshold")
            policy_rows = derive_gap_policy(
                rows_by_candidate[candidate_id],
                policy_id=policy_id,
                threshold=threshold,
                require_evidence_for_change=require_evidence_for_change,
            )
        policy_selection.append(
            {
                "policy_id": policy_id,
                "candidate_id": candidate_id,
                "trigger_mode": trigger_mode,
                "threshold": threshold,
                "translated_legacy_run_all_sentinel": legacy_sentinel,
            }
        )
        all_policy_rows.extend(policy_rows)
        policy_summaries[policy_id] = summarize_policy(policy_rows)
    _write_policy_rows(policy_rows_path, all_policy_rows)
    policy_summary_path.write_text(
        json.dumps(policy_summaries, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    source_commit = os.environ.get("EGOR_SOURCE_COMMIT", "").strip() or _git_output(
        "rev-parse", "HEAD"
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "tasks": len(tasks),
        "candidate_ids": sorted(seen_candidate_ids),
        "scoring_policy": (
            "one common programmatic scorer; failed final stages remain in "
            "denominator at quality zero"
        ),
        "evidence_policy": (
            "reject changed answers without cited, structurally valid "
            "specialist evidence"
            if strict_provenance_policy_ids
            else "audit changed answers for cited, structurally valid "
            "specialist evidence"
        ),
        "require_evidence_for_change": args.require_evidence_for_change,
        "strict_provenance_policy_ids": sorted(strict_provenance_policy_ids),
        "policy_selection": policy_selection,
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in dict.fromkeys(path.resolve() for path in input_paths)
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (
                matrix_path,
                summary_path,
                capability_path,
                policy_rows_path,
                policy_summary_path,
            )
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"scored {len(all_rows)} candidate-task rows in {output_dir}")
    return 0


def _resolve_policy_selection(
    policy: dict[str, Any],
    *,
    policy_id: str,
    candidate_id: str,
) -> tuple[str, float | None, bool]:
    """Resolve an explicit trigger mode and narrowly translate the frozen sentinel."""

    raw_mode = policy.get("trigger_mode")
    mode = str(raw_mode).strip().lower() if raw_mode is not None else ""
    if mode == "all":
        if "threshold" in policy:
            raise ValueError(
                f"policy {policy_id} trigger_mode=all must omit threshold"
            )
        return "all", None, False
    if mode not in {"", "gap"}:
        raise ValueError(f"policy {policy_id} has unknown trigger_mode: {mode}")

    threshold = float(policy.get("threshold", 0.05))
    if mode == "":
        expected_candidate = LEGACY_RUN_ALL_POLICIES.get(policy_id)
        if threshold == LEGACY_RUN_ALL_THRESHOLD:
            if candidate_id != expected_candidate:
                raise ValueError(
                    "legacy run-all sentinel is restricted to the frozen "
                    f"control pair: {policy_id}/{candidate_id}"
                )
            return "all", None, True
    return "gap", threshold, False


def _write_rows(path: Path, rows: list[CandidateRow]) -> None:
    fieldnames = [
        "source_id",
        "task_id",
        "candidate_id",
        "baseline_answer",
        "candidate_answer",
        "baseline_quality",
        "candidate_quality",
        "quality_delta",
        "residual_gaps_json",
        "valid_specialist_evidence_ids",
        "used_valid_specialist_evidence_ids",
        "unknown_used_evidence_ids",
        "actual_changed",
        "reported_changed",
        "change_flag_agrees",
        "evidence_supported_change",
        "calls",
        "total_tokens",
        "latency_seconds",
        "failed_stages",
        "baseline_calls",
        "baseline_total_tokens",
        "baseline_latency_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_id": row.source_id,
                    "task_id": row.task_id,
                    "candidate_id": row.candidate_id,
                    "baseline_answer": row.baseline_answer,
                    "candidate_answer": row.candidate_answer,
                    "baseline_quality": row.baseline_quality,
                    "candidate_quality": row.candidate_quality,
                    "quality_delta": row.quality_delta,
                    "residual_gaps_json": json.dumps(
                        {key.value: value for key, value in row.residual_gaps.items()},
                        sort_keys=True,
                    ),
                    "valid_specialist_evidence_ids": "|".join(
                        card.evidence_id for card in row.specialist_cards
                    ),
                    "used_valid_specialist_evidence_ids": "|".join(
                        row.synthesis_audit.used_valid_specialist_ids
                    ),
                    "unknown_used_evidence_ids": "|".join(
                        row.synthesis_audit.unknown_used_ids
                    ),
                    "actual_changed": int(row.synthesis_audit.actual_changed),
                    "reported_changed": int(row.synthesis_audit.reported_changed),
                    "change_flag_agrees": int(row.synthesis_audit.change_flag_agrees),
                    "evidence_supported_change": int(
                        row.synthesis_audit.evidence_supported_change
                    ),
                    "calls": row.calls,
                    "total_tokens": row.total_tokens,
                    "latency_seconds": row.latency_seconds,
                    "failed_stages": "|".join(row.failed_stages),
                    "baseline_calls": row.baseline_calls,
                    "baseline_total_tokens": row.baseline_total_tokens,
                    "baseline_latency_seconds": row.baseline_latency_seconds,
                }
            )


def _write_policy_rows(path: Path, rows: list[PolicyRow]) -> None:
    fieldnames = [
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["triggered"] = int(row.triggered)
            payload["strict_harm"] = int(row.strict_harm)
            payload["useful_repair"] = int(row.useful_repair)
            payload["provenance_rejected"] = int(row.provenance_rejected)
            payload["failed_stages"] = "|".join(row.failed_stages)
            writer.writerow(payload)


def _resolve_path(value: Any, base: Path) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError("config path must be non-empty")
    path = Path(text).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


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
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--require-evidence-for-change",
        action="store_true",
        help=(
            "Reject a changed synthesis answer unless it cites valid specialist "
            "evidence."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
