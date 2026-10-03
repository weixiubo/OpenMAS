#!/usr/bin/env python3
"""Score frozen matched-random and selector-triggered resynthesis controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openagora_experiments.egor.evaluation import PolicyRow, summarize_policy


TaskKey = tuple[str, str]
TEMPLATE_CANDIDATES = {
    "exact": "exact_repair",
    "exact_visual": "exact_visual_repair",
    "numerical": "numerical_repair",
    "verifier": "verifier_repair",
    "visual": "visual_repair",
}


def main() -> int:
    args = _parse_args()
    plan_path = Path(args.control_plans).resolve()
    matrix_path = Path(args.matrix_rows).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plans = _load_plans(plan_path)
    matrix = _load_matrix(matrix_path)
    rows = analyze_matched_controls(plans, matrix)
    rows_path = output_dir / "policy_rows.csv"
    _write_rows(rows_path, rows)
    summaries = {
        policy: summarize_policy([row for row in rows if row.policy_id == policy])
        for policy in sorted({row.policy_id for row in rows})
    }
    summary_path = output_dir / "policy_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "source_commit": os.environ.get("EGOR_SOURCE_COMMIT", "").strip(),
        "policies": sorted(summaries),
        "tasks_per_policy": len(plans),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (plan_path, matrix_path)
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (rows_path, summary_path)
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"scored two matched controls over {len(plans)} tasks")
    return 0


def analyze_matched_controls(
    plans: dict[TaskKey, dict[str, Any]],
    matrix: dict[tuple[str, str, str], dict[str, str]],
) -> list[PolicyRow]:
    results: list[PolicyRow] = []
    for key, plan in sorted(plans.items()):
        random_template = str(plan["matched_random_template_id"])
        random_candidate = (
            "baseline"
            if random_template == "baseline"
            else TEMPLATE_CANDIDATES[random_template]
        )
        results.append(
            _policy_row(
                key,
                matrix,
                policy_id="matched_random_repair",
                candidate_id=random_candidate,
                strict_certificate=True,
            )
        )
        results.append(
            _policy_row(
                key,
                matrix,
                policy_id="selector_trigger_resynthesis",
                candidate_id=str(plan["resynthesis_candidate_id"]),
                strict_certificate=False,
            )
        )
    return results


def _policy_row(
    key: TaskKey,
    matrix: dict[tuple[str, str, str], dict[str, str]],
    *,
    policy_id: str,
    candidate_id: str,
    strict_certificate: bool,
) -> PolicyRow:
    available = [row for item, row in matrix.items() if item[:2] == key]
    if not available:
        raise ValueError(f"matrix lacks task {key}")
    baseline = available[0]
    baseline_quality = float(baseline["baseline_quality"])
    if candidate_id == "baseline":
        return PolicyRow(
            source_id=key[0],
            task_id=key[1],
            policy_id=policy_id,
            candidate_id="baseline",
            triggered=False,
            baseline_quality=baseline_quality,
            policy_quality=baseline_quality,
            quality_delta=0.0,
            calls=float(baseline["baseline_calls"]),
            total_tokens=float(baseline["baseline_total_tokens"]),
            latency_seconds=float(baseline["baseline_latency_seconds"]),
            strict_harm=False,
            useful_repair=False,
            failed_stages=tuple(
                stage
                for stage in str(baseline.get("failed_stages", "")).split("|")
                if stage == "baseline"
            ),
            provenance_rejected=False,
        )
    row = matrix.get((*key, candidate_id))
    if row is None:
        raise ValueError(f"matrix lacks {candidate_id} for {key}")
    changed = _bool(row.get("actual_changed"))
    supported = _bool(row.get("evidence_supported_change"))
    rejected = strict_certificate and changed and not supported
    candidate_quality = float(row["candidate_quality"])
    policy_quality = baseline_quality if rejected else candidate_quality
    delta = policy_quality - baseline_quality
    return PolicyRow(
        source_id=key[0],
        task_id=key[1],
        policy_id=policy_id,
        candidate_id=candidate_id,
        triggered=True,
        baseline_quality=baseline_quality,
        policy_quality=policy_quality,
        quality_delta=delta,
        calls=float(row["calls"]),
        total_tokens=float(row["total_tokens"]),
        latency_seconds=float(row["latency_seconds"]),
        strict_harm=delta < -0.05,
        useful_repair=delta > 0.05,
        failed_stages=tuple(
            value for value in str(row.get("failed_stages", "")).split("|") if value
        ),
        provenance_rejected=rejected,
    )


def _load_plans(path: Path) -> dict[TaskKey, dict[str, Any]]:
    rows: dict[TaskKey, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("source_id", "")), str(row.get("task_id", "")))
            if not all(key) or key in rows:
                raise ValueError(f"invalid or duplicate control plan: {key}")
            rows[key] = row
    if not rows:
        raise ValueError(f"control plans are empty: {path}")
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
        raise ValueError(f"matrix is empty: {path}")
    return rows


def _write_rows(path: Path, rows: list[PolicyRow]) -> None:
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
            payload = asdict(row)
            payload["triggered"] = int(row.triggered)
            payload["strict_harm"] = int(row.strict_harm)
            payload["useful_repair"] = int(row.useful_repair)
            payload["provenance_rejected"] = int(row.provenance_rejected)
            payload["failed_stages"] = "|".join(row.failed_stages)
            writer.writerow(payload)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-plans", required=True)
    parser.add_argument("--matrix-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
