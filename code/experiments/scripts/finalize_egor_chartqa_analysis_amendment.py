#!/usr/bin/env python3
"""Validate and bind the outcome-blind ChartQA analysis amendment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PRIMARY_POLICY = "frozen_egor_selector"
TARGET_CONTROL = "standalone_candidate_value"
FROZEN_CANDIDATE_PLAN_SHA256 = (
    "e6295f660eccfe29acd4b14c0c943278ef09c5d1523e5fa12af0850e324ff3ca"
)


def main() -> int:
    args = _parse_args()
    report = finalize_amendment(
        result_root=Path(args.result_root).resolve(),
        generation_source_commit=args.generation_source_commit,
        analysis_commit=args.analysis_commit,
        expected_tasks=args.expected_tasks,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "analysis amendment integrity passed; "
        f"paper gate passed={report['paper_gate_passed']}"
    )
    return 0


def finalize_amendment(
    *,
    result_root: Path,
    generation_source_commit: str,
    analysis_commit: str,
    expected_tasks: int,
) -> dict[str, Any]:
    if not generation_source_commit.strip() or not analysis_commit.strip():
        raise ValueError("generation and analysis commits are required")
    if expected_tasks <= 0:
        raise ValueError("expected_tasks must be positive")
    paths = {
        "source_commit": result_root / "SOURCE_COMMIT",
        "generation_complete": result_root / "GENERATION_COMPLETE",
        "original_analysis_complete": result_root / "ANALYSIS_COMPLETE",
        "candidate_plan": result_root / "candidate_value_plan" / "plans.jsonl",
        "candidate_plan_manifest": (
            result_root / "candidate_value_plan" / "manifest.json"
        ),
        "candidate_policy_rows": (
            result_root / "candidate_value_analysis" / "policy_rows.csv"
        ),
        "candidate_policy_manifest": (
            result_root / "candidate_value_analysis" / "manifest.json"
        ),
        "policy_bundle": result_root / "policy_bundle.csv",
        "policy_bundle_manifest": result_root / "policy_bundle.csv.manifest.json",
        "paired_statistics": result_root / "paired_statistics.json",
        "alignment_manifest": (
            result_root / "evidence_alignment_audit" / "population_manifest.json"
        ),
        "paper_gate": result_root / "paper_gate_report.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    recorded_source = paths["source_commit"].read_text(encoding="utf-8").strip()
    if recorded_source != generation_source_commit:
        raise ValueError("generation source commit mismatch")
    if _sha256(paths["candidate_plan"]) != FROZEN_CANDIDATE_PLAN_SHA256:
        raise ValueError("candidate-value plan SHA-256 mismatch")

    candidate_manifest = _load_object(paths["candidate_plan_manifest"])
    if str(candidate_manifest.get("source_commit", "")) != generation_source_commit:
        raise ValueError("candidate plan source commit mismatch")
    _validate_manifest_hashes(candidate_manifest, section="outputs")

    primary_keys, target_keys = _policy_task_sets(paths["policy_bundle"])
    if len(primary_keys) != expected_tasks or target_keys != primary_keys:
        raise ValueError("amended policy bundle task population mismatch")
    candidate_rows = _single_policy_task_set(
        paths["candidate_policy_rows"], expected_policy=TARGET_CONTROL
    )
    if candidate_rows != primary_keys:
        raise ValueError("candidate policy rows do not match the primary population")
    candidate_policy_manifest = _load_object(paths["candidate_policy_manifest"])
    if str(candidate_policy_manifest.get("policy_id", "")) != TARGET_CONTROL:
        raise ValueError("candidate policy manifest has the wrong policy")
    if str(candidate_policy_manifest.get("source_commit", "")) != (
        generation_source_commit
    ):
        raise ValueError("candidate policy manifest source commit mismatch")

    policy_bundle_manifest = _load_object(paths["policy_bundle_manifest"])
    bundle_output = policy_bundle_manifest.get("output", {})
    if (
        str(policy_bundle_manifest.get("source_commit", "")) != generation_source_commit
        or not isinstance(bundle_output, dict)
        or Path(str(bundle_output.get("path", ""))).resolve()
        != paths["policy_bundle"].resolve()
        or str(bundle_output.get("sha256", "")) != _sha256(paths["policy_bundle"])
    ):
        raise ValueError("amended policy bundle manifest mismatch")

    statistics = _load_object(paths["paired_statistics"])
    contrast = statistics.get("contrasts", {}).get(TARGET_CONTROL, {})
    if (
        str(statistics.get("primary_policy", "")) != PRIMARY_POLICY
        or int(statistics.get("tasks", -1)) != expected_tasks
        or not isinstance(contrast, dict)
        or int(contrast.get("tasks", -1)) != expected_tasks
    ):
        raise ValueError("paired target contrast is missing or malformed")

    alignment = _load_object(paths["alignment_manifest"])
    if str(alignment.get("source_commit", "")) != generation_source_commit:
        raise ValueError("alignment packet generation commit mismatch")
    if str(alignment.get("analysis_commit", "")) != analysis_commit:
        raise ValueError("alignment packet analysis commit mismatch")
    if int(alignment.get("task_count", -1)) != expected_tasks:
        raise ValueError("alignment packet task count mismatch")
    item_count = int(alignment.get("audit_item_count", -1))
    population_counts = alignment.get("population_counts", {})
    exclusion_counts = alignment.get("exclusion_counts", {})
    if not isinstance(population_counts, dict) or not isinstance(
        exclusion_counts, dict
    ):
        raise ValueError("alignment population counts are malformed")
    if int(population_counts.get("accepted_changed", 0)) != item_count:
        raise ValueError("blinded packet is not the full accepted-change population")
    if (
        sum(int(value) for value in population_counts.values())
        + sum(int(value) for value in exclusion_counts.values())
        != expected_tasks
    ):
        raise ValueError("alignment populations do not partition every task")
    _validate_manifest_hashes(alignment, section="outputs")

    gate = _load_object(paths["paper_gate"])
    if (
        str(gate.get("source_commit", "")) != generation_source_commit
        or int(gate.get("tasks", -1)) != expected_tasks
        or str(gate.get("primary_policy", "")) != PRIMARY_POLICY
    ):
        raise ValueError("paper gate provenance or population mismatch")
    criterion_names = {
        str(item.get("name", ""))
        for item in gate.get("criteria", [])
        if isinstance(item, dict)
    }
    if "paired_target_vs_standalone_candidate_value" not in criterion_names:
        raise ValueError("paper gate lacks the frozen target criterion")

    bound_paths = [
        paths[name]
        for name in (
            "candidate_plan",
            "candidate_plan_manifest",
            "candidate_policy_rows",
            "candidate_policy_manifest",
            "policy_bundle",
            "policy_bundle_manifest",
            "paired_statistics",
            "alignment_manifest",
            "paper_gate",
        )
    ]
    return {
        "schema_version": 1,
        "generation_source_commit": generation_source_commit,
        "analysis_commit": analysis_commit,
        "expected_tasks": expected_tasks,
        "candidate_plan_sha256": FROZEN_CANDIDATE_PLAN_SHA256,
        "primary_policy": PRIMARY_POLICY,
        "target_control": TARGET_CONTROL,
        "paper_gate_passed": gate.get("passed") is True,
        "integrity_passed": True,
        "policy_counts": {
            PRIMARY_POLICY: len(primary_keys),
            TARGET_CONTROL: len(target_keys),
        },
        "alignment_population_counts": {
            key: int(value) for key, value in sorted(population_counts.items())
        },
        "alignment_exclusion_counts": {
            key: int(value) for key, value in sorted(exclusion_counts.items())
        },
        "artifacts": [
            {"path": str(path), "sha256": _sha256(path)} for path in bound_paths
        ],
    }


def _policy_task_sets(path: Path) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    policies: dict[str, set[tuple[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            policy = str(row.get("policy_id", "")).strip()
            key = (
                str(row.get("source_id", "")).strip(),
                str(row.get("task_id", "")).strip(),
            )
            if not policy or not all(key):
                raise ValueError("invalid policy bundle row")
            bucket = policies.setdefault(policy, set())
            if key in bucket:
                raise ValueError(f"duplicate policy task row: {policy}:{key}")
            bucket.add(key)
    missing = {PRIMARY_POLICY, TARGET_CONTROL} - set(policies)
    if missing:
        raise ValueError(f"amended policy bundle lacks policies: {sorted(missing)}")
    return policies[PRIMARY_POLICY], policies[TARGET_CONTROL]


def _single_policy_task_set(
    path: Path, *, expected_policy: str
) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("policy_id", "")).strip() != expected_policy:
                raise ValueError("unexpected policy in candidate rows")
            key = (
                str(row.get("source_id", "")).strip(),
                str(row.get("task_id", "")).strip(),
            )
            if not all(key) or key in keys:
                raise ValueError("invalid or duplicate candidate policy task")
            keys.add(key)
    return keys


def _validate_manifest_hashes(manifest: dict[str, Any], *, section: str) -> None:
    raw_entries = manifest.get(section, [])
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(f"manifest has no {section}")
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise ValueError(f"malformed manifest {section} entry")
        path = Path(str(entry.get("path", ""))).resolve()
        expected = str(entry.get("sha256", ""))
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"manifest hash mismatch: {path}")


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--generation-source-commit", required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--expected-tasks", type=int, default=240)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
