#!/usr/bin/env python3
"""Validate and bind a completed prospective EGOR transfer release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


COMMIT_RE = re.compile(r"[0-9a-f]{40}")
PRIMARY_POLICY = "frozen_egor_selector"
REQUIRED_GATE_CRITERIA = {
    "source_commit",
    "generation_audit",
    "plans_precede_repairs",
    "positive_heldout_effect",
    "pareto_vs_best_fixed_visual_gap",
    "pareto_vs_full_team_all",
    "pareto_vs_paid_llm_scheduler",
    "mechanism_vs_matched_random_repair",
    "mechanism_vs_selector_trigger_resynthesis",
    "mechanism_vs_baseline_resynthesis_all",
    "paired_target_vs_standalone_candidate_value",
    "strict_harm",
    "repair_rate",
    "completion",
}


def main() -> int:
    args = _parse_args()
    report = finalize_release(
        result_root=Path(args.result_root).resolve(),
        dataset_id=args.dataset_id,
        alignment_dataset_id=args.alignment_dataset_id,
        generation_source_commit=args.generation_source_commit,
        analysis_commit=args.analysis_commit,
        expected_tasks=args.expected_tasks,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(
        "prospective release integrity passed; "
        f"paper gate passed={report['paper_gate_passed']}"
    )
    return 0


def finalize_release(
    *,
    result_root: Path,
    dataset_id: str,
    alignment_dataset_id: str,
    generation_source_commit: str,
    analysis_commit: str,
    expected_tasks: int,
) -> dict[str, Any]:
    _validate_commit(generation_source_commit, "generation source commit")
    _validate_commit(analysis_commit, "analysis commit")
    if not dataset_id.strip() or not alignment_dataset_id.strip():
        raise ValueError("dataset identifiers are required")
    if expected_tasks <= 0:
        raise ValueError("expected_tasks must be positive")

    paths = {
        "source_commit": result_root / "SOURCE_COMMIT",
        "control_commit": result_root / "CONTROL_COMMIT",
        "plans_frozen": result_root / "PLANS_FROZEN",
        "generation_complete": result_root / "GENERATION_COMPLETE",
        "analysis_complete": result_root / "ANALYSIS_COMPLETE",
        "paper_gate": result_root / "paper_gate_report.json",
        "alignment_manifest": result_root
        / "evidence_alignment_audit"
        / "population_manifest.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    _require_commit(paths["source_commit"], generation_source_commit)
    _require_commit(paths["control_commit"], analysis_commit)

    gate = _load_object(paths["paper_gate"])
    if (
        int(gate.get("schema_version", -1)) != 1
        or str(gate.get("dataset_id", "")) != dataset_id
        or str(gate.get("source_commit", "")) != generation_source_commit
        or int(gate.get("tasks", -1)) != expected_tasks
        or str(gate.get("primary_policy", "")) != PRIMARY_POLICY
    ):
        raise ValueError("prospective paper gate provenance or population mismatch")
    criteria = gate.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("prospective paper gate has no criteria")
    criterion_names = {
        str(item.get("name", "")) for item in criteria if isinstance(item, dict)
    }
    missing = sorted(REQUIRED_GATE_CRITERIA - criterion_names)
    if missing:
        raise ValueError(f"prospective paper gate lacks criteria: {missing}")
    all_criteria_pass = all(
        isinstance(item, dict) and item.get("passed") is True for item in criteria
    )
    if (gate.get("passed") is True) != all_criteria_pass:
        raise ValueError("prospective paper gate summary conflicts with its criteria")
    _validate_artifacts(gate, "artifacts")

    alignment = _load_object(paths["alignment_manifest"])
    if (
        str(alignment.get("source_commit", "")) != generation_source_commit
        or str(alignment.get("analysis_commit", "")) != analysis_commit
        or str(alignment.get("dataset_id", "")) != alignment_dataset_id
        or str(alignment.get("policy_id", "")) != PRIMARY_POLICY
        or int(alignment.get("task_count", -1)) != expected_tasks
    ):
        raise ValueError("evidence-alignment provenance or population mismatch")
    _validate_artifacts(alignment, "inputs")
    _validate_artifacts(alignment, "outputs")

    bound_paths = list(paths.values())
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "alignment_dataset_id": alignment_dataset_id,
        "generation_source_commit": generation_source_commit,
        "analysis_commit": analysis_commit,
        "expected_tasks": expected_tasks,
        "primary_policy": PRIMARY_POLICY,
        "paper_gate_passed": gate.get("passed") is True,
        "integrity_passed": True,
        "criteria_passed": sum(
            isinstance(item, dict) and item.get("passed") is True
            for item in criteria
        ),
        "criteria_total": len(criteria),
        "artifacts": [
            {"path": str(path), "sha256": _sha256(path)} for path in bound_paths
        ],
    }


def _validate_artifacts(payload: dict[str, Any], section: str) -> None:
    entries = payload.get(section)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"manifest has no {section}")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"malformed {section} entry")
        path = Path(str(entry.get("path", ""))).resolve()
        if not path.is_file() or _sha256(path) != str(entry.get("sha256", "")):
            raise ValueError(f"bound artifact hash mismatch: {path}")


def _require_commit(path: Path, expected: str) -> None:
    if path.read_text(encoding="utf-8").strip() != expected:
        raise ValueError(f"commit mismatch: {path}")


def _validate_commit(value: str, label: str) -> None:
    if COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact 40-character Git commit")


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
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--alignment-dataset-id", required=True)
    parser.add_argument("--generation-source-commit", required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--expected-tasks", type=int, default=240)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
