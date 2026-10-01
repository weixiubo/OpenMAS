#!/usr/bin/env python3
"""Validate the isolated ChartQA strict-analysis recovery and amendment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from openagora_experiments.egor.analysis_recovery import sha256, verify_snapshot


TASK_SHA256 = (
    "0c2e28fd23ab69f85e6861f8fba58336cba59d00749367294a7f20e0fd4f3cf8"
)
MATRIX_CONFIG_SHA256 = (
    "2762e5056acd2ec112191c46a8d64010effa4d0a6f44650e75284effd53d4ad2"
)
EXECUTION_CONFIG_SHA256 = (
    "0418d35f5fb22605918144048ff0d9543ee5b6a18d32fa77fa1b98262d163543"
)
GENERATION_AUDIT_SHA256 = (
    "73ec4dc85d84231a51f30c319a919f812b2a09af4ed4f01352af420c8f21f826"
)
CANDIDATE_PLAN_SHA256 = (
    "e6295f660eccfe29acd4b14c0c943278ef09c5d1523e5fa12af0850e324ff3ca"
)
EXPECTED_CANDIDATES = {
    "baseline_resynthesis",
    "exact_repair",
    "exact_visual_repair",
    "full_team",
    "numerical_repair",
    "verifier_repair",
    "visual_repair",
}
EXPECTED_MATRIX_SELECTION = {
    "baseline_resynthesis_all": ("all", None, True),
    "best_fixed_visual_gap": ("gap", 0.05, False),
    "fixed_exact_gap": ("gap", 0.05, False),
    "fixed_exact_visual_gap": ("gap", 0.05, False),
    "fixed_numerical_gap": ("gap", 0.05, False),
    "fixed_verifier_gap": ("gap", 0.05, False),
    "full_team_all": ("all", None, True),
}
RUN_ALL_POLICIES = {"baseline_resynthesis_all", "full_team_all"}
REQUIRED_RELEASE_POLICIES = {
    "baseline_resynthesis_all",
    "best_fixed_visual_gap",
    "frozen_egor_selector",
    "full_team_all",
    "matched_random_repair",
    "paid_llm_scheduler",
    "selector_trigger_resynthesis",
    "standalone_candidate_value",
}


def finalize_recovery(
    *,
    generation_root: Path,
    recovery_root: Path,
    generation_commit: str,
    recovery_commit: str,
    amendment_commit: str,
    expected_tasks: int,
) -> dict[str, Any]:
    if expected_tasks <= 0:
        raise ValueError("expected_tasks must be positive")
    generation_root = generation_root.resolve()
    recovery_root = recovery_root.resolve()
    for marker in ("GENERATION_COMPLETE",):
        _require_file(generation_root / marker)
    for marker in ("ANALYSIS_COMPLETE", "ANALYSIS_AMENDMENT_COMPLETE"):
        _require_file(recovery_root / marker)
    _require_commit(generation_root / "SOURCE_COMMIT", generation_commit)
    _require_commit(recovery_root / "SOURCE_COMMIT", generation_commit)
    _require_commit(
        recovery_root / "ANALYSIS_RECOVERY_SOURCE_COMMIT", recovery_commit
    )
    _require_commit(
        recovery_root / "ANALYSIS_AMENDMENT_SOURCE_COMMIT", amendment_commit
    )

    original_paths = {
        "tasks": generation_root.parents[1]
        / "data"
        / "chartqa_fresh_v1"
        / "heldout240.jsonl",
        "matrix_config": generation_root / "matrix_config.json",
        "execution_config": generation_root / "execution_config.json",
        "generation_audit": generation_root / "generation_audit.json",
        "candidate_plan": generation_root / "candidate_value_plan" / "plans.jsonl",
    }
    expected_hashes = {
        "tasks": TASK_SHA256,
        "matrix_config": MATRIX_CONFIG_SHA256,
        "execution_config": EXECUTION_CONFIG_SHA256,
        "generation_audit": GENERATION_AUDIT_SHA256,
        "candidate_plan": CANDIDATE_PLAN_SHA256,
    }
    for name, path in original_paths.items():
        _require_sha(path, expected_hashes[name])

    snapshot_path = recovery_root / "protected_artifacts_snapshot.json"
    snapshot = _load_object(snapshot_path)
    verify_snapshot(snapshot)

    audit = _load_object(recovery_root / "generation_audit.json")
    if (
        sha256(recovery_root / "generation_audit.json") != GENERATION_AUDIT_SHA256
        or audit.get("passed") is not True
        or audit.get("outcome_blind") is not True
        or audit.get("require_manifest") is not True
        or int(audit.get("tasks", {}).get("count", -1)) != expected_tasks
    ):
        raise ValueError("frozen generation audit is incompatible")
    stages = audit.get("stages")
    if not isinstance(stages, dict) or len(stages) != 13:
        raise ValueError("generation audit does not bind all 13 stages")
    for stage_id, stage in stages.items():
        manifest = stage.get("manifest", {}) if isinstance(stage, dict) else {}
        if (
            stage.get("complete") is not True
            or int(stage.get("successful_tasks", -1)) != expected_tasks
            or manifest.get("compatible") is not True
            or manifest.get("git_commit") != generation_commit
        ):
            raise ValueError(f"generation stage is incomplete: {stage_id}")

    matrix_manifest_path = recovery_root / "matrix_analysis" / "manifest.json"
    matrix_manifest = _load_object(matrix_manifest_path)
    if (
        matrix_manifest.get("source_commit") != generation_commit
        or int(matrix_manifest.get("tasks", -1)) != expected_tasks
        or set(matrix_manifest.get("candidate_ids", [])) != EXPECTED_CANDIDATES
    ):
        raise ValueError("recovered matrix manifest is incompatible")
    _validate_manifest_hashes(matrix_manifest, section="outputs")
    selection = matrix_manifest.get("policy_selection")
    if not isinstance(selection, list):
        raise ValueError("matrix manifest lacks explicit policy selection semantics")
    observed_selection = {
        str(item.get("policy_id", "")): (
            item.get("trigger_mode"),
            item.get("threshold"),
            item.get("translated_legacy_run_all_sentinel"),
        )
        for item in selection
        if isinstance(item, dict)
    }
    if observed_selection != EXPECTED_MATRIX_SELECTION:
        raise ValueError("matrix run-all/gap policy semantics changed")

    matrix_sets, matrix_trigger_counts = _policy_task_sets(
        recovery_root / "matrix_analysis" / "policy_rows.csv"
    )
    if set(matrix_sets) != set(EXPECTED_MATRIX_SELECTION):
        raise ValueError("matrix policy population is incomplete")
    _validate_equal_task_sets(matrix_sets, expected_tasks=expected_tasks)
    if any(matrix_trigger_counts[name] != expected_tasks for name in RUN_ALL_POLICIES):
        raise ValueError("an explicit run-all control failed to trigger every task")

    bundle_path = recovery_root / "policy_bundle.csv"
    bundle_sets, _ = _policy_task_sets(bundle_path)
    if not REQUIRED_RELEASE_POLICIES.issubset(bundle_sets):
        raise ValueError("recovered policy bundle lacks a release policy")
    _validate_equal_task_sets(bundle_sets, expected_tasks=expected_tasks)
    bundle_manifest = _load_object(
        recovery_root / "policy_bundle.csv.manifest.json"
    )
    bundle_output = bundle_manifest.get("output")
    if (
        bundle_manifest.get("source_commit") != generation_commit
        or not isinstance(bundle_output, dict)
        or Path(str(bundle_output.get("path", ""))).resolve() != bundle_path
        or bundle_output.get("sha256") != sha256(bundle_path)
    ):
        raise ValueError("recovered policy bundle manifest mismatch")

    statistics_path = recovery_root / "paired_statistics.json"
    statistics = _load_object(statistics_path)
    if (
        statistics.get("primary_policy") != "frozen_egor_selector"
        or int(statistics.get("tasks", -1)) != expected_tasks
        or int(statistics.get("bootstrap_samples", -1)) != 100_000
        or int(statistics.get("seed", -1)) != 20_260_712
        or Path(str(statistics.get("policy_rows", ""))).resolve() != bundle_path
        or statistics.get("policy_rows_sha256") != sha256(bundle_path)
    ):
        raise ValueError("recovered paired statistics are not hash-bound")

    gate_path = recovery_root / "paper_gate_report.json"
    gate = _load_object(gate_path)
    if (
        gate.get("source_commit") != generation_commit
        or int(gate.get("tasks", -1)) != expected_tasks
        or gate.get("primary_policy") != "frozen_egor_selector"
    ):
        raise ValueError("recovered paper gate provenance mismatch")

    amendment_manifest_path = recovery_root / "analysis_amendment_manifest.json"
    amendment = _load_object(amendment_manifest_path)
    if (
        amendment.get("integrity_passed") is not True
        or amendment.get("generation_source_commit") != generation_commit
        or amendment.get("analysis_commit") != amendment_commit
        or int(amendment.get("expected_tasks", -1)) != expected_tasks
        or amendment.get("paper_gate_passed") != (gate.get("passed") is True)
    ):
        raise ValueError("frozen amendment manifest is incompatible")
    _validate_artifacts(amendment.get("artifacts"), within=recovery_root)

    alignment = _load_object(
        recovery_root / "evidence_alignment_audit" / "population_manifest.json"
    )
    if (
        alignment.get("source_commit") != generation_commit
        or alignment.get("analysis_commit") != amendment_commit
        or int(alignment.get("task_count", -1)) != expected_tasks
    ):
        raise ValueError("evidence-alignment audit provenance mismatch")

    bound_paths = [
        snapshot_path,
        matrix_manifest_path,
        recovery_root / "matrix_analysis" / "policy_rows.csv",
        bundle_path,
        recovery_root / "policy_bundle.csv.manifest.json",
        statistics_path,
        gate_path,
        amendment_manifest_path,
        recovery_root / "evidence_alignment_audit" / "population_manifest.json",
    ]
    return {
        "schema_version": 1,
        "generation_source_commit": generation_commit,
        "analysis_recovery_commit": recovery_commit,
        "analysis_amendment_commit": amendment_commit,
        "expected_tasks": expected_tasks,
        "strict_analysis_integrity_passed": True,
        "protected_artifacts_unchanged": True,
        "paper_gate_passed": gate.get("passed") is True,
        "publication_ready": gate.get("passed") is True,
        "matrix_policy_counts": {
            policy: len(keys) for policy, keys in sorted(matrix_sets.items())
        },
        "release_policy_counts": {
            policy: len(bundle_sets[policy])
            for policy in sorted(REQUIRED_RELEASE_POLICIES)
        },
        "artifacts": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in bound_paths
        ],
    }


def _policy_task_sets(
    path: Path,
) -> tuple[dict[str, set[tuple[str, str]]], dict[str, int]]:
    sets: dict[str, set[tuple[str, str]]] = {}
    trigger_counts: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            policy = str(row.get("policy_id", "")).strip()
            key = (
                str(row.get("source_id", "")).strip(),
                str(row.get("task_id", "")).strip(),
            )
            if not policy or not all(key) or key in sets.setdefault(policy, set()):
                raise ValueError(f"invalid or duplicate policy row: {policy}:{key}")
            sets[policy].add(key)
            trigger_counts[policy] = trigger_counts.get(policy, 0) + int(
                str(row.get("triggered", "0")).strip() in {"1", "true", "True"}
            )
    if not sets:
        raise ValueError(f"policy rows are empty: {path}")
    return sets, trigger_counts


def _validate_equal_task_sets(
    sets: dict[str, set[tuple[str, str]]], *, expected_tasks: int
) -> None:
    expected = next(iter(sets.values()))
    if len(expected) != expected_tasks:
        raise ValueError("policy task count mismatch")
    if any(keys != expected for keys in sets.values()):
        raise ValueError("policy task populations differ")


def _validate_manifest_hashes(manifest: dict[str, Any], *, section: str) -> None:
    entries = manifest.get(section)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"manifest has no {section}")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"malformed manifest {section} entry")
        _require_sha(
            Path(str(entry.get("path", ""))).resolve(),
            str(entry.get("sha256", "")),
        )


def _validate_artifacts(raw: Any, *, within: Path) -> None:
    if not isinstance(raw, list) or not raw:
        raise ValueError("manifest has no bound artifacts")
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("malformed artifact entry")
        path = Path(str(entry.get("path", ""))).resolve()
        try:
            path.relative_to(within)
        except ValueError as error:
            raise ValueError(f"recovery artifact escapes its root: {path}") from error
        _require_sha(path, str(entry.get("sha256", "")))


def _require_commit(path: Path, expected: str) -> None:
    _require_file(path)
    if path.read_text(encoding="utf-8").strip() != expected:
        raise ValueError(f"commit mismatch: {path}")


def _require_sha(path: Path, expected: str) -> None:
    _require_file(path)
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")


def _load_object(path: Path) -> dict[str, Any]:
    _require_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-result-root", required=True)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--generation-source-commit", required=True)
    parser.add_argument("--recovery-commit", required=True)
    parser.add_argument("--amendment-commit", required=True)
    parser.add_argument("--expected-tasks", type=int, default=240)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = finalize_recovery(
        generation_root=Path(args.generation_result_root),
        recovery_root=Path(args.recovery_root),
        generation_commit=args.generation_source_commit,
        recovery_commit=args.recovery_commit,
        amendment_commit=args.amendment_commit,
        expected_tasks=args.expected_tasks,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "analysis recovery integrity passed; "
        f"paper gate passed={report['paper_gate_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
