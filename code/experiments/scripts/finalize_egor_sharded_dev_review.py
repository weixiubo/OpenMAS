#!/usr/bin/env python3
"""Finalize the outcome-blind review of the EGOR sharded development smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


COMMIT_RE = re.compile(r"[0-9a-f]{40}")
HASH_RE = re.compile(r"[0-9a-f]{64}")
TASKS_SHA256 = "016510650d590a9890149d9988022cd7b6624897c9bde7f13107e6c6c1d036fc"
SOURCE_TREE_SHA256 = "8839dbb7669a694a23236f9d0bf8798de1dad047cc29999ca78d57add788b4e1"
RUNTIME_PACKAGES = {
    "accelerate": "1.13.0",
    "pillow": "11.3.0",
    "qwen-vl-utils": "0.0.14",
    "torch": "2.6.0+cu124",
    "transformers": "4.57.6",
}
RUNTIME_SOURCE_HASHES = {
    "openagora/providers/local_provider.py": "181b452e5901073777986567f301d7278f28023ead364c3241e2f7b88962888a",
    "openagora_experiments/egor/local_provider.py": "59b23c236a709f02e53e8dc1667eb197659a3997b24ebd4307a8f55b61d31204",
}
MODEL_METADATA_HASHES = {
    "qwen3-vl-8b-instruct": {
        "config.json": "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661",
        "model.safetensors.index.json": "520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070",
    },
    "qwen2.5-vl-7b-instruct": {
        "config.json": "77d9ec7321cc572e3579e2c84799c9cadaded63c49ce93b101733349fc330c43",
        "model.safetensors.index.json": "73b333b0b16e5286ddba615d2caebcd495cf7e616f52eb217a81781393d79de9",
    },
    "internvl3-8b-hf": {
        "config.json": "2f651cf5dcad81635fcd688dc8aa569797a3d829b9d0644f489e1c7272b1c637",
        "model.safetensors.index.json": "5551d292e2be31ed9c4fd016d99389127e55a6f2316519c7621cd2fdd9b0dfd4",
    },
}
STAGES = {
    "baseline": {
        "records": "baseline_qwen3.jsonl",
        "stage": "baseline",
        "model_name": "qwen3-vl-8b-instruct",
        "seed": 20260711,
    },
    "exact": {
        "records": "specialist_exact_qwen25.jsonl",
        "stage": "specialist",
        "model_name": "qwen2.5-vl-7b-instruct",
        "seed": 20260721,
    },
    "visual": {
        "records": "specialist_visual_internvl3.jsonl",
        "stage": "specialist",
        "model_name": "internvl3-8b-hf",
        "seed": 20260731,
    },
    "synthesis": {
        "records": "exact_repair.jsonl",
        "stage": "synthesis",
        "model_name": "qwen3-vl-8b-instruct",
        "seed": 20260811,
    },
}
FORBIDDEN_RUNTIME_MISMATCH_FIELDS = {
    "candidate_gpu_devices",
    "candidate_max_gpu_mem_gb",
    "candidate_git_commit",
}


def main() -> int:
    args = _parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if output.parent != root:
        raise SystemExit("development review output must be written inside its root")
    report = review_dev_smoke(
        root=root,
        source_commit=_commit(args.source_commit, "source commit"),
        control_commit=_commit(args.control_commit, "control commit"),
    )
    _write_json_atomic(output, report)
    if args.write_marker and report["passed"]:
        marker = root / "DEV_SMOKE_REVIEWED"
        marker_payload = _sha256(output) + "\n"
        if marker.exists() and marker.read_text(encoding="utf-8") != marker_payload:
            raise SystemExit("refusing to replace a non-matching review marker")
        _write_text_atomic(marker, marker_payload)
    print(
        "EGOR sharded development review "
        f"{'passed' if report['passed'] else 'failed'}: {output}"
    )
    if args.require_pass and not report["passed"]:
        return 2
    return 0


def review_dev_smoke(
    *, root: Path, source_commit: str, control_commit: str
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}
    historical_fields: Counter[str] = Counter()
    historical_passes: dict[str, bool] = {}

    _expect(errors, root.is_dir(), "missing_dev_root", path=str(root))
    _expect(
        errors,
        (root / "DEV_SMOKE_COMPLETE").is_file(),
        "missing_dev_smoke_complete",
    )
    _check_text_marker(errors, root / "SOURCE_COMMIT", source_commit, "source")
    _check_text_marker(errors, root / "CONTROL_COMMIT", control_commit, "control")

    exact_marker = (root / "HISTORICAL_EXACT_MATCH").is_file()
    drift_marker = (root / "HISTORICAL_DRIFT_RECORDED").is_file()
    _expect(
        errors,
        exact_marker != drift_marker,
        "historical_marker_exclusivity",
        exact_marker=exact_marker,
        drift_marker=drift_marker,
    )

    frozen = _load_object(root / "RUN_FROZEN.json", errors, "run_frozen", artifacts)
    _expect(errors, frozen.get("outcome_blind") is True, "run_not_outcome_blind")
    _expect(
        errors,
        frozen.get("source_commit_label") == source_commit,
        "run_source_commit_mismatch",
    )
    _expect(
        errors,
        frozen.get("control_commit") == control_commit,
        "run_control_commit_mismatch",
    )
    _expect(
        errors,
        frozen.get("physical_gpu_order") == [1, 0],
        "run_gpu_order_mismatch",
    )
    _expect(
        errors,
        _equal_number(frozen.get("max_gpu_mem_gb_per_visible_device"), 11),
        "run_memory_cap_mismatch",
    )
    _expect(
        errors,
        frozen.get("historical_exact_match_is_diagnostic") is True,
        "historical_match_not_diagnostic",
    )
    _expect(
        errors,
        frozen.get("full_chartqa_baseline_equivalence_required_before_repairs")
        is True,
        "full_baseline_gate_not_required",
    )
    _expect(
        errors,
        frozen.get("runtime_source_tree_sha256") == SOURCE_TREE_SHA256,
        "runtime_source_tree_mismatch",
    )
    allowed_pids = _positive_integer_set(frozen.get("preapproved_service_pids"))
    _expect(errors, bool(allowed_pids), "missing_preapproved_service_pids")

    audit = _load_object(
        root / "dev_smoke_audit.json", errors, "dev_smoke_audit", artifacts
    )
    _expect(errors, audit.get("outcome_blind") is True, "audit_not_outcome_blind")
    _expect(errors, audit.get("passed") is True, "development_audit_failed")
    _expect(errors, audit.get("require_manifest") is True, "manifest_audit_not_required")
    audit_tasks = audit.get("tasks", {})
    _expect(
        errors,
        isinstance(audit_tasks, dict)
        and audit_tasks.get("count") == 2
        and audit_tasks.get("subset_filter") is True
        and audit_tasks.get("sha256") == TASKS_SHA256,
        "audit_population_mismatch",
    )
    audit_stages = audit.get("stages", {})
    _expect(
        errors,
        isinstance(audit_stages, dict) and set(audit_stages) == set(STAGES),
        "audit_stage_set_mismatch",
    )

    for label, contract in STAGES.items():
        records = root / str(contract["records"])
        manifest_path = records.with_suffix(records.suffix + ".manifest.json")
        manifest = _load_object(
            manifest_path, errors, f"{label}_manifest", artifacts
        )
        _check_manifest(
            errors,
            label=label,
            manifest=manifest,
            contract=contract,
            source_commit=source_commit,
        )
        stage_audit = audit_stages.get(label, {}) if isinstance(audit_stages, dict) else {}
        _check_stage_audit(
            errors,
            label=label,
            stage_audit=stage_audit,
            source_commit=source_commit,
            seed=int(contract["seed"]),
        )

        gate_path = root / f"{label}.resource_gate.json"
        gate = _load_object(gate_path, errors, f"{label}_resource_gate", artifacts)
        _check_resource_gate(
            errors,
            label=label,
            gate=gate,
            allowed_pids=allowed_pids,
        )

        comparison_path = root / f"{label}.runtime_equivalence.json"
        comparison = _load_object(
            comparison_path,
            errors,
            f"{label}_historical_comparison",
            artifacts,
        )
        comparison_passed = comparison.get("passed") is True
        historical_passes[label] = comparison_passed
        _check_historical_comparison(
            errors,
            label=label,
            comparison=comparison,
            source_commit=source_commit,
        )
        mismatches = comparison.get("mismatches", [])
        if isinstance(mismatches, list):
            historical_fields.update(
                str(item.get("field", "unknown"))
                for item in mismatches
                if isinstance(item, dict)
            )

        receipt_path = root / f"{label}.STAGE_SHA256"
        _check_receipt(
            errors,
            root=root,
            path=receipt_path,
            expected_paths={gate_path, records, manifest_path, comparison_path},
            label=label,
            artifacts=artifacts,
        )
        _expect(
            errors,
            (root / f"{label}.STAGE_COMPLETE").is_file(),
            "missing_stage_complete",
            stage=label,
        )

    historical_exact_match = bool(historical_passes) and all(
        historical_passes.values()
    )
    _expect(
        errors,
        exact_marker == historical_exact_match,
        "historical_summary_marker_mismatch",
        reports_exact=historical_exact_match,
        exact_marker=exact_marker,
    )

    passed = not errors
    return {
        "schema_version": 1,
        "outcome_blind": True,
        "reads_task_answers": False,
        "reads_scores": False,
        "passed": passed,
        "authorizes_full_chartqa_baseline_equivalence": passed,
        "does_not_authorize": [
            "scheduler",
            "specialist",
            "synthesis",
            "heldout_scoring",
            "prospective_result_root",
        ],
        "source_commit_label": source_commit,
        "control_commit": control_commit,
        "runtime_contract": {
            "physical_gpu_order": [1, 0],
            "gpu_devices": "1,0",
            "max_gpu_mem_gb_per_visible_device": 11,
            "resource_readings_per_stage": 2,
            "resource_interval_seconds": 30,
        },
        "historical_reference_role": "diagnostic_only",
        "historical_exact_match": historical_exact_match,
        "historical_stage_passes": historical_passes,
        "historical_mismatch_field_counts": dict(sorted(historical_fields.items())),
        "errors": errors,
        "artifacts": dict(sorted(artifacts.items())),
    }


def _check_manifest(
    errors: list[dict[str, Any]],
    *,
    label: str,
    manifest: dict[str, Any],
    contract: dict[str, Any],
    source_commit: str,
) -> None:
    checks = {
        "stage": manifest.get("stage") == contract["stage"],
        "model_name": manifest.get("model_name") == contract["model_name"],
        "seed": manifest.get("seed") == contract["seed"],
        "tasks_sha256": manifest.get("tasks_sha256") == TASKS_SHA256,
        "gpu_devices": manifest.get("gpu_devices") == "1,0",
        "max_gpu_mem_gb": _equal_number(manifest.get("max_gpu_mem_gb"), 11),
        "requested": manifest.get("requested") == 2,
        "successful_total": manifest.get("successful_total") == 2,
        "git_commit": manifest.get("git_commit") == source_commit,
        "cublas_workspace_config": manifest.get("cublas_workspace_config")
        == ":4096:8",
        "runtime_fingerprint": _runtime_fingerprint_matches(
            manifest.get("runtime"), str(contract["model_name"])
        ),
    }
    _expect(
        errors,
        all(checks.values()),
        "candidate_manifest_contract_failed",
        stage=label,
        checks=checks,
    )


def _check_stage_audit(
    errors: list[dict[str, Any]],
    *,
    label: str,
    stage_audit: Any,
    source_commit: str,
    seed: int,
) -> None:
    if not isinstance(stage_audit, dict):
        _expect(errors, False, "invalid_stage_audit", stage=label)
        return
    manifest = stage_audit.get("manifest", {})
    checks = {
        "complete": stage_audit.get("complete") is True,
        "rows": stage_audit.get("rows") == 2,
        "successful_tasks": stage_audit.get("successful_tasks") == 2,
        "duplicate_attempt_rows": stage_audit.get("duplicate_attempt_rows") == 0,
        "status_counts": stage_audit.get("status_counts") == {"ok": 2},
        "expected_seed": stage_audit.get("expected_base_seed") == seed,
        "manifest_compatible": isinstance(manifest, dict)
        and manifest.get("compatible") is True,
        "manifest_commit": isinstance(manifest, dict)
        and manifest.get("git_commit") == source_commit,
    }
    _expect(
        errors,
        all(checks.values()),
        "stage_audit_contract_failed",
        stage=label,
        checks=checks,
    )


def _check_resource_gate(
    errors: list[dict[str, Any]],
    *,
    label: str,
    gate: dict[str, Any],
    allowed_pids: set[int],
) -> None:
    requirements = gate.get("requirements", {})
    readings = gate.get("readings", [])
    checks = {
        "outcome_blind": gate.get("outcome_blind") is True,
        "read_only": gate.get("read_only") is True,
        "passed": gate.get("passed") is True,
        "completed_readings": gate.get("completed_readings") == 2,
        "allowed_pids": isinstance(requirements, dict)
        and _positive_integer_set(requirements.get("allowed_compute_pids"))
        == allowed_pids,
        "minimum_free_mib": isinstance(requirements, dict)
        and requirements.get("minimum_free_mib") == {"0": 15000, "1": 18000},
        "max_utilization": isinstance(requirements, dict)
        and requirements.get("max_utilization_percent") == 5,
        "readings_required": isinstance(requirements, dict)
        and requirements.get("readings") == 2,
        "interval": isinstance(requirements, dict)
        and _equal_number(requirements.get("interval_seconds"), 30),
        "unknown_disallowed": isinstance(requirements, dict)
        and requirements.get("unknown_compute_pids_allowed") is False,
        "reading_records": isinstance(readings, list)
        and len(readings) == 2
        and all(
            isinstance(item, dict)
            and item.get("passed") is True
            and item.get("violations") == []
            for item in readings
        ),
    }
    _expect(
        errors,
        all(checks.values()),
        "resource_gate_contract_failed",
        stage=label,
        checks=checks,
    )


def _check_historical_comparison(
    errors: list[dict[str, Any]],
    *,
    label: str,
    comparison: dict[str, Any],
    source_commit: str,
) -> None:
    candidate = comparison.get("candidate_runtime", {})
    mismatch_fields = {
        str(item.get("field", ""))
        for item in comparison.get("mismatches", [])
        if isinstance(item, dict)
    }
    checks = {
        "outcome_blind": comparison.get("outcome_blind") is True,
        "tasks": comparison.get("tasks") == 2,
        "candidate_gpu_devices": isinstance(candidate, dict)
        and candidate.get("gpu_devices") == "1,0",
        "candidate_memory_cap": isinstance(candidate, dict)
        and _equal_number(candidate.get("max_gpu_mem_gb"), 11),
        "candidate_commit": isinstance(candidate, dict)
        and candidate.get("git_commit") == source_commit,
        "no_runtime_contract_mismatch": not (
            mismatch_fields & FORBIDDEN_RUNTIME_MISMATCH_FIELDS
        ),
    }
    _expect(
        errors,
        all(checks.values()),
        "historical_comparison_contract_failed",
        stage=label,
        checks=checks,
    )


def _check_receipt(
    errors: list[dict[str, Any]],
    *,
    root: Path,
    path: Path,
    expected_paths: set[Path],
    label: str,
    artifacts: dict[str, str],
) -> None:
    if not path.is_file():
        _expect(errors, False, "missing_stage_receipt", stage=label)
        return
    artifacts[f"{label}_receipt"] = _sha256(path)
    observed: set[Path] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        digest, separator, raw_path = line.partition("  ")
        if not separator or HASH_RE.fullmatch(digest) is None:
            _expect(
                errors,
                False,
                "invalid_stage_receipt_line",
                stage=label,
                line=line_number,
            )
            continue
        item = Path(raw_path).resolve()
        if not item.is_relative_to(root) or not item.is_file():
            _expect(
                errors,
                False,
                "receipt_path_outside_root_or_missing",
                stage=label,
                path=str(item),
            )
            continue
        observed.add(item)
        _expect(
            errors,
            _sha256(item) == digest,
            "receipt_hash_mismatch",
            stage=label,
            path=str(item),
        )
    _expect(
        errors,
        observed == {item.resolve() for item in expected_paths},
        "receipt_population_mismatch",
        stage=label,
        observed=sorted(str(item) for item in observed),
    )


def _load_object(
    path: Path,
    errors: list[dict[str, Any]],
    label: str,
    artifacts: dict[str, str],
) -> dict[str, Any]:
    if not path.is_file():
        _expect(errors, False, "missing_json_artifact", artifact=label)
        return {}
    artifacts[label] = _sha256(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _expect(
            errors,
            False,
            "invalid_json_artifact",
            artifact=label,
            error=type(exc).__name__,
        )
        return {}
    if not isinstance(payload, dict):
        _expect(errors, False, "json_artifact_not_object", artifact=label)
        return {}
    return payload


def _check_text_marker(
    errors: list[dict[str, Any]], path: Path, expected: str, label: str
) -> None:
    actual = path.read_text(encoding="utf-8").strip() if path.is_file() else None
    _expect(
        errors,
        actual == expected,
        f"{label}_marker_mismatch",
        observed=actual,
    )


def _expect(
    errors: list[dict[str, Any]], condition: bool, code: str, **details: Any
) -> None:
    if not condition:
        errors.append({"code": code, **details})


def _positive_integer_set(value: Any) -> set[int]:
    if not isinstance(value, list):
        return set()
    result: set[int] = set()
    for item in value:
        try:
            integer = int(item)
        except (TypeError, ValueError):
            return set()
        if integer <= 0:
            return set()
        result.add(integer)
    return result


def _runtime_fingerprint_matches(value: Any, model_name: str) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("packages") != RUNTIME_PACKAGES:
        return False
    raw_sources = value.get("source_files")
    if not isinstance(raw_sources, list):
        return False
    observed_sources = {
        suffix: next(
            (
                str(item.get("sha256", ""))
                for item in raw_sources
                if isinstance(item, dict)
                and str(item.get("path", "")).replace("\\", "/").endswith(suffix)
            ),
            "",
        )
        for suffix in RUNTIME_SOURCE_HASHES
    }
    if observed_sources != RUNTIME_SOURCE_HASHES:
        return False
    raw_metadata = value.get("model_metadata_files")
    if not isinstance(raw_metadata, list):
        return False
    observed_metadata = {
        str(item.get("name", "")): str(item.get("sha256", ""))
        for item in raw_metadata
        if isinstance(item, dict)
    }
    expected_metadata = MODEL_METADATA_HASHES.get(model_name)
    if expected_metadata is None or any(
        observed_metadata.get(name) != digest
        for name, digest in expected_metadata.items()
    ):
        return False
    return HASH_RE.fullmatch(str(value.get("model_weight_layout_sha256", ""))) is not None


def _equal_number(value: Any, expected: float) -> bool:
    try:
        observed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(observed) and math.isclose(observed, expected, abs_tol=1e-12)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit(value: str, label: str) -> str:
    if COMMIT_RE.fullmatch(str(value)) is None:
        raise ValueError(f"{label} is not an exact Git commit")
    return str(value)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--control-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--write-marker", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
