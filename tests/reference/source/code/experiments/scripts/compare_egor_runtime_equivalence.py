#!/usr/bin/env python3
"""Compare EGOR records across runtime placements without reading outcomes.

The comparator is intentionally outcome-blind: it consumes generation records
and their manifests, never task references or scores.  Runtime-only quantities
(latency, output path, GPU placement, and per-device memory cap) may differ;
the task order, prompts, seeds, raw generations, parsed records, token counts,
and all other audit fields must match exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


COMMIT_RE = re.compile(r"[0-9a-f]{40}")
ROW_KEY_FIELDS = ("source_id", "task_id")
ROW_EXACT_FIELDS = (
    "schema_version",
    "source_id",
    "task_id",
    "stage",
    "model_name",
    "seed",
    "request_id",
    "image_path",
    "system_prompt_sha256",
    "user_prompt_sha256",
    "status",
    "raw_output",
    "parsed",
    "parse_issues",
)
USAGE_EXACT_FIELDS = (
    "attempts",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)
AUDIT_RUNTIME_FIELDS = {"latency_ms", "gpu_devices", "max_gpu_mem_gb"}
MANIFEST_EXACT_FIELDS = (
    "stage",
    "tasks",
    "tasks_sha256",
    "model_name",
    "model_path",
    "model_profile",
    "seed",
    "requested",
    "cublas_workspace_config",
    "answer_leakage_policy",
)
MAX_REPORTED_MISMATCHES = 100


def main() -> int:
    args = _parse_args()
    reference_path = Path(args.reference).resolve()
    candidate_path = Path(args.candidate).resolve()
    reference_manifest = _manifest_path(
        args.reference_manifest, reference_path
    )
    candidate_manifest = _manifest_path(
        args.candidate_manifest, candidate_path
    )
    report = compare_runtime_equivalence(
        reference_records=_load_jsonl(reference_path),
        candidate_records=_load_jsonl(candidate_path),
        reference_manifest=_load_object(reference_manifest),
        candidate_manifest=_load_object(candidate_manifest),
        reference_sha256=_sha256(reference_path),
        candidate_sha256=_sha256(candidate_path),
        reference_manifest_sha256=_sha256(reference_manifest),
        candidate_manifest_sha256=_sha256(candidate_manifest),
        expected_tasks=args.expected_tasks,
        expected_candidate_gpu_devices=args.expected_candidate_gpu_devices,
        expected_candidate_max_gpu_mem_gb=args.expected_candidate_max_gpu_mem_gb,
        expected_candidate_commit=args.expected_candidate_commit,
    )
    output = Path(args.output).resolve()
    _write_json_atomic(output, report)
    print(
        "EGOR runtime equivalence "
        f"{'passed' if report['passed'] else 'failed'}: {output}"
    )
    if args.require_pass and not report["passed"]:
        raise SystemExit("runtime equivalence gate failed")
    return 0


def compare_runtime_equivalence(
    *,
    reference_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    reference_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
    reference_sha256: str,
    candidate_sha256: str,
    reference_manifest_sha256: str,
    candidate_manifest_sha256: str,
    expected_tasks: int,
    expected_candidate_gpu_devices: str,
    expected_candidate_max_gpu_mem_gb: float,
    expected_candidate_commit: str | None,
) -> dict[str, Any]:
    if expected_tasks <= 0:
        raise ValueError("expected task count must be positive")
    if not expected_candidate_gpu_devices.strip():
        raise ValueError("expected candidate GPU devices are required")
    if expected_candidate_max_gpu_mem_gb <= 0:
        raise ValueError("expected candidate memory cap must be positive")
    if expected_candidate_commit is not None:
        _commit(expected_candidate_commit, "expected candidate commit")
    for value, label in (
        (reference_sha256, "reference records hash"),
        (candidate_sha256, "candidate records hash"),
        (reference_manifest_sha256, "reference manifest hash"),
        (candidate_manifest_sha256, "candidate manifest hash"),
    ):
        _hash(value, label)

    reference_keys = _ordered_keys(reference_records, "reference")
    candidate_keys = _ordered_keys(candidate_records, "candidate")
    mismatches: list[dict[str, Any]] = []
    _compare_value(
        mismatches,
        scope="population",
        field="reference_task_count",
        reference=expected_tasks,
        candidate=len(reference_records),
    )
    _compare_value(
        mismatches,
        scope="population",
        field="candidate_task_count",
        reference=expected_tasks,
        candidate=len(candidate_records),
    )
    _compare_value(
        mismatches,
        scope="population",
        field="ordered_task_keys",
        reference=reference_keys,
        candidate=candidate_keys,
    )

    candidate_by_key = {
        key: row for key, row in zip(candidate_keys, candidate_records, strict=True)
    }
    for key, reference in zip(reference_keys, reference_records, strict=True):
        candidate = candidate_by_key.get(key)
        if candidate is None:
            continue
        scope = f"row:{key[0]}::{key[1]}"
        if reference.get("status") != "ok" or candidate.get("status") != "ok":
            _record_mismatch(
                mismatches,
                scope=scope,
                field="successful_status_required",
                reference=reference.get("status"),
                candidate=candidate.get("status"),
            )
        for field in ROW_EXACT_FIELDS:
            _compare_value(
                mismatches,
                scope=scope,
                field=field,
                reference=reference.get(field),
                candidate=candidate.get(field),
            )
        _compare_usage(
            mismatches,
            scope=scope,
            reference=reference.get("usage"),
            candidate=candidate.get("usage"),
        )
        _compare_audit_events(
            mismatches,
            scope=scope,
            reference=reference.get("audit_events"),
            candidate=candidate.get("audit_events"),
        )

    _compare_manifests(
        mismatches,
        reference=reference_manifest,
        candidate=candidate_manifest,
        expected_tasks=expected_tasks,
        expected_candidate_gpu_devices=expected_candidate_gpu_devices,
        expected_candidate_max_gpu_mem_gb=expected_candidate_max_gpu_mem_gb,
        expected_candidate_commit=expected_candidate_commit,
    )
    passed = not mismatches
    return {
        "schema_version": 1,
        "outcome_blind": True,
        "passed": passed,
        "tasks": expected_tasks,
        "exact_row_fields": list(ROW_EXACT_FIELDS),
        "exact_usage_fields": list(USAGE_EXACT_FIELDS),
        "ignored_runtime_fields": {
            "row": ["usage.latency_seconds"],
            "audit_event": sorted(AUDIT_RUNTIME_FIELDS),
            "manifest": [
                "created_at",
                "output",
                "gpu_devices",
                "max_gpu_mem_gb",
                "attempted_this_run",
                "successful_this_run",
                "successful_total",
            ],
        },
        "candidate_runtime": {
            "gpu_devices": candidate_manifest.get("gpu_devices"),
            "max_gpu_mem_gb": candidate_manifest.get("max_gpu_mem_gb"),
            "git_commit": candidate_manifest.get("git_commit"),
        },
        "artifacts": {
            "reference_records_sha256": reference_sha256,
            "candidate_records_sha256": candidate_sha256,
            "reference_manifest_sha256": reference_manifest_sha256,
            "candidate_manifest_sha256": candidate_manifest_sha256,
        },
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:MAX_REPORTED_MISMATCHES],
        "mismatches_truncated": len(mismatches) > MAX_REPORTED_MISMATCHES,
    }


def _compare_usage(
    mismatches: list[dict[str, Any]],
    *,
    scope: str,
    reference: Any,
    candidate: Any,
) -> None:
    if not isinstance(reference, dict) or not isinstance(candidate, dict):
        _record_mismatch(
            mismatches,
            scope=scope,
            field="usage_object",
            reference=reference,
            candidate=candidate,
        )
        return
    for field in USAGE_EXACT_FIELDS:
        _compare_value(
            mismatches,
            scope=scope,
            field=f"usage.{field}",
            reference=reference.get(field),
            candidate=candidate.get(field),
        )


def _compare_audit_events(
    mismatches: list[dict[str, Any]],
    *,
    scope: str,
    reference: Any,
    candidate: Any,
) -> None:
    if not isinstance(reference, list) or not isinstance(candidate, list):
        _record_mismatch(
            mismatches,
            scope=scope,
            field="audit_events_list",
            reference=reference,
            candidate=candidate,
        )
        return
    normalized_reference = [_normalize_audit_event(item) for item in reference]
    normalized_candidate = [_normalize_audit_event(item) for item in candidate]
    _compare_value(
        mismatches,
        scope=scope,
        field="audit_events_without_runtime_fields",
        reference=normalized_reference,
        candidate=normalized_candidate,
    )


def _normalize_audit_event(raw: Any) -> Any:
    if not isinstance(raw, dict):
        return raw
    return {key: value for key, value in raw.items() if key not in AUDIT_RUNTIME_FIELDS}


def _compare_manifests(
    mismatches: list[dict[str, Any]],
    *,
    reference: dict[str, Any],
    candidate: dict[str, Any],
    expected_tasks: int,
    expected_candidate_gpu_devices: str,
    expected_candidate_max_gpu_mem_gb: float,
    expected_candidate_commit: str | None,
) -> None:
    for field in MANIFEST_EXACT_FIELDS:
        if field not in reference and field not in candidate:
            continue
        _compare_value(
            mismatches,
            scope="manifest",
            field=field,
            reference=reference.get(field),
            candidate=candidate.get(field),
        )
    _compare_value(
        mismatches,
        scope="manifest",
        field="reference_requested",
        reference=expected_tasks,
        candidate=reference.get("requested"),
    )
    _compare_value(
        mismatches,
        scope="manifest",
        field="candidate_requested",
        reference=expected_tasks,
        candidate=candidate.get("requested"),
    )
    _compare_value(
        mismatches,
        scope="manifest",
        field="candidate_gpu_devices",
        reference=expected_candidate_gpu_devices,
        candidate=candidate.get("gpu_devices"),
    )
    candidate_cap = _finite_or_none(candidate.get("max_gpu_mem_gb"))
    if candidate_cap is None or not math.isclose(
        candidate_cap, expected_candidate_max_gpu_mem_gb, abs_tol=1e-12
    ):
        _record_mismatch(
            mismatches,
            scope="manifest",
            field="candidate_max_gpu_mem_gb",
            reference=expected_candidate_max_gpu_mem_gb,
            candidate=candidate.get("max_gpu_mem_gb"),
        )
    if expected_candidate_commit is not None:
        _compare_value(
            mismatches,
            scope="manifest",
            field="candidate_git_commit",
            reference=expected_candidate_commit,
            candidate=candidate.get("git_commit"),
        )

    reference_runtime = reference.get("runtime")
    candidate_runtime = candidate.get("runtime")
    if isinstance(reference_runtime, dict) and isinstance(candidate_runtime, dict):
        for field in (
            "model_metadata_files",
            "model_weight_layout",
            "model_weight_layout_sha256",
            "packages",
            "source_files",
        ):
            _compare_value(
                mismatches,
                scope="manifest.runtime",
                field=field,
                reference=reference_runtime.get(field),
                candidate=candidate_runtime.get(field),
            )


def _ordered_keys(rows: list[dict[str, Any]], label: str) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {index} is not an object")
        key = tuple(str(row.get(field, "")).strip() for field in ROW_KEY_FIELDS)
        if not all(key):
            raise ValueError(f"{label} row {index} lacks a task key")
        if key in seen:
            raise ValueError(f"{label} records contain duplicate task key: {key}")
        seen.add(key)
        keys.append(key)
    return keys


def _compare_value(
    mismatches: list[dict[str, Any]],
    *,
    scope: str,
    field: str,
    reference: Any,
    candidate: Any,
) -> None:
    if reference != candidate:
        _record_mismatch(
            mismatches,
            scope=scope,
            field=field,
            reference=reference,
            candidate=candidate,
        )


def _record_mismatch(
    mismatches: list[dict[str, Any]],
    *,
    scope: str,
    field: str,
    reference: Any,
    candidate: Any,
) -> None:
    mismatches.append(
        {
            "scope": scope,
            "field": field,
            "reference": _summary(reference),
            "candidate": _summary(candidate),
        }
    )


def _summary(value: Any) -> Any:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    if len(encoded) <= 240:
        return value
    return {
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "json_bytes": len(encoded.encode("utf-8")),
    }


def _manifest_path(value: str | None, records: Path) -> Path:
    if value:
        return Path(value).resolve()
    return records.with_suffix(records.suffix + ".manifest.json")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(payload)
    return rows


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(value: str, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", str(value)) is None:
        raise ValueError(f"{label} is not a SHA-256 digest")
    return str(value)


def _commit(value: str, label: str) -> str:
    if COMMIT_RE.fullmatch(str(value)) is None:
        raise ValueError(f"{label} is not an exact Git commit")
    return str(value)


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference-manifest")
    parser.add_argument("--candidate-manifest")
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--expected-candidate-gpu-devices", default="1,0")
    parser.add_argument(
        "--expected-candidate-max-gpu-mem-gb", type=float, default=11.0
    )
    parser.add_argument("--expected-candidate-commit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
