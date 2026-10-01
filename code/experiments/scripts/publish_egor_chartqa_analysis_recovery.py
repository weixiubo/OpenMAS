#!/usr/bin/env python3
"""Atomically publish a validated ChartQA analysis recovery for downstream use."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from openagora_experiments.egor.analysis_recovery import sha256, verify_snapshot


def publish_recovery(
    *,
    result_root: Path,
    recovery_root: Path,
    generation_commit: str,
    recovery_commit: str,
    amendment_commit: str,
    expected_tasks: int,
) -> dict[str, Any]:
    result_root = result_root.resolve()
    recovery_root = recovery_root.resolve()
    try:
        recovery_root.relative_to(result_root)
    except ValueError as error:
        raise ValueError(
            "recovery root must be inside the generation result root"
        ) from error
    _require_file(result_root / "GENERATION_COMPLETE")
    _require_file(recovery_root / "RECOVERY_COMPLETE")
    _require_commit(result_root / "SOURCE_COMMIT", generation_commit)
    _require_commit(recovery_root / "SOURCE_COMMIT", generation_commit)
    _require_commit(
        recovery_root / "ANALYSIS_RECOVERY_SOURCE_COMMIT", recovery_commit
    )
    _require_commit(
        recovery_root / "ANALYSIS_AMENDMENT_SOURCE_COMMIT", amendment_commit
    )

    recovery_manifest_path = recovery_root / "analysis_recovery_manifest.json"
    recovery = _load_object(recovery_manifest_path)
    if (
        recovery.get("strict_analysis_integrity_passed") is not True
        or recovery.get("protected_artifacts_unchanged") is not True
        or recovery.get("paper_gate_passed") is not True
        or recovery.get("publication_ready") is not True
        or recovery.get("generation_source_commit") != generation_commit
        or recovery.get("analysis_recovery_commit") != recovery_commit
        or recovery.get("analysis_amendment_commit") != amendment_commit
        or int(recovery.get("expected_tasks", -1)) != expected_tasks
    ):
        raise ValueError("analysis recovery is not publication-ready")
    _validate_artifacts(recovery.get("artifacts"), within=result_root)
    protected_snapshot = _load_object(
        recovery_root / "protected_artifacts_snapshot.json"
    )
    verify_snapshot(protected_snapshot)

    recovery_gate_path = recovery_root / "paper_gate_report.json"
    recovery_gate = _load_object(recovery_gate_path)
    if (
        recovery_gate.get("passed") is not True
        or recovery_gate.get("source_commit") != generation_commit
        or int(recovery_gate.get("tasks", -1)) != expected_tasks
    ):
        raise ValueError("recovered paper gate did not pass")

    published_gate_path = result_root / "paper_gate_report.json"
    _copy_once(recovery_gate_path, published_gate_path)
    bound_paths = [
        published_gate_path,
        recovery_manifest_path,
        recovery_root / "analysis_amendment_manifest.json",
        recovery_root / "policy_bundle.csv",
        recovery_root / "policy_bundle.csv.manifest.json",
        recovery_root / "paired_statistics.json",
        recovery_root / "matrix_analysis" / "manifest.json",
        recovery_root / "evidence_alignment_audit" / "population_manifest.json",
        recovery_root / "protected_artifacts_snapshot.json",
    ]
    release = {
        "schema_version": 2,
        "generation_source_commit": generation_commit,
        "analysis_commit": amendment_commit,
        "analysis_recovery_commit": recovery_commit,
        "expected_tasks": expected_tasks,
        "integrity_passed": True,
        "paper_gate_passed": True,
        "recovery_root": str(recovery_root),
        "artifacts": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in bound_paths
        ],
    }
    manifest_path = result_root / "analysis_amendment_manifest.json"
    _write_json_once(manifest_path, release)
    _write_text_once(
        result_root / "ANALYSIS_RECOVERY_SOURCE_COMMIT", recovery_commit + "\n"
    )
    _write_text_once(
        result_root / "ANALYSIS_AMENDMENT_SOURCE_COMMIT", amendment_commit + "\n"
    )
    verify_snapshot(protected_snapshot)
    for marker in (
        "ANALYSIS_RECOVERY_COMPLETE",
        "ANALYSIS_COMPLETE",
        "ANALYSIS_AMENDMENT_COMPLETE",
    ):
        _touch_once(result_root / marker)
    print("published validated ChartQA analysis recovery")
    return release


def _copy_once(source: Path, destination: Path) -> None:
    if destination.exists():
        _require_file(destination)
        if sha256(destination) != sha256(source):
            raise ValueError(f"refusing to overwrite different file: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale publication temporary file: {temporary}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_text_once(path, text)


def _write_text_once(path: Path, text: str) -> None:
    if path.exists():
        _require_file(path)
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"refusing to overwrite different file: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale publication temporary file: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _touch_once(path: Path) -> None:
    if path.exists():
        _require_file(path)
        return
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.close(descriptor)


def _validate_artifacts(raw: Any, *, within: Path) -> None:
    if not isinstance(raw, list) or not raw:
        raise ValueError("recovery manifest has no artifacts")
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("malformed recovery artifact entry")
        path = Path(str(entry.get("path", ""))).resolve()
        try:
            path.relative_to(within)
        except ValueError as error:
            raise ValueError(
                f"recovery artifact escapes result root: {path}"
            ) from error
        _require_file(path)
        if sha256(path) != str(entry.get("sha256", "")):
            raise ValueError(f"recovery artifact SHA-256 mismatch: {path}")


def _require_commit(path: Path, expected: str) -> None:
    _require_file(path)
    if path.read_text(encoding="utf-8").strip() != expected:
        raise ValueError(f"commit mismatch: {path}")


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
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--generation-source-commit", required=True)
    parser.add_argument("--recovery-commit", required=True)
    parser.add_argument("--amendment-commit", required=True)
    parser.add_argument("--expected-tasks", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    publish_recovery(
        result_root=Path(args.result_root),
        recovery_root=Path(args.recovery_root),
        generation_commit=args.generation_source_commit,
        recovery_commit=args.recovery_commit,
        amendment_commit=args.amendment_commit,
        expected_tasks=args.expected_tasks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
