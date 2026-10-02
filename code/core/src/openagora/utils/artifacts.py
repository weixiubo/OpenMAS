"""Artifact manifest/hash/export helpers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from openagora.types import RunResult
from openagora.utils.hash_utils import sha256_file, sha256_json


def build_file_hashes(paths: list[Path]) -> dict[str, str]:
    """Build sha256 map for existing files."""
    out: dict[str, str] = {}
    for path in paths:
        if path.exists() and path.is_file():
            out[path.name] = sha256_file(str(path))
    return out


def write_hash_manifest(path: Path, hashes: dict[str, str]) -> None:
    """Write deterministic hash manifest JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"hashes": hashes, "manifest_hash": sha256_json(hashes)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fingerprint_run_result(result: RunResult) -> str:
    """Return stable fingerprint for deterministic replay checks."""
    payload = {
        "task_id": result.task_id,
        "final_architecture": result.final_architecture.model_dump(mode="json"),
        "result": result.result,
        "ot_alignment": result.ot_alignment.model_dump(mode="json"),
        "metrics": result.metrics,
        "cache_used": result.cache_used,
        "negotiation_rounds": result.negotiation_rounds,
        "reconfig_count": result.reconfig_count,
        "override_used": result.override_used,
        "deadlock_breaker": result.deadlock_breaker,
    }
    return sha256_json(payload)


def export_run_bundle(
    *,
    run_id: str,
    manifest_path: Path,
    events_path: Path,
    result_path: Path,
    output_path: Path,
) -> Path:
    """Export run bundle to zip including hash manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hashes = build_file_hashes([manifest_path, events_path, result_path])
    hash_blob = json.dumps({"hashes": hashes}, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, arcname=f"{run_id}/manifest.json")
        zf.write(events_path, arcname=f"{run_id}/events.jsonl")
        zf.write(result_path, arcname=f"{run_id}/result.json")
        zf.writestr(f"{run_id}/hashes.json", hash_blob)

    return output_path


def read_run_result(path: Path) -> RunResult:
    """Load RunResult JSON from file."""
    return RunResult.model_validate_json(path.read_text(encoding="utf-8"))
