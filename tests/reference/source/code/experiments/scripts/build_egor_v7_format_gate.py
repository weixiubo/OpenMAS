#!/usr/bin/env python3
"""Combine two outcome-blind v7 format audits without opening any score."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_TASK_HASHES = {
    "chartqa": "92db333e6e93e1e1a34da43ef1f66afabf821a393fd0055cee89c6717440b603",
    "infographic": "ff81d0c0ba91673ae3aaa50819f833aa4f907bc51700ce2a561cbf9a11519506",
}


def main() -> int:
    args = _parse_args()
    report = build_gate(Path(args.root).resolve(), args.source_commit)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


def build_gate(root: Path, source_commit: str) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for name, expected_hash in EXPECTED_TASK_HASHES.items():
        source_root = root / "sources" / name
        audit_path = source_root / "audit.json"
        audit = (
            json.loads(audit_path.read_text(encoding="utf-8"))
            if audit_path.is_file()
            else {}
        )
        actual_commit = _read(source_root / "SOURCE_COMMIT")
        actual_hash = _read(source_root / "TASKS_SHA256")
        checks = {
            "source_complete": (source_root / "SOURCE_GENERATION_COMPLETE").is_file(),
            "format_complete": (source_root / "SOURCE_FORMAT_COMPLETE").is_file(),
            "format_not_failed": not (source_root / "SOURCE_FORMAT_FAILED").exists(),
            "generation_checksums": _verify_sha256sums(
                source_root / "SOURCE_GENERATION_SHA256SUMS", source_root
            ),
            "source_commit": actual_commit == source_commit,
            "task_hash": actual_hash == expected_hash,
            "audit_passed": audit.get("passed") is True,
            "audit_mode": audit.get("mode") == "format",
            "audit_outcome_blind": audit.get("outcome_fields_read") is False
            and audit.get("scores_computed") is False,
            "audit_commit": audit.get("source_commit") == source_commit,
        }
        sources[name] = {
            "passed": all(checks.values()),
            "checks": checks,
            "audit_sha256": _sha256(audit_path) if audit_path.is_file() else "",
            "verifier_triggered": audit.get("verifier_triggered"),
            "verifier_operational_completion": audit.get(
                "verifier_operational_completion"
            ),
        }
    return {
        "schema_version": 1,
        "claim_boundary": "unscored_format_calibration_only",
        "source_commit": source_commit,
        "outcome_blind": True,
        "scores_computed": False,
        "sources": sources,
        "passed": all(item["passed"] for item in sources.values()),
    }


def _verify_sha256sums(path: Path, root: Path) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, raw_path = line.partition("  ")
        if not separator or len(digest) != 64:
            return False
        candidate = Path(raw_path).resolve()
        if root.resolve() not in candidate.parents or not candidate.is_file():
            return False
        if _sha256(candidate) != digest:
            return False
    return True


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
