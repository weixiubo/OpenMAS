#!/usr/bin/env python3
"""Combine two outcome-blind EGOR-v10 format audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def build_gate(
    root: Path, source_commit: str, expected_task_hashes: dict[str, str]
) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for name, expected_hash in expected_task_hashes.items():
        source_root = root / "sources" / name
        audit_path = source_root / "audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
        checks = {
            "source_complete": (source_root / "SOURCE_GENERATION_COMPLETE").is_file(),
            "format_complete": (source_root / "SOURCE_FORMAT_COMPLETE").is_file(),
            "format_not_failed": not (source_root / "SOURCE_FORMAT_FAILED").exists(),
            "generation_checksums": _verify_sha256sums(
                source_root / "SOURCE_GENERATION_SHA256SUMS", source_root
            ),
            "source_commit": _read(source_root / "SOURCE_COMMIT") == source_commit,
            "task_hash": _read(source_root / "TASKS_SHA256") == expected_hash,
            "audit_passed": audit.get("passed") is True,
            "audit_mode": audit.get("mode") == "format",
            "audit_outcome_blind": audit.get("outcome_fields_read") is False
            and audit.get("scores_computed") is False,
            "audit_commit": audit.get("source_commit") == source_commit,
            "resolver_stage": audit.get("resolver_contract", {}).get("stage")
            == "verifier_v8",
            "resolver_file": audit.get("resolver_contract", {}).get("file")
            == "resolver_v10_qwen35.jsonl",
            "resolver_ids": audit.get("resolver_contract", {}).get("ids")
            == "repairability_triggered",
            "specialist_ids": audit.get("specialist_contract", {}).get("ids")
            == "resolver_changed",
            "specialists_blind_to_resolver": audit.get("specialist_contract", {}).get(
                "resolver_candidate_visible_to_specialists"
            )
            is False,
        }
        sources[name] = {
            "passed": all(checks.values()),
            "checks": checks,
            "audit_sha256": _sha256(audit_path) if audit_path.is_file() else "",
            "repairability_triggered": audit.get("repairability_triggered"),
            "resolver_changed": audit.get("resolver_changed"),
            "resolver_operational_completion": audit.get(
                "resolver_operational_completion"
            ),
            "resolver_native_transport_completion": audit.get(
                "resolver_native_transport_completion"
            ),
            "resolver_sanitized_abstentions": audit.get(
                "resolver_sanitized_abstentions"
            ),
        }
    return {
        "schema_version": 1,
        "claim_boundary": "unscored_v10_format_calibration_only",
        "source_commit": source_commit,
        "method": "egor_v10_two_support_blind_resolver",
        "outcome_blind": True,
        "scores_computed": False,
        "sources": sources,
        "passed": set(sources) == {"chartqa", "ocrbench"}
        and all(item["passed"] for item in sources.values()),
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


def main() -> int:
    args = _parse_args()
    report = build_gate(
        Path(args.root).resolve(),
        args.source_commit,
        {
            "chartqa": args.chartqa_tasks_sha256,
            "ocrbench": args.ocrbench_tasks_sha256,
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--chartqa-tasks-sha256", required=True)
    parser.add_argument("--ocrbench-tasks-sha256", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
