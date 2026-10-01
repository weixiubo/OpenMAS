#!/usr/bin/env python3
"""Audit the frozen EGOR-v10 split without using labels or outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from build_egor_v9_confirmation_data import collect_exclusions
from build_egor_v10_confirmation_data import (
    CHARTQA_CONFIRMATION_QUOTA,
    CHARTQA_FORMAT_QUOTA,
    OCRBENCH_CONFIRMATION_QUOTA,
)


EXPECTED_COUNTS = {
    "chartqa": {"format": 20, "confirmation": 230},
    "ocrbench": {"format": 30, "confirmation": 570},
}
EXPECTED_STRATA = {
    "chartqa": {
        "format": CHARTQA_FORMAT_QUOTA,
        "confirmation": CHARTQA_CONFIRMATION_QUOTA,
    },
    "ocrbench": {
        "format": {task_type: 5 for task_type in OCRBENCH_CONFIRMATION_QUOTA},
        "confirmation": OCRBENCH_CONFIRMATION_QUOTA,
    },
}
EXPECTED_BENCHMARKS = {
    "chartqa": "ChartQA-EGOR-V10-Fresh",
    "ocrbench": "OCRBench-v2-English-VQA-EGOR-V10-Fresh",
}
EXPECTED_METRICS = {"chartqa": "relaxed_accuracy", "ocrbench": "ocrbench_v2_vqa"}
EXPECTED_OFFICIAL_EVAL_COMMIT = "9ec46950c87e1b683cd3af9e6f6d45dea088ecee"


def audit_split(*, root: Path, exclusion_list: Path) -> tuple[dict[str, Any], int]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exclusion_paths = [
        Path(line).resolve()
        for line in exclusion_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    exclusions = collect_exclusions(exclusion_paths)

    all_rows: list[tuple[str, str, Mapping[str, Any]]] = []
    counts: dict[str, dict[str, int]] = {}
    strata: dict[str, dict[str, dict[str, int]]] = {}
    task_receipts_match = True
    for source in ("chartqa", "ocrbench"):
        counts[source] = {}
        strata[source] = {}
        for split in ("format", "confirmation"):
            receipt = manifest["sources"][source][split]
            path = Path(receipt["path"]).resolve()
            rows = _jsonl_rows(path)
            counts[source][split] = len(rows)
            stratum_key = "source_subset" if source == "chartqa" else "official_type"
            strata[source][split] = dict(
                Counter(str(row["metadata"][stratum_key]) for row in rows)
            )
            task_receipts_match = task_receipts_match and all(
                (
                    _is_within(path, root),
                    receipt["rows"] == len(rows),
                    receipt["sha256"] == _sha256(path),
                )
            )
            all_rows.extend((source, split, row) for row in rows)

    task_keys: list[tuple[str, str]] = []
    image_hashes: list[str] = []
    prior_overlap = Counter()
    image_integrity = True
    image_paths_inside_root = True
    row_contracts = True
    for source, _, row in all_rows:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, Mapping):
            row_contracts = False
            continue
        benchmark = str(row.get("benchmark", ""))
        task_id = str(row.get("task_id", ""))
        question = " ".join(str(row.get("question", "")).split()).strip()
        image_path = Path(str(row.get("image_path", ""))).resolve()
        recorded_hash = str(metadata.get("source_image_sha256", ""))
        task_keys.append((benchmark, task_id))
        image_hashes.append(recorded_hash)
        row_contracts = row_contracts and all(
            (
                benchmark == EXPECTED_BENCHMARKS[source],
                str(row.get("metric", "")) == EXPECTED_METRICS[source],
                bool(task_id),
                bool(question),
                len(recorded_hash) == 64,
                metadata.get("egor_v10_fresh_confirmation") is True,
            )
        )
        image_paths_inside_root = image_paths_inside_root and _is_within(image_path, root)
        image_integrity = image_integrity and image_path.is_file()
        if image_path.is_file():
            image_integrity = image_integrity and _sha256(image_path) == recorded_hash
        question_hash = hashlib.sha256(question.encode()).hexdigest()
        prior_overlap["question"] += question_hash in exclusions.question_hashes
        prior_overlap["image_hash"] += recorded_hash in exclusions.image_hashes
        if source == "chartqa":
            prior_overlap["image_basename"] += (
                str(metadata.get("source_image", "")) in exclusions.image_basenames
            )
        else:
            prior_overlap["source_id"] += (
                int(metadata.get("source_id", -1)) in exclusions.source_ids
            )

    declared_paths = sorted(str(path) for path in exclusion_paths)
    manifest_exclusions = manifest["exclusions"]["files"]
    exclusion_paths_bound = sorted(item["path"] for item in manifest_exclusions) == declared_paths
    exclusion_hashes_valid = all(
        Path(item["path"]).is_file()
        and _sha256(Path(item["path"])) == item["sha256"]
        for item in manifest_exclusions
    )
    checks = {
        "split_frozen": (root / "SPLIT_FROZEN").is_file(),
        "split_checksums": _verify_sums(root / "SPLIT_SHA256SUMS"),
        "image_checksums": _verify_sums(root / "IMAGE_SHA256SUMS"),
        "manifest_outcome_blind": manifest.get("outcome_fields_used_for_selection") is False,
        "denominators": counts == EXPECTED_COUNTS,
        "exact_stratum_quotas": strata == EXPECTED_STRATA,
        "task_receipts_match": task_receipts_match,
        "official_evaluator_bound": (
            manifest["sources"]["ocrbench"]["official_evaluator"]["commit"]
            == EXPECTED_OFFICIAL_EVAL_COMMIT
        ),
        "row_contracts": row_contracts,
        "unique_task_keys": len(task_keys) == 850 == len(set(task_keys)),
        "unique_images": len(image_hashes) == 850 == len(set(image_hashes)),
        "manifest_image_count": manifest.get("image_files") == 850,
        "image_paths_inside_root": image_paths_inside_root,
        "materialized_image_integrity": image_integrity,
        "exclusion_list_bound": exclusion_paths_bound,
        "exclusion_file_hashes": exclusion_hashes_valid,
        "source_file_hashes": _source_hashes_valid(manifest),
        "zero_prior_overlap": not any(prior_overlap.values()),
    }
    report = {
        "schema_version": 1,
        "claim_boundary": "outcome_blind_fresh_v10_split_integrity_audit",
        "fields_used_for_audit": [
            "benchmark",
            "task_id",
            "question",
            "metric",
            "image_path",
            "metadata source identifiers and strata",
        ],
        "outcome_fields_used": False,
        "scores_computed": False,
        "manifest_sha256": _sha256(manifest_path),
        "exclusion_list_sha256": _sha256(exclusion_list),
        "exclusion_files": len(exclusion_paths),
        "rows": len(all_rows),
        "counts": counts,
        "strata": strata,
        "prior_overlap_counts": dict(prior_overlap),
        "checks": checks,
        "passed": all(checks.values()),
    }
    return report, 0 if report["passed"] else 2


def _jsonl_rows(path: Path) -> list[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sums(path: Path) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, separator, raw_path = line.partition("  ")
        candidate = Path(raw_path)
        if not separator or not candidate.is_file() or _sha256(candidate) != expected:
            return False
    return True


def _source_hashes_valid(manifest: Mapping[str, Any]) -> bool:
    receipts: Sequence[Mapping[str, Any]] = [
        manifest["sources"]["chartqa"]["source_arrow"],
        *manifest["sources"]["ocrbench"]["source_arrows"],
    ]
    return all(
        Path(item["path"]).is_file()
        and Path(item["path"]).stat().st_size == item["bytes"]
        and _sha256(Path(item["path"])) == item["sha256"]
        for item in receipts
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def main() -> int:
    args = _parse_args()
    report, status = audit_split(
        root=Path(args.root).resolve(), exclusion_list=Path(args.exclusion_list).resolve()
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return status


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--exclusion-list", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
