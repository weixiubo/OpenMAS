#!/usr/bin/env python3
"""Audit fresh OCRBench-v2 capacity using only selection-visible fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from build_egor_v9_confirmation_data import (
    OCRBENCH_V9_TYPES,
    collect_exclusions,
)
from openagora_experiments.egor.ocrbench_splits import (
    OCRBenchCandidate,
    sha256,
)
from openagora_experiments.egor.ocrbench_v9_prior_filter import (
    filter_prior_questions,
)


VISIBLE_COLUMNS = ("image", "question", "dataset_name", "type", "id")


def audit_capacity(*, arrow_dir: Path, exclusion_list: Path) -> dict[str, Any]:
    exclusion_paths = [
        Path(line).resolve()
        for line in exclusion_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    exclusions = collect_exclusions(exclusion_paths)
    arrow_paths = sorted(arrow_dir.glob("*.arrow"))
    if not arrow_paths:
        raise FileNotFoundError(f"no Arrow shards in {arrow_dir}")

    candidates = _visible_candidates(arrow_paths)
    question_fresh, question_audit = filter_prior_questions(
        candidates,
        excluded_question_hashes=exclusions.question_hashes,
    )
    eligible = [
        row
        for row in question_fresh
        if row.source_id not in exclusions.source_ids
        and row.image_sha256 not in exclusions.image_hashes
    ]
    counts = Counter(row.task_type for row in eligible)
    unique_images = {
        task_type: len(
            {row.image_sha256 for row in eligible if row.task_type == task_type}
        )
        for task_type in sorted(counts)
    }
    return {
        "schema_version": 1,
        "claim_boundary": "outcome_blind_capacity_audit",
        "selection_visible_source_columns": list(VISIBLE_COLUMNS),
        "outcome_fields_used_for_selection_or_capacity": False,
        "scores_computed": False,
        "source_rows": len(candidates),
        **question_audit,
        "eligible_rows_by_type": dict(sorted(counts.items())),
        "eligible_unique_images_by_type": unique_images,
        "frozen_v9_types": list(OCRBENCH_V9_TYPES),
        "frozen_v9_unique_image_capacity": {
            task_type: unique_images.get(task_type, 0)
            for task_type in OCRBENCH_V9_TYPES
        },
        "infeasible_original_types": {
            task_type: unique_images.get(task_type, 0)
            for task_type in (
                "document classification en",
                "text recognition en",
            )
        },
        "exclusions": {
            "list_path": str(exclusion_list.resolve()),
            "list_sha256": sha256(exclusion_list),
            "files": len(exclusion_paths),
            "source_ids": len(exclusions.source_ids),
            "image_hashes": len(exclusions.image_hashes),
            "question_hashes": len(exclusions.question_hashes),
        },
        "source_arrows": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in arrow_paths
        ],
    }


def _visible_candidates(paths: list[Path]) -> list[OCRBenchCandidate]:
    try:
        from datasets import Dataset, Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("capacity audit requires the datasets package") from exc

    output: list[OCRBenchCandidate] = []
    for path in paths:
        dataset = Dataset.from_file(str(path)).select_columns(list(VISIBLE_COLUMNS))
        dataset = dataset.cast_column("image", Image(decode=False))
        for row_index in range(len(dataset)):
            row = dataset[row_index]
            payload = _image_bytes(row.get("image"), base_dir=path.parent)
            output.append(
                OCRBenchCandidate(
                    source_id=int(row["id"]),
                    question=str(row.get("question", "")).strip(),
                    answers=(),
                    dataset_name=str(row.get("dataset_name", "")).strip(),
                    task_type=str(row.get("type", "")).strip(),
                    image_sha256=hashlib.sha256(payload).hexdigest(),
                    shard_path=str(path),
                    row_index=row_index,
                )
            )
    return output


def _image_bytes(value: Any, *, base_dir: Path) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, Mapping):
        payload = value.get("bytes")
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        raw_path = value.get("path")
        if isinstance(raw_path, str) and raw_path:
            path = Path(raw_path)
            if not path.is_absolute():
                path = base_dir / path
            return path.read_bytes()
    raise ValueError("image is neither encoded bytes nor a readable path")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocrbench-arrow-dir", required=True)
    parser.add_argument("--exclusion-list", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = audit_capacity(
        arrow_dir=Path(args.ocrbench_arrow_dir).resolve(),
        exclusion_list=Path(args.exclusion_list).resolve(),
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
