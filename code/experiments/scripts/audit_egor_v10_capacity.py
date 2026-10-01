#!/usr/bin/env python3
"""Report outcome-blind v10 source capacity after applying exclusions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_egor_v9_confirmation_data as shared
from openagora_experiments.egor.ocrbench_v9_prior_filter import filter_prior_questions


def main() -> int:
    args = _parse_args()
    exclusion_list = Path(args.exclusion_list).resolve()
    exclusion_paths = [
        Path(line).resolve()
        for line in exclusion_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    exclusions = shared.collect_exclusions(exclusion_paths)
    chart_candidates = shared._index_chartqa(Path(args.chartqa_arrow).resolve())
    _, chart_audit = shared.select_chartqa_rows(
        chart_candidates,
        exclusions=exclusions,
        per_type=1,
        salt="egor-v10-capacity-only",
    )
    ocr_dir = Path(args.ocrbench_arrow_dir).resolve()
    ocr_candidates = shared._index_ocrbench(sorted(ocr_dir.glob("*.arrow")))
    ocr_filtered, question_audit = filter_prior_questions(
        ocr_candidates, excluded_question_hashes=exclusions.question_hashes
    )
    eligible_images = {task_type: set() for task_type in shared.OCRBENCH_V9_TYPES}
    excluded_source_ids = 0
    excluded_image_hashes = 0
    for row in ocr_filtered:
        if row.task_type not in eligible_images:
            continue
        if row.source_id in exclusions.source_ids:
            excluded_source_ids += 1
            continue
        if row.image_sha256 in exclusions.image_hashes:
            excluded_image_hashes += 1
            continue
        eligible_images[row.task_type].add(row.image_sha256)
    ocr_audit = {
        "eligible_unique_images_by_type": {
            key: len(value) for key, value in eligible_images.items()
        },
        "rows_excluded_by_source_id": excluded_source_ids,
        "rows_excluded_by_image_hash": excluded_image_hashes,
    }
    print(
        json.dumps(
            {
                "claim_boundary": "outcome_blind_capacity_only",
                "outcome_fields_used": False,
                "scores_computed": False,
                "exclusion_list": str(exclusion_list),
                "exclusion_list_sha256": shared.sha256(exclusion_list),
                "chartqa": chart_audit,
                "ocrbench": {**ocr_audit, **question_audit},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chartqa-arrow", required=True)
    parser.add_argument("--ocrbench-arrow-dir", required=True)
    parser.add_argument("--exclusion-list", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
