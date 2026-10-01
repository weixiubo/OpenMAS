#!/usr/bin/env python3
"""Report outcome-blind unique-image capacity after frozen exclusions."""

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
    chart_images = {task_type: set() for task_type in shared.CHARTQA_TYPES}
    chart_rows = {task_type: 0 for task_type in shared.CHARTQA_TYPES}
    chart_excluded_image = 0
    chart_excluded_question = 0
    for row in chart_candidates:
        if row.question_type not in chart_images:
            continue
        if (
            row.image_name in exclusions.image_basenames
            or row.image_sha256 in exclusions.image_hashes
        ):
            chart_excluded_image += 1
            continue
        if shared._text_sha256(
            shared._normalize_question(row.question)
        ) in exclusions.question_hashes:
            chart_excluded_question += 1
            continue
        chart_rows[row.question_type] += 1
        chart_images[row.question_type].add(row.image_sha256)
    chart_human = chart_images["human"]
    chart_augmented = chart_images["augmented"]
    chart_audit = {
        "eligible_rows_by_type": chart_rows,
        "eligible_unique_images_by_type": {
            key: len(value) for key, value in chart_images.items()
        },
        "eligible_unique_images_union": len(chart_human | chart_augmented),
        "shared_unique_images_between_types": len(
            chart_human & chart_augmented
        ),
        "rows_excluded_by_image": chart_excluded_image,
        "rows_excluded_by_question": chart_excluded_question,
        "source_rows": len(chart_candidates),
    }

    ocr_dir = Path(args.ocrbench_arrow_dir).resolve()
    ocr_candidates = shared._index_ocrbench(sorted(ocr_dir.glob("*.arrow")))
    ocr_filtered, question_audit = filter_prior_questions(
        ocr_candidates, excluded_question_hashes=exclusions.question_hashes
    )
    ocr_images = {task_type: set() for task_type in shared.OCRBENCH_V9_TYPES}
    ocr_excluded_source = 0
    ocr_excluded_image = 0
    for row in ocr_filtered:
        if row.task_type not in ocr_images:
            continue
        if row.source_id in exclusions.source_ids:
            ocr_excluded_source += 1
            continue
        if row.image_sha256 in exclusions.image_hashes:
            ocr_excluded_image += 1
            continue
        ocr_images[row.task_type].add(row.image_sha256)
    ocr_types = list(ocr_images)
    pairwise_overlap = {
        f"{ocr_types[i]} || {ocr_types[j]}": len(
            ocr_images[ocr_types[i]] & ocr_images[ocr_types[j]]
        )
        for i in range(len(ocr_types))
        for j in range(i + 1, len(ocr_types))
        if ocr_images[ocr_types[i]] & ocr_images[ocr_types[j]]
    }
    ocr_audit = {
        "eligible_unique_images_by_type": {
            key: len(value) for key, value in ocr_images.items()
        },
        "eligible_unique_images_union": len(set().union(*ocr_images.values())),
        "nonzero_pairwise_image_overlap": pairwise_overlap,
        "rows_excluded_by_source_id": ocr_excluded_source,
        "rows_excluded_by_image_hash": ocr_excluded_image,
        **question_audit,
    }

    print(
        json.dumps(
            {
                "schema_version": 2,
                "claim_boundary": "outcome_blind_unique_image_capacity_only",
                "capacity_fields_used": [
                    "prior-use identifiers",
                    "normalized-question hashes",
                    "image identities",
                    "official strata",
                ],
                "outcome_fields_used": False,
                "scores_computed": False,
                "exclusion_list": str(exclusion_list),
                "exclusion_list_sha256": shared.sha256(exclusion_list),
                "chartqa": chart_audit,
                "ocrbench": ocr_audit,
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
