#!/usr/bin/env python3
"""Build the outcome-blind fresh 800-task EGOR-v10 confirmation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_egor_v9_confirmation_data as shared


DEFAULT_SALT = "egor-aaai27-v10-two-support-confirmation-20260719"
CHARTQA_FORMAT_QUOTA = {"human": 10, "augmented": 10}
CHARTQA_CONFIRMATION_QUOTA = {"human": 18, "augmented": 212}
OCRBENCH_FORMAT_QUOTA = {
    "APP agent en": 5,
    "ASCII art classification en": 5,
    "cognition VQA en": 5,
    "diagram QA en": 5,
    "reasoning VQA en": 5,
    "science QA en": 5,
}
OCRBENCH_CONFIRMATION_QUOTA = {
    "APP agent en": 84,
    "ASCII art classification en": 84,
    "cognition VQA en": 101,
    "diagram QA en": 101,
    "reasoning VQA en": 100,
    "science QA en": 100,
}


def main() -> int:
    args = _parse_args()
    exclusion_list = Path(args.exclusion_list).resolve()
    exclusions = [
        Path(line).resolve()
        for line in exclusion_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not exclusions:
        raise ValueError("v10 exclusion list is empty")
    shared.CHARTQA_TASK_PREFIX = "chartqa-v10-test"
    manifest = shared.build_data(
        chartqa_arrow=Path(args.chartqa_arrow).resolve(),
        ocrbench_arrow_dir=Path(args.ocrbench_arrow_dir).resolve(),
        exclusion_paths=exclusions,
        output_root=Path(args.output_root).resolve(),
        salt=args.salt,
        chartqa_format_quota=CHARTQA_FORMAT_QUOTA,
        chartqa_confirmation_quota=CHARTQA_CONFIRMATION_QUOTA,
        ocrbench_format_quota=OCRBENCH_FORMAT_QUOTA,
        ocrbench_confirmation_quota=OCRBENCH_CONFIRMATION_QUOTA,
        chartqa_benchmark="ChartQA-EGOR-V10-Fresh",
        ocrbench_benchmark="OCRBench-v2-English-VQA-EGOR-V10-Fresh",
        metadata_flag="egor_v10_fresh_confirmation",
    )
    summary = {
        "claim_boundary": "outcome_blind_fresh_v10_split",
        "exclusion_list": str(exclusion_list),
        "exclusion_list_sha256": shared.sha256(exclusion_list),
        "sources": {
            source: {
                split: {
                    key: manifest["sources"][source][split][key]
                    for key in ("path", "rows", "sha256")
                }
                for split in ("format", "confirmation")
            }
            for source in ("chartqa", "ocrbench")
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chartqa-arrow", required=True)
    parser.add_argument("--ocrbench-arrow-dir", required=True)
    parser.add_argument("--exclusion-list", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
