#!/usr/bin/env python3
"""Analyze the frozen opened EGOR-v8 answer-only verifier diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_egor_opened_verifier_diagnostic_v1 as base


SOURCE_ORDER = ("chartqa", "infographic", "mathvista", "ocrbench")
SOURCE_SEEDS = {
    "chartqa": 2026078101,
    "infographic": 2026078102,
    "mathvista": 2026078103,
    "ocrbench": 2026078104,
}


def analyze(
    rows_by_source: dict[str, list[dict[str, Any]]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if set(rows_by_source) != set(SOURCE_ORDER):
        raise ValueError("v8 analysis requires exactly four opened sources")
    development = [
        row
        for source in ("chartqa", "infographic", "mathvista")
        for row in rows_by_source[source]
    ]
    ocrbench = rows_by_source["ocrbench"]
    all_rows = [row for source in SOURCE_ORDER for row in rows_by_source[source]]
    development_summary = base.summarize(
        development, bootstrap_samples=bootstrap_samples, seed=seed
    )
    ocrbench_summary = base.summarize(
        ocrbench, bootstrap_samples=bootstrap_samples, seed=seed + 1
    )
    completion_by_source = {
        source: fmean(
            bool(row["verifier_complete"])
            for row in rows_by_source[source]
            if row["triggered"]
        )
        for source in SOURCE_ORDER
    }
    accepted_sources = {
        row["source"] for row in all_rows if row["accept_overwrite"]
    }
    harmful_rejections = sum(
        row["core_delta"] < -0.05 and not row["accept_overwrite"]
        for row in all_rows
    )
    criteria = [
        base.criterion(
            "sourcewise_operational_completion",
            all(value >= 0.95 for value in completion_by_source.values()),
            completion_by_source,
            "each opened source >= 0.95",
        ),
        base.criterion(
            "development_signal",
            development_summary["verified_delta"] > 0.0
            and development_summary["verified_ci95_low"] >= 0.0
            and development_summary["verified_strict_harm_rate"] <= 0.02,
            {
                "delta": development_summary["verified_delta"],
                "ci95_low": development_summary["verified_ci95_low"],
                "strict_harm_rate": development_summary[
                    "verified_strict_harm_rate"
                ],
            },
            "delta > 0, CI low >= 0, strict harm <= 0.02",
        ),
        base.criterion(
            "opened_ocrbench_signal",
            ocrbench_summary["verified_delta"] > 0.0
            and ocrbench_summary["verified_strict_harm_rate"] == 0.0,
            {
                "delta": ocrbench_summary["verified_delta"],
                "strict_harm_rate": ocrbench_summary[
                    "verified_strict_harm_rate"
                ],
            },
            "point delta > 0 and zero strict harm",
        ),
        base.criterion(
            "accepted_source_breadth",
            sum(row["accept_overwrite"] for row in all_rows) >= 2
            and len(accepted_sources) >= 2,
            {
                "accepted": sum(row["accept_overwrite"] for row in all_rows),
                "sources": sorted(accepted_sources),
            },
            "at least two overwrites across at least two sources",
        ),
        base.criterion(
            "harm_rejection",
            harmful_rejections >= 1,
            harmful_rejections,
            "reject at least one opened harmful core overwrite",
        ),
    ]
    return {
        "schema_version": 1,
        "method": "egor_v8_answer_only_verified_overwrite",
        "claim_boundary": "opened_development_diagnostic_only",
        "v7_format_outputs_used": False,
        "verifier_generation_rows": sum(row["triggered"] for row in all_rows),
        "deployed_verifier_call_rows": sum(
            row["verifier_called"] for row in all_rows
        ),
        "completion_by_source": completion_by_source,
        "development_combined": development_summary,
        "ocrbench": ocrbench_summary,
        "sources": {
            source: base.summarize(
                rows_by_source[source], bootstrap_samples=0, seed=seed
            )
            for source in SOURCE_ORDER
        },
        "criteria": criteria,
        "passed": all(item["passed"] for item in criteria),
        "material_rows": [
            {
                key: row[key]
                for key in (
                    "source",
                    "task_id",
                    "core_delta",
                    "strict_pair_delta",
                    "verified_delta",
                    "accept_overwrite",
                    "decision_reason",
                    "corroborators",
                )
            }
            for row in all_rows
            if row["core_changed"] or row["verified_delta"] != 0.0
        ],
    }


def main() -> int:
    args = parse_args()
    rows_by_source = {
        source: base.replay_source(
            source=source,
            config=base.SOURCE_CONFIGS[source],
            legacy_root=Path(args.legacy_root).resolve(),
            method_root=Path(args.method_results).resolve(),
            verifier_root=Path(args.verifier_root).resolve(),
            answer_only=True,
            verifier_filename="answer_only_qwen35.jsonl",
        )
        for source in SOURCE_ORDER
    }
    report = analyze(
        rows_by_source,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-root", default="."
    )
    parser.add_argument(
        "--method-results",
        default="results",
    )
    parser.add_argument("--verifier-root", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260781)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
