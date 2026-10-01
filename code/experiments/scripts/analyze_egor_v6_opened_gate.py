#!/usr/bin/env python3
"""Replay sealed v5 outputs through the frozen v6 transport contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import analyze_egor_opened_verifier_diagnostic_v1 as v5
from openagora_experiments.egor.protocol import parse_verifier_output_v6
from openagora_experiments.egor.verification import (
    answers_agree,
    canonical_answer,
    decide_verified_overwrite,
    valid_evidence_cards,
)


def main() -> None:
    args = parse_args()
    legacy_root = Path(args.legacy_root).resolve()
    method_root = Path(args.method_results).resolve()
    verifier_root = Path(args.verifier_root).resolve()
    rows_by_source = {
        source: replay_source_v6(
            source=source,
            config=config,
            legacy_root=legacy_root,
            method_root=method_root,
            verifier_root=verifier_root,
        )
        for source, config in v5.SOURCE_CONFIGS.items()
    }
    development = [
        row
        for source in ("chartqa", "infographic", "mathvista")
        for row in rows_by_source[source]
    ]
    ocrbench = rows_by_source["ocrbench"]
    all_rows = [*development, *ocrbench]
    development_summary = v5.summarize(
        development,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    ocrbench_summary = v5.summarize(
        ocrbench,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 1,
    )
    completion_by_source = {
        source: fmean(
            row["verifier_operational"] for row in rows if row["triggered"]
        )
        for source, rows in rows_by_source.items()
    }
    accepted_sources = {
        row["source"] for row in all_rows if row["accept_overwrite"]
    }
    rejected_harms = sum(
        row["core_delta"] < -0.05 and not row["accept_overwrite"]
        for row in all_rows
    )
    criteria = [
        v5.criterion(
            "sourcewise_operational_completion",
            min(completion_by_source.values()) >= 0.95,
            completion_by_source,
            "every source >= 0.95; valid abstention counts as complete",
        ),
        v5.criterion(
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
        v5.criterion(
            "opened_ocrbench_signal",
            ocrbench_summary["verified_delta"] > 0.0
            and ocrbench_summary["verified_strict_harm_rate"] == 0.0,
            {
                "delta": ocrbench_summary["verified_delta"],
                "strict_harm_rate": ocrbench_summary[
                    "verified_strict_harm_rate"
                ],
            },
            "point delta > 0 and no strict harm",
        ),
        v5.criterion(
            "accepted_source_breadth",
            sum(row["accept_overwrite"] for row in all_rows) >= 2
            and len(accepted_sources) >= 2,
            {
                "accepted": sum(row["accept_overwrite"] for row in all_rows),
                "sources": sorted(accepted_sources),
            },
            "at least two overwrites across at least two sources",
        ),
        v5.criterion(
            "harm_rejection",
            rejected_harms >= 1,
            rejected_harms,
            "reject at least one harmful method-v3 overwrite",
        ),
    ]
    report = {
        "schema_version": 1,
        "claim_boundary": "opened_development_replay_only",
        "method": "egor_v6_verified_overwrite",
        "model_calls_made_by_replay": 0,
        "completion_definition": (
            "status ok plus valid v6 top-level transport; explicit and sanitized "
            "abstentions are complete and retain baseline"
        ),
        "completion_by_source": completion_by_source,
        "transport_modes": dict(
            Counter(
                row["transport_mode"]
                for row in all_rows
                if row["triggered"]
            )
        ),
        "sanitized_abstentions": sum(
            row["sanitized_abstention"] for row in all_rows if row["triggered"]
        ),
        "development_combined": development_summary,
        "ocrbench": ocrbench_summary,
        "sources": {
            source: v5.summarize(rows, bootstrap_samples=0, seed=args.seed)
            for source, rows in rows_by_source.items()
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
                    "transport_mode",
                    "sanitized_abstention",
                )
            }
            for row in all_rows
            if row["core_changed"] or row["verified_delta"] != 0.0
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def replay_source_v6(
    *,
    source: str,
    config: dict[str, str],
    legacy_root: Path,
    method_root: Path,
    verifier_root: Path,
) -> list[dict[str, Any]]:
    tasks = {
        task.task_id: task
        for task in v5.load_tasks(legacy_root / config["tasks"])
    }
    source_method = method_root / config["method"]
    if source == "ocrbench":
        source_legacy = source_method
        plans_path = source_method / "escalation_plan" / "plans.jsonl"
        current = (source_method / "ANALYSIS_CURRENT").read_text(
            encoding="utf-8"
        ).strip()
        if not current.startswith("analysis_attempts/") or ".." in current:
            raise ValueError("unsafe OCRBench analysis pointer")
        matrix_path = source_method / current / "matrix_analysis" / "matrix_rows.csv"
    else:
        source_legacy = legacy_root / config["legacy"]
        plans_path = source_method / "plan" / "plans.jsonl"
        matrix_path = source_method / "matrix_analysis" / "matrix_rows.csv"
    plans = {row["task_id"]: row for row in v5.load_jsonl(plans_path)}
    matrix = v5.load_core_matrix(matrix_path)
    exact = v5.latest(source_legacy / "specialist_exact_qwen25.jsonl")
    visual = v5.latest(source_legacy / "specialist_visual_internvl3.jsonl")
    verifier = v5.latest(verifier_root / source / "independent_qwen35.jsonl")

    rows: list[dict[str, Any]] = []
    for task_id, plan in sorted(plans.items()):
        task = tasks[task_id]
        core = matrix[task_id]
        triggered = plan["tier"] != "baseline"
        baseline_answer = core["baseline_answer"]
        core_answer = core["candidate_answer"]
        baseline_quality = float(core["baseline_quality"])
        core_quality = float(core["candidate_quality"])
        exact_parsed = v5.parsed(exact.get(task_id))
        visual_parsed = v5.parsed(visual.get(task_id))
        verifier_record = verifier.get(task_id)
        verifier_parsed, parser_issues = parse_verifier_output_v6(
            str(verifier_record.get("raw_output", "")) if verifier_record else ""
        )
        transport = verifier_parsed.get("transport", {})
        operational = bool(
            verifier_record
            and verifier_record.get("status") == "ok"
            and transport.get("operational_complete")
        )
        verifier_view = (
            {
                "candidate_answer": verifier_parsed.get("answer", ""),
                "evidence": verifier_parsed.get("evidence", []),
            }
            if operational
            else {"candidate_answer": "", "evidence": []}
        )
        options = [
            {"id": option_id, "text": text}
            for option_id, text in task.options
        ]
        core_changed = bool(
            triggered
            and canonical_answer(core_answer)
            and not answers_agree(
                baseline_answer,
                core_answer,
                metric=task.metric,
                options=options,
            )
        )
        decision = decide_verified_overwrite(
            baseline_answer=baseline_answer,
            core_answer=core_answer,
            verifier=verifier_view,
            specialists={"exact": exact_parsed, "visual": visual_parsed},
            metric=task.metric,
            options=options,
        )
        strict_pair = bool(
            core_changed
            and valid_evidence_cards(exact_parsed)
            and valid_evidence_cards(visual_parsed)
            and answers_agree(
                exact_parsed.get("candidate_answer", ""),
                visual_parsed.get("candidate_answer", ""),
                metric=task.metric,
                options=options,
            )
            and answers_agree(
                exact_parsed.get("candidate_answer", ""),
                core_answer,
                metric=task.metric,
                options=options,
            )
        )
        accept = bool(triggered and operational and decision.accept_overwrite)
        verified_quality = core_quality if accept else baseline_quality
        strict_pair_quality = core_quality if strict_pair else baseline_quality
        core_calls = (
            float(core["calls"])
            if triggered
            else float(core["baseline_calls"])
        )
        rows.append(
            {
                "source": source,
                "task_id": task_id,
                "triggered": triggered,
                "core_changed": core_changed,
                "verifier_complete": bool(not triggered or operational),
                "verifier_operational": operational,
                "transport_mode": str(transport.get("mode", "invalid")),
                "sanitized_abstention": bool(
                    transport.get("sanitized_abstention")
                ),
                "parser_issue_count": len(parser_issues),
                "accept_overwrite": accept,
                "decision_reason": decision.reason,
                "corroborators": list(decision.corroborating_specialists),
                "baseline_quality": baseline_quality,
                "core_delta": core_quality - baseline_quality if triggered else 0.0,
                "strict_pair_delta": strict_pair_quality - baseline_quality,
                "verified_delta": verified_quality - baseline_quality,
                "calls": core_calls + float(core_changed),
                "verifier_called": core_changed,
            }
        )
    return rows


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
    parser.add_argument("--seed", type=int, default=20260744)
    return parser.parse_args()


if __name__ == "__main__":
    main()
