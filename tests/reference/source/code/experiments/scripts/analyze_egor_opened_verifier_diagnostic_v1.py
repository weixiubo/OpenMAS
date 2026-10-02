#!/usr/bin/env python3
"""Analyze the frozen opened-data independent-verifier diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from openagora_experiments.egor.protocol import PilotTask
from openagora_experiments.egor.verification import (
    decide_answer_only_verified_overwrite,
    answers_agree,
    canonical_answer,
    decide_verified_overwrite,
    valid_evidence_cards,
)


SOURCE_CONFIGS = {
    "chartqa": {
        "tasks": "data/chartqa_fresh_v1/development120.jsonl",
        "legacy": "results/chartqa_development_v1",
        "method": "development_escalation_v1/chartqa",
    },
    "infographic": {
        "tasks": "data/infographic_fresh_v1/development60.jsonl",
        "legacy": "results/development_v1",
        "method": "development_escalation_v1/infographic",
    },
    "mathvista": {
        "tasks": "data/mathvista_development_v1/tasks60_capped900k.jsonl",
        "legacy": "results/mathvista_development_v1_capped900k",
        "method": "development_escalation_v1/mathvista",
    },
    "ocrbench": {
        "tasks": "data/ocrbench_transfer_v1/transfer240.jsonl",
        "legacy": "ocrbench_confirmation_recovery_v1",
        "method": "ocrbench_confirmation_recovery_v1",
    },
}


def main() -> None:
    args = parse_args()
    legacy_root = Path(args.legacy_root).resolve()
    method_root = Path(args.method_results).resolve()
    verifier_root = Path(args.verifier_root).resolve()
    rows_by_source = {
        source: replay_source(
            source=source,
            config=config,
            legacy_root=legacy_root,
            method_root=method_root,
            verifier_root=verifier_root,
        )
        for source, config in SOURCE_CONFIGS.items()
    }
    development = [
        row
        for source in ("chartqa", "infographic", "mathvista")
        for row in rows_by_source[source]
    ]
    ocrbench = rows_by_source["ocrbench"]
    dev_summary = summarize(
        development,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    ocr_summary = summarize(
        ocrbench,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 1,
    )
    accepted_sources = {
        row["source"]
        for row in [*development, *ocrbench]
        if row["accept_overwrite"]
    }
    verifier_completion = {
        "development": fmean(row["verifier_complete"] for row in development if row["triggered"]),
        "ocrbench": fmean(row["verifier_complete"] for row in ocrbench if row["triggered"]),
    }
    criteria = [
        criterion(
            "verifier_completion",
            min(verifier_completion.values()) >= 0.95,
            verifier_completion,
            "both development and OCRBench >= 0.95",
        ),
        criterion(
            "development_signal",
            dev_summary["verified_delta"] > 0.0
            and dev_summary["verified_ci95_low"] >= 0.0
            and dev_summary["verified_strict_harm_rate"] <= 0.02,
            {
                "delta": dev_summary["verified_delta"],
                "ci95_low": dev_summary["verified_ci95_low"],
                "strict_harm_rate": dev_summary["verified_strict_harm_rate"],
            },
            "delta > 0, CI low >= 0, strict harm <= 0.02",
        ),
        criterion(
            "opened_ocrbench_signal",
            ocr_summary["verified_delta"] > 0.0
            and ocr_summary["verified_strict_harm_rate"] == 0.0,
            {
                "delta": ocr_summary["verified_delta"],
                "strict_harm_rate": ocr_summary["verified_strict_harm_rate"],
            },
            "point delta > 0 and no strict harm",
        ),
        criterion(
            "accepted_source_breadth",
            sum(row["accept_overwrite"] for row in [*development, *ocrbench]) >= 2
            and len(accepted_sources) >= 2,
            {
                "accepted": sum(
                    row["accept_overwrite"] for row in [*development, *ocrbench]
                ),
                "sources": sorted(accepted_sources),
            },
            "at least two overwrites across at least two sources",
        ),
        criterion(
            "harm_rejection",
            sum(
                row["core_delta"] < -0.05 and not row["accept_overwrite"]
                for row in [*development, *ocrbench]
            )
            >= 1,
            sum(
                row["core_delta"] < -0.05 and not row["accept_overwrite"]
                for row in [*development, *ocrbench]
            ),
            "reject at least one harmful method-v3 overwrite",
        ),
    ]
    report = {
        "schema_version": 1,
        "claim_boundary": "opened_development_diagnostic_only",
        "primary_policy": "independently_verified_overwrite",
        "verifier_completion": verifier_completion,
        "verifier_completion_definition": (
            "status ok, exact compact JSON parse, non-empty answer, and at least one "
            "structurally valid evidence card"
        ),
        "development_combined": dev_summary,
        "ocrbench": ocr_summary,
        "sources": {
            source: summarize(rows, bootstrap_samples=0, seed=args.seed)
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
                )
            }
            for row in [*development, *ocrbench]
            if row["core_changed"] or row["verified_delta"] != 0.0
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def replay_source(
    *,
    source: str,
    config: dict[str, str],
    legacy_root: Path,
    method_root: Path,
    verifier_root: Path,
    answer_only: bool = False,
    verifier_filename: str = "independent_qwen35.jsonl",
) -> list[dict[str, Any]]:
    tasks = {
        task.task_id: task
        for task in load_tasks(legacy_root / config["tasks"])
    }
    source_method = method_root / config["method"]
    if source == "ocrbench":
        source_legacy = source_method
        plans_path = source_method / "escalation_plan" / "plans.jsonl"
        current = (source_method / "ANALYSIS_CURRENT").read_text(encoding="utf-8").strip()
        if not current.startswith("analysis_attempts/") or ".." in current:
            raise ValueError("unsafe OCRBench analysis pointer")
        matrix_path = source_method / current / "matrix_analysis" / "matrix_rows.csv"
    else:
        source_legacy = legacy_root / config["legacy"]
        plans_path = source_method / "plan" / "plans.jsonl"
        matrix_path = source_method / "matrix_analysis" / "matrix_rows.csv"
    plans = {row["task_id"]: row for row in load_jsonl(plans_path)}
    matrix = load_core_matrix(matrix_path)
    exact = latest(source_legacy / "specialist_exact_qwen25.jsonl")
    visual = latest(source_legacy / "specialist_visual_internvl3.jsonl")
    verifier = latest(verifier_root / source / verifier_filename)

    rows = []
    for task_id, plan in sorted(plans.items()):
        task = tasks[task_id]
        core = matrix[task_id]
        triggered = plan["tier"] != "baseline"
        baseline_answer = core["baseline_answer"]
        core_answer = core["candidate_answer"]
        baseline_quality = float(core["baseline_quality"])
        core_quality = float(core["candidate_quality"])
        exact_parsed = parsed(exact.get(task_id))
        visual_parsed = parsed(visual.get(task_id))
        verifier_record = verifier.get(task_id)
        verifier_parsed = parsed(verifier_record)
        verifier_view = {
            "candidate_answer": verifier_parsed.get("answer", ""),
            "evidence": verifier_parsed.get("evidence", []),
            "transport": verifier_parsed.get("transport", {}),
        }
        options = [
            {"id": option_id, "text": text}
            for option_id, text in task.options
        ]
        verifier_structurally_complete = bool(
            verifier_record
            and verifier_record.get("status") == "ok"
            and not verifier_record.get("parse_issues", [])
            and (
                bool(verifier_parsed.get("transport", {}).get("operational_complete"))
                if answer_only
                else bool(
                    str(verifier_parsed.get("answer", "")).strip()
                    and valid_evidence_cards(verifier_parsed)
                )
            )
        )
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
        decision_function = (
            decide_answer_only_verified_overwrite if answer_only else decide_verified_overwrite
        )
        decision = decision_function(
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
        accept = bool(triggered and decision.accept_overwrite)
        verified_quality = core_quality if accept else baseline_quality
        strict_pair_quality = core_quality if strict_pair else baseline_quality
        verifier_called = core_changed
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
                "verifier_complete": bool(
                    not triggered
                    or verifier_structurally_complete
                ),
                "accept_overwrite": accept,
                "decision_reason": decision.reason,
                "corroborators": list(decision.corroborating_specialists),
                "baseline_quality": baseline_quality,
                "core_delta": core_quality - baseline_quality if triggered else 0.0,
                "strict_pair_delta": strict_pair_quality - baseline_quality,
                "verified_delta": verified_quality - baseline_quality,
                "calls": core_calls + float(verifier_called),
                "verifier_called": verifier_called,
            }
        )
    return rows


def summarize(
    rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    verified = [float(row["verified_delta"]) for row in rows]
    core = [float(row["core_delta"]) for row in rows]
    pair = [float(row["strict_pair_delta"]) for row in rows]
    low, high = (
        bootstrap_interval(verified, samples=bootstrap_samples, seed=seed)
        if bootstrap_samples
        else (None, None)
    )
    return {
        "tasks": len(rows),
        "triggered": sum(row["triggered"] for row in rows),
        "core_changed": sum(row["core_changed"] for row in rows),
        "accepted_overwrites": sum(row["accept_overwrite"] for row in rows),
        "core_capped_delta": fmean(core),
        "strict_pair_delta": fmean(pair),
        "verified_delta": fmean(verified),
        "verified_ci95_low": low,
        "verified_ci95_high": high,
        "verified_wins": sum(value > 0.0 for value in verified),
        "verified_losses": sum(value < 0.0 for value in verified),
        "verified_ties": sum(value == 0.0 for value in verified),
        "verified_strict_harm_rate": fmean(value < -0.05 for value in verified),
        "mean_calls": fmean(float(row["calls"]) for row in rows),
        "decision_reasons": dict(Counter(row["decision_reason"] for row in rows)),
    }


def load_tasks(path: Path) -> list[PilotTask]:
    return [PilotTask.from_json(row) for row in load_jsonl(path)]


def load_core_matrix(path: Path) -> dict[str, dict[str, str]]:
    output = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["candidate_id"] == "core_pair":
                output[row["task_id"]] = row
    return output


def latest(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for row in load_jsonl(path):
        output[str(row["task_id"])] = row
    return output


def parsed(record: dict[str, Any] | None) -> dict[str, Any]:
    if not record or not isinstance(record.get("parsed"), dict):
        return {}
    return dict(record["parsed"])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def bootstrap_interval(
    values: list[float], *, samples: int, seed: int
) -> tuple[float, float]:
    generator = random.Random(seed)
    count = len(values)
    means = sorted(
        fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def criterion(name: str, passed: bool, actual: Any, required: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "actual": actual, "required": required}


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
