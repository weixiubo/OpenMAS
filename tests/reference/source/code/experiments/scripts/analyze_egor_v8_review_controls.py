#!/usr/bin/env python3
"""Post-hoc nested controls and audits for the sealed EGOR-v8 confirmation.

This analysis never changes the sealed confirmation result.  It reuses the
same generated records to isolate the verifier and structural-card conditions
one at a time, and labels every output as post-hoc review evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

import analyze_egor_v7_confirmation as shared
import analyze_egor_v8_confirmation as v8
from openagora_experiments.egor.runner import load_latest_stage_records, load_tasks
from openagora_experiments.egor.verification import (
    answers_agree,
    valid_evidence_cards,
)


POLICIES = (
    "core",
    "strict_pair",
    "specialist_only",
    "specialist_agreement_no_card",
    "blind_verifier_no_card",
    "egor_v8",
    "egor_v8_cited_card",
)


def main() -> int:
    args = _parse_args()
    sources = {
        "chartqa": (
            Path(args.chartqa_tasks).resolve(),
            Path(args.chartqa_root).resolve(),
        ),
        "infographic": (
            Path(args.infographic_tasks).resolve(),
            Path(args.infographic_root).resolve(),
        ),
    }
    primary, primary_rows = v8.analyze_confirmation(
        sources=sources,
        expected_commit=args.source_commit,
        tasks_per_source=args.tasks_per_source,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    rows = _build_control_rows(sources=sources, primary_rows=primary_rows)
    report = build_report(
        primary=primary,
        rows=rows,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _build_control_rows(
    *,
    sources: dict[str, tuple[Path, Path]],
    primary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    primary_by_key = {
        (str(row["source_id"]), str(row["task_id"])): row
        for row in primary_rows
    }
    output: list[dict[str, Any]] = []
    for source, (tasks_path, root) in sources.items():
        tasks = load_tasks(tasks_path)
        records = {
            name: load_latest_stage_records(root / filename)
            for name, filename in {
                "exact": "specialist_exact_qwen25.jsonl",
                "visual": "specialist_visual_internvl3.jsonl",
                "core": "core_pair.jsonl",
                "verifier": "verifier_v8_qwen35.jsonl",
            }.items()
        }
        plans = _load_selector_plans(root / "selector_plan" / "plans.jsonl")
        for task in tasks:
            key = (task.source_id, task.task_id)
            primary_row = primary_by_key[key]
            exact = _parsed(records["exact"].get(key))
            visual = _parsed(records["visual"].get(key))
            core = _parsed(records["core"].get(key))
            verifier = _parsed(records["verifier"].get(key))
            core_answer = str(core.get("answer", ""))
            options = [{"id": key, "text": text} for key, text in task.options]
            changed = bool(primary_row["core_changed"])

            specialist_agrees: dict[str, bool] = {}
            specialist_valid: dict[str, bool] = {}
            specialist_cited: dict[str, bool] = {}
            used_ids = {str(value) for value in core.get("used_evidence_ids", [])}
            for name, record in (("exact", exact), ("visual", visual)):
                specialist_agrees[name] = bool(
                    changed
                    and answers_agree(
                        str(record.get("candidate_answer", "")),
                        core_answer,
                        metric=task.metric,
                        options=options,
                    )
                )
                cards = valid_evidence_cards(record)
                specialist_valid[name] = bool(cards)
                card_ids = {str(card.get("evidence_id", "")) for card in cards}
                specialist_cited[name] = bool(card_ids & used_ids)

            agreement_h = any(specialist_agrees.values())
            structural_h = any(
                specialist_agrees[name] and specialist_valid[name]
                for name in specialist_agrees
            )
            cited_h = any(
                specialist_agrees[name] and specialist_cited[name]
                for name in specialist_agrees
            )
            transport = verifier.get("transport", {})
            blind_q = bool(
                changed
                and isinstance(transport, Mapping)
                and transport.get("operational_complete")
                and transport.get("decisive")
                and not transport.get("sanitized_abstention")
                and answers_agree(
                    str(verifier.get("answer", "")),
                    core_answer,
                    metric=task.metric,
                    options=options,
                )
            )

            baseline_quality = float(primary_row["baseline_quality"])
            core_quality = float(primary_row["core_quality"])
            values = {
                "core": core_quality,
                "strict_pair": float(primary_row["strict_pair_quality"]),
                "specialist_only": (
                    core_quality if changed and structural_h else baseline_quality
                ),
                "specialist_agreement_no_card": (
                    core_quality if changed and agreement_h else baseline_quality
                ),
                "blind_verifier_no_card": (
                    core_quality if changed and blind_q else baseline_quality
                ),
                "egor_v8": (
                    core_quality
                    if changed and blind_q and structural_h
                    else baseline_quality
                ),
                "egor_v8_cited_card": (
                    core_quality
                    if changed and blind_q and structural_h and cited_h
                    else baseline_quality
                ),
            }
            output.append(
                {
                    "source": source,
                    "source_id": task.source_id,
                    "task_id": task.task_id,
                    "baseline_quality": baseline_quality,
                    "core_changed": changed,
                    "repairability_triggered": bool(
                        primary_row["repairability_triggered"]
                    ),
                    "selector_template": str(
                        plans[key].get("selected_template_id", "")
                    ),
                    "specialist_agreement": agreement_h,
                    "structural_card_corroboration": structural_h,
                    "blind_verifier_agreement": blind_q,
                    "cited_card_corroboration": cited_h,
                    "corroborating_specialists": [
                        name
                        for name in specialist_agrees
                        if specialist_agrees[name] and specialist_valid[name]
                    ],
                    "used_evidence_ids": sorted(used_ids),
                    "core_answer": core_answer,
                    "verifier_answer": str(verifier.get("answer", "")),
                    "policy_quality": values,
                }
            )
    return output


def build_report(
    *,
    primary: dict[str, Any],
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    policy_summaries = {
        policy: _summarize(
            rows,
            policy=policy,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index,
        )
        for index, policy in enumerate(POLICIES)
    }
    changed = [row for row in rows if row["core_changed"]]
    accepted = [
        row
        for row in changed
        if row["blind_verifier_agreement"]
        and row["structural_card_corroboration"]
    ]
    exact_visual = [
        row
        for row in rows
        if row["repairability_triggered"]
        and row["selector_template"] == "exact_visual"
    ]
    other_action = [
        row
        for row in rows
        if row["repairability_triggered"]
        and row["selector_template"] != "exact_visual"
    ]
    return {
        "schema_version": 1,
        "claim_boundary": "posthoc_review_diagnostic_not_new_confirmation",
        "source_commit": primary["source_commit"],
        "sealed_primary_passed": primary["passed"],
        "tasks": len(rows),
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed_root": seed,
            "unit": "paired task with equal task weight",
        },
        "policies": policy_summaries,
        "nested_contrasts": {
            "verifier_effect_given_structural_corroboration": _contrast(
                rows,
                left="egor_v8",
                right="specialist_only",
                bootstrap_samples=bootstrap_samples,
                seed=seed + 101,
            ),
            "structural_card_effect_given_blind_verifier": _contrast(
                rows,
                left="egor_v8",
                right="blind_verifier_no_card",
                bootstrap_samples=bootstrap_samples,
                seed=seed + 102,
            ),
            "card_structure_effect_given_specialist_agreement": _contrast(
                rows,
                left="specialist_only",
                right="specialist_agreement_no_card",
                bootstrap_samples=bootstrap_samples,
                seed=seed + 103,
            ),
        },
        "decision_funnel": {
            "core_changed": len(changed),
            "specialist_agreement": sum(
                row["specialist_agreement"] for row in changed
            ),
            "structural_card_corroboration": sum(
                row["structural_card_corroboration"] for row in changed
            ),
            "blind_verifier_agreement": sum(
                row["blind_verifier_agreement"] for row in changed
            ),
            "egor_v8_accept": len(accepted),
            "accepted_with_cited_supporting_card": sum(
                row["cited_card_corroboration"] for row in accepted
            ),
        },
        "accepted_overwrite_harm_interval": {
            "observed_harms": sum(
                row["policy_quality"]["egor_v8"]
                < row["baseline_quality"] - 0.05
                for row in accepted
            ),
            "accepted_overwrites": len(accepted),
            "two_sided_clopper_pearson_95_upper": _zero_event_cp_upper(
                len(accepted), alpha=0.05
            ),
            "interpretation": "conditional uncertainty; not a safety guarantee",
        },
        "gate_action_audit": {
            "triggered": len(exact_visual) + len(other_action),
            "exact_visual_argmax": len(exact_visual),
            "other_action_argmax": len(other_action),
            "egor_v8_delta_pp_exact_visual_argmax_only": 100.0
            * fmean(
                (
                    row["policy_quality"]["egor_v8"]
                    - row["baseline_quality"]
                )
                if row["selector_template"] == "exact_visual"
                else 0.0
                for row in rows
            ),
            "egor_v8_delta_pp_other_action_argmax_only": 100.0
            * fmean(
                (
                    row["policy_quality"]["egor_v8"]
                    - row["baseline_quality"]
                )
                if row["selector_template"] != "exact_visual"
                else 0.0
                for row in rows
            ),
            "interpretation": (
                "The frozen gate predicts general repairability; its action "
                "argmax is discarded and is not evidence for the executed core."
            ),
        },
        "accepted_overwrite_audit": [
            {
                key: row[key]
                for key in (
                    "source",
                    "source_id",
                    "task_id",
                    "baseline_quality",
                    "core_answer",
                    "verifier_answer",
                    "corroborating_specialists",
                    "used_evidence_ids",
                    "cited_card_corroboration",
                )
            }
            | {
                "quality_delta": row["policy_quality"]["egor_v8"]
                - row["baseline_quality"]
            }
            for row in accepted
        ],
    }


def _summarize(
    rows: list[dict[str, Any]],
    *,
    policy: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    deltas = [
        row["policy_quality"][policy] - row["baseline_quality"] for row in rows
    ]
    low, high = shared._bootstrap_interval(
        deltas, samples=bootstrap_samples, seed=seed
    )
    accepted = [value for value in deltas if value != 0.0]
    return {
        "quality_delta": fmean(deltas),
        "ci95_low": low,
        "ci95_high": high,
        "wins": sum(value > 0.0 for value in deltas),
        "losses": sum(value < 0.0 for value in deltas),
        "ties": sum(value == 0.0 for value in deltas),
        "strict_harm_rate": fmean(value < -0.05 for value in deltas),
        "nonzero_overwrites": len(accepted),
        "exact_sign_p_two_sided": shared._exact_sign_p(
            sum(value > 0.0 for value in deltas),
            sum(value < 0.0 for value in deltas),
        ),
    }


def _contrast(
    rows: list[dict[str, Any]],
    *,
    left: str,
    right: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    deltas = [
        row["policy_quality"][left] - row["policy_quality"][right]
        for row in rows
    ]
    low, high = shared._bootstrap_interval(
        deltas, samples=bootstrap_samples, seed=seed
    )
    return {
        "left": left,
        "right": right,
        "mean_difference": fmean(deltas),
        "ci95_low": low,
        "ci95_high": high,
        "left_better_rows": sum(value > 0.0 for value in deltas),
        "left_worse_rows": sum(value < 0.0 for value in deltas),
    }


def _zero_event_cp_upper(count: int, *, alpha: float) -> float | None:
    if count <= 0:
        return None
    return 1.0 - (alpha / 2.0) ** (1.0 / count)


def _parsed(record: Mapping[str, Any] | None) -> dict[str, Any]:
    if not record or str(record.get("status", "")) != "ok":
        return {}
    value = record.get("parsed", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _load_selector_plans(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row.get("source_id", "")), str(row.get("task_id", "")))
        if not all(key) or key in rows:
            raise ValueError(f"invalid or duplicate selector plan: {key}")
        rows[key] = row
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chartqa-tasks", required=True)
    parser.add_argument("--chartqa-root", required=True)
    parser.add_argument("--infographic-tasks", required=True)
    parser.add_argument("--infographic-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--tasks-per-source", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260782)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
