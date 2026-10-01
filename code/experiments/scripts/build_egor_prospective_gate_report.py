#!/usr/bin/env python3
"""Build a hash-bound, outcome-read-only EGOR prospective paper gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


PRIMARY_POLICY = "frozen_egor_selector"
REQUIRED_CONTROLS = (
    "standalone_candidate_value",
    "paid_llm_scheduler",
    "best_fixed_visual_gap",
    "full_team_all",
    "baseline_resynthesis_all",
    "matched_random_repair",
    "selector_trigger_resynthesis",
)
EXPECTED_BOOTSTRAP_SAMPLES = 100_000
EXPECTED_STATISTICS_SEED = 20_260_712
NONINFERIORITY_MARGIN = 0.01
MIN_CALL_SAVING = 0.25
MAX_STRICT_HARM_RATE = 0.02
MIN_TRIGGER_RATE = 0.05
MAX_TRIGGER_RATE = 0.60
MIN_COMPLETE_RATE = 0.99


def main() -> int:
    args = _parse_args()
    root = Path(args.result_root).resolve()
    report = build_gate_report(
        result_root=root,
        dataset_id=args.dataset_id,
        expected_source_commit=args.expected_source_commit,
        expected_tasks=args.expected_tasks,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"prospective paper gate {'passed' if report['passed'] else 'failed'}: {output}"
    )
    if args.require_pass and not report["passed"]:
        return 2
    return 0


def build_gate_report(
    *,
    result_root: Path,
    dataset_id: str,
    expected_source_commit: str,
    expected_tasks: int = 240,
) -> dict[str, Any]:
    if not dataset_id.strip():
        raise ValueError("dataset_id is required")
    if expected_tasks <= 0:
        raise ValueError("expected_tasks must be positive")

    paths = {
        "source_commit": result_root / "SOURCE_COMMIT",
        "plans_frozen": result_root / "PLANS_FROZEN",
        "generation_complete": result_root / "GENERATION_COMPLETE",
        "analysis_complete": result_root / "ANALYSIS_COMPLETE",
        "generation_audit": result_root / "generation_audit.json",
        "policy_bundle": result_root / "policy_bundle.csv",
        "paired_statistics": result_root / "paired_statistics.json",
        "matrix_rows": result_root / "matrix_analysis" / "matrix_rows.csv",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    source_commit = paths["source_commit"].read_text(encoding="utf-8").strip()
    audit = _load_json(paths["generation_audit"])
    statistics = _load_json(paths["paired_statistics"])
    grouped = _load_policy_rows(paths["policy_bundle"])
    baseline_resources = _load_baseline_resources(paths["matrix_rows"])

    required_policies = (PRIMARY_POLICY, *REQUIRED_CONTROLS)
    missing_policies = sorted(set(required_policies) - set(grouped))
    if missing_policies:
        raise ValueError(f"policy bundle lacks required policies: {missing_policies}")
    primary_keys = set(grouped[PRIMARY_POLICY])
    if len(primary_keys) != expected_tasks:
        raise ValueError(
            f"primary policy has {len(primary_keys)} tasks, expected {expected_tasks}"
        )
    if set(baseline_resources) != primary_keys:
        raise ValueError("matrix baseline resources do not match primary task keys")
    for policy_id in required_policies:
        if set(grouped[policy_id]) != primary_keys:
            raise ValueError(f"task mismatch for policy {policy_id}")

    _validate_policy_rows(
        grouped,
        policy_ids=required_policies,
        task_keys=primary_keys,
    )

    if statistics.get("schema_version") != 1:
        raise ValueError("paired statistics schema is invalid")
    if statistics.get("primary_policy") != PRIMARY_POLICY:
        raise ValueError("paired statistics primary policy is not frozen EGOR")
    if int(statistics.get("tasks", -1)) != expected_tasks:
        raise ValueError("paired statistics task count is invalid")
    if int(statistics.get("bootstrap_samples", -1)) != EXPECTED_BOOTSTRAP_SAMPLES:
        raise ValueError("paired statistics bootstrap count is not frozen")
    if int(statistics.get("seed", -1)) != EXPECTED_STATISTICS_SEED:
        raise ValueError("paired statistics seed is not frozen")
    contrasts = statistics.get("contrasts")
    if not isinstance(contrasts, dict):
        raise ValueError("paired statistics contrasts are invalid")
    missing_contrasts = sorted({"baseline", *REQUIRED_CONTROLS} - set(contrasts))
    if missing_contrasts:
        raise ValueError(f"paired statistics lacks contrasts: {missing_contrasts}")
    _validate_statistics_binding(
        statistics,
        policy_rows_path=paths["policy_bundle"],
        grouped=grouped,
        required_controls=REQUIRED_CONTROLS,
        expected_tasks=expected_tasks,
    )

    summaries = {
        policy_id: _summarize_policy(
            grouped[policy_id], baseline_resources=baseline_resources
        )
        for policy_id in required_policies
    }
    primary = summaries[PRIMARY_POLICY]
    criteria: list[dict[str, Any]] = []
    _criterion(
        criteria,
        "source_commit",
        source_commit == expected_source_commit,
        source_commit,
        expected_source_commit,
    )
    _criterion(
        criteria,
        "generation_audit",
        audit.get("passed") is True
        and int(audit.get("tasks", {}).get("count", -1)) == expected_tasks,
        {"passed": audit.get("passed"), "tasks": audit.get("tasks", {}).get("count")},
        {"passed": True, "tasks": expected_tasks},
    )
    _criterion(
        criteria,
        "plans_precede_repairs",
        _plans_precede_repairs(result_root, paths["plans_frozen"]),
        "filesystem timestamps",
        "every decision plan and PLANS_FROZEN precede every specialist output",
    )
    baseline_contrast = _contrast(contrasts, "baseline", expected_tasks)
    _criterion(
        criteria,
        "positive_heldout_effect",
        baseline_contrast["ci95_low"] > 0.0,
        baseline_contrast,
        "paired 95% CI lower bound > 0",
    )
    for control_id in (
        "best_fixed_visual_gap",
        "full_team_all",
        "paid_llm_scheduler",
    ):
        contrast = _contrast(contrasts, control_id, expected_tasks)
        call_saving = (
            summaries[control_id]["incremental_calls"] - primary["incremental_calls"]
        )
        dominates = contrast["ci95_low"] > 0.0
        efficient_noninferior = (
            contrast["ci95_low"] >= -NONINFERIORITY_MARGIN
            and call_saving >= MIN_CALL_SAVING
        )
        _criterion(
            criteria,
            f"pareto_vs_{control_id}",
            dominates or efficient_noninferior,
            {
                "quality_contrast": contrast,
                "incremental_call_saving": call_saving,
            },
            (
                "quality CI lower bound > 0, or lower bound >= -0.01 with "
                "at least 0.25 incremental calls saved"
            ),
        )
    for control_id in (
        "matched_random_repair",
        "selector_trigger_resynthesis",
        "baseline_resynthesis_all",
    ):
        contrast = _contrast(contrasts, control_id, expected_tasks)
        _criterion(
            criteria,
            f"mechanism_vs_{control_id}",
            contrast["mean_delta"] > 0.0 and contrast["ci95_low"] >= 0.0,
            contrast,
            "positive paired mean with 95% CI lower bound >= 0",
        )
    target_contrast = _contrast(contrasts, "standalone_candidate_value", expected_tasks)
    _criterion(
        criteria,
        "paired_target_vs_standalone_candidate_value",
        target_contrast["mean_delta"] > 0.0 and target_contrast["ci95_low"] >= 0.0,
        target_contrast,
        "positive paired mean with 95% CI lower bound >= 0",
    )
    _criterion(
        criteria,
        "strict_harm",
        primary["strict_harm_rate"] <= MAX_STRICT_HARM_RATE,
        primary["strict_harm_rate"],
        f"<= {MAX_STRICT_HARM_RATE}",
    )
    _criterion(
        criteria,
        "repair_rate",
        MIN_TRIGGER_RATE <= primary["trigger_rate"] <= MAX_TRIGGER_RATE,
        primary["trigger_rate"],
        f"in [{MIN_TRIGGER_RATE}, {MAX_TRIGGER_RATE}]",
    )
    _criterion(
        criteria,
        "completion",
        primary["complete_rate"] >= MIN_COMPLETE_RATE,
        primary["complete_rate"],
        f">= {MIN_COMPLETE_RATE}",
    )

    artifact_paths = (
        paths["generation_audit"],
        paths["policy_bundle"],
        paths["paired_statistics"],
        paths["matrix_rows"],
        result_root / "selector_plan" / "plans.jsonl",
        result_root / "candidate_value_plan" / "plans.jsonl",
        result_root / "scheduler_plan" / "plans.jsonl",
        result_root / "matched_control_plans.jsonl",
    )
    for path in artifact_paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen result artifact: {path}")
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "source_commit": source_commit,
        "tasks": expected_tasks,
        "primary_policy": PRIMARY_POLICY,
        "thresholds": {
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "minimum_incremental_call_saving": MIN_CALL_SAVING,
            "maximum_strict_harm_rate": MAX_STRICT_HARM_RATE,
            "trigger_rate_interval": [MIN_TRIGGER_RATE, MAX_TRIGGER_RATE],
            "minimum_complete_rate": MIN_COMPLETE_RATE,
            "bootstrap_samples": EXPECTED_BOOTSTRAP_SAMPLES,
            "statistics_seed": EXPECTED_STATISTICS_SEED,
        },
        "summaries": summaries,
        "contrasts": {
            name: contrasts[name] for name in ("baseline", *REQUIRED_CONTROLS)
        },
        "criteria": criteria,
        "passed": all(item["passed"] for item in criteria),
        "artifacts": [
            {"path": str(path), "sha256": _sha256(path)} for path in artifact_paths
        ],
    }


def _load_policy_rows(
    path: Path,
) -> dict[str, dict[tuple[str, str], dict[str, str]]]:
    grouped: dict[str, dict[tuple[str, str], dict[str, str]]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            policy_id = str(row.get("policy_id", "")).strip()
            key = (
                str(row.get("source_id", "")).strip(),
                str(row.get("task_id", "")).strip(),
            )
            if not policy_id or not all(key) or key in grouped[policy_id]:
                raise ValueError(f"invalid or duplicate policy row: {policy_id}:{key}")
            grouped[policy_id][key] = row
    if not grouped:
        raise ValueError("policy bundle is empty")
    return dict(grouped)


def _load_baseline_resources(
    path: Path,
) -> dict[tuple[str, str], tuple[float, float, float]]:
    resources: dict[tuple[str, str], tuple[float, float, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["source_id"], row["task_id"])
            value = (
                _finite(row["baseline_calls"]),
                _finite(row["baseline_total_tokens"]),
                _finite(row["baseline_latency_seconds"]),
            )
            if key in resources and resources[key] != value:
                raise ValueError(f"inconsistent baseline resources for {key}")
            resources[key] = value
    if not resources:
        raise ValueError("matrix rows are empty")
    return resources


def _summarize_policy(
    rows: dict[tuple[str, str], dict[str, str]],
    *,
    baseline_resources: dict[tuple[str, str], tuple[float, float, float]],
) -> dict[str, float]:
    values = list(rows.items())
    calls = [_finite(row["calls"]) for _, row in values]
    tokens = [_finite(row["total_tokens"]) for _, row in values]
    latency = [_finite(row["latency_seconds"]) for _, row in values]
    baseline_calls = [baseline_resources[key][0] for key, _ in values]
    baseline_tokens = [baseline_resources[key][1] for key, _ in values]
    baseline_latency = [baseline_resources[key][2] for key, _ in values]
    return {
        "tasks": float(len(values)),
        "baseline_quality": fmean(
            _finite(row["baseline_quality"]) for _, row in values
        ),
        "policy_quality": fmean(_finite(row["policy_quality"]) for _, row in values),
        "quality_delta": fmean(_finite(row["quality_delta"]) for _, row in values),
        "trigger_rate": fmean(_boolean(row["triggered"]) for _, row in values),
        "strict_harm_rate": fmean(_boolean(row["strict_harm"]) for _, row in values),
        "useful_rate": fmean(_boolean(row["useful_repair"]) for _, row in values),
        "provenance_rejection_rate": fmean(
            _boolean(row["provenance_rejected"]) for _, row in values
        ),
        "complete_rate": fmean(
            not str(row["failed_stages"]).strip() for _, row in values
        ),
        "calls": fmean(calls),
        "incremental_calls": fmean(
            a - b for a, b in zip(calls, baseline_calls, strict=True)
        ),
        "total_tokens": fmean(tokens),
        "incremental_tokens": fmean(
            a - b for a, b in zip(tokens, baseline_tokens, strict=True)
        ),
        "latency_seconds": fmean(latency),
        "incremental_latency_seconds": fmean(
            a - b for a, b in zip(latency, baseline_latency, strict=True)
        ),
    }


def _validate_policy_rows(
    grouped: dict[str, dict[tuple[str, str], dict[str, str]]],
    *,
    policy_ids: tuple[str, ...],
    task_keys: set[tuple[str, str]],
) -> None:
    for key in sorted(task_keys):
        primary_row = grouped[PRIMARY_POLICY][key]
        expected_baseline = _rate(
            primary_row.get("baseline_quality"),
            f"baseline quality for {PRIMARY_POLICY}:{key}",
        )
        for policy_id in policy_ids:
            row = grouped[policy_id][key]
            baseline_quality = _rate(
                row.get("baseline_quality"),
                f"baseline quality for {policy_id}:{key}",
            )
            policy_quality = _rate(
                row.get("policy_quality"),
                f"policy quality for {policy_id}:{key}",
            )
            quality_delta = _finite(row.get("quality_delta"))
            if not math.isclose(
                baseline_quality, expected_baseline, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"baseline quality mismatch for {policy_id}:{key}")
            if not math.isclose(
                policy_quality - baseline_quality,
                quality_delta,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"quality delta mismatch for {policy_id}:{key}")
            for field in ("calls", "total_tokens", "latency_seconds"):
                value = _finite(row.get(field))
                if value < 0.0:
                    raise ValueError(f"negative {field} for {policy_id}:{key}: {value}")
            for field in (
                "triggered",
                "strict_harm",
                "useful_repair",
                "provenance_rejected",
            ):
                _boolean(row.get(field))


def _validate_statistics_binding(
    statistics: dict[str, Any],
    *,
    policy_rows_path: Path,
    grouped: dict[str, dict[tuple[str, str], dict[str, str]]],
    required_controls: tuple[str, ...],
    expected_tasks: int,
) -> None:
    recorded_path = str(statistics.get("policy_rows", "")).strip()
    recorded_sha256 = str(statistics.get("policy_rows_sha256", "")).strip()
    if (
        not recorded_path
        or Path(recorded_path).resolve() != policy_rows_path.resolve()
        or recorded_sha256 != _sha256(policy_rows_path)
    ):
        raise ValueError("paired statistics do not bind the exact policy bundle")

    triggered_tasks = sum(
        _boolean(row.get("triggered")) for row in grouped[PRIMARY_POLICY].values()
    )
    if _integer(statistics.get("triggered_tasks"), "triggered task count") != (
        triggered_tasks
    ):
        raise ValueError("paired statistics triggered-task count is inconsistent")

    contrasts = statistics["contrasts"]
    primary = grouped[PRIMARY_POLICY]
    for contrast_id in ("baseline", *required_controls):
        contrast = _contrast(contrasts, contrast_id, expected_tasks)
        if contrast_id == "baseline":
            expected_mean = fmean(
                _finite(row.get("policy_quality"))
                - _finite(row.get("baseline_quality"))
                for row in primary.values()
            )
        else:
            comparison = grouped[contrast_id]
            expected_mean = fmean(
                _finite(primary[key].get("policy_quality"))
                - _finite(comparison[key].get("policy_quality"))
                for key in sorted(primary)
            )
        if not math.isclose(
            _finite(contrast.get("mean_delta")),
            expected_mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"paired contrast mean disagrees with policy rows: {contrast_id}"
            )


def _contrast(
    contrasts: dict[str, Any], name: str, expected_tasks: int
) -> dict[str, Any]:
    raw = contrasts.get(name)
    if (
        not isinstance(raw, dict)
        or _integer(raw.get("tasks"), f"{name} task count") != expected_tasks
    ):
        raise ValueError(f"invalid paired contrast: {name}")
    mean_delta = _finite(raw.get("mean_delta"))
    ci95_low = _finite(raw.get("ci95_low"))
    ci95_high = _finite(raw.get("ci95_high"))
    if ci95_low > ci95_high:
        raise ValueError(f"invalid paired interval ordering: {name}")
    if (
        _integer(raw.get("bootstrap_samples"), f"{name} bootstrap count")
        != EXPECTED_BOOTSTRAP_SAMPLES
    ):
        raise ValueError(f"invalid paired bootstrap count: {name}")
    if _integer(raw.get("seed"), f"{name} statistics seed") != (
        EXPECTED_STATISTICS_SEED
    ):
        raise ValueError(f"invalid paired statistics seed: {name}")
    wins = _integer(raw.get("wins"), f"{name} wins")
    losses = _integer(raw.get("losses"), f"{name} losses")
    ties = _integer(raw.get("ties"), f"{name} ties")
    if wins + losses + ties != expected_tasks:
        raise ValueError(f"paired contrast counts do not partition tasks: {name}")
    for field in ("sign_p_one_sided", "sign_p_two_sided"):
        value = _finite(raw.get(field))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid {field} for paired contrast: {name}")
    raw["mean_delta"] = mean_delta
    raw["ci95_low"] = ci95_low
    raw["ci95_high"] = ci95_high
    return raw


def _plans_precede_repairs(root: Path, plans_frozen: Path) -> bool:
    plans = [
        plans_frozen,
        root / "selector_plan" / "plans.jsonl",
        root / "candidate_value_plan" / "plans.jsonl",
        root / "scheduler_plan" / "plans.jsonl",
        root / "matched_control_plans.jsonl",
    ]
    repairs = [
        root / "specialist_exact_qwen25.jsonl",
        root / "specialist_visual_internvl3.jsonl",
        root / "specialist_numerical_qwen25.jsonl",
        root / "specialist_verify_qwen3.jsonl",
    ]
    if not all(path.is_file() for path in plans):
        raise FileNotFoundError("one or more frozen plan artifacts are missing")
    if not all(path.is_file() for path in repairs):
        raise FileNotFoundError("one or more specialist outputs are missing")
    latest_plan_time = max(path.stat().st_mtime_ns for path in plans)
    return all(latest_plan_time <= path.stat().st_mtime_ns for path in repairs)


def _criterion(
    rows: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    requirement: Any,
) -> None:
    rows.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "requirement": requirement,
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value}")
    return result


def _rate(value: Any, label: str) -> float:
    result = _finite(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} lies outside [0, 1]: {value}")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not an integer: {value}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an integer: {value}") from exc
    if str(result) != str(value).strip() or result < 0:
        raise ValueError(f"{label} is not a non-negative integer: {value}")
    return result


def _boolean(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-tasks", type=int, default=240)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
