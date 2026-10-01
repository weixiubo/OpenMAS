#!/usr/bin/env python3
"""Build the frozen Qwen3.5 EGOR transfer report and submission gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import fmean
from typing import Any


COMMIT_RE = re.compile(r"[0-9a-f]{40}")
POLICY_ID = "qwen35_frozen_egor_selector"
QWEN3_POLICY_ID = "frozen_egor_selector"
EXPECTED_BOOTSTRAP_SAMPLES = 100_000
EXPECTED_SEED = 20_260_735


def main() -> int:
    args = _parse_args()
    report = build_report(
        result_root=Path(args.result_root).resolve(),
        chartqa_result_root=Path(args.chartqa_result_root).resolve(),
        generation_commit=args.generation_source_commit,
        analysis_commit=args.analysis_commit,
        expected_tasks=args.expected_tasks,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"stronger-backbone gate {'passed' if report['passed'] else 'failed'}: {output}"
    )
    return 2 if args.require_pass and not report["passed"] else 0


def build_report(
    *,
    result_root: Path,
    chartqa_result_root: Path,
    generation_commit: str,
    analysis_commit: str,
    expected_tasks: int,
) -> dict[str, Any]:
    _validate_commit(generation_commit, "generation source commit")
    _validate_commit(analysis_commit, "analysis commit")
    _require_commit(result_root / "SOURCE_COMMIT", generation_commit)
    _require_commit(result_root / "CONTROL_COMMIT", analysis_commit)
    for marker in ("PLANS_FROZEN", "GENERATION_COMPLETE"):
        _require_file(result_root / marker)

    paths = {
        "selected_audit": result_root / "selected_generation_audit.json",
        "plan_manifest": result_root / "selector_plan" / "manifest.json",
        "policy_rows": result_root / "qwen35_selector_analysis" / "policy_rows.csv",
        "policy_manifest": result_root / "qwen35_selector_analysis" / "manifest.json",
        "statistics": result_root / "paired_statistics.json",
        "alignment_manifest": result_root
        / "evidence_alignment_audit"
        / "population_manifest.json",
        "qwen3_policy_rows": chartqa_result_root
        / "selector_analysis"
        / "policy_rows.csv",
    }
    selected_audit = _load_object(paths["selected_audit"])
    plan_manifest = _load_object(paths["plan_manifest"])
    policy_manifest = _load_object(paths["policy_manifest"])
    statistics = _load_object(paths["statistics"])
    alignment = _load_object(paths["alignment_manifest"])

    if (
        selected_audit.get("passed") is not True
        or selected_audit.get("generation_source_commit") != generation_commit
        or selected_audit.get("analysis_commit") != analysis_commit
        or int(selected_audit.get("expected_tasks", -1)) != expected_tasks
    ):
        raise ValueError("selected-generation audit is incompatible")
    _validate_bound_artifacts(selected_audit, ("artifacts",))
    if (
        plan_manifest.get("source_commit") != generation_commit
        or int(plan_manifest.get("tasks", -1)) != expected_tasks
    ):
        raise ValueError("stronger-backbone plan manifest is incompatible")
    _validate_bound_artifacts(plan_manifest, ("inputs", "outputs"))
    if (
        policy_manifest.get("source_commit") != generation_commit
        or policy_manifest.get("policy_id") != POLICY_ID
        or int(policy_manifest.get("tasks", -1)) != expected_tasks
    ):
        raise ValueError("stronger-backbone policy manifest is incompatible")
    _validate_bound_artifacts(policy_manifest, ("inputs", "outputs"))
    if (
        alignment.get("source_commit") != generation_commit
        or alignment.get("analysis_commit") != analysis_commit
        or alignment.get("policy_id") != POLICY_ID
        or int(alignment.get("task_count", -1)) != expected_tasks
    ):
        raise ValueError("stronger-backbone alignment manifest is incompatible")
    _validate_bound_artifacts(alignment, ("inputs", "outputs"))

    if (
        statistics.get("primary_policy") != POLICY_ID
        or int(statistics.get("tasks", -1)) != expected_tasks
        or int(statistics.get("bootstrap_samples", -1)) != EXPECTED_BOOTSTRAP_SAMPLES
        or int(statistics.get("seed", -1)) != EXPECTED_SEED
    ):
        raise ValueError("stronger-backbone paired statistics are incompatible")
    if Path(str(statistics.get("policy_rows", ""))).resolve() != paths[
        "policy_rows"
    ].resolve() or str(statistics.get("policy_rows_sha256", "")) != _sha256(
        paths["policy_rows"]
    ):
        raise ValueError(
            "stronger-backbone statistics do not bind the exact policy rows"
        )
    contrast = statistics.get("contrasts", {}).get("baseline", {})
    if (
        not isinstance(contrast, dict)
        or int(contrast.get("tasks", -1)) != expected_tasks
    ):
        raise ValueError("stronger-backbone baseline contrast is missing")
    _validate_contrast(contrast, expected_tasks=expected_tasks)

    rows = _policy_rows(paths["policy_rows"], required_policy=POLICY_ID)
    qwen3_rows = _policy_rows(
        paths["qwen3_policy_rows"], required_policy=QWEN3_POLICY_ID
    )
    if len(rows) != expected_tasks or set(rows) != set(qwen3_rows):
        raise ValueError("Qwen3/Qwen3.5 task populations do not match")
    ordered = sorted(rows)
    for key in ordered:
        _validate_policy_row(rows[key], label=f"Qwen3.5:{key}")
        _validate_policy_row(qwen3_rows[key], label=f"Qwen3:{key}")
    qwen35_baseline = fmean(
        _rate(rows[key]["baseline_quality"], f"Qwen3.5 baseline:{key}")
        for key in ordered
    )
    qwen35_policy = fmean(
        _rate(rows[key]["policy_quality"], f"Qwen3.5 policy:{key}") for key in ordered
    )
    qwen3_baseline = fmean(
        _rate(qwen3_rows[key]["baseline_quality"], f"Qwen3 baseline:{key}")
        for key in ordered
    )
    mean_delta = qwen35_policy - qwen35_baseline
    if not math.isclose(
        mean_delta,
        _finite(contrast["mean_delta"], "paired mean"),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("policy rows and paired statistics disagree")
    strict_harm_rate = fmean(_boolean(rows[key]["strict_harm"]) for key in ordered)
    useful_repair_rate = fmean(_boolean(rows[key]["useful_repair"]) for key in ordered)
    trigger_rate = fmean(_boolean(rows[key]["triggered"]) for key in ordered)
    mean_calls = fmean(
        _nonnegative(rows[key]["calls"], f"total calls:{key}") for key in ordered
    )
    headroom = qwen35_baseline - qwen3_baseline

    criteria: list[dict[str, Any]] = []
    _criterion(criteria, "qwen35_baseline_headroom", headroom >= 0.0, headroom, 0.0)
    _criterion(criteria, "positive_policy_effect", mean_delta > 0.0, mean_delta, "> 0")
    _criterion(
        criteria,
        "paired_noninferiority",
        float(contrast["ci95_low"]) >= -0.01,
        float(contrast["ci95_low"]),
        -0.01,
    )
    _criterion(
        criteria,
        "strict_harm_rate",
        strict_harm_rate <= 0.05,
        strict_harm_rate,
        0.05,
    )
    _criterion(criteria, "mean_total_calls", mean_calls <= 2.0, mean_calls, 2.0)

    artifact_paths = [*paths.values()]
    return {
        "schema_version": 1,
        "generation_source_commit": generation_commit,
        "analysis_commit": analysis_commit,
        "tasks": expected_tasks,
        "policy_id": POLICY_ID,
        "qwen3_baseline_quality": qwen3_baseline,
        "qwen35_baseline_quality": qwen35_baseline,
        "qwen35_baseline_minus_qwen3": headroom,
        "qwen35_egor_quality": qwen35_policy,
        "qwen35_egor_delta": mean_delta,
        "ci95_low": float(contrast["ci95_low"]),
        "ci95_high": float(contrast["ci95_high"]),
        "trigger_rate": trigger_rate,
        "useful_repair_rate": useful_repair_rate,
        "strict_harm_rate": strict_harm_rate,
        "mean_total_calls": mean_calls,
        "criteria": criteria,
        "passed": all(item["passed"] for item in criteria),
        "artifacts": [
            {"path": str(path), "sha256": _sha256(path)} for path in artifact_paths
        ],
    }


def _policy_rows(
    path: Path, *, required_policy: str | None = None
) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if required_policy is not None and row.get("policy_id") != required_policy:
                raise ValueError(f"unexpected policy in {path}")
            key = (str(row.get("source_id", "")), str(row.get("task_id", "")))
            if not all(key) or key in rows:
                raise ValueError(f"invalid or duplicate policy task in {path}")
            rows[key] = row
    return rows


def _validate_policy_row(row: dict[str, str], *, label: str) -> None:
    baseline = _rate(row.get("baseline_quality"), f"{label} baseline quality")
    policy = _rate(row.get("policy_quality"), f"{label} policy quality")
    delta = _finite(row.get("quality_delta"), f"{label} quality delta")
    if not math.isclose(policy - baseline, delta, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"quality delta mismatch for {label}")
    for field in ("triggered", "strict_harm", "useful_repair"):
        _boolean(row.get(field))
    _nonnegative(row.get("calls"), f"{label} total calls")


def _validate_contrast(contrast: dict[str, Any], *, expected_tasks: int) -> None:
    if int(contrast.get("bootstrap_samples", -1)) != EXPECTED_BOOTSTRAP_SAMPLES:
        raise ValueError("stronger-backbone contrast has the wrong bootstrap count")
    if int(contrast.get("seed", -1)) != EXPECTED_SEED:
        raise ValueError("stronger-backbone contrast has the wrong statistics seed")
    low = _finite(contrast.get("ci95_low"), "paired CI lower bound")
    high = _finite(contrast.get("ci95_high"), "paired CI upper bound")
    _finite(contrast.get("mean_delta"), "paired mean")
    if low > high:
        raise ValueError("stronger-backbone paired interval is reversed")
    counts = [
        _integer(contrast.get(field), f"paired {field}")
        for field in ("wins", "losses", "ties")
    ]
    if sum(counts) != expected_tasks:
        raise ValueError("stronger-backbone paired counts do not partition tasks")
    for field in ("sign_p_one_sided", "sign_p_two_sided"):
        value = _finite(contrast.get(field), field)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid stronger-backbone {field}")


def _validate_bound_artifacts(payload: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        values = payload.get(key)
        if not isinstance(values, list):
            raise ValueError(f"manifest lacks artifact list: {key}")
        for raw in values:
            if not isinstance(raw, dict):
                raise ValueError(f"invalid artifact entry in {key}")
            path = Path(str(raw.get("path", ""))).resolve()
            if _sha256(path) != str(raw.get("sha256", "")):
                raise ValueError(f"bound artifact hash mismatch: {path}")


def _criterion(
    criteria: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    required: Any,
) -> None:
    criteria.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "required": required,
        }
    )


def _require_commit(path: Path, expected: str) -> None:
    _require_file(path)
    if path.read_text(encoding="utf-8").strip() != expected:
        raise ValueError(f"commit mismatch: {path}")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing required file: {path}")


def _load_object(path: Path) -> dict[str, Any]:
    _require_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite: {value}")
    return result


def _rate(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} lies outside [0, 1]: {value}")
    return result


def _nonnegative(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} is negative: {value}")
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
    raise ValueError(f"invalid Boolean value: {value}")


def _validate_commit(value: str, label: str) -> None:
    if COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact 40-character Git commit")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--chartqa-result-root", required=True)
    parser.add_argument("--generation-source-commit", required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--expected-tasks", type=int, default=240)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
