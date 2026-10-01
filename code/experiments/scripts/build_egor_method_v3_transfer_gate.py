#!/usr/bin/env python3
"""Build the preregistered, hash-bound EGOR-v3 OCRBench transfer gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


COMMIT_RE = re.compile(r"[0-9a-f]{40}")
HASH_RE = re.compile(r"[0-9a-f]{64}")
DATASET_ID = "ocrbench_v2_basicvqa_transfer_v3"
PRIMARY_POLICY = "evidence_escalation_v3"
CONTROL_POLICIES = ("core_pair_gate", "resynthesis_gate", "visual_gate")
EXPECTED_CONTRASTS = ("baseline", *CONTROL_POLICIES)
EXPECTED_TASKS_SHA256 = (
    "c31ac5dcce825c75a7e7ed26cdc0f941fd9c7d8caea141b06b7249dde2c76a74"
)
EXPECTED_BOOTSTRAP_SAMPLES = 100_000
EXPECTED_STATISTICS_SEED = 20_260_741
NONINFERIORITY_MARGIN = 0.01
MAX_STRICT_HARM_RATE = 0.02
MIN_REPAIR_RATE = 0.05
MAX_REPAIR_RATE = 0.60
MAX_INCREMENTAL_CALLS = 2.0
MIN_COMPLETE_RATE = 0.99


TaskKey = tuple[str, str]


def main() -> int:
    args = _parse_args()
    report = build_gate_report(
        result_root=Path(args.result_root).resolve(),
        expected_source_commit=args.expected_source_commit,
        analysis_commit=args.analysis_commit,
        expected_tasks=args.expected_tasks,
        expected_tasks_sha256=args.expected_tasks_sha256,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(
        "EGOR-v3 OCRBench transfer gate "
        f"{'passed' if report['passed'] else 'failed'}: {output}"
    )
    return 0 if report["passed"] or not args.require_pass else 2


def build_gate_report(
    *,
    result_root: Path,
    expected_source_commit: str,
    analysis_commit: str,
    expected_tasks: int = 240,
    expected_tasks_sha256: str = EXPECTED_TASKS_SHA256,
) -> dict[str, Any]:
    _commit(expected_source_commit, "generation source commit")
    _commit(analysis_commit, "analysis commit")
    _hash(expected_tasks_sha256, "task SHA-256")
    if expected_tasks <= 0:
        raise ValueError("expected_tasks must be positive")

    markers = {
        "source_commit": result_root / "SOURCE_COMMIT",
        "tasks_sha256": result_root / "TASKS_SHA256",
        "baseline_complete": result_root / "BASELINE_COMPLETE",
        "baseline_imported": result_root / "BASELINE_IMPORTED",
        "repairability_frozen": result_root / "REPAIRABILITY_FROZEN",
        "escalation_frozen": result_root / "ESCALATION_FROZEN",
        "generation_complete": result_root / "GENERATION_COMPLETE",
        "generation_checksums": result_root / "GENERATION_SHA256SUMS",
        "analysis_complete": result_root / "ANALYSIS_COMPLETE",
        "analysis_current": result_root / "ANALYSIS_CURRENT",
    }
    for name, path in markers.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    if markers["source_commit"].read_text(encoding="utf-8").strip() != expected_source_commit:
        raise ValueError("generation source commit marker disagrees")
    if markers["tasks_sha256"].read_text(encoding="utf-8").strip() != expected_tasks_sha256:
        raise ValueError("task SHA-256 marker disagrees")

    generation_entries = _verify_checksum_manifest(
        markers["generation_checksums"], allowed_root=result_root
    )
    required_generation = {
        "baseline_qwen3.jsonl",
        "repairability.jsonl",
        "repairability_task_ids.txt",
        "specialist_exact_qwen25.jsonl",
        "specialist_visual_internvl3.jsonl",
        "core_pair.jsonl",
        "escalation_plan/plans.jsonl",
        "specialist_numerical_qwen25.jsonl",
        "specialist_verify_qwen3.jsonl",
        "full_team.jsonl",
        "baseline_resynthesis.jsonl",
        "visual_control.jsonl",
    }
    generation_relatives = {
        path.relative_to(result_root).as_posix() for path in generation_entries
    }
    missing_generation = sorted(required_generation - generation_relatives)
    if missing_generation:
        raise ValueError(
            f"generation checksum manifest lacks required artifacts: {missing_generation}"
        )
    analysis_root = _resolve_analysis_root(result_root, markers["analysis_current"])
    analysis_checksum_path = analysis_root / "SHA256SUMS"
    if not (analysis_root / "COMPLETE").is_file() or not analysis_checksum_path.is_file():
        raise FileNotFoundError("published analysis attempt is incomplete")
    analysis_entries = _verify_checksum_manifest(
        analysis_checksum_path, allowed_root=analysis_root
    )

    paths = {
        "policy_rows": analysis_root / "combined_policy_rows.csv",
        "statistics": analysis_root / "paired_statistics.json",
        "matrix_rows": analysis_root / "matrix_analysis" / "matrix_rows.csv",
        "selected_manifest": analysis_root / "selected_analysis" / "manifest.json",
        "selected_audit": analysis_root / "selected_analysis" / "execution_audit_rows.csv",
        "selected_summary": analysis_root / "selected_analysis" / "policy_summary.json",
        "core_summary": analysis_root / "core_control_analysis" / "policy_summary.json",
        "resynthesis_summary": analysis_root
        / "resynthesis_control_analysis"
        / "policy_summary.json",
        "visual_summary": analysis_root
        / "visual_control_analysis"
        / "policy_summary.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing analysis artifact {name}: {path}")

    grouped = _load_policy_rows(paths["policy_rows"])
    expected_policies = {PRIMARY_POLICY, *CONTROL_POLICIES}
    if set(grouped) != expected_policies:
        raise ValueError(
            f"policy population mismatch: {sorted(grouped)} != {sorted(expected_policies)}"
        )
    task_keys = set(grouped[PRIMARY_POLICY])
    if len(task_keys) != expected_tasks:
        raise ValueError(
            f"primary policy has {len(task_keys)} tasks, expected {expected_tasks}"
        )
    if any(set(rows) != task_keys for rows in grouped.values()):
        raise ValueError("policy task populations disagree")

    baseline_calls = _load_baseline_calls(paths["matrix_rows"], task_keys=task_keys)
    summaries = {
        policy_id: _summarize(rows, baseline_calls=baseline_calls)
        for policy_id, rows in grouped.items()
    }
    _validate_summary_files(paths=paths, summaries=summaries)

    statistics = _load_object(paths["statistics"])
    contrasts = _validate_statistics(
        statistics,
        policy_rows_path=paths["policy_rows"],
        grouped=grouped,
        expected_tasks=expected_tasks,
    )
    selected_manifest = _load_object(paths["selected_manifest"])
    tier_counts = _validate_selected_manifest(
        selected_manifest,
        expected_source_commit=expected_source_commit,
        expected_tasks=expected_tasks,
    )
    row_tier_counts = _load_tier_counts(
        paths["selected_audit"], task_keys=task_keys
    )
    if row_tier_counts != tier_counts:
        raise ValueError("selected-policy tier counts disagree with policy rows")

    primary = summaries[PRIMARY_POLICY]
    criteria: list[dict[str, Any]] = []
    _criterion(
        criteria,
        "sealed_integrity",
        True,
        {
            "generation_files": len(generation_entries),
            "analysis_files": len(analysis_entries),
        },
        "all generation and analysis checksums valid",
    )
    _criterion(
        criteria,
        "positive_transfer_effect",
        contrasts["baseline"]["mean_delta"] > 0.0
        and contrasts["baseline"]["ci95_low"] > 0.0,
        contrasts["baseline"],
        "mean > 0 and paired 95% CI lower bound > 0",
    )
    _criterion(
        criteria,
        "executed_evidence_growth",
        tier_counts["full_team"] > 0
        and contrasts["core_pair_gate"]["mean_delta"] > 0.0
        and contrasts["core_pair_gate"]["ci95_low"] >= -NONINFERIORITY_MARGIN,
        {
            "full_team_tasks": tier_counts["full_team"],
            **contrasts["core_pair_gate"],
        },
        "at least one escalation, positive point effect vs core, CI low >= -0.01",
    )
    _criterion(
        criteria,
        "expertise_vs_resynthesis",
        contrasts["resynthesis_gate"]["mean_delta"] > 0.0
        and contrasts["resynthesis_gate"]["ci95_low"] >= -NONINFERIORITY_MARGIN,
        contrasts["resynthesis_gate"],
        "positive point effect vs matched resynthesis and CI low >= -0.01",
    )
    _criterion(
        criteria,
        "multimodal_vs_visual_only",
        contrasts["visual_gate"]["mean_delta"] > 0.0
        and contrasts["visual_gate"]["ci95_low"] >= -NONINFERIORITY_MARGIN,
        contrasts["visual_gate"],
        "positive point effect vs visual-only and CI low >= -0.01",
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
        MIN_REPAIR_RATE <= primary["trigger_rate"] <= MAX_REPAIR_RATE,
        primary["trigger_rate"],
        f"[{MIN_REPAIR_RATE}, {MAX_REPAIR_RATE}]",
    )
    _criterion(
        criteria,
        "incremental_call_budget",
        primary["incremental_calls"] <= MAX_INCREMENTAL_CALLS,
        primary["incremental_calls"],
        f"<= {MAX_INCREMENTAL_CALLS}",
    )
    _criterion(
        criteria,
        "completion",
        min(summary["complete_rate"] for summary in summaries.values())
        >= MIN_COMPLETE_RATE,
        {key: value["complete_rate"] for key, value in summaries.items()},
        f"every policy >= {MIN_COMPLETE_RATE}",
    )
    _criterion(
        criteria,
        "staged_breadth_population",
        all(tier_counts[name] > 0 for name in ("baseline", "core_pair", "full_team")),
        tier_counts,
        "baseline, core, and full tiers are all represented",
    )

    artifacts = [
        {"path": str(path), "sha256": _sha256(path)}
        for path in (*markers.values(), analysis_checksum_path, *paths.values())
    ]
    return {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "generation_source_commit": expected_source_commit,
        "analysis_commit": analysis_commit,
        "tasks": expected_tasks,
        "tasks_sha256": expected_tasks_sha256,
        "primary_policy": PRIMARY_POLICY,
        "thresholds": {
            "bootstrap_samples": EXPECTED_BOOTSTRAP_SAMPLES,
            "statistics_seed": EXPECTED_STATISTICS_SEED,
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "maximum_strict_harm_rate": MAX_STRICT_HARM_RATE,
            "repair_rate_range": [MIN_REPAIR_RATE, MAX_REPAIR_RATE],
            "maximum_incremental_calls": MAX_INCREMENTAL_CALLS,
            "minimum_complete_rate": MIN_COMPLETE_RATE,
        },
        "tier_counts": tier_counts,
        "summaries": summaries,
        "contrasts": contrasts,
        "criteria": criteria,
        "passed": all(item["passed"] for item in criteria),
        "artifacts": artifacts,
    }


def _resolve_analysis_root(result_root: Path, pointer_path: Path) -> Path:
    raw = pointer_path.read_text(encoding="utf-8").strip()
    relative = Path(raw)
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != "analysis_attempts"
        or ".." in relative.parts
    ):
        raise ValueError(f"unsafe analysis pointer: {raw}")
    resolved = (result_root / relative).resolve()
    if not resolved.is_relative_to(result_root.resolve()):
        raise ValueError("analysis pointer escapes result root")
    return resolved


def _verify_checksum_manifest(path: Path, *, allowed_root: Path) -> list[Path]:
    entries: list[Path] = []
    seen: set[Path] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or HASH_RE.fullmatch(parts[0]) is None:
            raise ValueError(f"malformed checksum line in {path}")
        raw_path = parts[1].lstrip("* ")
        artifact = Path(raw_path)
        if not artifact.is_absolute():
            artifact = path.parent / artifact
        artifact = artifact.resolve()
        if not artifact.is_relative_to(allowed_root.resolve()):
            raise ValueError(f"checksum path escapes allowed root: {artifact}")
        if artifact in seen or not artifact.is_file():
            raise ValueError(f"missing or duplicate checksum artifact: {artifact}")
        if _sha256(artifact) != parts[0]:
            raise ValueError(f"checksum mismatch: {artifact}")
        seen.add(artifact)
        entries.append(artifact)
    if not entries:
        raise ValueError(f"empty checksum manifest: {path}")
    return entries


def _load_policy_rows(path: Path) -> dict[str, dict[TaskKey, dict[str, str]]]:
    grouped: dict[str, dict[TaskKey, dict[str, str]]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            policy_id = str(row.get("policy_id", "")).strip()
            key = (str(row.get("source_id", "")).strip(), str(row.get("task_id", "")).strip())
            if not policy_id or not all(key) or key in grouped[policy_id]:
                raise ValueError(f"invalid or duplicate policy row: {policy_id}:{key}")
            _validate_policy_row(row, label=f"{policy_id}:{key}")
            grouped[policy_id][key] = row
    if not grouped:
        raise ValueError("policy rows are empty")
    return dict(grouped)


def _validate_policy_row(row: dict[str, str], *, label: str) -> None:
    baseline = _rate(row.get("baseline_quality"), f"{label} baseline quality")
    policy = _rate(row.get("policy_quality"), f"{label} policy quality")
    delta = _finite(row.get("quality_delta"), f"{label} quality delta")
    if not math.isclose(policy - baseline, delta, abs_tol=1e-12):
        raise ValueError(f"quality identity fails for {label}")
    for field in ("calls", "total_tokens", "latency_seconds"):
        if _finite(row.get(field), f"{label} {field}") < 0.0:
            raise ValueError(f"negative {field} for {label}")
    for field in ("triggered", "strict_harm", "useful_repair", "provenance_rejected"):
        _boolean(row.get(field), f"{label} {field}")


def _load_baseline_calls(path: Path, *, task_keys: set[TaskKey]) -> dict[TaskKey, float]:
    calls: dict[TaskKey, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["source_id"], row["task_id"])
            value = _finite(row.get("baseline_calls"), f"baseline calls {key}")
            if value < 0.0 or (key in calls and not math.isclose(calls[key], value)):
                raise ValueError(f"invalid baseline calls for {key}")
            calls[key] = value
    if set(calls) != task_keys:
        raise ValueError("matrix baseline population disagrees with policy rows")
    return calls


def _summarize(
    rows: dict[TaskKey, dict[str, str]], *, baseline_calls: dict[TaskKey, float]
) -> dict[str, float]:
    values = list(rows.items())
    return {
        "tasks": float(len(values)),
        "baseline_quality": fmean(float(row["baseline_quality"]) for _, row in values),
        "policy_quality": fmean(float(row["policy_quality"]) for _, row in values),
        "quality_delta": fmean(float(row["quality_delta"]) for _, row in values),
        "trigger_rate": fmean(_boolean(row["triggered"], "triggered") for _, row in values),
        "strict_harm_rate": fmean(_boolean(row["strict_harm"], "strict_harm") for _, row in values),
        "useful_rate": fmean(_boolean(row["useful_repair"], "useful_repair") for _, row in values),
        "provenance_rejection_rate": fmean(
            _boolean(row["provenance_rejected"], "provenance_rejected") for _, row in values
        ),
        "calls": fmean(float(row["calls"]) for _, row in values),
        "incremental_calls": fmean(
            float(row["calls"]) - baseline_calls[key] for key, row in values
        ),
        "complete_rate": fmean(
            not str(row.get("failed_stages", "")).strip() for _, row in values
        ),
    }


def _validate_summary_files(
    *, paths: dict[str, Path], summaries: dict[str, dict[str, float]]
) -> None:
    mapping = {
        "selected_summary": PRIMARY_POLICY,
        "core_summary": "core_pair_gate",
        "resynthesis_summary": "resynthesis_gate",
        "visual_summary": "visual_gate",
    }
    for path_name, policy_id in mapping.items():
        stored = _load_object(paths[path_name])
        for field in (
            "tasks",
            "baseline_quality",
            "policy_quality",
            "quality_delta",
            "trigger_rate",
            "strict_harm_rate",
            "useful_rate",
            "provenance_rejection_rate",
            "calls",
            "complete_rate",
        ):
            if not math.isclose(
                _finite(stored.get(field), f"{policy_id} stored {field}"),
                summaries[policy_id][field],
                abs_tol=1e-12,
            ):
                raise ValueError(f"stored summary disagrees for {policy_id}:{field}")


def _validate_statistics(
    statistics: dict[str, Any],
    *,
    policy_rows_path: Path,
    grouped: dict[str, dict[TaskKey, dict[str, str]]],
    expected_tasks: int,
) -> dict[str, dict[str, Any]]:
    if (
        statistics.get("schema_version") != 1
        or statistics.get("primary_policy") != PRIMARY_POLICY
        or int(statistics.get("tasks", -1)) != expected_tasks
        or int(statistics.get("bootstrap_samples", -1)) != EXPECTED_BOOTSTRAP_SAMPLES
        or int(statistics.get("seed", -1)) != EXPECTED_STATISTICS_SEED
    ):
        raise ValueError("paired statistics contract disagrees")
    if Path(str(statistics.get("policy_rows", ""))).resolve() != policy_rows_path.resolve():
        raise ValueError("paired statistics points to another policy file")
    if statistics.get("policy_rows_sha256") != _sha256(policy_rows_path):
        raise ValueError("paired statistics does not bind the exact policy rows")
    raw = statistics.get("contrasts")
    if not isinstance(raw, dict) or set(raw) != set(EXPECTED_CONTRASTS):
        raise ValueError("paired statistics contrast population disagrees")

    primary = grouped[PRIMARY_POLICY]
    result: dict[str, dict[str, Any]] = {}
    for contrast_id in EXPECTED_CONTRASTS:
        item = raw[contrast_id]
        if not isinstance(item, dict) or int(item.get("tasks", -1)) != expected_tasks:
            raise ValueError(f"malformed contrast: {contrast_id}")
        comparison = (
            {key: float(row["baseline_quality"]) for key, row in primary.items()}
            if contrast_id == "baseline"
            else {key: float(row["policy_quality"]) for key, row in grouped[contrast_id].items()}
        )
        mean = fmean(float(primary[key]["policy_quality"]) - comparison[key] for key in primary)
        stored_mean = _finite(item.get("mean_delta"), f"{contrast_id} mean")
        low = _finite(item.get("ci95_low"), f"{contrast_id} CI low")
        high = _finite(item.get("ci95_high"), f"{contrast_id} CI high")
        if not math.isclose(mean, stored_mean, abs_tol=1e-12):
            raise ValueError(f"contrast mean disagrees with rows: {contrast_id}")
        if low > high:
            raise ValueError(f"reversed interval: {contrast_id}")
        result[contrast_id] = {
            "tasks": expected_tasks,
            "mean_delta": stored_mean,
            "ci95_low": low,
            "ci95_high": high,
            "wins": int(item.get("wins", -1)),
            "losses": int(item.get("losses", -1)),
            "ties": int(item.get("ties", -1)),
        }
        if (
            result[contrast_id]["wins"]
            + result[contrast_id]["losses"]
            + result[contrast_id]["ties"]
            != expected_tasks
        ):
            raise ValueError(f"contrast counts do not partition tasks: {contrast_id}")
    return result


def _load_tier_counts(
    path: Path, *, task_keys: set[TaskKey]
) -> dict[str, int]:
    allowed_tiers = {"baseline", "core_pair", "full_team"}
    tiers: dict[TaskKey, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row.get("source_id", "")).strip(),
                str(row.get("task_id", "")).strip(),
            )
            tier = str(row.get("tier", "")).strip()
            if not all(key) or key in tiers:
                raise ValueError(f"invalid or duplicate selected audit row: {key}")
            if tier not in allowed_tiers:
                raise ValueError(f"invalid selected audit tier: {tier}")
            tiers[key] = tier
    if set(tiers) != task_keys:
        raise ValueError("selected audit task population disagrees with policy rows")
    counts = {tier: 0 for tier in allowed_tiers}
    for tier in tiers.values():
        counts[tier] += 1
    return counts


def _validate_selected_manifest(
    manifest: dict[str, Any], *, expected_source_commit: str, expected_tasks: int
) -> dict[str, int]:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("source_commit") != expected_source_commit
        or manifest.get("policy_id") != PRIMARY_POLICY
        or int(manifest.get("tasks", -1)) != expected_tasks
        or manifest.get("require_evidence_for_change") is not True
    ):
        raise ValueError("selected-policy manifest contract disagrees")
    raw = manifest.get("tier_counts")
    allowed_tiers = {"baseline", "core_pair", "full_team"}
    if not isinstance(raw, dict) or not set(raw).issubset(allowed_tiers):
        raise ValueError("selected-policy tier population disagrees")
    result = {key: int(raw.get(key, 0)) for key in allowed_tiers}
    if any(value < 0 for value in result.values()) or sum(result.values()) != expected_tasks:
        raise ValueError("selected-policy tier counts are invalid")
    return result


def _criterion(
    output: list[dict[str, Any]], name: str, passed: bool, actual: Any, required: Any
) -> None:
    output.append({"name": name, "passed": bool(passed), "actual": actual, "required": required})


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {label}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite value for {label}")
    return result


def _rate(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"rate outside [0,1] for {label}")
    return result


def _boolean(value: Any, label: str) -> bool:
    text = str(value).strip().lower()
    if text not in {"0", "1", "false", "true"}:
        raise ValueError(f"invalid Boolean for {label}: {value}")
    return text in {"1", "true"}


def _commit(value: str, label: str) -> str:
    if COMMIT_RE.fullmatch(str(value)) is None:
        raise ValueError(f"{label} must be an exact 40-character Git commit")
    return str(value)


def _hash(value: str, label: str) -> str:
    if HASH_RE.fullmatch(str(value)) is None:
        raise ValueError(f"{label} must be an exact SHA-256")
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--expected-tasks", type=int, default=240)
    parser.add_argument("--expected-tasks-sha256", default=EXPECTED_TASKS_SHA256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
