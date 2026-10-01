#!/usr/bin/env python3
"""Analyze the frozen prospective EGOR policy with and without fallback."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

from openagora_experiments.egor.statistics import paired_bootstrap_contrast


COMMIT_RE = re.compile(r"[0-9a-f]{40}")
HASH_RE = re.compile(r"[0-9a-f]{64}")
DATASET_ID = "chartqa_heldout_v1"
CERTIFIED_POLICY = "frozen_egor_selector"
NO_CERTIFICATE_POLICY = "frozen_egor_selector_no_certificate"
BOOTSTRAP_SAMPLES = 100_000
BOOTSTRAP_SEED = 20_260_712
HARM_TOLERANCE = 0.05
USEFUL_MARGIN = 0.05
REQUIRED_FIELDS = (
    "source_id",
    "task_id",
    "policy_id",
    "candidate_id",
    "selected_action_ids",
    "combination_id",
    "triggered",
    "baseline_quality",
    "policy_quality",
    "quality_delta",
    "calls",
    "total_tokens",
    "latency_seconds",
    "strict_harm",
    "useful_repair",
    "provenance_rejected",
    "failed_stages",
)


def main() -> int:
    args = _parse_args()
    certified_rows_path = Path(args.certified_rows).resolve()
    certified_manifest_path = Path(args.certified_manifest).resolve()
    no_certificate_rows_path = Path(args.no_certificate_rows).resolve()
    no_certificate_manifest_path = Path(args.no_certificate_manifest).resolve()
    output_rows_path = Path(args.output_rows).resolve()
    output_report_path = Path(args.output_report).resolve()
    source_commit = _commit(args.generation_source_commit, "generation commit")
    analysis_commit = _commit(args.analysis_commit, "analysis commit")

    rows, report = build_control(
        certified_rows_path=certified_rows_path,
        certified_manifest=_load_object(certified_manifest_path),
        no_certificate_rows_path=no_certificate_rows_path,
        no_certificate_manifest=_load_object(no_certificate_manifest_path),
        generation_source_commit=source_commit,
        analysis_commit=analysis_commit,
        expected_tasks=args.expected_tasks,
    )
    _write_rows(output_rows_path, rows)
    report.update(
        {
            "certified_rows_sha256": _sha256(certified_rows_path),
            "certified_manifest_sha256": _sha256(certified_manifest_path),
            "no_certificate_rows_sha256": _sha256(no_certificate_rows_path),
            "no_certificate_manifest_sha256": _sha256(
                no_certificate_manifest_path
            ),
            "output_rows_sha256": _sha256(output_rows_path),
        }
    )
    _write_text(
        output_report_path,
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(
        "analyzed fixed-decision certificate control over "
        f"{report['tasks']} tasks"
    )
    return 0


def build_control(
    *,
    certified_rows_path: Path,
    certified_manifest: dict[str, Any],
    no_certificate_rows_path: Path,
    no_certificate_manifest: dict[str, Any],
    generation_source_commit: str,
    analysis_commit: str,
    expected_tasks: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generation_source_commit = _commit(
        generation_source_commit, "generation commit"
    )
    analysis_commit = _commit(analysis_commit, "analysis commit")
    if expected_tasks <= 0:
        raise ValueError("expected task count must be positive")
    certified_inputs = _validate_manifest(
        certified_manifest,
        rows_path=certified_rows_path,
        expected_policy=CERTIFIED_POLICY,
        expected_gate=True,
        generation_source_commit=generation_source_commit,
        expected_tasks=expected_tasks,
    )
    no_certificate_inputs = _validate_manifest(
        no_certificate_manifest,
        rows_path=no_certificate_rows_path,
        expected_policy=NO_CERTIFICATE_POLICY,
        expected_gate=False,
        generation_source_commit=generation_source_commit,
        expected_tasks=expected_tasks,
    )
    if certified_inputs != no_certificate_inputs:
        raise ValueError("certified and no-certificate scorings use different inputs")

    certified = _load_rows(certified_rows_path, expected_policy=CERTIFIED_POLICY)
    no_certificate = _load_rows(
        no_certificate_rows_path,
        expected_policy=NO_CERTIFICATE_POLICY,
    )
    if set(certified) != set(no_certificate) or len(certified) != expected_tasks:
        raise ValueError("certificate-control task population mismatch")

    output_rows: list[dict[str, Any]] = []
    for key in sorted(certified):
        output_rows.append(_compare_row(certified[key], no_certificate[key]))
    summary = _summarize(output_rows)
    contrast = paired_bootstrap_contrast(
        [float(row["no_certificate_quality"]) for row in output_rows],
        [float(row["certified_quality"]) for row in output_rows],
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    report = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "generation_source_commit": generation_source_commit,
        "analysis_commit": analysis_commit,
        "tasks": expected_tasks,
        "certified_policy_id": CERTIFIED_POLICY,
        "no_certificate_policy_id": NO_CERTIFICATE_POLICY,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "fixed_decision_invariant_passed": True,
        "shared_input_artifacts": [
            {"path": path, "sha256": digest}
            for path, digest in certified_inputs
        ],
        "summary": summary,
        "no_certificate_minus_certified": asdict(contrast),
    }
    return output_rows, report


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    rows_path: Path,
    expected_policy: str,
    expected_gate: bool,
    generation_source_commit: str,
    expected_tasks: int,
) -> tuple[tuple[str, str], ...]:
    if (
        _integer(manifest.get("schema_version"), "manifest schema") != 1
        or manifest.get("source_commit") != generation_source_commit
        or manifest.get("policy_id") != expected_policy
        or _integer(manifest.get("tasks"), "manifest task count")
        != expected_tasks
        or manifest.get("strict_provenance_gate") is not expected_gate
    ):
        raise ValueError(f"incompatible policy manifest: {expected_policy}")
    inputs = _artifact_map(manifest.get("inputs"), label="manifest inputs")
    outputs = _artifact_map(manifest.get("outputs"), label="manifest outputs")
    resolved_rows = str(rows_path.resolve())
    if outputs.get(resolved_rows) != _sha256(rows_path):
        raise ValueError(f"manifest does not bind policy rows: {rows_path}")
    return tuple(sorted(inputs.items()))


def _artifact_map(raw: Any, *, label: str) -> dict[str, str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label} are missing")
    result: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"malformed artifact in {label}")
        path = Path(str(item.get("path", ""))).resolve()
        digest = str(item.get("sha256", ""))
        _hash(digest, f"{label} hash")
        normalized = str(path)
        if normalized in result:
            raise ValueError(f"duplicate artifact in {label}: {path}")
        if _sha256(path) != digest:
            raise ValueError(f"artifact hash mismatch in {label}: {path}")
        result[normalized] = digest
    return result


def _load_rows(
    path: Path, *, expected_policy: str
) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(
            field not in reader.fieldnames for field in REQUIRED_FIELDS
        ):
            raise ValueError(f"policy rows lack required fields: {path}")
        for raw in reader:
            key = (str(raw.get("source_id", "")), str(raw.get("task_id", "")))
            if not all(key) or key in rows:
                raise ValueError(f"invalid or duplicate policy task: {key}")
            if raw.get("policy_id") != expected_policy:
                raise ValueError(f"unexpected policy ID in {path}")
            rows[key] = raw
    if not rows:
        raise ValueError(f"policy rows are empty: {path}")
    return rows


def _compare_row(
    certified: dict[str, str], no_certificate: dict[str, str]
) -> dict[str, Any]:
    key = (certified["source_id"], certified["task_id"])
    for field in (
        "source_id",
        "task_id",
        "candidate_id",
        "selected_action_ids",
        "combination_id",
        "triggered",
        "failed_stages",
    ):
        if certified[field] != no_certificate[field]:
            raise ValueError(f"fixed decision differs for {key}: {field}")
    for field in (
        "baseline_quality",
        "calls",
        "total_tokens",
        "latency_seconds",
    ):
        certified_value = _finite(certified[field])
        no_certificate_value = _finite(no_certificate[field])
        if field != "baseline_quality" and (
            certified_value < 0.0 or no_certificate_value < 0.0
        ):
            raise ValueError(f"negative resource for {key}: {field}")
        if not math.isclose(
            certified_value,
            no_certificate_value,
            abs_tol=1e-12,
        ):
            raise ValueError(f"fixed resource/state differs for {key}: {field}")

    certified_values = _outcomes(certified, label=f"certified {key}")
    raw_values = _outcomes(no_certificate, label=f"no-certificate {key}")
    rejected = _binary(certified["provenance_rejected"], "provenance rejection")
    if _binary(no_certificate["provenance_rejected"], "raw provenance rejection"):
        raise ValueError(f"no-certificate row is marked rejected: {key}")
    triggered = _binary(certified["triggered"], "triggered")
    baseline = _finite(certified["baseline_quality"])
    if rejected:
        if not triggered:
            raise ValueError(f"untriggered policy row is certificate-rejected: {key}")
        if not math.isclose(certified_values["quality"], baseline, abs_tol=1e-12):
            raise ValueError(f"certificate rejection does not fall back: {key}")
    else:
        for field in ("quality", "delta", "strict_harm", "useful_repair"):
            if certified_values[field] != raw_values[field]:
                raise ValueError(
                    f"non-rejected outcome changes without certificate removal: {key}"
                )

    return {
        "source_id": key[0],
        "task_id": key[1],
        "triggered": triggered,
        "certificate_rejected": rejected,
        "baseline_quality": baseline,
        "certified_quality": certified_values["quality"],
        "no_certificate_quality": raw_values["quality"],
        "certified_delta": certified_values["delta"],
        "no_certificate_delta": raw_values["delta"],
        "no_certificate_minus_certified": (
            raw_values["quality"] - certified_values["quality"]
        ),
        "certified_strict_harm": certified_values["strict_harm"],
        "no_certificate_strict_harm": raw_values["strict_harm"],
        "certified_useful": certified_values["useful_repair"],
        "no_certificate_useful": raw_values["useful_repair"],
    }


def _outcomes(raw: dict[str, str], *, label: str) -> dict[str, float | bool]:
    baseline = _finite(raw["baseline_quality"])
    quality = _finite(raw["policy_quality"])
    delta = _finite(raw["quality_delta"])
    if not 0.0 <= baseline <= 1.0 or not 0.0 <= quality <= 1.0:
        raise ValueError(f"quality outside [0,1]: {label}")
    if not math.isclose(quality - baseline, delta, abs_tol=1e-12):
        raise ValueError(f"quality delta is incoherent: {label}")
    strict_harm = _binary(raw["strict_harm"], f"strict harm: {label}")
    useful = _binary(raw["useful_repair"], f"useful repair: {label}")
    if strict_harm is not (delta < -HARM_TOLERANCE):
        raise ValueError(f"strict-harm label is incoherent: {label}")
    if useful is not (delta > USEFUL_MARGIN):
        raise ValueError(f"useful label is incoherent: {label}")
    return {
        "quality": quality,
        "delta": delta,
        "strict_harm": strict_harm,
        "useful_repair": useful,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rejected = [row for row in rows if row["certificate_rejected"]]
    return {
        "certificate_rejections": len(rejected),
        "certificate_rejection_rate": len(rejected) / len(rows),
        "rejected_raw_useful": sum(
            float(row["no_certificate_delta"]) > USEFUL_MARGIN for row in rejected
        ),
        "rejected_raw_strict_harm": sum(
            float(row["no_certificate_delta"]) < -HARM_TOLERANCE
            for row in rejected
        ),
        "rejected_raw_score_neutral": sum(
            -HARM_TOLERANCE
            <= float(row["no_certificate_delta"])
            <= USEFUL_MARGIN
            for row in rejected
        ),
        "certified_quality_delta": fmean(
            float(row["certified_delta"]) for row in rows
        ),
        "no_certificate_quality_delta": fmean(
            float(row["no_certificate_delta"]) for row in rows
        ),
        "certified_strict_harm_rate": fmean(
            bool(row["certified_strict_harm"]) for row in rows
        ),
        "no_certificate_strict_harm_rate": fmean(
            bool(row["no_certificate_strict_harm"]) for row in rows
        ),
        "certified_useful_rate": fmean(
            bool(row["certified_useful"]) for row in rows
        ),
        "no_certificate_useful_rate": fmean(
            bool(row["no_certificate_useful"]) for row in rows
        ),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _load_object(path: Path) -> dict[str, Any]:
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


def _commit(value: Any, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if COMMIT_RE.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be an exact 40-character Git commit")
    return normalized


def _hash(value: Any, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if HASH_RE.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be an exact SHA-256")
    return normalized


def _integer(value: Any, label: str) -> int:
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid {label}") from error
    if not math.isfinite(numeric) or numeric != float(result) or result < 0:
        raise ValueError(f"invalid {label}")
    return result


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"non-numeric value: {value}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite value: {value}")
    return result


def _binary(value: Any, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"invalid binary {label}: {value}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certified-rows", required=True)
    parser.add_argument("--certified-manifest", required=True)
    parser.add_argument("--no-certificate-rows", required=True)
    parser.add_argument("--no-certificate-manifest", required=True)
    parser.add_argument("--generation-source-commit", required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--expected-tasks", type=int, default=240)
    parser.add_argument("--output-rows", required=True)
    parser.add_argument("--output-report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
