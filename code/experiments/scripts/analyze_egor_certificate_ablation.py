#!/usr/bin/env python3
"""Evaluate EGOR with identical decisions but without certificate fallback."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from openagora_experiments.egor.statistics import paired_bootstrap_contrast


ROOT = Path(__file__).resolve().parents[3]
TaskKey = tuple[str, str]
CandidateKey = tuple[str, str, str]


@dataclass(frozen=True)
class FrozenTrainingRow:
    source_id: str
    task_id: str
    baseline_quality: float
    strict_advantages: dict[str, float]


def main() -> int:
    args = _parse_args()
    training_path = Path(args.training_rows).resolve()
    config_path = Path(args.selector_config).resolve()
    fit_manifest_path = Path(args.fit_manifest).resolve()
    reference_path = Path(args.reference_rows).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_object(config_path)
    fit_manifest = _load_object(fit_manifest_path)
    _require_manifest_sha(
        fit_manifest,
        section="outputs",
        path=training_path,
        logical_name="training_rows.csv",
    )
    _require_manifest_sha(
        fit_manifest,
        section="outputs",
        path=reference_path,
        logical_name="cross_validated_rows.csv",
    )
    _require_manifest_sha(
        fit_manifest,
        section="inputs",
        path=config_path,
        logical_name=config_path.name,
    )
    training = load_training_rows(training_path)
    matrix, matrix_paths = load_candidate_rows(
        config,
        config_path=config_path,
        fit_manifest=fit_manifest,
    )
    reference = load_reference_rows(reference_path)
    candidate_ids = _candidate_ids(config)
    assert_frozen_population(
        training,
        matrix,
        reference,
        candidate_ids=candidate_ids,
    )

    policy_rows = build_policy_rows(training, matrix, reference)
    summary = {
        "schema_version": 1,
        "protocol": "fixed_decision_inference_time_certificate_removal",
        "definition": (
            "Reuse every frozen nested-LOSO action and call; for changed answers "
            "without valid recruited evidence, compare deterministic baseline "
            "fallback with the raw candidate score"
        ),
        "selected_policy": summarize_policy(policy_rows),
        "complete_catalog": summarize_catalog(training, matrix),
    }
    statistics = build_statistics(
        policy_rows,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    rows_path = output_dir / "certificate_policy_rows.csv"
    summary_path = output_dir / "certificate_summary.json"
    statistics_path = output_dir / "certificate_statistics.json"
    write_policy_rows(rows_path, policy_rows)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    statistics_path.write_text(
        json.dumps(statistics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    outputs = (rows_path, summary_path, statistics_path)
    manifest = {
        "schema_version": 1,
        "source_commit": os.environ.get("EGOR_ANALYSIS_COMMIT", "").strip()
        or _git_output("rev-parse", "HEAD"),
        "fit_source_commit": fit_manifest.get("source_commit"),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (
                training_path,
                config_path,
                fit_manifest_path,
                reference_path,
                *matrix_paths,
            )
        ],
        "outputs": [{"path": str(path), "sha256": _sha256(path)} for path in outputs],
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "decision_boundary": (
            "selected action, repair trigger, calls, and all frozen outcomes are "
            "unchanged; only the deterministic certificate fallback is removed"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote certificate fallback control over {len(policy_rows)} tasks to "
        f"{output_dir}"
    )
    return 0


def load_training_rows(path: Path) -> dict[TaskKey, FrozenTrainingRow]:
    rows: dict[TaskKey, FrozenTrainingRow] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            key = (raw["source_id"], raw["task_id"])
            if not all(key) or key in rows:
                raise ValueError(f"invalid or duplicate training row: {key}")
            advantages = json.loads(raw["strict_advantages_json"])
            if not isinstance(advantages, dict) or not advantages:
                raise ValueError(f"invalid strict advantages for {key}")
            rows[key] = FrozenTrainingRow(
                source_id=key[0],
                task_id=key[1],
                baseline_quality=float(raw["baseline_quality"]),
                strict_advantages={
                    str(candidate): float(value)
                    for candidate, value in advantages.items()
                },
            )
    if not rows:
        raise ValueError("training rows are empty")
    return rows


def load_candidate_rows(
    config: dict[str, Any],
    *,
    config_path: Path,
    fit_manifest: dict[str, Any],
) -> tuple[dict[CandidateKey, dict[str, str]], list[Path]]:
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("selector config requires sources")
    rows: dict[CandidateKey, dict[str, str]] = {}
    paths: list[Path] = []
    allowed = set(_candidate_ids(config))
    for raw_source in sources:
        if not isinstance(raw_source, dict):
            raise ValueError("selector source must be an object")
        source_id = str(raw_source.get("source_id", "")).strip()
        matrices = raw_source.get("matrices")
        if not source_id or not isinstance(matrices, list) or not matrices:
            raise ValueError("selector source lacks source_id/matrices")
        for raw_matrix in matrices:
            if not isinstance(raw_matrix, dict):
                raise ValueError("matrix config must be an object")
            matrix_path = _resolve_path(raw_matrix.get("path"), config_path.parent)
            mapping = raw_matrix.get("candidate_mapping")
            if not isinstance(mapping, dict):
                raise ValueError("matrix config lacks candidate_mapping")
            _require_manifest_sha(
                fit_manifest,
                section="inputs",
                path=matrix_path,
                logical_name=str(matrix_path),
                match_full_path=True,
            )
            paths.append(matrix_path)
            canonical = {str(key): str(value) for key, value in mapping.items()}
            with matrix_path.open(encoding="utf-8", newline="") as handle:
                for raw in csv.DictReader(handle):
                    candidate_id = canonical.get(str(raw.get("candidate_id", "")))
                    if candidate_id not in allowed:
                        continue
                    task_id = str(raw.get("task_id", "")).strip()
                    key = (source_id, task_id, candidate_id)
                    if not task_id or key in rows:
                        raise ValueError(f"duplicate/empty candidate row: {key}")
                    rows[key] = {
                        **raw,
                        "source_id": source_id,
                        "candidate_id": candidate_id,
                    }
    if not rows:
        raise ValueError("candidate matrix is empty")
    return rows, list(dict.fromkeys(paths))


def load_reference_rows(path: Path) -> dict[TaskKey, dict[str, str]]:
    rows: dict[TaskKey, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["source_id"], row["task_id"])
            if not all(key) or key in rows:
                raise ValueError(f"invalid or duplicate reference row: {key}")
            rows[key] = row
    if not rows:
        raise ValueError("reference rows are empty")
    return rows


def assert_frozen_population(
    training: dict[TaskKey, FrozenTrainingRow],
    matrix: dict[CandidateKey, dict[str, str]],
    reference: dict[TaskKey, dict[str, str]],
    *,
    candidate_ids: tuple[str, ...],
) -> None:
    if set(training) != set(reference):
        raise ValueError("training and reference task populations differ")
    expected = {
        (*key, candidate_id) for key in training for candidate_id in candidate_ids
    }
    if set(matrix) != expected:
        missing = sorted(expected - set(matrix))[:5]
        extra = sorted(set(matrix) - expected)[:5]
        raise ValueError(
            f"candidate population mismatch; missing={missing}, extra={extra}"
        )
    for key, item in training.items():
        if set(item.strict_advantages) != set(candidate_ids):
            raise ValueError(f"candidate IDs differ for {key}")
        for candidate_id in candidate_ids:
            raw = matrix[(*key, candidate_id)]
            baseline = float(raw["baseline_quality"])
            candidate = float(raw["candidate_quality"])
            rejected = _certificate_rejected(raw)
            strict_delta = 0.0 if rejected else candidate - baseline
            if not math.isclose(
                baseline,
                item.baseline_quality,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"baseline quality mismatch for {(*key, candidate_id)}"
                )
            if not math.isclose(
                strict_delta,
                item.strict_advantages[candidate_id],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"strict advantage mismatch for {(*key, candidate_id)}"
                )
        selected = str(reference[key].get("selected_id", ""))
        if selected != "baseline" and selected not in candidate_ids:
            raise ValueError(
                f"reference selects unknown candidate for {key}: {selected}"
            )


def build_policy_rows(
    training: dict[TaskKey, FrozenTrainingRow],
    matrix: dict[CandidateKey, dict[str, str]],
    reference: dict[TaskKey, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(training):
        item = training[key]
        selected = str(reference[key]["selected_id"])
        if selected == "baseline":
            certified_delta = 0.0
            no_certificate_delta = 0.0
            rejected = False
            actual_changed = False
            evidence_supported = False
        else:
            raw = matrix[(*key, selected)]
            certified_delta = float(reference[key]["quality_delta"])
            no_certificate_delta = float(raw["candidate_quality"]) - float(
                raw["baseline_quality"]
            )
            rejected = _certificate_rejected(raw)
            actual_changed = _bool(raw.get("actual_changed"))
            evidence_supported = _bool(raw.get("evidence_supported_change"))
            expected = item.strict_advantages[selected]
            if not math.isclose(
                certified_delta,
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"reference delta mismatch for {key}")
        rows.append(
            {
                "source_id": key[0],
                "task_id": key[1],
                "selected_id": selected,
                "baseline_quality": item.baseline_quality,
                "certified_quality": item.baseline_quality + certified_delta,
                "no_certificate_quality": (
                    item.baseline_quality + no_certificate_delta
                ),
                "certified_delta": certified_delta,
                "no_certificate_delta": no_certificate_delta,
                "incremental_calls": float(reference[key]["incremental_calls"]),
                "triggered": int(selected != "baseline"),
                "certificate_rejected": int(rejected),
                "actual_changed": int(actual_changed),
                "evidence_supported_change": int(evidence_supported),
                "certified_strict_harm": int(certified_delta < -0.05),
                "no_certificate_strict_harm": int(no_certificate_delta < -0.05),
                "certified_useful": int(certified_delta > 0.05),
                "no_certificate_useful": int(no_certificate_delta > 0.05),
            }
        )
    return rows


def summarize_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("policy rows are empty")
    selected = [str(row["selected_id"]) for row in rows]
    return {
        "tasks": len(rows),
        "repair_rate": fmean(item != "baseline" for item in selected),
        "incremental_calls": fmean(float(row["incremental_calls"]) for row in rows),
        "selection_counts": dict(sorted(Counter(selected).items())),
        "certificate_rejections": sum(int(row["certificate_rejected"]) for row in rows),
        "certified_quality_delta": fmean(float(row["certified_delta"]) for row in rows),
        "no_certificate_quality_delta": fmean(
            float(row["no_certificate_delta"]) for row in rows
        ),
        "certified_strict_harm_rate": fmean(
            int(row["certified_strict_harm"]) for row in rows
        ),
        "no_certificate_strict_harm_rate": fmean(
            int(row["no_certificate_strict_harm"]) for row in rows
        ),
        "certified_useful_rate": fmean(int(row["certified_useful"]) for row in rows),
        "no_certificate_useful_rate": fmean(
            int(row["no_certificate_useful"]) for row in rows
        ),
    }


def summarize_catalog(
    training: dict[TaskKey, FrozenTrainingRow],
    matrix: dict[CandidateKey, dict[str, str]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key, raw in sorted(matrix.items()):
        candidate_id = key[2]
        strict_delta = training[key[:2]].strict_advantages[candidate_id]
        raw_delta = float(raw["candidate_quality"]) - float(raw["baseline_quality"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "rejected": _certificate_rejected(raw),
                "strict_delta": strict_delta,
                "raw_delta": raw_delta,
            }
        )
    rejected = [row for row in rows if row["rejected"]]
    return {
        "candidate_rows": len(rows),
        "certificate_rejections": len(rejected),
        "rejections_by_candidate": dict(
            sorted(Counter(row["candidate_id"] for row in rejected).items())
        ),
        "rejected_raw_harms": sum(row["raw_delta"] < -0.05 for row in rejected),
        "rejected_raw_useful": sum(row["raw_delta"] > 0.05 for row in rejected),
        "raw_strict_harms": sum(row["raw_delta"] < -0.05 for row in rows),
        "certified_strict_harms": sum(row["strict_delta"] < -0.05 for row in rows),
        "raw_useful": sum(row["raw_delta"] > 0.05 for row in rows),
        "certified_useful": sum(row["strict_delta"] > 0.05 for row in rows),
        "raw_quality_delta": fmean(float(row["raw_delta"]) for row in rows),
        "certified_quality_delta": fmean(float(row["strict_delta"]) for row in rows),
    }


def build_statistics(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    certified = [float(row["certified_quality"]) for row in rows]
    no_certificate = [float(row["no_certificate_quality"]) for row in rows]
    return {
        "schema_version": 1,
        "tasks": len(rows),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "no_certificate_minus_certified": asdict(
            paired_bootstrap_contrast(
                no_certificate,
                certified,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        ),
    }


def write_policy_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "source_id",
        "task_id",
        "selected_id",
        "baseline_quality",
        "certified_quality",
        "no_certificate_quality",
        "certified_delta",
        "no_certificate_delta",
        "incremental_calls",
        "triggered",
        "certificate_rejected",
        "actual_changed",
        "evidence_supported_change",
        "certified_strict_harm",
        "no_certificate_strict_harm",
        "certified_useful",
        "no_certificate_useful",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _candidate_ids(config: dict[str, Any]) -> tuple[str, ...]:
    raw = config.get("candidate_ids")
    if not isinstance(raw, list) or not raw:
        raise ValueError("selector config requires candidate_ids")
    values = tuple(str(value) for value in raw)
    if len(set(values)) != len(values) or "baseline" in values:
        raise ValueError("candidate_ids must be unique nonbaseline IDs")
    return values


def _certificate_rejected(row: dict[str, str]) -> bool:
    return _bool(row.get("actual_changed")) and not _bool(
        row.get("evidence_supported_change")
    )


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _require_manifest_sha(
    manifest: dict[str, Any],
    *,
    section: str,
    path: Path,
    logical_name: str,
    match_full_path: bool = False,
) -> None:
    records = manifest.get(section)
    if not isinstance(records, list):
        raise ValueError(f"fit manifest lacks {section}")
    matches = []
    for row in records:
        if not isinstance(row, dict):
            continue
        raw = str(row.get("path", ""))
        if raw == logical_name if match_full_path else Path(raw).name == logical_name:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            f"fit manifest must bind exactly one {section} record for {logical_name}"
        )
    actual = _sha256(path)
    expected = str(matches[0].get("sha256", ""))
    if actual != expected:
        raise ValueError(f"manifest SHA mismatch for {path}: {actual} != {expected}")


def _resolve_path(value: Any, base: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("matrix path is empty")
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-rows", required=True)
    parser.add_argument("--selector-config", required=True)
    parser.add_argument("--fit-manifest", required=True)
    parser.add_argument("--reference-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_260_713)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
