#!/usr/bin/env python3
"""Run an answer-blind CAGO diagnostic on a complete frozen outcome matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from openagora_experiments.cago.data import CounterfactualMatrix, load_counterfactual_matrix
from openagora_experiments.cago.evaluation import UtilitySpec, run_cago_evaluation
from openagora_experiments.cago.splits import source_stratified_split


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit("CAGO config must contain a mapping")
    if not bool(config.get("diagnostic_only", False)):
        raise SystemExit("This runner requires diagnostic_only: true; formal runs use a separate frozen config")

    task_paths = {
        str(source_id): _resolve_path(str(source_payload["task_path"]))
        for source_id, source_payload in dict(config.get("sources", {})).items()
    }
    outcome_paths = {
        str(organization_id): {
            source_id: _resolve_path(str(organization_payload["path_template"]).format(source=source_id))
            for source_id in task_paths
        }
        for organization_id, organization_payload in dict(config.get("organizations", {})).items()
    }
    matrix = load_counterfactual_matrix(task_paths=task_paths, outcome_paths=outcome_paths)
    split_payload = dict(config.get("split", {}))
    split = source_stratified_split(
        matrix.task_keys,
        train_fraction=float(split_payload.get("train_fraction", 0.5)),
        calibration_fraction=float(split_payload.get("calibration_fraction", 0.25)),
        salt=str(split_payload.get("salt", "cago-diagnostic-v1")),
    )

    _write_json(output_dir / "matrix_audit.json", asdict(matrix.audit))
    _write_json(
        output_dir / "split.json",
        {
            "train": [key.stable_id for key in split.train],
            "calibration": [key.stable_id for key in split.calibration],
            "test": [key.stable_id for key in split.test],
            "counts_by_source": split.counts_by_source(),
        },
    )
    _write_normalized_outcomes(output_dir / "normalized_outcomes.csv", matrix)

    baseline_id = str(config.get("baseline_id", "")).strip()
    alpha = float(config.get("alpha", 0.1))
    margin = float(config.get("margin", 0.0))
    ridge_l2 = float(config.get("ridge_l2", 1.0))
    regime_results: dict[str, dict[str, object]] = {}
    regimes = dict(config.get("utility_regimes", {}))
    if not regimes:
        raise SystemExit("at least one utility_regime is required")
    for regime_name, regime_payload in regimes.items():
        utility = UtilitySpec(**dict(regime_payload))
        result = run_cago_evaluation(
            matrix=matrix,
            split=split,
            baseline_id=baseline_id,
            utility=utility,
            alpha=alpha,
            margin=margin,
            ridge_l2=ridge_l2,
        )
        safe_name = _slug(str(regime_name))
        result_payload = result.to_dict()
        result_payload["utility_spec"] = asdict(utility)
        regime_results[str(regime_name)] = result_payload
        _write_json(output_dir / f"{safe_name}_result.json", result_payload)
        _write_csv(
            output_dir / f"{safe_name}_decisions.csv",
            [decision.to_csv_row() for decision in result.cago_decisions],
        )

    manifest = {
        "artifact": "cago_diagnostic",
        "diagnostic_only": True,
        "paper_evidence_admissible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": _portable_path(config_path),
        "config_sha256": _sha256(config_path),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(_git_output("status", "--porcelain")),
        "baseline_id": baseline_id,
        "organization_ids": list(matrix.organization_ids),
        "task_count": matrix.audit.task_count,
        "outcome_count": matrix.audit.outcome_count,
        "reported_score_disagreements": matrix.audit.reported_score_disagreements,
        "input_sha256": matrix.audit.input_sha256,
        "answer_leakage_policy": (
            "answers and references are used only for uniform post-selection scoring; "
            "the predictor consumes visible_feature_vector(question, options, question_type)"
        ),
        "limitations": [
            "reuses legacy executions and is therefore diagnostic, not formal paper evidence",
            "contains one seed per task-organization cell",
            "uses text-visible features because archived image paths are anonymized",
        ],
        "regimes": regime_results,
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(_console_summary(matrix=matrix, regime_results=regime_results), indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "code" / "experiments" / "configs" / "cago_diagnostic_330.yaml"),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _write_normalized_outcomes(path: Path, matrix: CounterfactualMatrix) -> None:
    rows: list[dict[str, object]] = []
    for key in matrix.task_keys:
        for organization_id in matrix.organization_ids:
            outcome = matrix.outcome(key, organization_id)
            rows.append(
                {
                    "source_id": key.source_id,
                    "task_id": key.task_id,
                    "organization_id": organization_id,
                    "quality": outcome.quality,
                    "reported_quality": outcome.reported_quality,
                    "reported_score_difference": outcome.reported_quality - outcome.quality,
                    "tokens": outcome.tokens,
                    "latency_seconds": outcome.latency_seconds,
                    "cost_usd": outcome.cost_usd,
                    "run_id": outcome.run_id,
                }
            )
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _console_summary(
    *,
    matrix: CounterfactualMatrix,
    regime_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "task_count": matrix.audit.task_count,
        "organization_count": matrix.audit.organization_count,
        "outcome_count": matrix.audit.outcome_count,
        "reported_score_disagreements": matrix.audit.reported_score_disagreements,
        "regimes": {
            name: {
                "best_static_id": payload["best_static_id"],
                "calibration_correction": payload["calibration_correction"],
                "empirical_simultaneous_coverage": payload["empirical_simultaneous_coverage"],
                "summaries": payload["summaries"],
            }
            for name, payload in regime_results.items()
        },
    }


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
