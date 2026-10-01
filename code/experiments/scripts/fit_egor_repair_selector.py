#!/usr/bin/env python3
"""Fit and audit a source-blind counterfactual EGOR repair selector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from openagora_experiments.egor.evaluation import residual_gap_vector
from openagora_experiments.egor.runner import load_latest_stage_records, load_tasks
from openagora_experiments.egor.selection import (
    CounterfactualRepairSelector,
    RepairSelectionConfig,
    RepairTrainingExample,
    SelectorTarget,
    evaluate_selector,
    select_config_by_source_cv,
    visible_repair_feature_vector,
)


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    examples, input_paths = load_examples(config, config_path=config_path)
    grid = config.get("grid", {})
    if not isinstance(grid, dict):
        raise ValueError("config.grid must be an object")
    selected, choices = _select_config(examples, grid, target="paired_advantage")
    selector = CounterfactualRepairSelector.fit(examples, config=selected.config)
    cv_rows, cv_protocol = nested_or_feasibility_rows(
        examples,
        grid,
        target="paired_advantage",
        fallback_config=selected.config,
    )
    standalone_selected, standalone_choices = _select_config(
        examples,
        grid,
        target="standalone_quality",
    )
    standalone_selector = CounterfactualRepairSelector.fit(
        examples,
        config=standalone_selected.config,
        target="standalone_quality",
    )
    standalone_rows, standalone_cv_protocol = nested_or_feasibility_rows(
        examples,
        grid,
        target="standalone_quality",
        fallback_config=standalone_selected.config,
    )
    call_price = float(grid.get("evaluation_call_price", 0.01))
    baseline_rows = best_fixed_cross_validated_rows(
        examples,
        minimum_gap=selected.config.minimum_gap,
        evaluation_call_price=call_price,
    )

    selector_path = output_dir / "selector.json"
    grid_path = output_dir / "cv_grid.json"
    rows_path = output_dir / "cross_validated_rows.csv"
    fixed_rows_path = output_dir / "best_fixed_cross_validated_rows.csv"
    training_path = output_dir / "training_rows.csv"
    standalone_selector_path = output_dir / "selector_standalone_value.json"
    standalone_grid_path = output_dir / "cv_grid_standalone_value.json"
    standalone_rows_path = output_dir / "cross_validated_rows_standalone_value.csv"
    summary_path = output_dir / "summary.json"
    selector_path.write_text(
        json.dumps(selector.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    grid_path.write_text(
        json.dumps(
            {
                "selected": _choice_payload(selected),
                "choices": [_choice_payload(item) for item in choices],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    standalone_selector_path.write_text(
        json.dumps(standalone_selector.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    standalone_grid_path.write_text(
        json.dumps(
            {
                "target": "standalone_quality",
                "selected": _choice_payload(standalone_selected),
                "choices": [_choice_payload(item) for item in standalone_choices],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_decision_rows(rows_path, cv_rows)
    _write_decision_rows(standalone_rows_path, standalone_rows)
    _write_decision_rows(fixed_rows_path, baseline_rows)
    _write_training_rows(training_path, examples)
    summary = build_summary(
        examples=examples,
        selector=selector,
        selected_choice=selected,
        selector_rows=cv_rows,
        standalone_selected_choice=standalone_selected,
        standalone_rows=standalone_rows,
        cv_protocol=cv_protocol,
        standalone_cv_protocol=standalone_cv_protocol,
        fixed_rows=baseline_rows,
        evaluation_call_price=call_price,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    output_paths = (
        selector_path,
        grid_path,
        rows_path,
        fixed_rows_path,
        training_path,
        standalone_selector_path,
        standalone_grid_path,
        standalone_rows_path,
        summary_path,
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("EGOR_SOURCE_COMMIT", "").strip()
        or _git_output("rev-parse", "HEAD"),
        "tasks": len(examples),
        "sources": sorted({item.source_id for item in examples}),
        "candidate_ids": list(selector.candidate_ids),
        "analysis_controls": ["standalone_quality_selector"],
        "cross_validation_protocol": cv_protocol,
        "selection_boundary": (
            "source_id is used only to define leave-one-source-out development folds; "
            "gold answers and realized candidate outcomes are absent from test-time features"
        ),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in dict.fromkeys([config_path.resolve(), *input_paths])
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in output_paths
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    nested_summary = summary["source_cross_validated_selector"]
    print(
        f"fit {len(selector.candidate_ids)} repair heads on {len(examples)} tasks; "
        f"{cv_protocol} delta={nested_summary['mean_quality_delta']:.6f}, "
        f"calls={nested_summary['mean_incremental_calls']:.6f}, "
        f"harm={nested_summary['strict_harm_rate']:.6f}; "
        "global deployment config serialized separately"
    )
    return 0


def load_examples(
    config: dict[str, Any],
    *,
    config_path: Path,
) -> tuple[list[RepairTrainingExample], list[Path]]:
    candidate_ids = tuple(sorted(str(item) for item in config.get("candidate_ids", [])))
    if not candidate_ids:
        raise ValueError("config.candidate_ids must be non-empty")
    raw_sources = config.get("sources", [])
    if not isinstance(raw_sources, list) or len(raw_sources) < 2:
        raise ValueError("config.sources must contain at least two source objects")
    examples: list[RepairTrainingExample] = []
    input_paths: list[Path] = []
    seen_sources: set[str] = set()
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict):
            raise ValueError("each source config must be an object")
        source_id = str(raw_source.get("source_id", "")).strip()
        if not source_id or source_id in seen_sources:
            raise ValueError(f"source_id must be non-empty and unique: {source_id}")
        seen_sources.add(source_id)
        task_path = _resolve_path(raw_source.get("tasks"), config_path.parent)
        baseline_path = _resolve_path(raw_source.get("baseline"), config_path.parent)
        tasks = load_tasks(task_path)
        baseline_records = load_latest_stage_records(baseline_path)
        candidate_rows, matrix_paths = _load_candidate_rows(
            raw_source.get("matrices", []),
            base=config_path.parent,
        )
        input_paths.extend([task_path, baseline_path, *matrix_paths])
        for task in tasks:
            key = (task.source_id, task.task_id)
            baseline_record = baseline_records.get(key)
            if baseline_record is None or baseline_record.get("status") != "ok":
                raise ValueError(
                    f"missing successful baseline for {source_id}::{task.task_id}"
                )
            parsed = baseline_record.get("parsed", {})
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"malformed baseline payload for {source_id}::{task.task_id}"
                )
            gaps = residual_gap_vector(parsed)
            advantages: dict[str, float] = {}
            calls: dict[str, float] = {}
            baseline_quality: float | None = None
            for candidate_id in candidate_ids:
                row = candidate_rows.get((candidate_id, task.task_id))
                if row is None:
                    raise ValueError(
                        f"missing candidate {candidate_id} for {source_id}::{task.task_id}"
                    )
                row_baseline_quality = float(row.get("baseline_quality", 0.0))
                if baseline_quality is None:
                    baseline_quality = row_baseline_quality
                elif abs(baseline_quality - row_baseline_quality) > 1e-9:
                    raise ValueError(
                        "candidate matrices disagree on baseline quality for "
                        f"{source_id}::{task.task_id}"
                    )
                advantages[candidate_id] = strict_advantage(row)
                calls[candidate_id] = max(
                    0.0,
                    float(row.get("calls", 0.0))
                    - float(row.get("baseline_calls", 1.0)),
                )
            examples.append(
                RepairTrainingExample(
                    source_id=source_id,
                    task_id=task.task_id,
                    features=visible_repair_feature_vector(task, parsed),
                    gap_max=max(gaps.values(), default=0.0),
                    strict_advantages=advantages,
                    incremental_calls=calls,
                    baseline_quality=float(baseline_quality or 0.0),
                )
            )
    return examples, input_paths


def strict_advantage(row: dict[str, str]) -> float:
    baseline = float(row.get("baseline_quality", 0.0))
    candidate = float(row.get("candidate_quality", 0.0))
    changed = _bool(row.get("actual_changed"))
    supported = _bool(row.get("evidence_supported_change"))
    strict_quality = baseline if changed and not supported else candidate
    return strict_quality - baseline


def cross_validated_rows(
    examples: list[RepairTrainingExample],
    config: RepairSelectionConfig,
    *,
    target: SelectorTarget = "paired_advantage",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for held_source in sorted({item.source_id for item in examples}):
        train = [item for item in examples if item.source_id != held_source]
        held = [item for item in examples if item.source_id == held_source]
        selector = CounterfactualRepairSelector.fit(
            train,
            config=config,
            target=target,
        )
        rows.extend(
            _decision_rows(
                selector,
                held,
                selection_protocol="global_config_loso_feasibility",
                fold_config=config,
            )
        )
    return sorted(rows, key=lambda row: (row["source_id"], row["task_id"]))


def nested_or_feasibility_rows(
    examples: list[RepairTrainingExample],
    grid: dict[str, Any],
    *,
    target: SelectorTarget,
    fallback_config: RepairSelectionConfig,
) -> tuple[list[dict[str, Any]], str]:
    sources = sorted({item.source_id for item in examples})
    if len(sources) < 3:
        return (
            cross_validated_rows(examples, fallback_config, target=target),
            "global_config_loso_feasibility",
        )

    rows: list[dict[str, Any]] = []
    for held_source in sources:
        train = [item for item in examples if item.source_id != held_source]
        held = [item for item in examples if item.source_id == held_source]
        inner_selected, _ = _select_config(train, grid, target=target)
        selector = CounterfactualRepairSelector.fit(
            train,
            config=inner_selected.config,
            target=target,
        )
        rows.extend(
            _decision_rows(
                selector,
                held,
                selection_protocol="nested_loso",
                fold_config=inner_selected.config,
            )
        )
    return (
        sorted(rows, key=lambda row: (row["source_id"], row["task_id"])),
        "nested_loso",
    )


def _decision_rows(
    selector: CounterfactualRepairSelector,
    held: list[RepairTrainingExample],
    *,
    selection_protocol: str,
    fold_config: RepairSelectionConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in held:
        decision = selector.select(item.features, gap_max=item.gap_max)
        delta = (
            float(item.strict_advantages[decision.selected_id])
            if decision.repaired
            else 0.0
        )
        calls = (
            float(item.incremental_calls[decision.selected_id])
            if decision.repaired
            else 0.0
        )
        rows.append(
            {
                "source_id": item.source_id,
                "task_id": item.task_id,
                "selected_id": decision.selected_id,
                "reason": decision.reason,
                "selection_protocol": selection_protocol,
                "selector_target": selector.target,
                "fold_config_json": json.dumps(asdict(fold_config), sort_keys=True),
                "gap_max": item.gap_max,
                "quality_delta": delta,
                "incremental_calls": calls,
                "strict_harm": int(delta < -0.05),
                "useful_repair": int(delta > 0.05),
                "predicted_advantages_json": json.dumps(
                    decision.predicted_advantages, sort_keys=True
                ),
                "net_advantage_scores_json": json.dumps(
                    decision.net_advantage_scores, sort_keys=True
                ),
            }
        )
    return rows


def best_fixed_cross_validated_rows(
    examples: list[RepairTrainingExample],
    *,
    minimum_gap: float,
    evaluation_call_price: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_ids = tuple(sorted(examples[0].strict_advantages))
    for held_source in sorted({item.source_id for item in examples}):
        train = [item for item in examples if item.source_id != held_source]
        held = [item for item in examples if item.source_id == held_source]
        utilities = {"baseline": 0.0}
        for candidate_id in candidate_ids:
            utilities[candidate_id] = fmean(
                (
                    item.strict_advantages[candidate_id]
                    - evaluation_call_price * item.incremental_calls[candidate_id]
                )
                if item.gap_max > minimum_gap
                else 0.0
                for item in train
            )
        selected = min(utilities, key=lambda item: (-utilities[item], item))
        for item in held:
            repaired = selected != "baseline" and item.gap_max > minimum_gap
            chosen = selected if repaired else "baseline"
            delta = float(item.strict_advantages[selected]) if repaired else 0.0
            calls = float(item.incremental_calls[selected]) if repaired else 0.0
            rows.append(
                {
                    "source_id": item.source_id,
                    "task_id": item.task_id,
                    "selected_id": chosen,
                    "reason": f"training_selected_fixed::{selected}",
                    "selection_protocol": "nested_best_fixed",
                    "selector_target": "best_fixed",
                    "fold_config_json": "{}",
                    "gap_max": item.gap_max,
                    "quality_delta": delta,
                    "incremental_calls": calls,
                    "strict_harm": int(delta < -0.05),
                    "useful_repair": int(delta > 0.05),
                    "predicted_advantages_json": "{}",
                    "net_advantage_scores_json": json.dumps(utilities, sort_keys=True),
                }
            )
    return sorted(rows, key=lambda row: (row["source_id"], row["task_id"]))


def build_summary(
    *,
    examples: list[RepairTrainingExample],
    selector: CounterfactualRepairSelector,
    selected_choice: Any,
    selector_rows: list[dict[str, Any]],
    standalone_selected_choice: Any,
    standalone_rows: list[dict[str, Any]],
    cv_protocol: str,
    standalone_cv_protocol: str,
    fixed_rows: list[dict[str, Any]],
    evaluation_call_price: float,
) -> dict[str, Any]:
    selector_summary = _summarize_rows(selector_rows, evaluation_call_price)
    standalone_summary = _summarize_rows(
        standalone_rows,
        evaluation_call_price,
    )
    fixed_summary = _summarize_rows(fixed_rows, evaluation_call_price)
    return {
        "schema_version": 1,
        "tasks": len(examples),
        "sources": sorted({item.source_id for item in examples}),
        "candidate_ids": list(selector.candidate_ids),
        "selected_config": asdict(selected_choice.config),
        "standalone_value_selected_config": asdict(standalone_selected_choice.config),
        "cross_validation_protocol": cv_protocol,
        "standalone_cross_validation_protocol": standalone_cv_protocol,
        "source_cross_validated_selector": selector_summary,
        "source_cross_validated_standalone_value": standalone_summary,
        "source_cross_validated_best_fixed": fixed_summary,
        "paired_selector_minus_standalone_value": {
            "quality_delta": selector_summary["mean_quality_delta"]
            - standalone_summary["mean_quality_delta"],
            "incremental_calls": selector_summary["mean_incremental_calls"]
            - standalone_summary["mean_incremental_calls"],
            "utility": selector_summary["mean_utility"]
            - standalone_summary["mean_utility"],
        },
        "selector_minus_best_fixed": {
            "quality_delta": selector_summary["mean_quality_delta"]
            - fixed_summary["mean_quality_delta"],
            "incremental_calls": selector_summary["mean_incremental_calls"]
            - fixed_summary["mean_incremental_calls"],
            "utility": selector_summary["mean_utility"] - fixed_summary["mean_utility"],
        },
        "all_development_in_sample": asdict(evaluate_selector(selector, examples)),
        "per_source_in_sample": {
            source_id: asdict(
                evaluate_selector(
                    selector,
                    [item for item in examples if item.source_id == source_id],
                )
            )
            for source_id in sorted({item.source_id for item in examples})
        },
    }


def _summarize_rows(
    rows: list[dict[str, Any]], evaluation_call_price: float
) -> dict[str, Any]:
    deltas = [float(row["quality_delta"]) for row in rows]
    calls = [float(row["incremental_calls"]) for row in rows]
    selections = [str(row["selected_id"]) for row in rows]
    mean_delta = fmean(deltas)
    mean_calls = fmean(calls)
    return {
        "tasks": len(rows),
        "mean_quality_delta": mean_delta,
        "mean_incremental_calls": mean_calls,
        "mean_utility": mean_delta - evaluation_call_price * mean_calls,
        "repair_rate": sum(item != "baseline" for item in selections) / len(rows),
        "strict_harm_rate": sum(value < -0.05 for value in deltas) / len(rows),
        "useful_repair_rate": sum(value > 0.05 for value in deltas) / len(rows),
        "selection_counts": dict(sorted(_counts(selections).items())),
        "by_source": {
            source_id: _summarize_rows(
                [row for row in rows if row["source_id"] == source_id],
                evaluation_call_price,
            )
            for source_id in sorted({str(row["source_id"]) for row in rows})
        }
        if len({str(row["source_id"]) for row in rows}) > 1
        else {},
    }


def _load_candidate_rows(
    raw_matrices: Any,
    *,
    base: Path,
) -> tuple[dict[tuple[str, str], dict[str, str]], list[Path]]:
    if not isinstance(raw_matrices, list) or not raw_matrices:
        raise ValueError("each source requires a non-empty matrices list")
    rows: dict[tuple[str, str], dict[str, str]] = {}
    paths: list[Path] = []
    for raw in raw_matrices:
        if not isinstance(raw, dict) or not isinstance(
            raw.get("candidate_mapping"), dict
        ):
            raise ValueError("each matrix requires path and candidate_mapping")
        path = _resolve_path(raw.get("path"), base)
        paths.append(path)
        mapping = {
            str(key): str(value) for key, value in raw["candidate_mapping"].items()
        }
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_candidate = str(row.get("candidate_id", ""))
                if raw_candidate not in mapping:
                    continue
                key = (mapping[raw_candidate], str(row.get("task_id", "")))
                if not key[1] or key in rows:
                    raise ValueError(f"duplicate or empty mapped candidate row: {key}")
                rows[key] = row
    return rows, paths


def _write_decision_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "source_id",
        "task_id",
        "selected_id",
        "reason",
        "selection_protocol",
        "selector_target",
        "fold_config_json",
        "gap_max",
        "quality_delta",
        "incremental_calls",
        "strict_harm",
        "useful_repair",
        "predicted_advantages_json",
        "net_advantage_scores_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_training_rows(path: Path, examples: list[RepairTrainingExample]) -> None:
    fieldnames = [
        "source_id",
        "task_id",
        "gap_max",
        "baseline_quality",
        "features_json",
        "strict_advantages_json",
        "incremental_calls_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in examples:
            writer.writerow(
                {
                    "source_id": item.source_id,
                    "task_id": item.task_id,
                    "gap_max": item.gap_max,
                    "baseline_quality": item.baseline_quality,
                    "features_json": json.dumps(item.features),
                    "strict_advantages_json": json.dumps(
                        item.strict_advantages, sort_keys=True
                    ),
                    "incremental_calls_json": json.dumps(
                        item.incremental_calls, sort_keys=True
                    ),
                }
            )


def _choice_payload(choice: Any) -> dict[str, Any]:
    payload = asdict(choice)
    payload["config"] = asdict(choice.config)
    return payload


def _select_config(
    examples: list[RepairTrainingExample],
    grid: dict[str, Any],
    *,
    target: SelectorTarget,
) -> tuple[Any, tuple[Any, ...]]:
    threshold_key = (
        "standalone_value_thresholds"
        if target == "standalone_quality"
        else "thresholds"
    )
    return select_config_by_source_cv(
        examples,
        l2_values=_float_list(grid, "l2_values"),
        thresholds=_float_list(grid, threshold_key),
        cost_weights=_float_list(grid, "cost_weights"),
        evaluation_call_price=float(grid.get("evaluation_call_price", 0.01)),
        maximum_harm_rate=float(grid.get("maximum_harm_rate", 0.05)),
        minimum_repair_rate=float(grid.get("minimum_repair_rate", 0.05)),
        minimum_gap=float(grid.get("minimum_gap", 0.05)),
        target=target,
    )


def _float_list(payload: dict[str, Any], key: str) -> tuple[float, ...]:
    raw = payload.get(key, [])
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"config.grid.{key} must be a non-empty list")
    return tuple(float(item) for item in raw)


def _resolve_path(value: Any, base: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("config path must be non-empty")
    path = Path(text).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _counts(values: list[str]) -> dict[str, int]:
    return {value: values.count(value) for value in sorted(set(values))}


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
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
