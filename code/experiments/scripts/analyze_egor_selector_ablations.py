#!/usr/bin/env python3
"""Run hash-bound nested source-held-out selector ablations for EGOR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from openagora_experiments.egor.selection import (
    REPAIR_FEATURE_NAMES,
    CounterfactualRepairSelector,
    RepairTrainingExample,
    select_config_by_source_cv,
)
from openagora_experiments.egor.statistics import paired_bootstrap_contrast


ROOT = Path(__file__).resolve().parents[3]
QUESTION_FEATURES = (
    "question_chars",
    "question_words",
    "digit_tokens",
    "math_symbols",
    "option_count",
    "option_text_chars",
    "is_multiple_choice",
    "has_percent",
    "has_units",
    "ocr_term_hits",
    "chart_term_hits",
    "math_term_hits",
    "spatial_term_hits",
    "yes_no_form",
)
GAP_FEATURES = (
    "gap_max",
    "gap_sum",
    "gap_count",
    *(name for name in REPAIR_FEATURE_NAMES if name.startswith("gap::")),
)


@dataclass(frozen=True)
class AblationSpec:
    ablation_id: str
    definition: str
    keep_features: tuple[str, ...] | None = None
    use_gap_gate: bool = True
    force_repair: bool = False
    remove_cost: bool = False


ABLATIONS = (
    AblationSpec(
        ablation_id="full",
        definition="complete frozen feature map, cost score, gap stop, and abstention",
    ),
    AblationSpec(
        ablation_id="no_gap_vector",
        definition=(
            "zero residual-gap aggregates/vector and replace the hard-gap input with "
            "one constant; demand/evidence counts remain"
        ),
        keep_features=tuple(
            name for name in REPAIR_FEATURE_NAMES if name not in GAP_FEATURES
        ),
        use_gap_gate=False,
    ),
    AblationSpec(
        ablation_id="confidence_only",
        definition="retain only baseline self-confidence; use no residual-gap gate",
        keep_features=("baseline_confidence",),
        use_gap_gate=False,
    ),
    AblationSpec(
        ablation_id="question_only",
        definition="retain only frozen question-shape/lexical features",
        keep_features=QUESTION_FEATURES,
        use_gap_gate=False,
    ),
    AblationSpec(
        ablation_id="no_cost",
        definition="set every selector cost weight to zero under the same nested grid",
        remove_cost=True,
    ),
    AblationSpec(
        ablation_id="no_abstention",
        definition=(
            "always execute the best-scoring nonbaseline repair, bypassing both the "
            "gap stop and score threshold"
        ),
        force_repair=True,
    ),
)


def main() -> int:
    args = _parse_args()
    training_path = Path(args.training_rows).resolve()
    grid_path = Path(args.grid_config).resolve()
    fit_manifest_path = Path(args.fit_manifest).resolve()
    reference_path = Path(args.reference_rows).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
        path=grid_path,
        logical_name=grid_path.name,
    )
    examples = load_training_examples(training_path)
    config = _load_object(grid_path)
    grid = config.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("grid config lacks an object-valued grid")

    all_rows: dict[str, list[dict[str, Any]]] = {}
    for spec in ABLATIONS:
        transformed = transform_examples(examples, spec)
        all_rows[spec.ablation_id] = nested_ablation_rows(
            transformed,
            grid=grid,
            spec=spec,
        )
    assert_reference_reproduction(all_rows["full"], reference_path)

    policy_rows_path = output_dir / "ablation_policy_rows.csv"
    summary_path = output_dir / "ablation_summary.json"
    statistics_path = output_dir / "ablation_statistics.json"
    write_policy_rows(policy_rows_path, all_rows)
    summaries = {
        spec.ablation_id: summarize_rows(all_rows[spec.ablation_id], grid=grid)
        for spec in ABLATIONS
    }
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "nested_leave_one_source_out",
                "tasks": len(examples),
                "sources": sorted({item.source_id for item in examples}),
                "definitions": {
                    spec.ablation_id: {
                        "definition": spec.definition,
                        "kept_features": (
                            list(spec.keep_features)
                            if spec.keep_features is not None
                            else list(REPAIR_FEATURE_NAMES)
                        ),
                        "use_gap_gate": spec.use_gap_gate,
                        "force_repair": spec.force_repair,
                        "remove_cost": spec.remove_cost,
                    }
                    for spec in ABLATIONS
                },
                "summaries": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    statistics_path.write_text(
        json.dumps(
            build_statistics(
                all_rows,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs = (policy_rows_path, summary_path, statistics_path)
    manifest = {
        "schema_version": 1,
        "source_commit": os.environ.get("EGOR_ANALYSIS_COMMIT", "").strip()
        or _git_output("rev-parse", "HEAD"),
        "fit_source_commit": fit_manifest.get("source_commit"),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (
                training_path,
                grid_path,
                fit_manifest_path,
                reference_path,
            )
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in outputs
        ],
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "reference_reproduced": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(ABLATIONS)} nested ablations over {len(examples)} tasks "
        f"to {output_dir}"
    )
    return 0


def load_training_examples(path: Path) -> list[RepairTrainingExample]:
    examples: list[RepairTrainingExample] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            examples.append(
                RepairTrainingExample(
                    source_id=row["source_id"],
                    task_id=row["task_id"],
                    features=tuple(float(value) for value in json.loads(row["features_json"])),
                    gap_max=float(row["gap_max"]),
                    baseline_quality=float(row["baseline_quality"]),
                    strict_advantages={
                        str(key): float(value)
                        for key, value in json.loads(
                            row["strict_advantages_json"]
                        ).items()
                    },
                    incremental_calls={
                        str(key): float(value)
                        for key, value in json.loads(
                            row["incremental_calls_json"]
                        ).items()
                    },
                )
            )
    if not examples:
        raise ValueError("training rows are empty")
    keys = [(item.source_id, item.task_id) for item in examples]
    if len(keys) != len(set(keys)):
        raise ValueError("training rows contain duplicate source/task keys")
    if len({item.source_id for item in examples}) < 3:
        raise ValueError("nested ablations require at least three source groups")
    return examples


def transform_examples(
    examples: list[RepairTrainingExample], spec: AblationSpec
) -> list[RepairTrainingExample]:
    keep = set(spec.keep_features or REPAIR_FEATURE_NAMES)
    unknown = keep - set(REPAIR_FEATURE_NAMES)
    if unknown:
        raise ValueError(f"ablation references unknown features: {sorted(unknown)}")
    return [
        RepairTrainingExample(
            source_id=item.source_id,
            task_id=item.task_id,
            features=tuple(
                value if name in keep else 0.0
                for name, value in zip(
                    REPAIR_FEATURE_NAMES, item.features, strict=True
                )
            ),
            gap_max=item.gap_max if spec.use_gap_gate else 1.0,
            strict_advantages=item.strict_advantages,
            incremental_calls=item.incremental_calls,
            baseline_quality=item.baseline_quality,
        )
        for item in examples
    ]


def nested_ablation_rows(
    examples: list[RepairTrainingExample],
    *,
    grid: dict[str, Any],
    spec: AblationSpec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = sorted({item.source_id for item in examples})
    cost_weights = (0.0,) if spec.remove_cost else _float_values(grid, "cost_weights")
    for held_source in sources:
        train = [item for item in examples if item.source_id != held_source]
        held = [item for item in examples if item.source_id == held_source]
        selected, _ = select_config_by_source_cv(
            train,
            l2_values=_float_values(grid, "l2_values"),
            thresholds=_float_values(grid, "thresholds"),
            cost_weights=cost_weights,
            evaluation_call_price=float(grid.get("evaluation_call_price", 0.01)),
            maximum_harm_rate=float(grid.get("maximum_harm_rate", 0.05)),
            minimum_repair_rate=float(grid.get("minimum_repair_rate", 0.05)),
            minimum_gap=float(grid.get("minimum_gap", 0.05)),
            force_repair=spec.force_repair,
        )
        selector = CounterfactualRepairSelector.fit(train, config=selected.config)
        for item in held:
            decision = selector.select(
                item.features,
                gap_max=item.gap_max,
                force_repair=spec.force_repair,
            )
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
                    "ablation_id": spec.ablation_id,
                    "policy_id": f"ablation::{spec.ablation_id}",
                    "selected_id": decision.selected_id,
                    "reason": decision.reason,
                    "fold_config_json": json.dumps(
                        asdict(selected.config), sort_keys=True
                    ),
                    "baseline_quality": item.baseline_quality,
                    "policy_quality": item.baseline_quality + delta,
                    "quality_delta": delta,
                    "incremental_calls": calls,
                    "triggered": int(decision.repaired),
                    "strict_harm": int(delta < -0.05),
                    "useful_repair": int(delta > 0.05),
                }
            )
    return sorted(rows, key=lambda row: (row["source_id"], row["task_id"]))


def assert_reference_reproduction(
    rows: list[dict[str, Any]], reference_path: Path
) -> None:
    reference: dict[tuple[str, str], dict[str, str]] = {}
    with reference_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["source_id"], row["task_id"])
            if key in reference:
                raise ValueError(f"duplicate reference task: {key}")
            reference[key] = row
    actual = {(row["source_id"], row["task_id"]): row for row in rows}
    if set(actual) != set(reference):
        raise ValueError("full ablation keys do not reproduce the frozen reference")
    for key, row in actual.items():
        expected = reference[key]
        if row["selected_id"] != expected["selected_id"]:
            raise ValueError(f"full selector decision drift for {key}")
        for field in ("quality_delta", "incremental_calls"):
            if abs(float(row[field]) - float(expected[field])) > 1e-12:
                raise ValueError(f"full selector {field} drift for {key}")


def summarize_rows(rows: list[dict[str, Any]], *, grid: dict[str, Any]) -> dict[str, Any]:
    deltas = [float(row["quality_delta"]) for row in rows]
    calls = [float(row["incremental_calls"]) for row in rows]
    selections = [str(row["selected_id"]) for row in rows]
    call_price = float(grid.get("evaluation_call_price", 0.01))
    return {
        "tasks": len(rows),
        "quality_delta": fmean(deltas),
        "incremental_calls": fmean(calls),
        "utility": fmean(deltas) - call_price * fmean(calls),
        "repair_rate": fmean(item != "baseline" for item in selections),
        "strict_harm_rate": fmean(value < -0.05 for value in deltas),
        "useful_repair_rate": fmean(value > 0.05 for value in deltas),
        "selection_counts": dict(sorted(Counter(selections).items())),
    }


def build_statistics(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    ordered_ids = [spec.ablation_id for spec in ABLATIONS]
    task_keys = [
        (row["source_id"], row["task_id"]) for row in grouped["full"]
    ]
    qualities = {
        ablation_id: {
            (row["source_id"], row["task_id"]): float(row["policy_quality"])
            for row in grouped[ablation_id]
        }
        for ablation_id in ordered_ids
    }
    baseline = {
        (row["source_id"], row["task_id"]): float(row["baseline_quality"])
        for row in grouped["full"]
    }
    if any(set(values) != set(task_keys) for values in qualities.values()):
        raise ValueError("ablation task populations do not match")
    effects_vs_baseline = {}
    full_minus_ablation = {}
    for ablation_id in ordered_ids:
        effects_vs_baseline[ablation_id] = asdict(
            paired_bootstrap_contrast(
                [qualities[ablation_id][key] for key in task_keys],
                [baseline[key] for key in task_keys],
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        )
        if ablation_id != "full":
            full_minus_ablation[ablation_id] = asdict(
                paired_bootstrap_contrast(
                    [qualities["full"][key] for key in task_keys],
                    [qualities[ablation_id][key] for key in task_keys],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                )
            )
    return {
        "schema_version": 1,
        "tasks": len(task_keys),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "effects_vs_baseline": effects_vs_baseline,
        "full_minus_ablation": full_minus_ablation,
    }


def write_policy_rows(
    path: Path, grouped: dict[str, list[dict[str, Any]]]
) -> None:
    fieldnames = [
        "source_id",
        "task_id",
        "ablation_id",
        "policy_id",
        "selected_id",
        "reason",
        "fold_config_json",
        "baseline_quality",
        "policy_quality",
        "quality_delta",
        "incremental_calls",
        "triggered",
        "strict_harm",
        "useful_repair",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for spec in ABLATIONS:
            writer.writerows(grouped[spec.ablation_id])


def _require_manifest_sha(
    manifest: dict[str, Any],
    *,
    section: str,
    path: Path,
    logical_name: str,
) -> None:
    records = manifest.get(section)
    if not isinstance(records, list):
        raise ValueError(f"fit manifest lacks {section}")
    matches = [
        row
        for row in records
        if isinstance(row, dict) and Path(str(row.get("path", ""))).name == logical_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"fit manifest must bind exactly one {section} record named {logical_name}"
        )
    expected = str(matches[0].get("sha256", ""))
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"manifest SHA mismatch for {logical_name}: {actual} != {expected}"
        )


def _float_values(payload: dict[str, Any], key: str) -> tuple[float, ...]:
    raw = payload.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"grid.{key} must be a non-empty list")
    return tuple(float(value) for value in raw)


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
    parser.add_argument("--grid-config", required=True)
    parser.add_argument("--fit-manifest", required=True)
    parser.add_argument("--reference-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_260_713)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
