#!/usr/bin/env python3
"""Evaluate a typed coverage-only EGOR planner with nested source holdout."""

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

from openagora.mechanisms.evidence_gap_repair import CapabilityCard, EvidenceType
from openagora_experiments.egor.planning import fit_capability_cards
from openagora_experiments.egor.selection import (
    REPAIR_FEATURE_NAMES,
    RepairTrainingExample,
)
from openagora_experiments.egor.statistics import paired_bootstrap_contrast


ROOT = Path(__file__).resolve().parents[3]
ACTION_CANDIDATES = {
    "exact": "exact",
    "numerical": "numerical",
    "verifier": "verifier",
    "visual": "visual",
}
ADMISSIBLE_GAP_TYPES = {
    "exact": (EvidenceType.EXACT_SPAN,),
    "numerical": (EvidenceType.NUMERICAL_DERIVATION,),
    "verifier": (EvidenceType.INDEPENDENT_VERIFICATION,),
    "visual": (EvidenceType.VISUAL_RELATION,),
}
TEMPLATE_ACTIONS = {
    "exact": ("exact",),
    "exact_visual": ("exact", "visual"),
    "numerical": ("numerical",),
    "verifier": ("verifier",),
    "visual": ("visual",),
}
GAP_FEATURE_INDEX = {
    gap_type: REPAIR_FEATURE_NAMES.index(f"gap::{gap_type.value}")
    for gap_type in EvidenceType
}


@dataclass(frozen=True)
class CoverageConfig:
    threshold: float
    cost_weight: float


@dataclass(frozen=True)
class CoverageChoice:
    config: CoverageConfig
    objective: float
    quality_delta: float
    incremental_calls: float
    repair_rate: float
    strict_harm_rate: float
    useful_repair_rate: float
    eligible: bool


def main() -> int:
    args = _parse_args()
    training_path = Path(args.training_rows).resolve()
    config_path = Path(args.selector_config).resolve()
    fit_manifest_path = Path(args.fit_manifest).resolve()
    reference_path = Path(args.reference_rows).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selector_config = _load_object(config_path)
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
    examples = load_training_examples(training_path)
    capability_rows, matrix_paths = load_capability_rows(
        selector_config,
        config_path=config_path,
        fit_manifest=fit_manifest,
    )
    assert_capability_population(examples, capability_rows)
    grid = selector_config.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("selector config lacks an object-valued grid")

    policy_rows, fold_audit = nested_coverage_rows(
        examples,
        capability_rows=capability_rows,
        grid=grid,
        minimum_eligible=args.minimum_eligible,
    )
    reference = load_reference_rows(reference_path)
    statistics = build_statistics(
        policy_rows,
        reference_rows=reference,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    policy_path = output_dir / "coverage_policy_rows.csv"
    summary_path = output_dir / "coverage_summary.json"
    statistics_path = output_dir / "coverage_statistics.json"
    audit_path = output_dir / "coverage_fold_audit.json"
    write_policy_rows(policy_path, policy_rows)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "nested_leave_one_source_out_typed_coverage",
                "definition": (
                    "Beta(1,1) typed closure supplies fitted on outer-training "
                    "sources only; finite executed template catalog; coverage gain "
                    "minus an inner-selected outer-training mean call-cost penalty; "
                    "explicit baseline abstention"
                ),
                "minimum_eligible": args.minimum_eligible,
                "summary": summarize_rows(policy_rows, grid=grid),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    statistics_path.write_text(
        json.dumps(statistics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(fold_audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    outputs = (policy_path, summary_path, statistics_path, audit_path)
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
        "minimum_eligible": args.minimum_eligible,
        "source_blind_boundary": (
            "source identity defines nested folds only; each held decision uses its "
            "residual gap vector, outer-training capability supplies and mean "
            "template calls, and the inner-selected threshold/cost weight"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote nested typed-coverage control over {len(policy_rows)} tasks to "
        f"{output_dir}"
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
                    features=tuple(
                        float(value) for value in json.loads(row["features_json"])
                    ),
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
    return examples


def load_capability_rows(
    config: dict[str, Any],
    *,
    config_path: Path,
    fit_manifest: dict[str, Any],
) -> tuple[list[dict[str, str]], list[Path]]:
    sources = config.get("sources")
    if not isinstance(sources, list) or len(sources) < 3:
        raise ValueError("selector config requires at least three sources")
    rows: list[dict[str, str]] = []
    paths: list[Path] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_source in sources:
        if not isinstance(raw_source, dict):
            raise ValueError("selector source must be an object")
        source_id = str(raw_source.get("source_id", "")).strip()
        matrices = raw_source.get("matrices")
        if not source_id or not isinstance(matrices, list):
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
                for row in csv.DictReader(handle):
                    candidate_id = canonical.get(str(row.get("candidate_id", "")))
                    if candidate_id not in ACTION_CANDIDATES.values():
                        continue
                    task_id = str(row.get("task_id", "")).strip()
                    key = (source_id, task_id, candidate_id)
                    if not task_id or key in seen:
                        raise ValueError(f"duplicate/empty capability row: {key}")
                    seen.add(key)
                    rows.append(
                        {
                            **row,
                            "source_id": source_id,
                            "candidate_id": candidate_id,
                        }
                    )
    if not rows:
        raise ValueError("no singleton capability rows were loaded")
    return rows, list(dict.fromkeys(paths))


def assert_capability_population(
    examples: list[RepairTrainingExample], rows: list[dict[str, str]]
) -> None:
    expected = {
        (item.source_id, item.task_id, candidate_id)
        for item in examples
        for candidate_id in ACTION_CANDIDATES.values()
    }
    actual = {(row["source_id"], row["task_id"], row["candidate_id"]) for row in rows}
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise ValueError(
            f"capability population mismatch; missing={missing}, extra={extra}"
        )
    gaps: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["source_id"], row["task_id"])
        canonical = json.dumps(_gap_vector(row), separators=(",", ":"), sort_keys=True)
        if key in gaps and gaps[key] != canonical:
            raise ValueError(f"candidate rows disagree on residual gaps for {key}")
        gaps[key] = canonical
    for item in examples:
        key = (item.source_id, item.task_id)
        matrix_gaps = _gap_vector_from_json(gaps[key])
        feature_gaps = _gap_vector_from_example(item)
        if any(
            not math.isclose(
                matrix_gaps.get(gap_type, 0.0),
                feature_gaps[gap_type],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for gap_type in EvidenceType
        ):
            raise ValueError(f"matrix and frozen features disagree on gaps for {key}")


def nested_coverage_rows(
    examples: list[RepairTrainingExample],
    *,
    capability_rows: list[dict[str, str]],
    grid: dict[str, Any],
    minimum_eligible: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    folds: dict[str, Any] = {}
    sources = sorted({item.source_id for item in examples})
    for held_source in sources:
        train = [item for item in examples if item.source_id != held_source]
        held = [item for item in examples if item.source_id == held_source]
        train_capability = [
            row for row in capability_rows if row["source_id"] != held_source
        ]
        selected, choices = select_config_by_source_cv(
            train,
            capability_rows=train_capability,
            grid=grid,
            minimum_eligible=minimum_eligible,
        )
        cards, estimates = fit_capability_cards(
            train_capability,
            action_candidates=ACTION_CANDIDATES,
            admissible_gap_types=ADMISSIBLE_GAP_TYPES,
            minimum_gap=float(grid.get("minimum_gap", 0.05)),
            minimum_eligible=minimum_eligible,
        )
        template_costs = _template_costs(train)
        output.extend(
            decision_rows(
                held,
                cards=cards,
                config=selected.config,
                minimum_gap=float(grid.get("minimum_gap", 0.05)),
                template_costs=template_costs,
            )
        )
        folds[held_source] = {
            "selected": _choice_payload(selected),
            "choices": [_choice_payload(choice) for choice in choices],
            "capability_cards": [_card_payload(card) for card in cards],
            "capability_estimates": [
                {**asdict(estimate), "gap_type": estimate.gap_type.value}
                for estimate in estimates
            ],
            "training_sources": sorted({item.source_id for item in train}),
            "template_costs": template_costs,
        }
    return (
        sorted(output, key=lambda row: (row["source_id"], row["task_id"])),
        {
            "schema_version": 1,
            "protocol": "nested_leave_one_source_out_typed_coverage",
            "folds": folds,
        },
    )


def select_config_by_source_cv(
    examples: list[RepairTrainingExample],
    *,
    capability_rows: list[dict[str, str]],
    grid: dict[str, Any],
    minimum_eligible: int,
) -> tuple[CoverageChoice, tuple[CoverageChoice, ...]]:
    sources = sorted({item.source_id for item in examples})
    if len(sources) < 2:
        raise ValueError("coverage config selection requires two source groups")
    choices: list[CoverageChoice] = []
    for threshold in _float_values(grid, "thresholds"):
        for cost_weight in _float_values(grid, "cost_weights"):
            config = CoverageConfig(threshold=threshold, cost_weight=cost_weight)
            rows: list[dict[str, Any]] = []
            for held_source in sources:
                train = [item for item in examples if item.source_id != held_source]
                held = [item for item in examples if item.source_id == held_source]
                training_capability = [
                    row for row in capability_rows if row["source_id"] != held_source
                ]
                cards, _ = fit_capability_cards(
                    training_capability,
                    action_candidates=ACTION_CANDIDATES,
                    admissible_gap_types=ADMISSIBLE_GAP_TYPES,
                    minimum_gap=float(grid.get("minimum_gap", 0.05)),
                    minimum_eligible=minimum_eligible,
                )
                template_costs = _template_costs(train)
                rows.extend(
                    decision_rows(
                        held,
                        cards=cards,
                        config=config,
                        minimum_gap=float(grid.get("minimum_gap", 0.05)),
                        template_costs=template_costs,
                    )
                )
            summary = summarize_rows(rows, grid=grid)
            eligible = summary["strict_harm_rate"] <= float(
                grid.get("maximum_harm_rate", 0.05)
            ) and summary["repair_rate"] >= float(grid.get("minimum_repair_rate", 0.05))
            choices.append(
                CoverageChoice(
                    config=config,
                    objective=summary["utility"] if eligible else float("-inf"),
                    quality_delta=summary["quality_delta"],
                    incremental_calls=summary["incremental_calls"],
                    repair_rate=summary["repair_rate"],
                    strict_harm_rate=summary["strict_harm_rate"],
                    useful_repair_rate=summary["useful_repair_rate"],
                    eligible=eligible,
                )
            )
    eligible_choices = [choice for choice in choices if choice.eligible]
    if not eligible_choices:
        raise ValueError("no typed-coverage config satisfies frozen constraints")
    selected = min(
        eligible_choices,
        key=lambda choice: (
            -choice.objective,
            -choice.quality_delta,
            choice.incremental_calls,
            -choice.config.threshold,
            choice.config.cost_weight,
        ),
    )
    return selected, tuple(choices)


def decision_rows(
    examples: list[RepairTrainingExample],
    *,
    cards: list[CapabilityCard],
    config: CoverageConfig,
    minimum_gap: float,
    template_costs: dict[str, float],
) -> list[dict[str, Any]]:
    card_map = {card.action_id: card for card in cards}
    rows: list[dict[str, Any]] = []
    for item in examples:
        gap_vector = _gap_vector_from_example(item)
        scores = {
            template_id: coverage_gain(
                gap_vector,
                actions=actions,
                cards=card_map,
            )
            - config.cost_weight * template_costs[template_id]
            for template_id, actions in TEMPLATE_ACTIONS.items()
        }
        best = min(
            scores,
            key=lambda candidate_id: (
                -scores[candidate_id],
                template_costs[candidate_id],
                candidate_id,
            ),
        )
        if max(gap_vector.values(), default=0.0) <= minimum_gap:
            selected = "baseline"
            reason = "no_residual_gap"
        elif scores[best] <= config.threshold:
            selected = "baseline"
            reason = "insufficient_coverage_gain"
        else:
            selected = best
            reason = "positive_typed_coverage_gain"
        delta = (
            float(item.strict_advantages[selected]) if selected != "baseline" else 0.0
        )
        calls = (
            float(item.incremental_calls[selected]) if selected != "baseline" else 0.0
        )
        rows.append(
            {
                "source_id": item.source_id,
                "task_id": item.task_id,
                "policy_id": "coverage_only",
                "selected_id": selected,
                "reason": reason,
                "fold_config_json": json.dumps(asdict(config), sort_keys=True),
                "baseline_quality": item.baseline_quality,
                "policy_quality": item.baseline_quality + delta,
                "quality_delta": delta,
                "incremental_calls": calls,
                "triggered": int(selected != "baseline"),
                "strict_harm": int(delta < -0.05),
                "useful_repair": int(delta > 0.05),
                "coverage_scores_json": json.dumps(scores, sort_keys=True),
            }
        )
    return rows


def coverage_gain(
    gaps: dict[EvidenceType, float],
    *,
    actions: tuple[str, ...],
    cards: dict[str, CapabilityCard],
) -> float:
    return sum(
        gap
        * (1.0 - math.prod(1.0 - cards[action].supply(gap_type) for action in actions))
        for gap_type, gap in gaps.items()
    )


def summarize_rows(
    rows: list[dict[str, Any]], *, grid: dict[str, Any]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("coverage rows are empty")
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
    rows: list[dict[str, Any]],
    *,
    reference_rows: dict[tuple[str, str], dict[str, str]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    ordered = [(row["source_id"], row["task_id"]) for row in rows]
    if set(ordered) != set(reference_rows):
        raise ValueError("coverage and full-selector task populations differ")
    coverage = [float(row["policy_quality"]) for row in rows]
    baseline = [float(row["baseline_quality"]) for row in rows]
    full = [
        float(row["baseline_quality"]) + float(reference_rows[key]["quality_delta"])
        for row, key in zip(rows, ordered, strict=True)
    ]
    return {
        "schema_version": 1,
        "tasks": len(rows),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "coverage_minus_baseline": asdict(
            paired_bootstrap_contrast(
                coverage,
                baseline,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        ),
        "full_minus_coverage": asdict(
            paired_bootstrap_contrast(
                full,
                coverage,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        ),
    }


def load_reference_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["source_id"], row["task_id"])
            if key in rows:
                raise ValueError(f"duplicate reference row: {key}")
            rows[key] = row
    return rows


def write_policy_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "source_id",
        "task_id",
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
        "coverage_scores_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _gap_vector(row: dict[str, str]) -> dict[EvidenceType, float]:
    return _gap_vector_from_json(str(row.get("residual_gaps_json", "{}")))


def _gap_vector_from_json(raw: str) -> dict[EvidenceType, float]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("residual_gaps_json must encode an object")
    return {EvidenceType(str(key)): float(value) for key, value in payload.items()}


def _gap_vector_from_example(
    item: RepairTrainingExample,
) -> dict[EvidenceType, float]:
    return {
        gap_type: float(item.features[index])
        for gap_type, index in GAP_FEATURE_INDEX.items()
    }


def _template_costs(
    examples: list[RepairTrainingExample],
) -> dict[str, float]:
    if not examples:
        raise ValueError("template cost fitting requires training examples")
    missing = [
        template_id
        for template_id in TEMPLATE_ACTIONS
        if any(template_id not in item.incremental_calls for item in examples)
    ]
    if missing:
        raise ValueError(f"training rows lack template costs: {sorted(missing)}")
    return {
        template_id: fmean(
            float(item.incremental_calls[template_id]) for item in examples
        )
        for template_id in TEMPLATE_ACTIONS
    }


def _card_payload(card: CapabilityCard) -> dict[str, Any]:
    return {
        "action_id": card.action_id,
        "estimated_cost": card.estimated_cost,
        "supply_by_type": {
            key.value: value for key, value in sorted(card.supply_by_type.items())
        },
    }


def _choice_payload(choice: CoverageChoice) -> dict[str, Any]:
    payload = {**asdict(choice), "config": asdict(choice.config)}
    if not math.isfinite(choice.objective):
        payload["objective"] = None
    return payload


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


def _float_values(payload: dict[str, Any], key: str) -> tuple[float, ...]:
    raw = payload.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"grid.{key} must be a non-empty list")
    return tuple(float(value) for value in raw)


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
    parser.add_argument("--minimum-eligible", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_260_713)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
