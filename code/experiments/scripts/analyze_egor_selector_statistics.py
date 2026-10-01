#!/usr/bin/env python3
"""Compute paired uncertainty for nested-LOSO EGOR selector rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from openagora_experiments.egor.statistics import paired_bootstrap_contrast


TaskKey = tuple[str, str]


@dataclass(frozen=True)
class SelectorRow:
    source_id: str
    task_id: str
    selected_id: str
    quality_delta: float
    incremental_calls: float
    strict_harm: int


def main() -> int:
    args = _parse_args()
    selector_path = Path(args.selector_rows).resolve()
    fixed_path = Path(args.best_fixed_rows).resolve()
    standalone_path = Path(args.standalone_rows).resolve()
    selector = _load_rows(selector_path)
    fixed = _load_rows(fixed_path)
    standalone = _load_rows(standalone_path)
    _require_same_tasks(selector, fixed, "best fixed")
    _require_same_tasks(selector, standalone, "standalone value")

    contrasts = {
        "selector_minus_baseline_quality": _paired(
            selector,
            None,
            left=lambda row: row.quality_delta,
            right=lambda row: 0.0,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "selector_minus_best_fixed_quality": _paired(
            selector,
            fixed,
            left=lambda row: row.quality_delta,
            right=lambda row: row.quality_delta,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 1,
        ),
        "selector_minus_best_fixed_utility": _paired(
            selector,
            fixed,
            left=lambda row: _utility(row, args.evaluation_call_price),
            right=lambda row: _utility(row, args.evaluation_call_price),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 2,
        ),
        "selector_minus_standalone_quality": _paired(
            selector,
            standalone,
            left=lambda row: row.quality_delta,
            right=lambda row: row.quality_delta,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 3,
        ),
        "selector_minus_standalone_utility": _paired(
            selector,
            standalone,
            left=lambda row: _utility(row, args.evaluation_call_price),
            right=lambda row: _utility(row, args.evaluation_call_price),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 4,
        ),
        "calls_saved_vs_best_fixed": _paired(
            fixed,
            selector,
            left=lambda row: row.incremental_calls,
            right=lambda row: row.incremental_calls,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 5,
        ),
        "calls_saved_vs_standalone": _paired(
            standalone,
            selector,
            left=lambda row: row.incremental_calls,
            right=lambda row: row.incremental_calls,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 6,
        ),
    }
    sources = sorted({key[0] for key in selector})
    by_source = {
        source: _paired(
            {key: row for key, row in selector.items() if key[0] == source},
            None,
            left=lambda row: row.quality_delta,
            right=lambda row: 0.0,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 100 + index,
        )
        for index, source in enumerate(sources)
    }
    output = {
        "schema_version": 1,
        "inputs": {
            "selector": _file_record(selector_path),
            "best_fixed": _file_record(fixed_path),
            "standalone_value": _file_record(standalone_path),
        },
        "tasks": len(selector),
        "sources": sources,
        "evaluation_call_price": args.evaluation_call_price,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "selector_counts": {
            "selections": dict(
                sorted(Counter(row.selected_id for row in selector.values()).items())
            ),
            "strict_harms": sum(row.strict_harm for row in selector.values()),
        },
        "contrasts": contrasts,
        "selector_minus_baseline_quality_by_source": by_source,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(contrasts)} pooled contrasts to {output_path}")
    return 0


def _load_rows(path: Path) -> dict[TaskKey, SelectorRow]:
    rows: dict[TaskKey, SelectorRow] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            key = (str(raw.get("source_id", "")), str(raw.get("task_id", "")))
            if not all(key):
                raise ValueError(f"selector row lacks source/task ID: {path}")
            if key in rows:
                raise ValueError(f"duplicate selector task {key}: {path}")
            rows[key] = SelectorRow(
                source_id=key[0],
                task_id=key[1],
                selected_id=str(raw.get("selected_id", "")),
                quality_delta=float(raw["quality_delta"]),
                incremental_calls=float(raw["incremental_calls"]),
                strict_harm=int(raw["strict_harm"]),
            )
    if not rows:
        raise ValueError(f"selector rows are empty: {path}")
    return rows


def _paired(
    left_rows: dict[TaskKey, SelectorRow],
    right_rows: dict[TaskKey, SelectorRow] | None,
    *,
    left: Callable[[SelectorRow], float],
    right: Callable[[SelectorRow], float],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float | int]:
    keys = sorted(left_rows)
    if right_rows is not None and set(keys) != set(right_rows):
        raise ValueError("paired contrast task mismatch")
    left_values = [left(left_rows[key]) for key in keys]
    right_values = [
        right(right_rows[key] if right_rows is not None else left_rows[key])
        for key in keys
    ]
    return asdict(
        paired_bootstrap_contrast(
            left_values,
            right_values,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    )


def _require_same_tasks(
    selector: dict[TaskKey, SelectorRow],
    comparison: dict[TaskKey, SelectorRow],
    label: str,
) -> None:
    if set(selector) != set(comparison):
        missing = sorted(set(selector) - set(comparison))[:5]
        extra = sorted(set(comparison) - set(selector))[:5]
        raise ValueError(f"{label} task mismatch: missing={missing}, extra={extra}")


def _utility(row: SelectorRow, call_price: float) -> float:
    return row.quality_delta - call_price * row.incremental_calls


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-rows", required=True)
    parser.add_argument("--best-fixed-rows", required=True)
    parser.add_argument("--standalone-rows", required=True)
    parser.add_argument("--evaluation-call-price", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_260_712)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
