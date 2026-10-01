#!/usr/bin/env python3
"""Apply the preregistered hard gate to an EGOR selector summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> int:
    args = _parse_args()
    summary_path = Path(args.summary).resolve()
    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()
    summary = _object(summary_path)
    config = _object(config_path)
    selector = _mapping(summary, "source_cross_validated_selector")
    fixed = _mapping(summary, "source_cross_validated_best_fixed")
    standalone = _mapping(summary, "source_cross_validated_standalone_value")
    selection_counts = _mapping(selector, "selection_counts")
    nonbaseline = [
        key
        for key, value in selection_counts.items()
        if key != "baseline" and int(value) > 0
    ]
    selector_utility = float(selector["mean_utility"])
    fixed_utility = float(fixed["mean_utility"])
    standalone_utility = float(standalone["mean_utility"])

    checks = {
        "nested_protocol": _check(
            summary.get("cross_validation_protocol")
            == config["expected_cross_validation_protocol"],
            summary.get("cross_validation_protocol"),
            config["expected_cross_validation_protocol"],
        ),
        "source_count": _check(
            len(summary.get("sources", [])) == int(config["expected_sources"]),
            len(summary.get("sources", [])),
            int(config["expected_sources"]),
        ),
        "task_count": _check(
            int(selector["tasks"]) >= int(config["minimum_tasks"]),
            int(selector["tasks"]),
            f">={int(config['minimum_tasks'])}",
        ),
        "positive_mean_quality": _check(
            float(selector["mean_quality_delta"])
            > float(config["minimum_mean_quality_delta_exclusive"]),
            float(selector["mean_quality_delta"]),
            f">{float(config['minimum_mean_quality_delta_exclusive'])}",
        ),
        "utility_over_best_fixed": _check(
            selector_utility - fixed_utility
            > float(config["minimum_utility_over_best_fixed_exclusive"]),
            selector_utility - fixed_utility,
            f">{float(config['minimum_utility_over_best_fixed_exclusive'])}",
        ),
        "utility_over_standalone": _check(
            selector_utility - standalone_utility
            >= float(config["minimum_utility_over_standalone_inclusive"]),
            selector_utility - standalone_utility,
            f">={float(config['minimum_utility_over_standalone_inclusive'])}",
        ),
        "strict_harm": _check(
            float(selector["strict_harm_rate"])
            <= float(config["maximum_strict_harm_rate"]),
            float(selector["strict_harm_rate"]),
            f"<={float(config['maximum_strict_harm_rate'])}",
        ),
        "baseline_selected": _check(
            not bool(config["require_baseline_selection"])
            or int(selection_counts.get("baseline", 0)) > 0,
            int(selection_counts.get("baseline", 0)),
            ">0",
        ),
        "repair_diversity": _check(
            len(nonbaseline) >= int(config["minimum_distinct_nonbaseline_repairs"]),
            sorted(nonbaseline),
            f">={int(config['minimum_distinct_nonbaseline_repairs'])} distinct repairs",
        ),
    }
    passed = all(bool(item["passed"]) for item in checks.values())
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"summary field {key} must be an object")
    return value


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _check(passed: bool, actual: Any, required: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "actual": actual, "required": required}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
