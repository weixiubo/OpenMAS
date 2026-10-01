from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_selector_gate_writes_pass_and_fail_reports(tmp_path: Path) -> None:
    config = tmp_path / "gate.json"
    config.write_text(
        json.dumps(
            {
                "expected_cross_validation_protocol": "nested_loso",
                "expected_sources": 3,
                "minimum_tasks": 240,
                "maximum_strict_harm_rate": 0.05,
                "minimum_mean_quality_delta_exclusive": 0.0,
                "minimum_utility_over_best_fixed_exclusive": 0.0,
                "minimum_utility_over_standalone_inclusive": 0.0,
                "require_baseline_selection": True,
                "minimum_distinct_nonbaseline_repairs": 2,
            }
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    payload = _passing_summary()
    summary.write_text(json.dumps(payload), encoding="utf-8")
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "check_egor_selector_gate.py"
    )
    passed_report = tmp_path / "passed.json"

    passed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--summary",
            str(summary),
            "--config",
            str(config),
            "--output",
            str(passed_report),
        ],
        check=False,
    )

    assert passed.returncode == 0
    assert json.loads(passed_report.read_text(encoding="utf-8"))["passed"] is True

    payload["source_cross_validated_selector"]["selection_counts"] = {
        "baseline": 120,
        "exact": 120,
    }
    summary.write_text(json.dumps(payload), encoding="utf-8")
    failed_report = tmp_path / "failed.json"
    failed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--summary",
            str(summary),
            "--config",
            str(config),
            "--output",
            str(failed_report),
        ],
        check=False,
    )
    report = json.loads(failed_report.read_text(encoding="utf-8"))

    assert failed.returncode == 2
    assert report["passed"] is False
    assert report["checks"]["repair_diversity"]["passed"] is False


def _passing_summary() -> dict:
    return {
        "cross_validation_protocol": "nested_loso",
        "sources": ["chart", "info", "math"],
        "source_cross_validated_selector": {
            "tasks": 240,
            "mean_quality_delta": 0.03,
            "mean_incremental_calls": 0.5,
            "mean_utility": 0.025,
            "strict_harm_rate": 0.01,
            "selection_counts": {"baseline": 120, "exact": 60, "visual": 60},
        },
        "source_cross_validated_best_fixed": {
            "mean_utility": 0.015,
        },
        "source_cross_validated_standalone_value": {
            "mean_utility": 0.02,
        },
    }
