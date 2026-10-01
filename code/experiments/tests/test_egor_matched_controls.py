from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from openagora_experiments.egor.evaluation import PolicyRow


def test_matched_random_preserves_trigger_and_template_multiset() -> None:
    module = _load_script("build_egor_matched_control_plans.py")

    plans = [
        _selector_plan("t1", "exact"),
        _selector_plan("t2", "visual"),
        _selector_plan("t3", "baseline"),
    ]

    controls = module.build_matched_control_plans(plans, seed=7)

    assert [row["selector_triggered"] for row in controls] == [True, True, False]
    assert sorted(
        row["matched_random_template_id"]
        for row in controls
        if row["selector_triggered"]
    ) == ["exact", "visual"]
    assert controls[2]["matched_random_template_id"] == "baseline"


def test_matched_analysis_applies_certificate_only_to_random_repair() -> None:
    module = _load_script("analyze_egor_matched_controls.py")

    plans = {
        ("fixture", "t1"): {
            "matched_random_template_id": "exact",
            "resynthesis_candidate_id": "baseline_resynthesis",
        },
        ("fixture", "t2"): {
            "matched_random_template_id": "baseline",
            "resynthesis_candidate_id": "baseline",
        },
    }
    matrix: dict[tuple[str, str, str], dict[str, str]] = {}
    for task_id in ("t1", "t2"):
        matrix[("fixture", task_id, "visual_repair")] = _matrix_row(
            task_id, "visual_repair", candidate_quality=0.0
        )
    matrix[("fixture", "t1", "exact_repair")] = _matrix_row(
        "t1", "exact_repair", candidate_quality=1.0, supported=False
    )
    matrix[("fixture", "t1", "baseline_resynthesis")] = _matrix_row(
        "t1", "baseline_resynthesis", candidate_quality=0.5, supported=False
    )

    rows = module.analyze_matched_controls(plans, matrix)
    by_policy: dict[str, list[PolicyRow]] = {}
    for row in rows:
        by_policy.setdefault(row.policy_id, []).append(row)

    random_t1 = next(
        row for row in by_policy["matched_random_repair"] if row.task_id == "t1"
    )
    resynthesis_t1 = next(
        row for row in by_policy["selector_trigger_resynthesis"] if row.task_id == "t1"
    )
    assert random_t1.policy_quality == 0.0
    assert random_t1.provenance_rejected is True
    assert resynthesis_t1.policy_quality == 0.5
    assert resynthesis_t1.provenance_rejected is False


def _load_script(filename: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selector_plan(task_id: str, template: str) -> dict[str, Any]:
    return {
        "source_id": "fixture",
        "task_id": task_id,
        "selected_template_id": template,
    }


def _matrix_row(
    task_id: str,
    candidate_id: str,
    *,
    candidate_quality: float,
    supported: bool = True,
) -> dict[str, str]:
    return {
        "source_id": "fixture",
        "task_id": task_id,
        "candidate_id": candidate_id,
        "baseline_quality": "0.0",
        "candidate_quality": str(candidate_quality),
        "actual_changed": "1",
        "evidence_supported_change": str(int(supported)),
        "calls": "3",
        "total_tokens": "30",
        "latency_seconds": "3",
        "failed_stages": "",
        "baseline_calls": "1",
        "baseline_total_tokens": "10",
        "baseline_latency_seconds": "1",
    }
