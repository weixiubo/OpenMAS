from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_egor_answer_only_v8.py"
SPEC = importlib.util.spec_from_file_location("analyze_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(
    source: str,
    task_id: str,
    *,
    complete: bool = True,
    accepted: bool = True,
    core_delta: float = 0.1,
    verified_delta: float = 0.1,
) -> dict[str, Any]:
    return {
        "source": source,
        "task_id": task_id,
        "triggered": True,
        "core_changed": True,
        "verifier_complete": complete,
        "accept_overwrite": accepted,
        "decision_reason": "accepted" if accepted else "retained",
        "corroborators": ["exact"] if accepted else [],
        "baseline_quality": 0.5,
        "core_delta": core_delta,
        "strict_pair_delta": 0.0,
        "verified_delta": verified_delta,
        "calls": 5.0,
        "verifier_called": True,
    }


def passing_rows() -> dict[str, list[dict[str, Any]]]:
    return {
        "chartqa": [
            row("chartqa", "c1"),
            row(
                "chartqa",
                "c2",
                accepted=False,
                core_delta=-0.1,
                verified_delta=0.0,
            ),
        ],
        "infographic": [row("infographic", "i1")],
        "mathvista": [row("mathvista", "m1")],
        "ocrbench": [row("ocrbench", "o1")],
    }


def test_v8_gate_passes_only_with_sourcewise_completion_and_opened_signal() -> None:
    report = MODULE.analyze(passing_rows(), bootstrap_samples=1000, seed=9)
    assert report["passed"]
    assert report["v7_format_outputs_used"] is False
    assert report["completion_by_source"] == {
        "chartqa": 1.0,
        "infographic": 1.0,
        "mathvista": 1.0,
        "ocrbench": 1.0,
    }
    assert report["verifier_generation_rows"] == 5
    assert report["deployed_verifier_call_rows"] == 5


def test_v8_gate_fails_closed_on_one_source_completion_failure() -> None:
    rows = passing_rows()
    rows["infographic"][0]["verifier_complete"] = False
    report = MODULE.analyze(rows, bootstrap_samples=1000, seed=9)
    assert not report["passed"]
    completion = next(
        item for item in report["criteria"]
        if item["name"] == "sourcewise_operational_completion"
    )
    assert not completion["passed"]
