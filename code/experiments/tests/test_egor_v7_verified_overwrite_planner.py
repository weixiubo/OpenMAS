from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plan_egor_v7_verified_overwrite.py"
)
SPEC = importlib.util.spec_from_file_location("plan_egor_v7_verified_overwrite", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
VisibleTask = module.VisibleTask
plan_verified_overwrites = module.plan_verified_overwrites


def record(answer: str, *, status: str = "ok") -> dict[str, Any]:
    return {"status": status, "parsed": {"answer": answer}}


def test_planner_calls_verifier_only_for_a_real_core_overwrite() -> None:
    tasks = [
        VisibleTask("s", "a", "relaxed_accuracy", ()),
        VisibleTask("s", "b", "relaxed_accuracy", ()),
        VisibleTask("s", "c", "relaxed_accuracy", ()),
        VisibleTask("s", "d", "relaxed_accuracy", ()),
    ]
    repairability = {
        ("s", "a"): {"repairability_triggered": False},
        ("s", "b"): {"repairability_triggered": True},
        ("s", "c"): {"repairability_triggered": True},
        ("s", "d"): {"repairability_triggered": True},
    }
    baseline = {
        ("s", "a"): record("10"),
        ("s", "b"): record("10"),
        ("s", "c"): record("10"),
        ("s", "d"): record("10"),
    }
    core = {
        ("s", "b"): record("10.4"),
        ("s", "c"): record("12"),
        ("s", "d"): record("12", status="error"),
    }

    plans = plan_verified_overwrites(
        tasks=tasks,
        repairability=repairability,
        baseline=baseline,
        core=core,
    )

    assert [row["reason"] for row in plans] == [
        "repairability_gate_abstained",
        "no_baseline_overwrite",
        "core_proposes_overwrite",
        "core_unavailable",
    ]
    assert [row["task_id"] for row in plans if row["verifier_triggered"]] == ["c"]


def test_planner_has_no_reference_or_outcome_argument() -> None:
    task = VisibleTask("s", "a", "accuracy", ())
    repairability = {("s", "a"): {"repairability_triggered": True}}
    inputs = {
        "tasks": [task],
        "repairability": repairability,
        "baseline": {("s", "a"): record("old")},
        "core": {("s", "a"): record("new")},
    }

    first = plan_verified_overwrites(**inputs)
    second = plan_verified_overwrites(**inputs)

    assert first == second
    assert first[0]["verifier_triggered"] is True
    assert set(first[0]) == {
        "schema_version",
        "source_id",
        "task_id",
        "repairability_triggered",
        "verifier_triggered",
        "reason",
    }
