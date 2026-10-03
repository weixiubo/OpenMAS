from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from openagora_experiments.cago.linear import RidgeModel
from openagora_experiments.egor.execution import (
    build_scheduler_repair_plans,
    build_selector_repair_plans,
    task_lists_from_plans,
)
from openagora_experiments.egor.protocol import PilotTask
from openagora_experiments.egor.selection import (
    REPAIR_FEATURE_NAMES,
    CounterfactualRepairSelector,
    RepairSelectionConfig,
)


def test_plans_are_source_blind_and_emit_only_selected_stages() -> None:
    selector = _selector()
    tasks = [_task("source-a", "1"), _task("source-b", "2"), _task("source-c", "3")]
    baselines = {
        ("source-a", "1"): _baseline(open_gap=True),
        ("source-b", "2"): _baseline(open_gap=True),
        ("source-c", "3"): _baseline(open_gap=False),
    }

    plans = build_selector_repair_plans(
        tasks,
        baseline_records=baselines,
        selector=selector,
        template_actions={"exact": ("exact",), "visual": ("visual",)},
    )

    assert plans[0]["feature_vector"] == plans[1]["feature_vector"]
    assert (
        plans[0]["selected_template_id"] == plans[1]["selected_template_id"] == "exact"
    )
    assert plans[2]["selected_template_id"] == "baseline"
    actions, combinations = task_lists_from_plans(
        plans,
        action_ids=("exact", "visual"),
        combination_ids=("exact", "visual"),
    )
    assert actions == {"exact": ["1", "2"], "visual": []}
    assert combinations == {"exact": ["1", "2"], "visual": []}


def test_policy_planner_script_writes_auditable_task_lists(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    selector_path = tmp_path / "selector.json"
    action_spec_path = tmp_path / "actions.json"
    output_dir = tmp_path / "plan"
    task = _task("fixture", "t1")
    tasks_path.write_text(
        json.dumps(
            {
                "task_id": task.task_id,
                "benchmark": task.source_id,
                "question": task.question,
                "image_path": task.image_path,
                "answer": "secret gold",
                "reference_answers": ["secret gold"],
                "metric": "anls",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_path.write_text(
        json.dumps(
            {"source_id": "fixture", "task_id": "t1", **_baseline(open_gap=True)}
        )
        + "\n",
        encoding="utf-8",
    )
    selector_path.write_text(json.dumps(_selector().to_dict()), encoding="utf-8")
    action_spec_path.write_text(
        json.dumps(
            {
                "actions": {"exact": {}, "visual": {}},
                "templates": {
                    "exact": {"action_ids": ["exact"]},
                    "visual": {"action_ids": ["visual"]},
                },
            }
        ),
        encoding="utf-8",
    )
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "plan_egor_selector_policy.py"
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--tasks",
            str(tasks_path),
            "--baseline",
            str(baseline_path),
            "--selector",
            str(selector_path),
            "--action-spec",
            str(action_spec_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    plan = json.loads((output_dir / "plans.jsonl").read_text(encoding="utf-8"))
    assert plan["selected_template_id"] == "exact"
    assert plan["selected_action_ids"] == ["exact"]
    assert "answer" not in plan
    assert (output_dir / "actions_task_ids" / "exact.txt").read_text() == "t1\n"
    assert (output_dir / "actions_task_ids" / "visual.txt").read_text() == ""
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection_counts"] == {"exact": 1}
    assert (
        "absent from the serialized feature schema"
        in manifest["source_id_feature_policy"]
    )

    scheduler_path = tmp_path / "scheduler.jsonl"
    scheduler_path.write_text(
        json.dumps(
            {
                "source_id": "fixture",
                "task_id": "t1",
                "status": "ok",
                "parsed": {
                    "selected_template_id": "visual",
                    "confidence": 0.7,
                    "target_gap_types": ["visual_relation"],
                    "rationale": "inspect the relation",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scheduler_output = tmp_path / "scheduler-plan"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--tasks",
            str(tasks_path),
            "--baseline",
            str(baseline_path),
            "--scheduler-records",
            str(scheduler_path),
            "--action-spec",
            str(action_spec_path),
            "--output-dir",
            str(scheduler_output),
        ],
        check=True,
    )
    scheduler_plan = json.loads(
        (scheduler_output / "plans.jsonl").read_text(encoding="utf-8")
    )
    assert scheduler_plan["selected_template_id"] == "visual"
    assert scheduler_plan["selected_action_ids"] == ["visual"]


def test_scheduler_records_convert_to_the_same_execution_contract() -> None:
    task = _task("fixture", "t1")
    key = (task.source_id, task.task_id)
    plans = build_scheduler_repair_plans(
        [task],
        baseline_records={key: _baseline(open_gap=True)},
        scheduler_records={
            key: {
                "status": "ok",
                "parsed": {
                    "selected_template_id": "visual",
                    "confidence": 0.7,
                    "target_gap_types": ["visual_relation"],
                    "rationale": "inspect the relation",
                },
            }
        },
        template_actions={"exact": ("exact",), "visual": ("visual",)},
    )

    assert plans[0]["selected_template_id"] == "visual"
    assert plans[0]["selected_action_ids"] == ["visual"]
    assert plans[0]["combination_id"] == "visual"
    assert plans[0]["scheduler_confidence"] == 0.7


def _selector() -> CounterfactualRepairSelector:
    width = len(REPAIR_FEATURE_NAMES)
    return CounterfactualRepairSelector(
        config=RepairSelectionConfig(l2=1.0, threshold=0.0, cost_weight=0.0),
        models={
            "exact": _constant_model(width, 0.2),
            "visual": _constant_model(width, 0.1),
        },
        estimated_incremental_calls={"exact": 1.0, "visual": 1.0},
    )


def _constant_model(width: int, value: float) -> RidgeModel:
    return RidgeModel(
        feature_means=(0.0,) * width,
        feature_scales=(1.0,) * width,
        weights=(value, *((0.0,) * width)),
        l2=1.0,
    )


def _task(source_id: str, task_id: str) -> PilotTask:
    return PilotTask(
        task_id=task_id,
        source_id=source_id,
        question="What text is shown?",
        image_path="/tmp/fixture.png",
        mime_type="image/png",
        metric="anls",
        answer="hidden",
        reference_answers=("hidden",),
        options=(),
        metadata={},
    )


def _baseline(*, open_gap: bool) -> dict[str, object]:
    return {
        "status": "ok",
        "parsed": {
            "answer": "draft",
            "confidence": 1.0,
            "demands": [],
            "evidence": [],
            "open_gap_types": ["exact_span"] if open_gap else [],
        },
    }
