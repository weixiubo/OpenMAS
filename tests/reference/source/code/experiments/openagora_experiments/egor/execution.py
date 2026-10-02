"""Prospective execution plans for a frozen counterfactual repair selector."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from openagora_experiments.egor.evaluation import residual_gap_vector
from openagora_experiments.egor.planning import combination_id
from openagora_experiments.egor.protocol import PilotTask
from openagora_experiments.egor.selection import (
    REPAIR_FEATURE_NAMES,
    CounterfactualRepairSelector,
    visible_repair_feature_vector,
)


def build_selector_repair_plans(
    tasks: Sequence[PilotTask],
    *,
    baseline_records: Mapping[tuple[str, str], Mapping[str, Any]],
    selector: CounterfactualRepairSelector,
    template_actions: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Choose one repair template per task from baseline-visible state only.

    Paper mapping: the deployment procedure under ``sec:evidence-certificate``.
    The call sequence is baseline -> source-blind selection -> at most one
    repair template -> one synthesis; the reference and candidate outcomes are
    unavailable here by construction.
    """

    _validate_template_actions(selector, template_actions)
    feature_schema_sha256 = hashlib.sha256(
        json.dumps(REPAIR_FEATURE_NAMES, separators=(",", ":")).encode()
    ).hexdigest()
    plans: list[dict[str, Any]] = []
    for task in tasks:
        key = (task.source_id, task.task_id)
        record = baseline_records.get(key)
        if record is None or str(record.get("status", "")) != "ok":
            plans.append(
                {
                    "schema_version": 2,
                    "source_id": task.source_id,
                    "task_id": task.task_id,
                    "baseline_status": (
                        "missing" if record is None else str(record.get("status", ""))
                    ),
                    "selected_template_id": "baseline",
                    "selected_action_ids": [],
                    "combination_id": "baseline",
                    "residual_gaps": {},
                    "gap_max": 0.0,
                    "predicted_advantages": {},
                    "net_advantage_scores": {},
                    "reason": "baseline_stage_failed",
                    "feature_schema_sha256": feature_schema_sha256,
                    "feature_vector": [],
                }
            )
            continue
        raw_parsed = record.get("parsed", {})
        baseline = raw_parsed if isinstance(raw_parsed, Mapping) else {}
        features = visible_repair_feature_vector(task, baseline)
        gaps = residual_gap_vector(baseline)
        gap_max = max(gaps.values(), default=0.0)
        # task.source_id is used for record lookup and audit only; it is never
        # an argument to the feature map or selector.
        decision = selector.select(features, gap_max=gap_max)
        actions = (
            tuple(template_actions[decision.selected_id]) if decision.repaired else ()
        )
        plans.append(
            {
                "schema_version": 2,
                "source_id": task.source_id,
                "task_id": task.task_id,
                "baseline_status": "ok",
                "selected_template_id": decision.selected_id,
                "selected_action_ids": list(actions),
                "combination_id": combination_id(actions),
                "residual_gaps": {
                    key.value: value for key, value in sorted(gaps.items())
                },
                "gap_max": gap_max,
                "predicted_advantages": dict(decision.predicted_advantages),
                "net_advantage_scores": dict(decision.net_advantage_scores),
                "reason": decision.reason,
                "feature_schema_sha256": feature_schema_sha256,
                "feature_vector": list(features),
            }
        )
    return plans


def task_lists_from_plans(
    plans: Sequence[Mapping[str, Any]],
    *,
    action_ids: Sequence[str],
    combination_ids: Sequence[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build exact stage task lists without consulting labels or outcomes."""

    actions = {action_id: [] for action_id in action_ids}
    combinations = {item: [] for item in combination_ids}
    for plan in plans:
        task_id = str(plan.get("task_id", "")).strip()
        if not task_id:
            raise ValueError("repair plan lacks task_id")
        selected_actions = plan.get("selected_action_ids", [])
        if not isinstance(selected_actions, list):
            raise ValueError("selected_action_ids must be a list")
        selected = tuple(str(value) for value in selected_actions)
        combination = combination_id(selected)
        if str(plan.get("combination_id", "")) != combination:
            raise ValueError(f"combination mismatch for task {task_id}")
        for action_id in selected:
            if action_id not in actions:
                raise ValueError(f"plan selects unknown action {action_id}")
            actions[action_id].append(task_id)
        if selected:
            if combination not in combinations:
                raise ValueError(f"plan selects unknown combination {combination}")
            combinations[combination].append(task_id)
    return actions, combinations


def build_scheduler_repair_plans(
    tasks: Sequence[PilotTask],
    *,
    baseline_records: Mapping[tuple[str, str], Mapping[str, Any]],
    scheduler_records: Mapping[tuple[str, str], Mapping[str, Any]],
    template_actions: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Convert outcome-blind LLM scheduler records into execution plans."""

    plans: list[dict[str, Any]] = []
    for task in tasks:
        key = (task.source_id, task.task_id)
        baseline = baseline_records.get(key)
        scheduler = scheduler_records.get(key)
        baseline_ok = baseline is not None and str(baseline.get("status", "")) == "ok"
        scheduler_ok = (
            scheduler is not None and str(scheduler.get("status", "")) == "ok"
        )
        if not baseline_ok or not scheduler_ok:
            selected_template = "baseline"
            actions: tuple[str, ...] = ()
            reason = (
                "baseline_stage_failed" if not baseline_ok else "scheduler_stage_failed"
            )
            parsed: Mapping[str, Any] = {}
        else:
            raw_parsed = scheduler.get("parsed", {}) if scheduler is not None else {}
            parsed = raw_parsed if isinstance(raw_parsed, Mapping) else {}
            selected_template = str(parsed.get("selected_template_id", "baseline"))
            if selected_template == "baseline":
                actions = ()
            elif selected_template in template_actions:
                actions = tuple(template_actions[selected_template])
            else:
                selected_template = "baseline"
                actions = ()
            reason = str(parsed.get("rationale", "")).strip() or "llm_scheduler"
        plans.append(
            {
                "schema_version": 2,
                "source_id": task.source_id,
                "task_id": task.task_id,
                "baseline_status": (
                    "missing" if baseline is None else str(baseline.get("status", ""))
                ),
                "scheduler_status": (
                    "missing" if scheduler is None else str(scheduler.get("status", ""))
                ),
                "selected_template_id": selected_template,
                "selected_action_ids": list(actions),
                "combination_id": combination_id(actions),
                "reason": reason,
                "scheduler_confidence": float(parsed.get("confidence", 0.0) or 0.0),
                "target_gap_types": list(parsed.get("target_gap_types", [])),
            }
        )
    return plans


def _validate_template_actions(
    selector: CounterfactualRepairSelector,
    template_actions: Mapping[str, Sequence[str]],
) -> None:
    if set(template_actions) != set(selector.candidate_ids):
        raise ValueError("execution templates must exactly match selector candidates")
    for template_id, actions in template_actions.items():
        normalized = tuple(str(value).strip() for value in actions)
        if not normalized or any(not value for value in normalized):
            raise ValueError(f"template {template_id} requires non-empty actions")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"template {template_id} repeats an action")
