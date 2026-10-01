#!/usr/bin/env python3
"""Audit a plan-selected EGOR execution without requiring a full matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_egor_stage_completeness import audit_stage_matrix  # noqa: E402


COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def main() -> int:
    args = _parse_args()
    report = build_selected_execution_audit(
        tasks_path=Path(args.tasks).resolve(),
        baseline_path=Path(args.baseline).resolve(),
        plans_path=Path(args.plans).resolve(),
        plan_manifest_path=Path(args.plan_manifest).resolve(),
        execution_config_path=Path(args.execution_config).resolve(),
        generation_commit=args.generation_source_commit,
        analysis_commit=args.analysis_commit,
        expected_tasks=args.expected_tasks,
        baseline_seed=args.baseline_seed,
        action_seed=args.action_seed,
        synthesis_seed=args.synthesis_seed,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"selected execution audit {'passed' if report['passed'] else 'failed'}: "
        f"{output}"
    )
    return 0 if report["passed"] else 2


def build_selected_execution_audit(
    *,
    tasks_path: Path,
    baseline_path: Path,
    plans_path: Path,
    plan_manifest_path: Path,
    execution_config_path: Path,
    generation_commit: str,
    analysis_commit: str,
    expected_tasks: int,
    baseline_seed: int,
    action_seed: int,
    synthesis_seed: int,
) -> dict[str, Any]:
    _validate_commit(generation_commit, "generation source commit")
    _validate_commit(analysis_commit, "analysis commit")
    task_keys = _task_keys(tasks_path)
    plans = _jsonl_objects(plans_path)
    if len(task_keys) != expected_tasks or len(plans) != expected_tasks:
        raise ValueError("stronger-backbone task/plan count mismatch")
    plan_keys = [(_text(row, "source_id"), _text(row, "task_id")) for row in plans]
    if plan_keys != task_keys:
        raise ValueError("repair plans do not preserve the complete task order")

    plan_manifest = _load_object(plan_manifest_path)
    if (
        plan_manifest.get("source_commit") != generation_commit
        or int(plan_manifest.get("tasks", -1)) != expected_tasks
    ):
        raise ValueError("repair-plan manifest has incompatible provenance")
    _validate_bound_artifacts(plan_manifest, keys=("inputs", "outputs"))

    config = _load_object(execution_config_path)
    actions = _path_map(config.get("actions"), execution_config_path)
    combinations = _path_map(config.get("combinations"), execution_config_path)
    expected_actions = {key: [] for key in actions}
    expected_combinations = {key: [] for key in combinations}
    for row in plans:
        task_id = _text(row, "task_id")
        selected_raw = row.get("selected_action_ids")
        if not isinstance(selected_raw, list):
            raise ValueError(f"plan selected_action_ids is invalid: {task_id}")
        selected = tuple(str(value) for value in selected_raw)
        combination = "__".join(sorted(selected)) if selected else "baseline"
        if row.get("combination_id") != combination:
            raise ValueError(f"plan combination mismatch: {task_id}")
        for action_id in selected:
            if action_id not in expected_actions:
                raise ValueError(f"plan selects unknown action: {action_id}")
            expected_actions[action_id].append(task_id)
        if selected:
            if combination not in expected_combinations:
                raise ValueError(f"plan selects unknown combination: {combination}")
            expected_combinations[combination].append(task_id)

    result_root = execution_config_path.parent.resolve()
    plan_root = plans_path.parent.resolve()
    list_artifacts: list[Path] = []
    stage_reports: dict[str, dict[str, Any]] = {}
    baseline_report = audit_stage_matrix(
        tasks_path=tasks_path,
        stages={"baseline": baseline_path},
        expected_seeds={"baseline": baseline_seed},
        require_manifest=True,
    )
    _require_generation_commit(baseline_report, "baseline", generation_commit)
    stage_reports["baseline"] = baseline_report

    for namespace, expected, paths, seed in (
        ("actions", expected_actions, actions, action_seed),
        ("combinations", expected_combinations, combinations, synthesis_seed),
    ):
        for stage_id, selected_ids in sorted(expected.items()):
            list_path = plan_root / f"{namespace}_task_ids" / f"{stage_id}.txt"
            list_artifacts.append(list_path)
            if _task_id_lines(list_path) != selected_ids:
                raise ValueError(f"selected task list mismatch: {namespace}/{stage_id}")
            output_path = paths[stage_id]
            if not _is_within(output_path, result_root):
                raise ValueError(f"stage output escapes result root: {output_path}")
            report_key = f"{namespace}/{stage_id}"
            if selected_ids:
                report = audit_stage_matrix(
                    tasks_path=tasks_path,
                    stages={stage_id: output_path},
                    expected_seeds={stage_id: seed},
                    require_manifest=True,
                    allowed_task_ids=set(selected_ids),
                )
                _require_generation_commit(report, stage_id, generation_commit)
                stage_reports[report_key] = report
            else:
                if not output_path.is_file() or output_path.stat().st_size != 0:
                    raise ValueError(
                        f"empty selected stage is not an empty file: {output_path}"
                    )
                manifest_path = output_path.with_suffix(
                    output_path.suffix + ".manifest.json"
                )
                if manifest_path.exists():
                    raise ValueError(
                        f"empty selected stage has a stale manifest: {manifest_path}"
                    )
                stage_reports[report_key] = {
                    "passed": True,
                    "outcome_blind": True,
                    "tasks": {"count": 0, "subset_filter": True},
                    "stages": {},
                }

    passed = all(bool(report.get("passed")) for report in stage_reports.values())
    artifacts = [
        tasks_path,
        baseline_path,
        baseline_path.with_suffix(baseline_path.suffix + ".manifest.json"),
        plans_path,
        plan_manifest_path,
        execution_config_path,
        *list_artifacts,
    ]
    for path in (*actions.values(), *combinations.values()):
        artifacts.append(path)
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        if manifest_path.exists():
            artifacts.append(manifest_path)
    return {
        "schema_version": 1,
        "generation_source_commit": generation_commit,
        "analysis_commit": analysis_commit,
        "expected_tasks": expected_tasks,
        "passed": passed,
        "outcome_blind": True,
        "selection_counts": {
            "actions": {
                key: len(value) for key, value in sorted(expected_actions.items())
            },
            "combinations": {
                key: len(value) for key, value in sorted(expected_combinations.items())
            },
            "baseline": expected_tasks
            - sum(len(value) for value in expected_combinations.values()),
        },
        "stage_reports": stage_reports,
        "artifacts": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in dict.fromkeys(artifacts)
        ],
    }


def _require_generation_commit(
    report: dict[str, Any], stage_id: str, expected: str
) -> None:
    stages = report.get("stages", {})
    stage = stages.get(stage_id, {}) if isinstance(stages, dict) else {}
    manifest = stage.get("manifest", {}) if isinstance(stage, dict) else {}
    if report.get("passed") is not True or manifest.get("git_commit") != expected:
        raise ValueError(f"stage audit/commit failed: {stage_id}")


def _validate_bound_artifacts(
    payload: dict[str, Any], *, keys: tuple[str, ...]
) -> None:
    for key in keys:
        values = payload.get(key)
        if not isinstance(values, list):
            raise ValueError(f"manifest lacks artifact list: {key}")
        for raw in values:
            if not isinstance(raw, dict):
                raise ValueError(f"invalid artifact entry in {key}")
            path = Path(str(raw.get("path", ""))).resolve()
            if _sha256(path) != str(raw.get("sha256", "")):
                raise ValueError(f"bound artifact hash mismatch: {path}")


def _task_keys(path: Path) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for row in _jsonl_objects(path):
        source_id = str(row.get("source_id") or row.get("benchmark") or "").strip()
        task_id = _text(row, "task_id")
        keys.append((source_id, task_id))
    if len(keys) != len(set(keys)):
        raise ValueError("task file contains duplicate source/task keys")
    return keys


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object: {path}")
            rows.append(row)
    return rows


def _task_id_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"missing selected task list: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _path_map(raw: Any, config_path: Path) -> dict[str, Path]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("execution config path map is empty")
    result: dict[str, Path] = {}
    for key, value in raw.items():
        path = Path(str(value)).expanduser()
        result[str(key)] = (
            path if path.is_absolute() else config_path.parent / path
        ).resolve()
    return result


def _text(row: dict[str, Any], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"row lacks {key}")
    return value


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_commit(value: str, label: str) -> None:
    if COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact 40-character Git commit")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--plan-manifest", required=True)
    parser.add_argument("--execution-config", required=True)
    parser.add_argument("--generation-source-commit", required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--expected-tasks", type=int, default=240)
    parser.add_argument("--baseline-seed", type=int, required=True)
    parser.add_argument("--action-seed", type=int, required=True)
    parser.add_argument("--synthesis-seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
