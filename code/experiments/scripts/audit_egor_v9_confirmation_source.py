#!/usr/bin/env python3
"""Audit one EGOR-v9 source without reading labels or scoring outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import audit_egor_v7_confirmation_source as shared


STAGE_SPECS = {
    "baseline": {
        "file": "baseline_qwen3.jsonl",
        "stage": "baseline",
        "model_name": "qwen3-vl-8b-instruct",
        "model_arg": "qwen3_model",
        "seed_offset": 10,
        "ids": "all",
    },
    "exact": {
        "file": "specialist_exact_qwen25.jsonl",
        "stage": "specialist",
        "model_name": "qwen2_5_vl_7b_instruct",
        "model_arg": "qwen25_model",
        "seed_offset": 30,
        "ids": "repairability",
    },
    "visual": {
        "file": "specialist_visual_internvl3.jsonl",
        "stage": "specialist",
        "model_name": "internvl3-8b-hf",
        "model_arg": "internvl_model",
        "seed_offset": 31,
        "ids": "repairability",
    },
    "core_ablation": {
        "file": "core_pair_ablation.jsonl",
        "stage": "synthesis",
        "model_name": "qwen3-vl-8b-instruct",
        "model_arg": "qwen3_model",
        "seed_offset": 40,
        "ids": "repairability",
    },
    "resolver": {
        "file": "resolver_v9_qwen35.jsonl",
        "stage": "verifier_v8",
        "model_name": "qwen3.5-9b",
        "model_arg": "qwen35_model",
        "seed_offset": 50,
        "ids": "repairability",
    },
}


def audit_source(
    *,
    tasks_path: Path,
    root: Path,
    repairability_path: Path,
    source_commit: str,
    mode: str,
    expected_tasks: int,
    expected_tasks_sha256: str,
    seed_root: int,
    model_paths: dict[str, str],
) -> dict[str, Any]:
    task_keys, task_indices = shared._load_task_keys(tasks_path)
    actual_tasks_sha256 = shared._sha256(tasks_path)
    repairability = shared._load_plan(
        repairability_path, decision="repairability_triggered"
    )
    task_key_set = set(task_keys)
    plan_checks = {"repairability_keys": set(repairability) == task_key_set}
    id_sets = {
        "all": task_key_set,
        "repairability": {
            key
            for key, row in repairability.items()
            if row["repairability_triggered"]
        },
    }
    stages = {
        name: shared._audit_stage(
            path=root / str(spec["file"]),
            tasks_path=tasks_path,
            expected_keys=id_sets[str(spec["ids"])],
            task_indices=task_indices,
            expected_stage=str(spec["stage"]),
            expected_model_name=str(spec["model_name"]),
            expected_model_path=model_paths[str(spec["model_arg"])],
            expected_seed=seed_root + int(spec["seed_offset"]),
            expected_commit=source_commit,
            expect_cpu_offload=name == "resolver",
        )
        for name, spec in STAGE_SPECS.items()
    }
    resolver_expected = len(id_sets["repairability"])
    resolver_operational = stages["resolver"]["operational_records"]
    completion = (
        resolver_operational / resolver_expected if resolver_expected else 0.0
    )
    task_checks = {
        "count": len(task_keys) == expected_tasks,
        "sha256": actual_tasks_sha256 == expected_tasks_sha256,
        "single_source": len({source for source, _ in task_keys}) == 1,
        "unique_task_ids": len({task_id for _, task_id in task_keys})
        == len(task_keys),
    }
    integrity_passed = (
        all(task_checks.values())
        and all(plan_checks.values())
        and all(stage["integrity_complete"] for stage in stages.values())
    )
    format_gate = {
        "all_nonresolver_rows_successful": all(
            stages[name]["successful_records"]
            == stages[name]["expected_records"]
            for name in ("baseline", "exact", "visual", "core_ablation")
        ),
        "resolver_was_exercised": resolver_expected > 0,
        "resolver_operational_completion": completion >= 0.95,
    }
    passed = integrity_passed and (
        all(format_gate.values()) if mode == "format" else True
    )
    return {
        "schema_version": 1,
        "mode": mode,
        "passed": passed,
        "integrity_passed": integrity_passed,
        "outcome_blind": True,
        "allowed_task_fields_read": [
            "benchmark/source_id",
            "task_id",
            "image_path",
        ],
        "outcome_fields_read": False,
        "scores_computed": False,
        "resolver_contract": {
            "stage": "verifier_v8",
            "file": "resolver_v9_qwen35.jsonl",
            "ids": "repairability_triggered",
        },
        "source_commit": source_commit,
        "tasks": {
            "path": str(tasks_path),
            "sha256": actual_tasks_sha256,
            "count": len(task_keys),
            "checks": task_checks,
        },
        "plan_checks": plan_checks,
        "repairability_triggered": resolver_expected,
        "resolver_triggered": resolver_expected,
        "resolver_operational_completion": completion,
        "format_gate": format_gate,
        "stages": stages,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--repairability", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--mode", choices=["format", "confirmation"], required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--expected-tasks-sha256", required=True)
    parser.add_argument("--seed-root", type=int, required=True)
    parser.add_argument("--qwen3-model", required=True)
    parser.add_argument("--qwen25-model", required=True)
    parser.add_argument("--internvl-model", required=True)
    parser.add_argument("--qwen35-model", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = audit_source(
        tasks_path=Path(args.tasks).resolve(),
        root=Path(args.root).resolve(),
        repairability_path=Path(args.repairability).resolve(),
        source_commit=args.source_commit,
        mode=args.mode,
        expected_tasks=args.expected_tasks,
        expected_tasks_sha256=args.expected_tasks_sha256,
        seed_root=args.seed_root,
        model_paths={
            "qwen3_model": str(Path(args.qwen3_model).resolve()),
            "qwen25_model": str(Path(args.qwen25_model).resolve()),
            "internvl_model": str(Path(args.internvl_model).resolve()),
            "qwen35_model": str(Path(args.qwen35_model).resolve()),
        },
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
