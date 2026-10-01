#!/usr/bin/env python3
"""Run one resumable EGOR generation stage with one local multimodal model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from openagora.mechanisms.evidence_gap_repair import EvidenceType
from openagora_experiments.egor import local_provider as egor_local_provider_module
from openagora_experiments.egor.local_provider import EgorLocalModelProvider
from openagora_experiments.egor import runtime_types as egor_runtime_types_module
from openagora_experiments.egor.protocol import PilotTask
from openagora_experiments.egor.runner import (
    load_successful_stage_records,
    load_tasks,
    run_baseline_stage,
    run_scheduler_stage,
    run_specialist_stage,
    run_synthesis_stage,
    run_verifier_stage,
    run_verifier_v6_stage,
    run_verifier_v7_stage,
    run_verifier_v8_stage,
)


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    args = _parse_args()
    task_path = Path(args.tasks).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allowed_task_ids = (
        _load_task_ids(Path(args.task_ids).resolve()) if args.task_ids else None
    )
    tasks, task_indices = _load_requested_tasks(
        task_path,
        allowed_task_ids=allowed_task_ids,
        limit=args.limit,
    )
    completed = (
        _successful_keys(output_path) if args.resume and output_path.exists() else set()
    )
    tasks_to_run = [
        task for task in tasks if (task.source_id, task.task_id) not in completed
    ]
    if not tasks_to_run:
        _write_manifest(
            output_path=output_path,
            task_path=task_path,
            args=args,
            requested=len(tasks),
            attempted=0,
            successful=0,
        )
        print("All requested tasks already have successful records.")
        return 0

    provider = EgorLocalModelProvider(
        args.model_name,
        backend="transformers-local",
        model_path=args.model_path,
        model_profile=args.model_profile,
        max_retries=args.max_retries,
        max_concurrency=1,
        max_gpu_mem_gb=args.max_gpu_mem_gb,
        gpu_devices=args.gpu_devices,
        allow_download=False,
        allow_cpu_offload=args.allow_cpu_offload,
    )
    if args.stage == "baseline":
        records = run_baseline_stage(
            provider=provider,
            tasks=tasks_to_run,
            seed=args.seed,
            task_indices=task_indices,
        )
    elif args.stage == "verifier":
        records = run_verifier_stage(
            provider=provider,
            tasks=tasks_to_run,
            seed=args.seed,
            task_indices=task_indices,
        )
    elif args.stage == "verifier_v6":
        records = run_verifier_v6_stage(
            provider=provider,
            tasks=tasks_to_run,
            seed=args.seed,
            task_indices=task_indices,
        )
    elif args.stage == "verifier_v7":
        records = run_verifier_v7_stage(
            provider=provider,
            tasks=tasks_to_run,
            seed=args.seed,
            task_indices=task_indices,
        )
    elif args.stage == "verifier_v8":
        records = run_verifier_v8_stage(
            provider=provider,
            tasks=tasks_to_run,
            seed=args.seed,
            task_indices=task_indices,
        )
    elif args.stage == "scheduler":
        if not args.baseline or not args.action_spec:
            raise SystemExit("scheduler requires --baseline and --action-spec")
        records = run_scheduler_stage(
            provider=provider,
            tasks=tasks_to_run,
            baseline_records=load_successful_stage_records(Path(args.baseline)),
            template_catalog=_scheduler_catalog(Path(args.action_spec).resolve()),
            seed=args.seed,
            task_indices=task_indices,
        )
    elif args.stage == "specialist":
        if not args.baseline:
            raise SystemExit("--baseline is required for specialist stage")
        specialist_type = EvidenceType(args.specialist_type)
        records = run_specialist_stage(
            provider=provider,
            tasks=tasks_to_run,
            baseline_records=load_successful_stage_records(Path(args.baseline)),
            specialist_type=specialist_type,
            specialist_id=args.specialist_id or specialist_type.value,
            seed=args.seed,
            task_indices=task_indices,
        )
    else:
        if not args.baseline or not args.candidate_id:
            raise SystemExit("synthesis requires --baseline and --candidate-id")
        records = run_synthesis_stage(
            provider=provider,
            tasks=tasks_to_run,
            baseline_records=load_successful_stage_records(Path(args.baseline)),
            specialist_record_sets=[
                load_successful_stage_records(Path(path))
                for path in args.specialist_inputs
            ],
            candidate_id=args.candidate_id,
            seed=args.seed,
            task_indices=task_indices,
        )

    mode = "a" if args.resume and output_path.exists() else "w"
    written = 0
    successes = 0
    with output_path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            written += 1
            successes += int(record.get("status") == "ok")
            print(
                f"[{args.stage}] {record.get('source_id')}::{record.get('task_id')} "
                f"status={record.get('status')}",
                flush=True,
            )
    _write_manifest(
        output_path=output_path,
        task_path=task_path,
        args=args,
        requested=len(tasks),
        attempted=written,
        successful=successes,
    )
    return 0 if successes == written else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "baseline",
            "verifier",
            "verifier_v6",
            "verifier_v7",
            "verifier_v8",
            "scheduler",
            "specialist",
            "synthesis",
        ],
        required=True,
    )
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline", default="")
    parser.add_argument(
        "--specialist-type",
        choices=[item.value for item in EvidenceType],
        default="exact_span",
    )
    parser.add_argument("--specialist-id", default="")
    parser.add_argument("--specialist-inputs", nargs="*", default=[])
    parser.add_argument("--candidate-id", default="")
    parser.add_argument("--action-spec", default="")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--model-profile", choices=["small", "medium", "large"], default="medium"
    )
    parser.add_argument("--gpu-devices", required=True)
    parser.add_argument("--max-gpu-mem-gb", type=float, default=20.0)
    parser.add_argument("--allow-cpu-offload", action="store_true")
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def _successful_keys(path: Path) -> set[tuple[str, str]]:
    return set(load_successful_stage_records(path))


def _load_task_ids(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _load_requested_tasks(
    path: Path,
    *,
    allowed_task_ids: set[str] | None,
    limit: int,
) -> tuple[list[PilotTask], dict[tuple[str, str], int]]:
    all_tasks = load_tasks(path)
    original_indices = {
        (task.source_id, task.task_id): index
        for index, task in enumerate(all_tasks)
    }
    tasks = [
        task
        for task in all_tasks
        if allowed_task_ids is None or task.task_id in allowed_task_ids
    ]
    if limit > 0:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("no tasks selected")
    return tasks, original_indices


def _write_manifest(
    *,
    output_path: Path,
    task_path: Path,
    args: argparse.Namespace,
    requested: int,
    attempted: int,
    successful: int,
) -> None:
    source_commit = os.environ.get("EGOR_SOURCE_COMMIT", "").strip() or _git_output(
        "rev-parse", "HEAD"
    )
    git_status = _git_output("status", "--porcelain")
    payload: dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "tasks": str(task_path),
        "tasks_sha256": _sha256(task_path),
        "output": str(output_path),
        "model_name": args.model_name,
        "model_path": args.model_path,
        "model_profile": args.model_profile,
        "gpu_devices": args.gpu_devices,
        "max_gpu_mem_gb": args.max_gpu_mem_gb,
        "allow_cpu_offload": bool(getattr(args, "allow_cpu_offload", False)),
        "seed": args.seed,
        "seed_schedule": "base_plus_original_requested_index",
        "resume_policy": "skip successful keys; retry all other keys append-only",
        "requested": requested,
        "attempted_this_run": attempted,
        "successful_this_run": successful,
        "successful_total": len(_successful_keys(output_path))
        if output_path.exists()
        else 0,
        "git_commit": source_commit,
        "git_worktree_dirty": bool(git_status),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "runtime": _runtime_fingerprint(Path(args.model_path)),
        "answer_leakage_policy": "generation prompts are constructed only from question, options, image, and prior stage records",
    }
    action_spec = str(getattr(args, "action_spec", "")).strip()
    if action_spec:
        action_spec_path = Path(action_spec).resolve()
        payload["action_spec"] = str(action_spec_path)
        payload["action_spec_sha256"] = _sha256(action_spec_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_output(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_fingerprint(model_path: Path) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "accelerate", "pillow", "qwen-vl-utils"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    source_files = []
    for module in (egor_local_provider_module, egor_runtime_types_module):
        path = Path(str(module.__file__)).resolve()
        source_files.append({"path": str(path), "sha256": _sha256(path)})
    model_files = []
    if model_path.is_dir():
        patterns = (
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "*.index.json",
        )
        selected = {
            path.resolve() for pattern in patterns for path in model_path.glob(pattern)
        }
        model_files = [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(selected)
        ]
        weight_layout = [
            {"name": path.name, "bytes": path.stat().st_size}
            for path in sorted(model_path.glob("*.safetensors"))
        ]
    else:
        weight_layout = []
    weight_layout_sha256 = hashlib.sha256(
        json.dumps(weight_layout, sort_keys=True).encode()
    ).hexdigest()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "source_files": source_files,
        "model_metadata_files": model_files,
        "model_weight_layout": weight_layout,
        "model_weight_layout_sha256": weight_layout_sha256,
    }


def _scheduler_catalog(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actions = payload.get("actions", {}) if isinstance(payload, dict) else {}
    templates = payload.get("templates", {}) if isinstance(payload, dict) else {}
    if not isinstance(actions, dict) or not isinstance(templates, dict):
        raise ValueError("scheduler action spec requires actions and templates objects")
    catalog: list[dict[str, Any]] = []
    for template_id, raw_template in sorted(templates.items()):
        if not isinstance(raw_template, dict) or not isinstance(
            raw_template.get("action_ids"), list
        ):
            raise ValueError(f"template {template_id} requires action_ids")
        action_ids = [str(value) for value in raw_template["action_ids"]]
        evidence_types: list[str] = []
        for action_id in action_ids:
            raw_action = actions.get(action_id)
            if not isinstance(raw_action, dict):
                raise ValueError(
                    f"template {template_id} references unknown action {action_id}"
                )
            evidence_types.append(str(raw_action.get("evidence_type", "")))
        catalog.append(
            {
                "template_id": str(template_id),
                "action_ids": action_ids,
                "evidence_types": evidence_types,
            }
        )
    return catalog


if __name__ == "__main__":
    raise SystemExit(main())
