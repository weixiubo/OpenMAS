"""Configured EGOR stage execution with explicit inputs and resumable records."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import uuid4

from openagora.mechanisms.evidence_gap_repair import EvidenceType
from openagora.resources import ResourcePaths, sha256_file
from pydantic import BaseModel, ConfigDict, Field, model_validator

from openagora_experiments.egor.catalog import load_scheduler_catalog
from openagora_experiments.egor.local_provider import EgorLocalModelProvider
from openagora_experiments.egor.runner import (
    MultimodalProvider,
    load_successful_stage_records,
    load_tasks,
    run_baseline_stage,
    run_scheduler_stage,
    run_specialist_stage,
    run_synthesis_stage,
    run_verifier_v8_stage,
)
from openagora_experiments.egor.task_model import PilotTask


class StageConfig(BaseModel):
    """A stage references resource paths, never neighbouring repositories."""

    model_config = ConfigDict(extra="forbid")
    workflow: Literal["egor-stage"] = "egor-stage"
    stage: Literal["baseline", "verifier", "scheduler", "specialist", "synthesis"]
    tasks: str
    model_path: str
    model_name: str
    model_revision: str
    output: str | None = None
    run_id: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
    )
    seed: int = 20260711
    limit: int = Field(default=0, ge=0)
    resume: bool = False
    baseline: str | None = None
    specialist_inputs: list[str] = Field(default_factory=list)
    specialist_type: EvidenceType = EvidenceType.EXACT_SPAN
    specialist_id: str = ""
    candidate_id: str = ""
    action_spec: str | None = None
    gpu_devices: str = "cpu"
    max_gpu_mem_gb: float = Field(default=20.0, gt=0)
    allow_cpu_offload: bool = False
    max_retries: int = Field(default=0, ge=0)
    input_sha256: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_stage(self) -> StageConfig:
        if self.stage in {"scheduler", "specialist", "synthesis"} and not self.baseline:
            raise ValueError(f"{self.stage} requires baseline records")
        if self.stage == "scheduler" and not self.action_spec:
            raise ValueError("scheduler requires action_spec")
        if self.stage == "synthesis" and not self.candidate_id:
            raise ValueError("synthesis requires candidate_id")
        if not self.run_id or Path(self.run_id).name != self.run_id or self.run_id in {".", ".."}:
            raise ValueError("run_id must be a single directory name")
        return self


def resolve_stage(payload: dict[str, Any], config_path: Path, paths: ResourcePaths) -> StageConfig:
    cfg = StageConfig.model_validate(payload)
    for name in ("tasks", "model_path", "output", "baseline", "action_spec"):
        value = getattr(cfg, name)
        if value:
            setattr(cfg, name, str(paths.expand(value, base=config_path.parent)))
    cfg.specialist_inputs = [str(paths.expand(value, base=config_path.parent)) for value in cfg.specialist_inputs]
    if cfg.output is None:
        cfg.output = str(paths.runs / cfg.run_id / (cfg.stage + ".jsonl"))
    cfg.input_sha256 = {
        str(paths.expand(path, base=config_path.parent)): digest for path, digest in cfg.input_sha256.items()
    }
    return cfg


def stage_problems(cfg: StageConfig, *, check_packages: bool = True) -> list[str]:
    problems = []
    for path in (cfg.tasks, cfg.baseline, cfg.action_spec, *cfg.specialist_inputs):
        if path and not Path(path).is_file():
            problems.append(f"Input missing: {path}; restore it and set the corresponding path in --config")
    if not Path(cfg.model_path).is_dir():
        problems.append(f"Model directory missing: {cfg.model_path}; restore revision {cfg.model_revision}")
    if check_packages:
        for package in ("torch", "transformers", "PIL"):
            if importlib.util.find_spec(package) is None:
                problems.append(f"Optional dependency {package} missing; uv sync --group inference")
    for path, expected in cfg.input_sha256.items():
        if not Path(path).is_file() or sha256_file(Path(path)) != expected:
            problems.append(f"Input checksum mismatch or missing file: {path}; restore the exact manifest version")
    return problems


class StageArguments(TypedDict):
    provider: MultimodalProvider
    tasks: list[PilotTask]
    seed: int
    task_indices: dict[tuple[str, str], int]


def stage_records(
    cfg: StageConfig, provider: MultimodalProvider, tasks: list[PilotTask], indices: dict[tuple[str, str], int]
) -> Iterable[dict[str, Any]]:
    common: StageArguments = {"provider": provider, "tasks": tasks, "seed": cfg.seed, "task_indices": indices}
    if cfg.stage == "baseline":
        return run_baseline_stage(**common)
    if cfg.stage == "verifier":
        return run_verifier_v8_stage(**common)
    baseline = load_successful_stage_records(Path(str(cfg.baseline)))
    if cfg.stage == "specialist":
        return run_specialist_stage(
            **common,
            baseline_records=baseline,
            specialist_type=cfg.specialist_type,
            specialist_id=cfg.specialist_id or cfg.specialist_type.value,
        )
    if cfg.stage == "scheduler":
        catalog = load_scheduler_catalog(Path(str(cfg.action_spec)))
        return run_scheduler_stage(**common, baseline_records=baseline, template_catalog=catalog)
    return run_synthesis_stage(
        **common,
        baseline_records=baseline,
        specialist_record_sets=[load_successful_stage_records(Path(p)) for p in cfg.specialist_inputs],
        candidate_id=cfg.candidate_id,
    )


def execute_stage(cfg: StageConfig, *, provider: MultimodalProvider | None = None) -> dict[str, Any]:
    """A supplied provider enables offline contract tests; real runs preflight first."""
    problems = stage_problems(cfg, check_packages=provider is None)
    if problems:
        raise ValueError("; ".join(problems))
    all_tasks = load_tasks(Path(cfg.tasks))
    indices = {(task.source_id, task.task_id): index for index, task in enumerate(all_tasks)}
    tasks = all_tasks[: cfg.limit] if cfg.limit else all_tasks
    output = Path(str(cfg.output))
    binding = _run_binding(cfg)
    _check_resume(output, cfg.resume, binding)
    completed = set(load_successful_stage_records(output)) if cfg.resume and output.exists() else set()
    pending = [task for task in tasks if (task.source_id, task.task_id) not in completed]
    for task in pending:
        image = Path(task.image_path)
        if not image.is_file():
            raise ValueError(f"Task image missing: {image}; restore the images referenced by {cfg.tasks}")
    if not pending:
        return {"requested": len(tasks), "attempted": 0, "successful": 0, "output": str(output)}
    if provider is None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        provider = EgorLocalModelProvider(
            cfg.model_name,
            model_path=cfg.model_path,
            max_retries=cfg.max_retries,
            max_concurrency=1,
            max_gpu_mem_gb=cfg.max_gpu_mem_gb,
            gpu_devices=cfg.gpu_devices,
            allow_download=False,
            allow_cpu_offload=cfg.allow_cpu_offload,
        )
    records = stage_records(cfg, provider, pending, indices)
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_manifest(output, {"status": "running", "binding": binding})
    try:
        counts = _write_records(output, records, resume=cfg.resume)
    except BaseException:
        _save_manifest(output, {"status": "interrupted", "binding": binding})
        raise
    manifest = {
        "schema_version": 1,
        "source_identity": "openmas-configured-stages-v1",
        "config": cfg.model_dump(mode="json"),
        "task_sha256": sha256_file(Path(cfg.tasks)),
        "requested": len(tasks),
        **counts,
        "output": str(output),
    }
    manifest.update(status="completed", binding=binding)
    _save_manifest(output, manifest)
    return manifest


def _write_records(output: Path, records: Iterable[dict[str, Any]], *, resume: bool) -> dict[str, int]:
    attempted = successful = 0
    with output.open("a" if resume else "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            attempted += 1
            successful += int(record.get("status") == "ok")
    return {"attempted": attempted, "successful": successful}


def _run_binding(cfg: StageConfig) -> dict[str, Any]:
    source = Path(__file__).parent
    source_hashes = {
        name: sha256_file(source / name)
        for name in (
            "job.py",
            "runner.py",
            "prompts.py",
            "parsing.py",
            "task_model.py",
            "values.py",
            "catalog.py",
            "local_provider.py",
            "model_inputs.py",
            "provider_errors.py",
        )
    }
    inputs = {
        path: sha256_file(Path(path))
        for path in (cfg.tasks, cfg.baseline, cfg.action_spec, *cfg.specialist_inputs)
        if path
    }
    config = cfg.model_dump(mode="json", exclude={"resume", "limit", "output", "run_id"})
    return {
        "config": config,
        "inputs": inputs,
        "source_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
    }


def _check_resume(output: Path, resume: bool, binding: dict[str, Any]) -> None:
    if not output.exists():
        return
    if not resume:
        raise FileExistsError(f"Output exists: {output}; use a new run_id or enable resume")
    manifest = output.with_suffix(".manifest.json")
    if not manifest.is_file():
        raise ValueError(f"Resume manifest missing: {manifest}; cannot verify the existing records")
    if json.loads(manifest.read_text()).get("binding") != binding:
        raise ValueError("Resume input/config/source mismatch; start a new run instead of mixing evidence")


def _save_manifest(output: Path, manifest: dict[str, Any]) -> None:
    target = output.with_suffix(".manifest.json")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(target)
