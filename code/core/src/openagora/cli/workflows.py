"""CLI dispatch boundary between the core runtime and optional experiment jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from openagora.resources import ResourcePaths


def experiment_payload(config: str) -> dict[str, Any] | None:
    payload = yaml.safe_load(Path(config).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a YAML mapping")
    workflow = payload.get("workflow")
    if workflow is None:
        return None
    if workflow != "egor-stage":
        raise ValueError(f"Unsupported workflow {workflow!r}; supported: egor-stage (omit for runtime)")
    return payload


def run_experiment(payload: dict[str, Any], config: str, paths: ResourcePaths, *, check_only: bool) -> dict[str, Any]:
    from openagora_experiments.egor.job import execute_stage, resolve_stage, stage_problems

    cfg = resolve_stage(payload, Path(config).resolve(), paths)
    problems = stage_problems(cfg)
    if check_only:
        return {
            "status": "missing_requirements" if problems else "ok",
            "paths": paths.as_strings(),
            "workflow": cfg.workflow,
            "problems": problems,
        }
    if problems:
        from openagora.core.errors import ExternalDependencyError

        raise ExternalDependencyError("; ".join(problems), code="resource_preflight_failed", retryable=False)
    result: dict[str, Any] = execute_stage(cfg)
    return result
