"""Read-only environment and resource diagnostics, before engine construction."""

from __future__ import annotations

import importlib.util
import json
import os
from importlib.resources import files
from pathlib import Path
from typing import Any

from openagora.config.models import RunConfig
from openagora.core.errors import ExternalDependencyError
from openagora.resources import ResourcePaths, resource_report
from openagora.templates.runtime import build_template_runtime


def environment_report(paths: ResourcePaths, cfg: RunConfig | None = None) -> dict[str, Any]:
    catalog = json.loads(files("openagora").joinpath("resource_catalog.json").read_text())
    report: dict[str, Any] = {
        "status": "ok",
        "paths": paths.as_strings(),
        "resources": resource_report(catalog, paths),
        "problems": [],
    }
    if cfg is None:
        return report
    runtime = build_template_runtime(cfg)
    report.update(
        config_path=cfg.config_path,
        provider_mode=cfg.provider.provider_mode,
        provider_name=cfg.provider.name,
        template_id=runtime.template_id,
        template_version=runtime.version,
        db_path=cfg.paths.db_path,
        persona_count=len(runtime.assign_personas(cfg=cfg, worker_count=cfg.runtime.worker_count)),
    )
    problems = provider_problems(cfg)
    if cfg.provenance.dependency_lock_path and not Path(cfg.provenance.dependency_lock_path).is_file():
        problems.append(f"Dependency lock missing: {cfg.provenance.dependency_lock_path}")
    report["problems"] = problems
    report["status"] = "missing_requirements" if problems else "ok"
    return report


def provider_problems(cfg: RunConfig) -> list[str]:
    provider = cfg.provider
    problems = []
    if provider.provider_mode == "local" and provider.local_backend == "transformers-local":
        for name in ("torch", "transformers"):
            if importlib.util.find_spec(name) is None:
                problems.append(f"Optional dependency {name} missing; run uv sync --group inference")
        path = Path(provider.local_model_path) if provider.local_model_path else None
        if not provider.local_allow_download and (path is None or not path.is_dir()):
            problems.append(
                "Local model directory missing; set provider.local_model_path to {models}/<model> "
                "after restoring the pinned model listed in resource_catalog.json"
            )
    if provider.provider_mode == "live" and provider.name.lower() == "openai":
        if importlib.util.find_spec("openai") is None:
            problems.append("OpenAI client missing; run uv sync --group services")
        if not os.environ.get("OPENAI_API_KEY"):
            problems.append("OPENAI_API_KEY is unset; configure it locally before a live run")
    return problems


def require_run_resources(cfg: RunConfig, *, evaluation: bool = False) -> None:
    problems = provider_problems(cfg)
    if evaluation and not Path(cfg.eval.dataset_path).is_file():
        problems.append(f"Dataset missing: {cfg.eval.dataset_path}; set eval.dataset_path in --config")
    if problems:
        raise ExternalDependencyError("; ".join(problems), code="resource_preflight_failed", retryable=False)


def require_catalog_resources(report: dict[str, Any], names: list[str]) -> None:
    resources = {row["id"]: row for row in report["resources"]}
    for name in names:
        resource = resources.get(name)
        if resource is None:
            report["problems"].append(f"Unknown resource ID: {name}")
        elif resource["status"] not in {"ready", "present_unverified"}:
            report["problems"].append(f"{name}: {resource['status']}; {resource['how_to_get']}")
