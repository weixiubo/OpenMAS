"""YAML config loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from openagora.config.models import RunConfig
from openagora.resources import ResourcePaths, resolve_paths


def load_run_config(path: str, *, resources: ResourcePaths | None = None) -> RunConfig:
    """Load and validate a runtime config from YAML."""
    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    if not isinstance(payload, dict):
        raise ValueError("Top-level YAML must be a mapping")
    payload["config_path"] = str(cfg_path)
    cfg = RunConfig.model_validate(payload)
    return resolve_config_paths(cfg, resources or resolve_paths())


def resolve_config_paths(cfg: RunConfig, resources: ResourcePaths) -> RunConfig:
    """Explicit YAML paths are config-relative; unspecified outputs use research roots."""
    base = Path(cfg.config_path).parent if cfg.config_path else resources.data
    clone = cfg.model_copy(deep=True)
    for field, default in {"db_path": "{runs}/openagora.db", "artifacts_dir": "{runs}"}.items():
        value = getattr(cfg.paths, field) if field in cfg.paths.model_fields_set else default
        setattr(clone.paths, field, str(resources.expand(value, base=base)))
    for field in type(cfg.eval).model_fields:
        if not (field.endswith("_path") or field.startswith("output_")):
            continue
        value = getattr(cfg.eval, field)
        if field not in cfg.eval.model_fields_set:
            root = "runs" if field.startswith("output_") else "data"
            value = "{" + root + "}/" + Path(value).name
        setattr(clone.eval, field, str(resources.expand(value, base=base)))
    if cfg.templates.registry_path:
        clone.templates.registry_path = str(resources.expand(cfg.templates.registry_path, base=base))
    if cfg.provider.local_model_path:
        clone.provider.local_model_path = str(resources.expand(cfg.provider.local_model_path, base=base))
    if cfg.provenance.dependency_lock_path:
        clone.provenance.dependency_lock_path = str(resources.expand(cfg.provenance.dependency_lock_path, base=base))
    return clone


def dump_config(cfg: RunConfig) -> dict[str, Any]:
    """Return config payload for hashing and manifesting."""
    return cfg.model_dump(mode="json", exclude={"config_path"})
