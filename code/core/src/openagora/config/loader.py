"""YAML config loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from openagora.config.models import RunConfig


def load_run_config(path: str) -> RunConfig:
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
    return cfg.with_paths_resolved()


def dump_config(cfg: RunConfig) -> dict[str, Any]:
    """Return config payload for hashing and manifesting."""
    return cfg.model_dump(mode="json", exclude={"config_path"})
