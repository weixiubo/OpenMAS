"""Reproducibility helpers."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from openagora.config.loader import dump_config
from openagora.config.models import RunConfig
from openagora.types import RunManifest
from openagora.utils.hash_utils import sha256_file, sha256_json


def safe_git_commit() -> str:
    """Return current git commit hash, or 'unknown'."""
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def resolve_dependency_lock_hash(candidates: Iterable[str]) -> str:
    """Hash first existing lock candidate."""
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_file():
            return sha256_file(str(path))
    return "unknown"


def build_manifest(
    run_id: str,
    cfg: RunConfig,
    lock_candidates: Iterable[str],
    *,
    event_schema_version: str,
) -> RunManifest:
    """Build run manifest."""
    payload_hash = sha256_json(dump_config(cfg))
    return RunManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=safe_git_commit(),
        dependency_lock_hash=resolve_dependency_lock_hash(lock_candidates),
        config_hash=payload_hash,
        seed=cfg.provider.seed,
        provider_name=cfg.provider.name,
        provider_mode=cfg.provider.provider_mode,
        model_name=cfg.provider.model,
        temperature=cfg.provider.temperature,
        data_version=cfg.runtime.data_version,
        template_id=cfg.templates.template_id,
        template_version=cfg.templates.version,
        event_schema_version=event_schema_version,
    )


def write_manifest(manifest: RunManifest, path: str) -> None:
    """Persist manifest JSON to file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest.model_dump(mode="json"), fp, ensure_ascii=False, indent=2)
