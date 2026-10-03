"""Template bundle loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from openagora.templates.models import TemplateBundle


def bundled_template_path() -> Path:
    """Return default template bundle path shipped with the package."""
    return Path(__file__).resolve().parent / "default_templates.yaml"


def load_template_bundle(
    *,
    registry_path: str | None,
    template_id: str,
    version: str,
) -> TemplateBundle:
    """Load and validate one versioned template bundle."""
    source_path = Path(registry_path) if registry_path else bundled_template_path()
    if not source_path.exists():
        raise FileNotFoundError(f"Template registry not found: {source_path}")
    with source_path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    if not isinstance(payload, dict):
        raise ValueError("Template registry YAML must be a mapping")
    bundle = _extract_bundle(payload, template_id=template_id, version=version)
    bundle.validate_required_templates()
    return bundle


def _extract_bundle(payload: dict[str, Any], *, template_id: str, version: str) -> TemplateBundle:
    # Supports single-bundle and multi-bundle registry formats.
    if "bundles" in payload:
        bundles = payload.get("bundles")
        if not isinstance(bundles, list):
            raise ValueError("Template registry bundles must be a list")
        for candidate in bundles:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("template_id") == template_id and candidate.get("version") == version:
                return TemplateBundle.model_validate(candidate)
        raise ValueError(f"Template bundle not found: template_id={template_id}, version={version}")
    bundle = TemplateBundle.model_validate(payload)
    if bundle.template_id != template_id or bundle.version != version:
        raise ValueError(
            f"Template bundle mismatch. expected {template_id}:{version}, got {bundle.template_id}:{bundle.version}"
        )
    return bundle
