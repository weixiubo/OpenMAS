"""Template rendering runtime with persona assignment."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Mapping

from openagora.config.models import RunConfig
from openagora.core.interfaces import Observer
from openagora.templates.loader import load_template_bundle
from openagora.templates.models import PromptTemplate, TemplateBundle
from openagora.types import RunEvent


@dataclass(frozen=True)
class RenderedPrompt:
    template_key: str
    system_prompt: str
    user_prompt: str
    template_id: str
    template_version: str


class TemplateRuntime:
    """Load templates, validate variables, render prompts and assign personas."""

    def __init__(self, bundle: TemplateBundle, observer: Observer | None = None) -> None:
        self._bundle = bundle
        self._observer = observer

    @property
    def template_id(self) -> str:
        return self._bundle.template_id

    @property
    def version(self) -> str:
        return self._bundle.version

    def set_observer(self, observer: Observer) -> None:
        self._observer = observer

    def assign_personas(self, *, cfg: RunConfig, worker_count: int) -> dict[int, str]:
        defaults = self._bundle.persona_defaults
        fixed = cfg.templates.fixed_personas or defaults.fixed_personas
        pool = cfg.templates.persona_pool or defaults.persona_pool or fixed
        if not pool:
            pool = ["generalist_worker"]
        assignments: dict[int, str] = {}
        if cfg.templates.persona_strategy == "fixed":
            if not fixed:
                fixed = pool
            for worker_id in range(1, worker_count + 1):
                assignments[worker_id] = fixed[(worker_id - 1) % len(fixed)]
            return assignments
        rng = random.Random(cfg.provider.seed)
        for worker_id in range(1, worker_count + 1):
            assignments[worker_id] = rng.choice(pool)
        return assignments

    def render(self, template_key: str, variables: Mapping[str, Any]) -> RenderedPrompt:
        template = self._bundle.templates.get(template_key)
        if template is None:
            raise KeyError(f"Template key not found: {template_key}")
        rendered = _render_template(template_key, template, variables, self._bundle)
        if self._observer is not None:
            self._observer.on_event(
                RunEvent(
                    event_type="template_rendered",
                    payload={
                        "template_id": self._bundle.template_id,
                        "template_version": self._bundle.version,
                        "template_key": template_key,
                        "variables": sorted(str(key) for key in variables.keys()),
                        "system_prompt_preview": rendered.system_prompt[:200],
                        "user_prompt_preview": rendered.user_prompt[:200],
                    },
                )
            )
        return rendered


def build_template_runtime(cfg: RunConfig, observer: Observer | None = None) -> TemplateRuntime:
    """Build runtime from config."""
    bundle = load_template_bundle(
        registry_path=cfg.templates.registry_path,
        template_id=cfg.templates.template_id,
        version=cfg.templates.version,
    )
    return TemplateRuntime(bundle, observer=observer)


def render_json(value: Any) -> str:
    """Serialize any value for template variables."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _render_template(
    template_key: str,
    template: PromptTemplate,
    variables: Mapping[str, Any],
    bundle: TemplateBundle,
) -> RenderedPrompt:
    missing = [name for name in template.required_variables if name not in variables]
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Template `{template_key}` missing required variables: {names}")
    normalized = {key: _normalize_value(value) for key, value in variables.items()}
    try:
        system_prompt = template.system_prompt.format_map(normalized)
        user_prompt = template.user_prompt.format_map(normalized)
    except KeyError as exc:
        raise ValueError(f"Template `{template_key}` unresolved variable: {exc.args[0]}") from exc
    return RenderedPrompt(
        template_key=template_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        template_id=bundle.template_id,
        template_version=bundle.version,
    )


def _normalize_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    return render_json(value)
