"""Plugin registry and discovery for providers/adapters."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable

from openagora.adapters.qa_tools import LocalToolAdapter, QAToolTaskAdapter
from openagora.config.models import RunConfig
from openagora.core.errors import ConfigurationError
from openagora.core.interfaces import ModelProvider, TaskAdapter, ToolAdapter
from openagora.providers.local_provider import LocalModelProvider
from openagora.providers.mock import MockModelProvider
from openagora.providers.openai_provider import OpenAIModelProvider

ProviderFactory = Callable[[RunConfig], ModelProvider]
TaskAdapterFactory = Callable[[RunConfig], TaskAdapter]
ToolAdapterFactory = Callable[[RunConfig], ToolAdapter]


@dataclass
class PluginRegistry:
    """Factory registry for pluggable runtime components."""

    provider_factories: dict[str, ProviderFactory] = field(default_factory=dict)
    task_adapter_factories: dict[str, TaskAdapterFactory] = field(default_factory=dict)
    tool_adapter_factories: dict[str, ToolAdapterFactory] = field(default_factory=dict)

    def register_provider(self, name: str, factory: ProviderFactory) -> None:
        self.provider_factories[name.lower()] = factory

    def register_task_adapter(self, name: str, factory: TaskAdapterFactory) -> None:
        self.task_adapter_factories[name.lower()] = factory

    def register_tool_adapter(self, name: str, factory: ToolAdapterFactory) -> None:
        self.tool_adapter_factories[name.lower()] = factory

    def discover(self, modules: list[str]) -> None:
        for module_name in modules:
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - env dependent
                raise ConfigurationError(
                    f"Failed loading plugin module `{module_name}`: {exc}",
                    code="plugin_discovery_error",
                ) from exc

    def create_provider(self, cfg: RunConfig) -> ModelProvider:
        key = cfg.provider.name.lower()
        if cfg.provider.provider_mode == "mock":
            key = "mock"
        elif cfg.provider.provider_mode == "local":
            key = "local"
        factory = self.provider_factories.get(key)
        if factory is None:
            raise ConfigurationError(f"Unknown provider plugin: {key}", code="unknown_provider_plugin")
        return factory(cfg)

    def create_task_adapter(self, cfg: RunConfig) -> TaskAdapter:
        key = cfg.plugins.task_adapter.lower()
        factory = self.task_adapter_factories.get(key)
        if factory is None:
            raise ConfigurationError(f"Unknown task adapter plugin: {key}", code="unknown_task_adapter_plugin")
        return factory(cfg)

    def create_tool_adapter(self, cfg: RunConfig) -> ToolAdapter:
        key = cfg.plugins.tool_adapter.lower()
        factory = self.tool_adapter_factories.get(key)
        if factory is None:
            raise ConfigurationError(f"Unknown tool adapter plugin: {key}", code="unknown_tool_adapter_plugin")
        return factory(cfg)


def build_default_registry() -> PluginRegistry:
    """Construct registry with built-in plugins."""
    registry = PluginRegistry()
    registry.register_provider("mock", _build_mock_provider)
    registry.register_provider("openai", _build_openai_provider)
    registry.register_provider("local", _build_local_provider)
    registry.register_task_adapter("qa_tools", _build_qa_task_adapter)
    registry.register_tool_adapter("local_tools", _build_local_tool_adapter)
    return registry


def _build_mock_provider(cfg: RunConfig) -> ModelProvider:
    return MockModelProvider(cfg.provider.model)


def _build_openai_provider(cfg: RunConfig) -> ModelProvider:
    return OpenAIModelProvider(
        cfg.provider.model,
        timeout_s=cfg.provider.timeout_s,
        max_retries=cfg.provider.max_retries,
        retry_backoff_base=cfg.provider.retry_backoff_base,
        max_concurrency=cfg.provider.max_concurrency,
        enable_idempotency_key=cfg.provider.enable_idempotency_key,
    )


def _build_local_provider(cfg: RunConfig) -> ModelProvider:
    return LocalModelProvider(
        cfg.provider.model,
        backend=cfg.provider.local_backend,
        model_path=cfg.provider.local_model_path,
        model_profile=cfg.provider.local_model_profile,
        endpoint=cfg.provider.local_endpoint,
        timeout_s=cfg.provider.timeout_s,
        max_retries=cfg.provider.max_retries,
        retry_backoff_base=cfg.provider.retry_backoff_base,
        max_concurrency=cfg.provider.max_concurrency,
        max_gpu_mem_gb=cfg.provider.max_gpu_mem_gb,
        gpu_devices=cfg.provider.gpu_devices,
        allow_download=cfg.provider.local_allow_download,
    )


def _build_qa_task_adapter(cfg: RunConfig) -> TaskAdapter:
    del cfg
    return QAToolTaskAdapter()


def _build_local_tool_adapter(cfg: RunConfig) -> ToolAdapter:
    del cfg
    return LocalToolAdapter()
