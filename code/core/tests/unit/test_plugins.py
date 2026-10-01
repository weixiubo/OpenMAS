from __future__ import annotations

import pytest

from openagora.config.models import RunConfig
from openagora.core.errors import ConfigurationError
from openagora.plugins.registry import build_default_registry


def _config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "provider": {"name": "mock", "provider_mode": "mock"},
            "task": {
                "task_id": "plugin-test",
                "prompt": "Use tools to answer: what is 2 + 3?",
                "metadata": {"reference_answer": "5"},
            },
        }
    )


def test_default_registry_builds_builtin_components() -> None:
    cfg = _config()
    registry = build_default_registry()
    provider = registry.create_provider(cfg)
    task_adapter = registry.create_task_adapter(cfg)
    tool_adapter = registry.create_tool_adapter(cfg)

    assert provider.model_name == "mock-leader-v1"
    assert task_adapter.__class__.__name__ == "QAToolTaskAdapter"
    assert tool_adapter.__class__.__name__ == "LocalToolAdapter"


def test_registry_discovery_invalid_module() -> None:
    registry = build_default_registry()
    with pytest.raises(ConfigurationError):
        registry.discover(["openagora.non_existing_plugin_module"])


def test_registry_creates_local_provider_when_mode_local() -> None:
    cfg = RunConfig.model_validate(
        {
            "provider": {
                "name": "local",
                "model": "sshleifer/tiny-gpt2",
                "provider_mode": "local",
                "local_backend": "vllm-local",
                "local_endpoint": "http://127.0.0.1:8000/v1",
            },
            "task": {
                "task_id": "local-provider-test",
                "prompt": "answer",
                "metadata": {"reference_answer": "answer"},
            },
        }
    )
    registry = build_default_registry()
    provider = registry.create_provider(cfg)
    assert provider.__class__.__name__ == "LocalModelProvider"
