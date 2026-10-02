"""Stable protocol interfaces."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from openagora.config.models import RunConfig
from openagora.types import (
    ArchitectureSpec,
    CacheEntry,
    RunEvent,
    TaskSignature,
    TaskSpec,
)


@runtime_checkable
class ModelProvider(Protocol):
    """Abstraction over LLM providers."""

    model_name: str

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        seed: int,
        request_id: str | None = None,
    ) -> str:
        """Return model completion text."""


@runtime_checkable
class ToolAdapter(Protocol):
    """Abstraction over external tools."""

    def execute(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute one tool call and return a structured response."""


@runtime_checkable
class TaskAdapter(Protocol):
    """Task-domain adapter for analysis, decomposition and execution."""

    def analyze_task(self, task: TaskSpec) -> TaskSignature:
        """Estimate task signature."""

    def decompose_task(self, task: TaskSpec, architecture: ArchitectureSpec) -> list[dict[str, Any]]:
        """Return subtasks for the current step policy."""

    def execute_subtask(
        self,
        *,
        task: TaskSpec,
        subtask: dict[str, Any],
        architecture: ArchitectureSpec,
        tools: ToolAdapter,
        seed: int,
    ) -> dict[str, Any]:
        """Execute one subtask."""

    def evaluate_result(
        self,
        *,
        task: TaskSpec,
        step_results: list[dict[str, Any]],
        architecture: ArchitectureSpec,
    ) -> dict[str, Any]:
        """Evaluate final task outcome."""


@runtime_checkable
class CacheStore(Protocol):
    """Cache repository abstraction."""

    def lookup(self, signature: TaskSignature, threshold: float, type_weight: float) -> CacheEntry | None:
        """Find reusable cache entry."""

    def upsert(
        self,
        *,
        signature: TaskSignature,
        architecture: ArchitectureSpec,
        performance: float,
        ot_alignment: float,
        source: str,
    ) -> CacheEntry:
        """Write a new cache version or update current one."""

    def mark_needs_revalidation(self, task_type: str) -> None:
        """Mark current version as needing revalidation."""

    def rollback_if_needed(self, task_type: str) -> None:
        """Rollback current version if previous performs better."""

    def get_current(self, task_type: str) -> CacheEntry | None:
        """Get current version."""


@runtime_checkable
class Observer(Protocol):
    """Runtime observer for structured events."""

    def on_event(self, event: RunEvent) -> None:
        """Handle one event."""


@runtime_checkable
class Engine(Protocol):
    """Public engine interface."""

    def run(self, task: TaskSpec, run_config: RunConfig) -> Any:
        """Run one task."""

    def replay(self, run_id: str) -> Any:
        """Replay one completed run from persisted artifact."""
