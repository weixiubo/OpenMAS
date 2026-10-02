"""Provider audit helpers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from openagora.core.interfaces import ModelProvider, Observer
from openagora.types import RunEvent


@runtime_checkable
class _AuditableProvider(Protocol):
    def pop_audit_events(self) -> list[dict[str, Any]]:
        """Return and clear buffered provider audit events."""


def emit_provider_audit(provider: ModelProvider, observer: Observer, *, stage: str) -> None:
    """Flush provider internal audit entries into runtime event stream."""
    if not isinstance(provider, _AuditableProvider):
        return
    for entry in provider.pop_audit_events():
        payload: dict[str, Any] = {"stage": stage}
        payload.update(entry)
        observer.on_event(RunEvent(event_type="provider_call_audit", payload=payload))
