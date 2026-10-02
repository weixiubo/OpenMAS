"""Audit integrity checks for run events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openagora.core.errors import ConfigurationError

REQUIRED_EVENT_KEYS = {"event_type", "payload", "schema_version", "ts"}


def load_events(path: Path) -> list[dict[str, Any]]:
    """Load event jsonl entries."""
    if not path.exists():
        raise ConfigurationError(f"Events file not found: {path}", code="events_missing")
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        events.append(json.loads(text))
    return events


def assert_event_integrity(*, events_path: Path, schema_version: str) -> dict[str, int]:
    """Validate event schema consistency and required fields."""
    events = load_events(events_path)
    if not events:
        raise ConfigurationError("No events found for run", code="events_empty")

    redacted_hits = 0
    prompt_truncations = 0
    for event in events:
        missing = REQUIRED_EVENT_KEYS - set(event.keys())
        if missing:
            raise ConfigurationError(
                f"Event missing keys {sorted(missing)} in {events_path}",
                code="event_schema_invalid",
            )
        if str(event.get("schema_version")) != schema_version:
            raise ConfigurationError(
                f"Event schema mismatch: expected {schema_version}",
                code="event_schema_mismatch",
            )

        payload = event.get("payload")
        if isinstance(payload, dict):
            redacted_hits += _count_redacted(payload)
            prompt_truncations += _count_truncated(payload)

    return {
        "event_count": len(events),
        "redacted_hits": redacted_hits,
        "prompt_truncations": prompt_truncations,
    }


def _count_redacted(payload: dict[str, Any]) -> int:
    hits = 0
    for value in payload.values():
        if value == "***REDACTED***":
            hits += 1
        elif isinstance(value, dict):
            hits += _count_redacted(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    hits += _count_redacted(item)
    return hits


def _count_truncated(payload: dict[str, Any]) -> int:
    hits = 0
    for value in payload.values():
        if isinstance(value, str) and value.endswith("<truncated>"):
            hits += 1
        elif isinstance(value, dict):
            hits += _count_truncated(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    hits += _count_truncated(item)
    return hits
