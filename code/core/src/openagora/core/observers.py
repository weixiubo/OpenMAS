"""Observer implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from openagora.core.interfaces import Observer
from openagora.storage.sqlite_store import SQLiteStore
from openagora.types import RunEvent


class CompositeObserver(Observer):
    """Dispatch events to a set of observers."""

    def __init__(self, observers: Iterable[Observer]) -> None:
        self._observers = list(observers)

    def on_event(self, event: RunEvent) -> None:
        for observer in self._observers:
            observer.on_event(event)


class JsonlObserver(Observer):
    """Append events to a JSONL file."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def on_event(self, event: RunEvent) -> None:
        with self._path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")


class StorageObserver(Observer):
    """Persist events into SQLite store."""

    def __init__(self, store: SQLiteStore, run_id: str) -> None:
        self._store = store
        self._run_id = run_id

    def on_event(self, event: RunEvent) -> None:
        self._store.save_event(self._run_id, event)


class RedactingObserver(Observer):
    """Redact sensitive data and truncate long prompts before downstream persistence."""

    _REDACT_KEYS = {"api_key", "authorization", "token", "secret", "password"}

    def __init__(self, downstream: Observer, *, max_prompt_chars: int = 240) -> None:
        self._downstream = downstream
        self._max_prompt_chars = max_prompt_chars

    def on_event(self, event: RunEvent) -> None:
        payload = _sanitize_payload(event.payload, self._max_prompt_chars)
        self._downstream.on_event(event.model_copy(update={"payload": payload}))


def _sanitize_payload(payload: dict[str, Any], max_prompt_chars: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        lowered = key.lower()
        if any(flag in lowered for flag in RedactingObserver._REDACT_KEYS):
            out[key] = "***REDACTED***"
            continue
        if isinstance(value, str) and "prompt" in lowered and len(value) > max_prompt_chars:
            out[key] = value[:max_prompt_chars] + "...<truncated>"
            continue
        out[key] = _sanitize_value(value, max_prompt_chars)
    return out


def _sanitize_value(value: Any, max_prompt_chars: int) -> Any:
    if isinstance(value, dict):
        return _sanitize_payload(value, max_prompt_chars)
    if isinstance(value, list):
        return [_sanitize_value(item, max_prompt_chars) for item in value]
    if isinstance(value, str) and len(value) > max_prompt_chars * 4:
        return value[: max_prompt_chars * 4] + "...<truncated>"
    return value
