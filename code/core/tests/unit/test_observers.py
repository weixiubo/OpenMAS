from __future__ import annotations

from openagora.core.observers import RedactingObserver
from openagora.types import RunEvent


class _Collector:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def on_event(self, event: RunEvent) -> None:
        self.events.append(event)


def test_redacting_observer_masks_sensitive_and_truncates() -> None:
    sink = _Collector()
    observer = RedactingObserver(sink, max_prompt_chars=8)
    observer.on_event(
        RunEvent(
            event_type="x",
            payload={
                "api_key": "secret",
                "user_prompt": "abcdefghijklmnopqrstuvwxyz",
                "normal": "ok",
            },
        )
    )
    payload = sink.events[0].payload
    assert payload["api_key"] == "***REDACTED***"
    assert payload["user_prompt"].endswith("<truncated>")
    assert payload["normal"] == "ok"
