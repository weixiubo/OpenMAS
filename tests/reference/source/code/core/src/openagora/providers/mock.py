"""Deterministic mock provider."""

from __future__ import annotations

from typing import Any

from openagora.utils.hash_utils import sha256_text


class MockModelProvider:
    """Simple deterministic text provider for tests and local runs."""

    def __init__(self, model_name: str = "mock-leader-v1") -> None:
        self.model_name = model_name
        self._audit_buffer: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        seed: int,
        request_id: str | None = None,
    ) -> str:
        fingerprint = sha256_text(
            f"{self.model_name}|{system_prompt}|{user_prompt}|{temperature:.2f}|{seed}|{request_id or ''}"
        )
        self._audit_buffer.append(
            {
                "provider": "mock",
                "model_name": self.model_name,
                "request_id": request_id,
                "attempts": 1,
                "retry_count": 0,
                "status": "ok",
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
            }
        )
        return f"mock:{fingerprint[:16]}"

    def pop_audit_events(self) -> list[dict[str, Any]]:
        events = list(self._audit_buffer)
        self._audit_buffer.clear()
        return events
