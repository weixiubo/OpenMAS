"""OpenAI provider adapter with retries, timeout and error normalization."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from openagora.core.errors import ExternalDependencyError


@dataclass(frozen=True)
class _ErrorKind:
    code: str
    retryable: bool


class ProviderInvocationError(ExternalDependencyError):
    """Normalized provider invocation error."""

    def __init__(self, message: str, *, code: str, retryable: bool) -> None:
        super().__init__(message, code=code, retryable=retryable)


class OpenAIModelProvider:
    """OpenAI Chat Completions adapter (optional dependency)."""

    def __init__(
        self,
        model_name: str,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        max_concurrency: int = 4,
        enable_idempotency_key: bool = True,
    ) -> None:
        self.model_name = model_name
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._retry_backoff_base = retry_backoff_base
        self._max_concurrency = max_concurrency
        self._enable_idempotency_key = enable_idempotency_key
        self._client: Any | None = None
        self._audit_buffer: list[dict[str, Any]] = []
        self._semaphore = threading.BoundedSemaphore(max_concurrency)

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ExternalDependencyError(
                "OPENAI_API_KEY is required when provider runs in live mode",
                code="missing_openai_api_key",
                retryable=False,
            )
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - import failures are environment-dependent
            raise ExternalDependencyError(
                "OpenAI package is not installed. Install with: pip install 'openagora[openai]'",
                code="openai_package_missing",
                retryable=False,
            ) from exc
        self._client = OpenAI(api_key=api_key, timeout=self._timeout_s, max_retries=0)
        return self._client

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        seed: int,
        request_id: str | None = None,
    ) -> str:
        client = self._ensure_client()
        headers: dict[str, str] | None = None
        if request_id and self._enable_idempotency_key:
            headers = {"Idempotency-Key": request_id}
        start_ts = time.perf_counter()
        final_error: ProviderInvocationError | None = None

        with self._semaphore:
            for attempt in range(self._max_retries + 1):
                try:
                    response = client.chat.completions.create(
                        model=self.model_name,
                        temperature=temperature,
                        seed=seed,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        extra_headers=headers,
                    )
                    content = response.choices[0].message.content
                    text = "" if content is None else str(content)
                    self._audit_buffer.append(
                        {
                            "provider": "openai",
                            "model_name": self.model_name,
                            "request_id": request_id,
                            "status": "ok",
                            "attempts": attempt + 1,
                            "retry_count": attempt,
                            "latency_ms": round((time.perf_counter() - start_ts) * 1000, 2),
                            "system_prompt_chars": len(system_prompt),
                            "user_prompt_chars": len(user_prompt),
                            "max_concurrency": self._max_concurrency,
                        }
                    )
                    return text
                except Exception as exc:  # pragma: no cover - exercised with mocked error classes
                    kind = self._classify_error(exc)
                    final_error = ProviderInvocationError(
                        f"OpenAI call failed: {kind.code}",
                        code=kind.code,
                        retryable=kind.retryable,
                    )
                    if kind.retryable and attempt < self._max_retries:
                        time.sleep(self._retry_backoff_base * (2**attempt))
                        continue
                    break

        assert final_error is not None
        self._audit_buffer.append(
            {
                "provider": "openai",
                "model_name": self.model_name,
                "request_id": request_id,
                "status": "error",
                "attempts": self._max_retries + 1,
                "retry_count": self._max_retries,
                "error_code": final_error.code,
                "error_retryable": final_error.retryable,
                "latency_ms": round((time.perf_counter() - start_ts) * 1000, 2),
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
                "max_concurrency": self._max_concurrency,
            }
        )
        raise final_error

    def _classify_error(self, exc: Exception) -> _ErrorKind:
        name = exc.__class__.__name__
        status_code = int(getattr(exc, "status_code", 0) or 0)
        if status_code == 429 or name in {"RateLimitError"}:
            return _ErrorKind(code="rate_limit", retryable=True)
        if status_code >= 500:
            return _ErrorKind(code="server_error", retryable=True)
        if name in {"APITimeoutError", "TimeoutError"}:
            return _ErrorKind(code="timeout", retryable=True)
        if name in {"APIConnectionError", "ConnectionError"}:
            return _ErrorKind(code="connection_error", retryable=True)
        if 400 <= status_code < 500:
            return _ErrorKind(code=f"client_error_{status_code}", retryable=False)
        return _ErrorKind(code=f"unknown_error_{name}", retryable=False)

    def pop_audit_events(self) -> list[dict[str, Any]]:
        events = list(self._audit_buffer)
        self._audit_buffer.clear()
        return events


