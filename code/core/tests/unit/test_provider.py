from __future__ import annotations

import pytest

from openagora.providers.openai_provider import OpenAIModelProvider, ProviderInvocationError


class _RateLimitError(Exception):
    status_code = 429


class _ClientError(Exception):
    status_code = 400


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, *, fail_times: int, exc_type: type[Exception]) -> None:
        self._fail_times = fail_times
        self._count = 0
        self._exc_type = exc_type

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        self._count += 1
        if self._count <= self._fail_times:
            raise self._exc_type("boom")
        return _Response("ok")


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _Client:
    def __init__(self, completions: _Completions) -> None:
        self.chat = _Chat(completions)


def test_openai_provider_retries_and_audits(monkeypatch) -> None:
    provider = OpenAIModelProvider("gpt-4o-mini", max_retries=2, retry_backoff_base=0.001)
    completions = _Completions(fail_times=1, exc_type=_RateLimitError)
    monkeypatch.setattr(provider, "_ensure_client", lambda: _Client(completions))
    out = provider.complete(system_prompt="s", user_prompt="u", temperature=0.0, seed=1, request_id="r1")
    assert out == "ok"
    audit = provider.pop_audit_events()
    assert audit
    assert audit[-1]["retry_count"] == 1


def test_openai_provider_non_retryable_error(monkeypatch) -> None:
    provider = OpenAIModelProvider("gpt-4o-mini", max_retries=2, retry_backoff_base=0.001)
    completions = _Completions(fail_times=3, exc_type=_ClientError)
    monkeypatch.setattr(provider, "_ensure_client", lambda: _Client(completions))
    with pytest.raises(ProviderInvocationError) as exc:
        provider.complete(system_prompt="s", user_prompt="u", temperature=0.0, seed=1)
    assert "client_error_400" in str(exc.value)
