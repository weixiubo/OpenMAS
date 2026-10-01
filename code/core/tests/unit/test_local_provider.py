from __future__ import annotations

import io
import json
import sys
import types
import urllib.error

import pytest

from openagora.providers.local_provider import LocalModelProvider, LocalProviderInvocationError


class _DummyResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        del exc_type, exc_val, exc_tb

    def read(self) -> bytes:
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")


def test_local_provider_vllm_success(monkeypatch) -> None:
    provider = LocalModelProvider(
        "local-model",
        backend="vllm-local",
        endpoint="http://127.0.0.1:8000/v1",
    )

    def _fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        del request, timeout
        return _DummyResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    out = provider.complete(system_prompt="sys", user_prompt="usr", temperature=0.0, seed=42)
    assert out == "ok"
    audit = provider.pop_audit_events()
    assert audit
    assert audit[-1]["status"] == "ok"


def test_local_provider_vllm_http_error(monkeypatch) -> None:
    provider = LocalModelProvider(
        "local-model",
        backend="vllm-local",
        endpoint="http://127.0.0.1:8000/v1",
        max_retries=0,
    )

    def _fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        del request, timeout
        raise urllib.error.HTTPError(
            url="http://127.0.0.1:8000/v1/chat/completions",
            code=503,
            msg="svc down",
            hdrs=None,
            fp=io.BytesIO(b""),
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    with pytest.raises(LocalProviderInvocationError) as exc:
        provider.complete(system_prompt="sys", user_prompt="usr", temperature=0.0, seed=42)
    assert exc.value.code == "vllm_http_503"


def test_local_provider_transformers_model_load_uses_torch_dtype_and_low_cpu_mem(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class _FakeModel:
        dtype = "bf16"

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):  # type: ignore[no-untyped-def]
            del model_id, kwargs
            return object()

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):  # type: ignore[no-untyped-def]
            del model_id
            captured_kwargs.update(kwargs)
            return _FakeModel()

    def _fake_pipeline(task, model, tokenizer, device):  # type: ignore[no-untyped-def]
        del task, model, tokenizer, device

        def _runner(prompt, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            return [{"generated_text": f"{prompt}OK"}]

        return _runner

    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=_AutoModelForCausalLM,
        AutoTokenizer=_AutoTokenizer,
        pipeline=_fake_pipeline,
    )
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        manual_seed=lambda seed: None,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    provider = LocalModelProvider(
        "local-model",
        backend="transformers-local",
        model_path="/tmp/offline-model",
        max_retries=0,
    )
    out = provider.complete(system_prompt="sys", user_prompt="usr", temperature=0.0, seed=42)
    assert out == "OK"
    assert captured_kwargs.get("torch_dtype") == "auto"
    assert captured_kwargs.get("low_cpu_mem_usage") is True
    assert captured_kwargs.get("local_files_only") is True
    assert captured_kwargs.get("trust_remote_code") is True


def test_local_provider_transformers_oom_error_code_unchanged(monkeypatch) -> None:
    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):  # type: ignore[no-untyped-def]
            del model_id, kwargs
            return object()

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):  # type: ignore[no-untyped-def]
            del model_id, kwargs
            raise RuntimeError("CUDA out of memory during model init")

    def _fake_pipeline(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return None

    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=_AutoModelForCausalLM,
        AutoTokenizer=_AutoTokenizer,
        pipeline=_fake_pipeline,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    provider = LocalModelProvider(
        "local-model",
        backend="transformers-local",
        model_path="/tmp/offline-model",
        max_retries=0,
    )
    with pytest.raises(LocalProviderInvocationError) as exc:
        provider.complete(system_prompt="sys", user_prompt="usr", temperature=0.0, seed=42)
    assert exc.value.code == "cuda_oom"


@pytest.fixture(autouse=True)
def isolated_transformers_pipeline_cache(monkeypatch):
    """A cached fake pipeline must not bypass another test's model loader."""
    import openagora.providers.local_provider as local_provider
    monkeypatch.setattr(local_provider, "_TRANSFORMERS_PIPELINE_CACHE", {})
