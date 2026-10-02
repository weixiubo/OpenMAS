from __future__ import annotations

import os
import sys
import types
from typing import Any

import pytest
from openagora_experiments.egor.local_provider import EgorLocalModelProvider, _max_new_tokens_for_request
from openagora_experiments.egor.model_inputs import (
    _first_image_artifact,
    _first_image_path,
    _prepare_internvl_hf_inputs,
    _prepare_qwen_vl_inputs,
    _uses_qwen35_native_path,
)
from openagora_experiments.egor.provider_errors import EgorLocalProviderError
from openagora_experiments.egor.runtime_types import ArtifactSpec


class FakeProcessor:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] | None = None
        self.kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.messages = messages
        self.kwargs = kwargs
        return "prepared"


class FakeQwenProcessor(FakeProcessor):
    def __call__(self, **kwargs: Any) -> str:
        self.kwargs = kwargs
        return "qwen-inputs"


def test_internvl_uses_native_multimodal_chat_template() -> None:
    processor = FakeProcessor()

    prepared = _prepare_internvl_hf_inputs(
        processor=processor,
        system_prompt="system contract",
        user_prompt="inspect independently",
        image_path="/data/image.png",
    )

    assert prepared == "prepared"
    assert processor.messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "/data/image.png"},
                {"type": "text", "text": "system contract\n\ninspect independently"},
            ],
        }
    ]
    assert processor.kwargs == {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }


def test_qwen_uses_native_multimodal_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = FakeQwenProcessor()
    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        types.SimpleNamespace(process_vision_info=lambda messages: (["processed-image"], [])),
    )
    artifact = ArtifactSpec(
        artifact_id="image",
        kind="image",
        path="/data/image.png",
        mime_type="image/png",
    )

    prepared = _prepare_qwen_vl_inputs(
        processor=processor,
        system_prompt="system contract",
        user_prompt="inspect independently",
        image_artifact=artifact,
    )

    assert prepared == "qwen-inputs"
    assert processor.messages == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "system contract"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/data/image.png"},
                {"type": "text", "text": "inspect independently"},
            ],
        },
    ]
    assert processor.kwargs == {
        "text": ["prepared"],
        "images": ["processed-image"],
        "return_tensors": "pt",
        "padding": True,
    }


def test_frozen_request_token_budgets() -> None:
    assert _max_new_tokens_for_request("egor::proposal::baseline") == 384
    assert _max_new_tokens_for_request("egor::finalize::exact") == 384
    assert _max_new_tokens_for_request("egor::answer_only_verifier_v8::x") == 64
    assert _max_new_tokens_for_request("egor::route::llm_scheduler") == 128


def test_internvl_requires_an_image_artifact() -> None:
    with pytest.raises(EgorLocalProviderError, match="requires one image"):
        _first_image_path([])

    assert (
        _first_image_path(
            [
                ArtifactSpec(
                    artifact_id="image",
                    kind="image",
                    path="/data/image.png",
                    mime_type="image/png",
                )
            ]
        )
        == "/data/image.png"
    )


def test_egor_provider_fails_fast_instead_of_cpu_fallback_on_placement_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        device = "cpu"
        hf_device_map = None

        def to(self, device: str) -> FakeModel:
            del device
            raise RuntimeError("CUDA out of memory during model.to")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)),
    )
    provider = EgorLocalModelProvider(
        "qwen3-vl",
        backend="transformers-local",
        model_path="/tmp/qwen3-vl",
        gpu_devices="0",
        max_retries=0,
    )

    with pytest.raises(EgorLocalProviderError) as exc:
        provider._place_model_for_requested_device(FakeModel())

    assert exc.value.code == "cuda_oom"


def test_egor_multi_gpu_loader_enforces_explicit_per_device_memory_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcessorFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> object:
            assert model_id == "/models/qwen3-vl"
            assert kwargs == {
                "local_files_only": True,
                "trust_remote_code": True,
            }
            return object()

    class FakeModel:
        hf_device_map = {
            "visual": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "lm_head": 1,
        }

        def __init__(self) -> None:
            self.eval_called = False

        def eval(self) -> None:
            self.eval_called = True

        def to(self, device: str) -> FakeModel:
            raise AssertionError(f"sharded model must not be moved to {device}")

    class FakeModelFactory:
        calls: list[tuple[str, dict[str, Any]]] = []

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeModel:
            cls.calls.append((model_id, kwargs))
            return FakeModel()

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoProcessor=FakeProcessorFactory,
            AutoModelForImageTextToText=FakeModelFactory,
        ),
    )
    provider = EgorLocalModelProvider(
        "qwen3-vl",
        backend="transformers-local",
        model_path="/models/qwen3-vl",
        gpu_devices="1,0",
        max_gpu_mem_gb=11,
        max_retries=0,
    )

    model, _ = provider._ensure_transformers_multimodal_bundle()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,0"
    assert provider._build_max_memory_map() == {
        0: "11GiB",
        1: "11GiB",
        "cpu": "96GiB",
    }
    assert FakeModelFactory.calls == [
        (
            "/models/qwen3-vl",
            {
                "local_files_only": True,
                "trust_remote_code": True,
                "torch_dtype": "auto",
                "low_cpu_mem_usage": True,
                "device_map": "auto",
                "max_memory": {0: "11GiB", 1: "11GiB", "cpu": "96GiB"},
            },
        )
    ]
    assert model.eval_called is True


def test_egor_single_gpu_cpu_offload_uses_a_bounded_device_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcessorFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> object:
            del model_id, kwargs
            return object()

    class FakeModel:
        hf_device_map = {"visual": 0, "model.layers.0": 0, "lm_head": "cpu"}

        def eval(self) -> None:
            return None

        def to(self, device: str) -> FakeModel:
            raise AssertionError(f"offloaded model must not be moved to {device}")

    class FakeModelFactory:
        calls: list[dict[str, Any]] = []

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeModel:
            del model_id
            cls.calls.append(kwargs)
            return FakeModel()

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoProcessor=FakeProcessorFactory,
            AutoModelForImageTextToText=FakeModelFactory,
        ),
    )
    provider = EgorLocalModelProvider(
        "qwen3.5-9b",
        backend="transformers-local",
        model_path="/models/qwen3.5-9b",
        gpu_devices="0",
        max_gpu_mem_gb=18,
        allow_cpu_offload=True,
    )

    provider._ensure_transformers_multimodal_bundle()

    assert provider._build_max_memory_map() == {0: "18GiB", "cpu": "96GiB"}
    assert FakeModelFactory.calls == [
        {
            "local_files_only": True,
            "trust_remote_code": True,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "max_memory": {0: "18GiB", "cpu": "96GiB"},
        }
    ]


@pytest.mark.parametrize("gpu_devices,max_memory", [("0,1", 18), ("0", 0)])
def test_egor_cpu_offload_rejects_an_ambiguous_placement(gpu_devices: str, max_memory: float) -> None:
    with pytest.raises(ValueError, match="CPU offload requires one GPU"):
        EgorLocalModelProvider(
            "qwen3.5-9b",
            gpu_devices=gpu_devices,
            max_gpu_mem_gb=max_memory,
            allow_cpu_offload=True,
        )


def test_provider_failure_is_recorded_in_audit() -> None:
    class FailingProvider(EgorLocalModelProvider):
        def _complete_transformers_multimodal(self, **kwargs: Any) -> str:
            del kwargs
            raise EgorLocalProviderError(
                "fixture failure",
                code="fixture_error",
                retryable=False,
            )

    provider = FailingProvider(
        "fixture",
        model_path="/models/fixture",
        gpu_devices="cpu",
    )
    artifact = ArtifactSpec(
        artifact_id="image",
        kind="image",
        path="/data/image.png",
        mime_type="image/png",
    )

    with pytest.raises(EgorLocalProviderError, match="fixture failure"):
        provider.complete_multimodal(
            system_prompt="system",
            user_prompt="user",
            artifacts=[artifact],
            temperature=0.0,
            seed=7,
            request_id="fixture",
        )

    assert provider.pop_audit_events() == [
        {
            "provider": "egor-local",
            "backend": "transformers-local",
            "model_name": "fixture",
            "model_profile": "medium",
            "model": "/models/fixture",
            "mode": "multimodal",
            "modality": "multimodal",
            "request_id": "fixture",
            "status": "error",
            "attempts": 1,
            "retry_count": 0,
            "latency_ms": pytest.approx(0.0, abs=100.0),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "system_prompt_chars": 6,
            "user_prompt_chars": 4,
            "artifact_count": 1,
            "allow_download": False,
            "gpu_devices": "cpu",
            "max_gpu_mem_gb": 0.0,
            "allow_cpu_offload": False,
            "error_code": "fixture_error",
            "error_retryable": False,
        }
    ]


def test_qwen35_uses_native_multimodal_path_and_requires_image() -> None:
    assert _uses_qwen35_native_path("/models/Qwen3.5-9B")
    assert _uses_qwen35_native_path("Qwen3_5ForConditionalGeneration")
    assert not _uses_qwen35_native_path("Qwen3-VL-8B-Instruct")
    artifact = ArtifactSpec(
        artifact_id="image",
        kind="image",
        path="/data/image.png",
        mime_type="image/png",
    )
    assert _first_image_artifact([artifact]) == artifact
    with pytest.raises(EgorLocalProviderError, match="requires one image"):
        _first_image_artifact([])
