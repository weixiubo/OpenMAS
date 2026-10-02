"""Self-contained local multimodal provider used by formal EGOR runs.

The experiment package deliberately owns this narrow adapter instead of
depending on the much broader OpenAGORA runtime.  A clean artifact therefore
needs only Transformers, PyTorch, Pillow, and (for Qwen-VL) qwen-vl-utils.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openagora_experiments.egor.runtime_types import ArtifactSpec


_DEFAULT_MAX_NEW_TOKENS = 128
_ANSWER_ONLY_MAX_NEW_TOKENS = 64
_SMALL_STRUCTURED_MAX_NEW_TOKENS = 192
_MEDIUM_STRUCTURED_MAX_NEW_TOKENS = 384


@dataclass(frozen=True)
class _ErrorKind:
    code: str
    retryable: bool


class EgorLocalProviderError(RuntimeError):
    """Normalized failure raised by the frozen EGOR inference adapter."""

    def __init__(self, message: str, *, code: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _max_new_tokens_for_request(request_id: str | None) -> int:
    key = str(request_id or "").strip().lower()
    if "answer_only_verifier" in key:
        return _ANSWER_ONLY_MAX_NEW_TOKENS
    if "proposal" in key or "negotiation_finalize" in key or "finalize" in key:
        return _MEDIUM_STRUCTURED_MAX_NEW_TOKENS
    if "feedback" in key or "conflict_summary" in key or "ballot" in key:
        return _SMALL_STRUCTURED_MAX_NEW_TOKENS
    return _DEFAULT_MAX_NEW_TOKENS


def _apply_deterministic_inference_settings(seed: int) -> None:
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # Older library versions may not expose every deterministic switch.
        return


class EgorLocalModelProvider:
    """Run one frozen local VLM with strict placement and append-only audits."""

    def __init__(
        self,
        model_name: str,
        *,
        backend: str = "transformers-local",
        model_path: str = "",
        model_profile: str = "medium",
        max_retries: int = 0,
        retry_backoff_base: float = 0.5,
        max_concurrency: int = 1,
        max_gpu_mem_gb: float = 0.0,
        gpu_devices: str = "",
        allow_download: bool = False,
        allow_cpu_offload: bool = False,
        **unused: Any,
    ) -> None:
        del unused
        if backend != "transformers-local":
            raise ValueError("formal EGOR generation supports transformers-local only")
        if max_retries < 0 or max_concurrency < 1:
            raise ValueError("invalid retry or concurrency setting")
        self.model_name = model_name
        self._backend = backend
        self._model_path = model_path.strip()
        self._model_profile = model_profile
        self._max_retries = max_retries
        self._retry_backoff_base = retry_backoff_base
        self._max_gpu_mem_gb = max_gpu_mem_gb
        self._gpu_devices = gpu_devices.strip()
        self._allow_cpu_offload = bool(allow_cpu_offload)
        self._allow_download = allow_download
        requested_devices = self._requested_gpu_slots()
        if self._allow_cpu_offload and (
            len(requested_devices) != 1
            or self._max_gpu_mem_gb <= 0
            or requested_devices[0].lower() in {"cpu", "none", "-1"}
        ):
            raise ValueError("CPU offload requires one GPU and a positive memory cap")
        self._audit_buffer: list[dict[str, Any]] = []
        self._last_call_usage: dict[str, Any] | None = None
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._transformers_multimodal_bundle: tuple[Any, Any] | None = None
        if self._gpu_devices:
            if self._gpu_devices.lower() in {"cpu", "none", "-1"}:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            else:
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", self._gpu_devices)

    def complete_multimodal(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        artifacts: list[ArtifactSpec],
        temperature: float,
        seed: int,
        request_id: str | None = None,
    ) -> str:
        """Generate one multimodal completion and retain its resource audit."""

        start = time.perf_counter()
        final_error: EgorLocalProviderError | None = None
        attempts = 0
        with self._semaphore:
            for attempt in range(self._max_retries + 1):
                attempts = attempt + 1
                self._last_call_usage = None
                try:
                    text = self._complete_transformers_multimodal(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        artifacts=artifacts,
                        temperature=temperature,
                        seed=seed,
                        request_id=request_id,
                    )
                    self._append_audit_event(
                        request_id=request_id,
                        attempts=attempts,
                        retry_count=attempt,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        artifacts=artifacts,
                    )
                    return text
                except EgorLocalProviderError as exc:
                    final_error = exc
                    if exc.retryable and attempt < self._max_retries:
                        time.sleep(self._retry_backoff_base * (2**attempt))
                        continue
                    break

        assert final_error is not None
        self._append_audit_event(
            request_id=request_id,
            attempts=attempts,
            retry_count=max(0, attempts - 1),
            latency_ms=(time.perf_counter() - start) * 1000,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            artifacts=artifacts,
            error=final_error,
        )
        raise final_error

    def pop_audit_events(self) -> list[dict[str, Any]]:
        events = list(self._audit_buffer)
        self._audit_buffer.clear()
        return events

    def _complete_transformers_multimodal(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        artifacts: list[ArtifactSpec],
        temperature: float,
        seed: int,
        request_id: str | None,
    ) -> str:
        model_id = self._resolve_model_identifier()
        try:
            _apply_deterministic_inference_settings(seed)
            model, processor = self._ensure_transformers_multimodal_bundle()
            image_artifact = _first_image_artifact(artifacts)
            if _uses_internvl_path(model_id, processor=processor, model=model):
                inputs = _prepare_internvl_hf_inputs(
                    processor=processor,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_path=image_artifact.path,
                )
                inputs = _move_internvl_inputs(inputs=inputs, model=model)
            else:
                inputs = _prepare_qwen_vl_inputs(
                    processor=processor,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_artifact=image_artifact,
                    enable_thinking=(
                        False if _uses_qwen35_native_path(model_id) else None
                    ),
                )
                inputs = _move_inputs_to_model_device(inputs=inputs, model=model)

            input_ids = inputs.get("input_ids")
            prompt_tokens = int(input_ids.shape[-1])
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": _max_new_tokens_for_request(request_id),
                "do_sample": temperature > 0.0,
            }
            if temperature > 0.0:
                generation_kwargs["temperature"] = max(temperature, 1e-4)
            outputs = model.generate(**inputs, **generation_kwargs)
            generated = outputs[:, prompt_tokens:]
            completion_tokens = int(generated.shape[-1])
            if _uses_internvl_path(model_id, processor=processor, model=model):
                text = str(
                    processor.decode(generated[0], skip_special_tokens=True)
                ).strip()
            else:
                text = str(
                    processor.batch_decode(generated, skip_special_tokens=True)[0]
                ).strip()
            self._record_call_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=model_id,
            )
            return text or "local:mm_empty_completion"
        except EgorLocalProviderError:
            raise
        except Exception as exc:
            kind = self._classify_error(exc)
            detail = " ".join(str(exc).split())[:500] or type(exc).__name__
            raise EgorLocalProviderError(
                f"EGOR multimodal call failed: {kind.code}: {detail}",
                code=kind.code,
                retryable=kind.retryable,
            ) from exc

    def _ensure_transformers_multimodal_bundle(self) -> tuple[Any, Any]:
        if self._transformers_multimodal_bundle is not None:
            return self._transformers_multimodal_bundle
        model_id = self._resolve_model_identifier()
        try:
            import transformers  # type: ignore

            auto_processor = getattr(transformers, "AutoProcessor")
            model_class = (
                getattr(transformers, "AutoModelForImageTextToText", None)
                or getattr(transformers, "AutoModelForVision2Seq", None)
                or getattr(transformers, "AutoModelForCausalLM", None)
            )
            if model_class is None:
                raise EgorLocalProviderError(
                    "no multimodal model class is available",
                    code="multimodal_model_class_missing",
                    retryable=False,
                )
            local_only = not self._allow_download
            processor = auto_processor.from_pretrained(
                model_id,
                local_files_only=local_only,
                trust_remote_code=True,
            )
            load_kwargs: dict[str, Any] = {
                "local_files_only": local_only,
                "trust_remote_code": True,
                "torch_dtype": "auto",
                "low_cpu_mem_usage": True,
            }
            if self._uses_device_map():
                load_kwargs["device_map"] = "auto"
                max_memory = self._build_max_memory_map()
                if max_memory:
                    load_kwargs["max_memory"] = max_memory
            try:
                model = model_class.from_pretrained(model_id, **load_kwargs)
            except ImportError as exc:
                if "requires Accelerate" not in str(exc):
                    raise
                if self._uses_device_map():
                    raise EgorLocalProviderError(
                        "sharded or offloaded loading requires accelerate",
                        code="accelerate_missing_for_multi_gpu",
                        retryable=False,
                    ) from exc
                load_kwargs.pop("low_cpu_mem_usage", None)
                load_kwargs.pop("max_memory", None)
                model = model_class.from_pretrained(model_id, **load_kwargs)
            model = self._place_model_for_requested_device(model)
            if callable(getattr(model, "eval", None)):
                model.eval()
            self._transformers_multimodal_bundle = (model, processor)
            return self._transformers_multimodal_bundle
        except EgorLocalProviderError:
            raise
        except Exception as exc:
            kind = self._classify_error(exc)
            message = " ".join(str(exc).split())[:500] or type(exc).__name__
            if not self._allow_download and _looks_like_missing_model(message):
                kind = _ErrorKind("local_model_missing", False)
            raise EgorLocalProviderError(
                f"EGOR model initialization failed: {kind.code}: {message}",
                code=kind.code,
                retryable=kind.retryable,
            ) from exc

    def _place_model_for_requested_device(self, model: Any) -> Any:
        """Require the requested single GPU instead of silently using CPU."""

        requested = self._requested_gpu_slots()
        force_cpu = self._gpu_devices.lower() in {"cpu", "none", "-1"}
        if not requested or force_cpu or self._uses_device_map():
            return model
        if _resolve_model_input_device(model) is not None:
            return model
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                raise EgorLocalProviderError(
                    "EGOR requested a GPU but CUDA is unavailable",
                    code="cuda_unavailable",
                    retryable=False,
                )
            if not callable(getattr(model, "to", None)):
                raise EgorLocalProviderError(
                    "EGOR model cannot be placed on the requested GPU",
                    code="device_placement_unsupported",
                    retryable=False,
                )
            model = model.to("cuda:0")
            _set_explicit_model_input_device(model, "cuda:0")
            if _resolve_model_input_device(model) is None:
                raise EgorLocalProviderError(
                    "EGOR model remained on CPU after explicit placement",
                    code="device_placement_failed",
                    retryable=False,
                )
            return model
        except EgorLocalProviderError:
            raise
        except Exception as exc:
            kind = self._classify_error(exc)
            raise EgorLocalProviderError(
                f"EGOR GPU placement failed: {kind.code}",
                code=kind.code,
                retryable=kind.retryable,
            ) from exc

    def _resolve_model_identifier(self) -> str:
        return self._model_path or self.model_name.strip()

    def _requested_gpu_slots(self) -> list[str]:
        if not self._gpu_devices or self._gpu_devices.lower() in {"cpu", "none", "-1"}:
            return []
        return [item.strip() for item in self._gpu_devices.split(",") if item.strip()]

    def _uses_multi_gpu(self) -> bool:
        return len(self._requested_gpu_slots()) > 1

    def _uses_device_map(self) -> bool:
        return self._uses_multi_gpu() or self._allow_cpu_offload

    def _build_max_memory_map(self) -> dict[Any, str] | None:
        if not self._uses_device_map() or self._max_gpu_mem_gb <= 0:
            return None
        per_gpu = f"{max(1, int(self._max_gpu_mem_gb))}GiB"
        if self._allow_cpu_offload:
            result: dict[Any, str] = {0: per_gpu}
        else:
            result = {
                index: per_gpu for index, _ in enumerate(self._requested_gpu_slots())
            }
        result["cpu"] = "96GiB"
        return result

    def _record_call_usage(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> None:
        self._last_call_usage = {
            "model": model,
            "prompt_tokens": max(0, int(prompt_tokens)),
            "completion_tokens": max(0, int(completion_tokens)),
            "total_tokens": max(0, int(prompt_tokens) + int(completion_tokens)),
        }

    def _append_audit_event(
        self,
        *,
        request_id: str | None,
        attempts: int,
        retry_count: int,
        latency_ms: float,
        system_prompt: str,
        user_prompt: str,
        artifacts: list[ArtifactSpec],
        error: EgorLocalProviderError | None = None,
    ) -> None:
        usage = dict(self._last_call_usage or {})
        self._last_call_usage = None
        entry: dict[str, Any] = {
            "provider": "egor-local",
            "backend": self._backend,
            "model_name": self.model_name,
            "model_profile": self._model_profile,
            "model": usage.get("model", self._resolve_model_identifier()),
            "mode": "multimodal",
            "modality": "multimodal",
            "request_id": request_id,
            "status": "ok" if error is None else "error",
            "attempts": attempts,
            "retry_count": retry_count,
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
            "system_prompt_chars": len(system_prompt),
            "user_prompt_chars": len(user_prompt),
            "artifact_count": len(artifacts),
            "allow_download": self._allow_download,
            "gpu_devices": self._gpu_devices,
            "max_gpu_mem_gb": self._max_gpu_mem_gb,
            "allow_cpu_offload": self._allow_cpu_offload,
        }
        if error is not None:
            entry["error_code"] = error.code
            entry["error_retryable"] = error.retryable
        self._audit_buffer.append(entry)

    @staticmethod
    def _classify_error(exc: Exception) -> _ErrorKind:
        text = str(exc).lower()
        if "timeout" in text:
            return _ErrorKind("timeout", True)
        if "connection" in text:
            return _ErrorKind("connection_error", True)
        if "expected all tensors to be on the same device" in text:
            return _ErrorKind("device_mismatch", True)
        if "found at least two devices" in text:
            return _ErrorKind("device_mismatch", True)
        if "out of memory" in text:
            return _ErrorKind("cuda_oom", False)
        if "device-side assert" in text or "cuda error" in text:
            return _ErrorKind("cuda_runtime_error", False)
        return _ErrorKind(f"unknown_error_{exc.__class__.__name__}", False)


def _prepare_qwen_vl_inputs(
    *,
    processor: Any,
    system_prompt: str,
    user_prompt: str,
    image_artifact: ArtifactSpec,
    enable_thinking: bool | None = None,
) -> Any:
    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt.strip()}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_artifact.path},
                {"type": "text", "text": user_prompt.strip() or "Describe the image."},
            ],
        }
    )
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    chat_text = processor.apply_chat_template(messages, **template_kwargs)
    image_inputs, video_inputs = _maybe_process_qwen_vision_info(
        messages,
        image_artifact=image_artifact,
    )
    kwargs: dict[str, Any] = {
        "text": [chat_text],
        "return_tensors": "pt",
        "padding": True,
    }
    if image_inputs is not None:
        kwargs["images"] = image_inputs
    if video_inputs:
        kwargs["videos"] = video_inputs
    return processor(**kwargs)


def _prepare_internvl_hf_inputs(
    *,
    processor: Any,
    system_prompt: str,
    user_prompt: str,
    image_path: str,
) -> Any:
    """Follow the official InternVL3-HF apply-chat-template contract."""

    instruction = "\n\n".join(
        item for item in (system_prompt.strip(), user_prompt.strip()) if item
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": instruction or "Describe the image."},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def _move_internvl_inputs(*, inputs: Any, model: Any) -> Any:
    device = _resolve_model_input_device(model)
    try:
        import torch  # type: ignore

        if device is not None and hasattr(inputs, "to"):
            return inputs.to(device, dtype=torch.bfloat16)
    except Exception:
        pass
    return _move_inputs_to_model_device(inputs=inputs, model=model)


def _move_inputs_to_model_device(*, inputs: Any, model: Any) -> Any:
    device = _resolve_model_input_device(model)
    if device is None:
        return inputs
    candidate = inputs
    if hasattr(candidate, "to"):
        try:
            candidate = candidate.to(device)
        except Exception:
            pass
    data = getattr(candidate, "data", None)
    if isinstance(data, Mapping):
        for key, value in list(data.items()):
            data[key] = _move_value_to_device(value, device=device)
        return candidate
    if hasattr(candidate, "items"):
        return {
            key: _move_value_to_device(value, device=device)
            for key, value in candidate.items()
        }
    return _move_value_to_device(candidate, device=device)


def _move_value_to_device(value: Any, *, device: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _move_value_to_device(item, device=device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_value_to_device(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device=device) for item in value)
    if hasattr(value, "to"):
        try:
            return value.to(device)
        except Exception:
            return value
    return value


def _resolve_model_input_device(model: Any) -> Any | None:
    hinted = _normalize_model_device(getattr(model, "_egor_input_device", None))
    if hinted is not None:
        return hinted
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for device in device_map.values():
            normalized = _normalize_model_device(device)
            if normalized is not None:
                return normalized
    direct = _normalize_model_device(getattr(model, "device", None))
    if direct is not None:
        return direct
    try:
        return _normalize_model_device(next(model.parameters()).device)
    except Exception:
        return None


def _normalize_model_device(device: Any) -> Any | None:
    if device is None:
        return None
    if isinstance(device, int):
        return f"cuda:{device}"
    text = str(device).strip().lower()
    if text in {"", "cpu", "disk", "meta"}:
        return None
    if text.isdigit():
        return f"cuda:{text}"
    return device


def _set_explicit_model_input_device(model: Any, device: Any) -> None:
    try:
        setattr(model, "_egor_input_device", device)
    except Exception:
        return


def _maybe_process_qwen_vision_info(
    messages: list[dict[str, Any]],
    *,
    image_artifact: ArtifactSpec,
) -> tuple[Any, Any]:
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore

        return process_vision_info(messages)
    except Exception:
        return [_load_image(image_artifact.path)], None


def _load_image(path: str) -> Any:
    try:
        from PIL import Image
    except Exception as exc:
        raise EgorLocalProviderError(
            "Pillow is required for multimodal inference",
            code="pillow_missing",
            retryable=False,
        ) from exc
    with Image.open(path) as image:
        return image.convert("RGB")


def _first_image_artifact(artifacts: Sequence[ArtifactSpec]) -> ArtifactSpec:
    for artifact in artifacts:
        if artifact.kind.lower() == "image" and artifact.path.strip():
            return artifact
    raise EgorLocalProviderError(
        "multimodal generation requires one image artifact",
        code="missing_image_artifact",
        retryable=False,
    )


def _first_image_path(artifacts: Sequence[ArtifactSpec]) -> str:
    return _first_image_artifact(artifacts).path


def _uses_qwen35_native_path(model_id: str) -> bool:
    lowered = model_id.strip().lower()
    return "qwen3.5" in lowered or "qwen3_5" in lowered


def _uses_internvl_path(model_id: str, *, processor: Any, model: Any) -> bool:
    lowered = " ".join(
        (
            model_id,
            processor.__class__.__name__,
            model.__class__.__name__,
            str(getattr(processor, "name_or_path", "")),
            str(getattr(model, "name_or_path", "")),
        )
    ).lower()
    return "internvl" in lowered


def _looks_like_missing_model(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "not found",
            "no such file",
            "does not appear to have a file named",
            "is not a local folder",
            "incorrect path_or_model_id",
        )
    )
