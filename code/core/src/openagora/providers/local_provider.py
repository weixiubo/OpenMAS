"""Local model provider adapters for zero-cash server runs."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

from openagora.core.errors import ExternalDependencyError

LocalBackend = Literal["transformers-local", "vllm-local"]
LocalProfile = Literal["small", "medium", "large"]

_PROFILE_DEFAULT_MODEL: dict[LocalProfile, str] = {
    "small": "sshleifer/tiny-gpt2",
    "medium": "distilgpt2",
    "large": "gpt2-medium",
}
_PROFILE_MIN_MEM_GB: dict[LocalProfile, float] = {
    "small": 4.0,
    "medium": 12.0,
    "large": 24.0,
}
_PROFILE_DESC_ORDER: tuple[LocalProfile, ...] = ("large", "medium", "small")
_TRANSFORMERS_PIPELINE_CACHE: dict[tuple[str, str, str], Any] = {}
_TRANSFORMERS_PIPELINE_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class _ErrorKind:
    code: str
    retryable: bool


class LocalProviderInvocationError(ExternalDependencyError):
    """Normalized local-provider invocation error."""

    def __init__(self, message: str, *, code: str, retryable: bool) -> None:
        super().__init__(message, code=code, retryable=retryable)


class LocalModelProvider:
    """Local provider with transformers/vLLM backends."""

    def __init__(
        self,
        model_name: str,
        *,
        backend: LocalBackend = "transformers-local",
        model_path: str = "",
        model_profile: LocalProfile = "small",
        endpoint: str = "http://127.0.0.1:8000/v1",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        max_concurrency: int = 4,
        max_gpu_mem_gb: float = 0.0,
        gpu_devices: str = "",
        allow_download: bool = False,
    ) -> None:
        self.model_name = model_name
        self._backend = backend
        self._model_path = model_path.strip()
        self._model_profile = model_profile
        self._effective_profile = model_profile
        self._endpoint = endpoint.rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._retry_backoff_base = retry_backoff_base
        self._max_concurrency = max_concurrency
        self._max_gpu_mem_gb = max_gpu_mem_gb
        self._gpu_devices = gpu_devices.strip()
        self._allow_download = allow_download
        self._audit_buffer: list[dict[str, Any]] = []
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._transformers_pipeline: Any | None = None
        if self._gpu_devices:
            # Keep local runs pinned to explicit devices when requested.
            if self._gpu_devices.lower() in {"cpu", "none", "-1"}:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            else:
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", self._gpu_devices)

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        seed: int,
        request_id: str | None = None,
    ) -> str:
        start_ts = time.perf_counter()
        final_error: LocalProviderInvocationError | None = None
        attempts = 0
        with self._semaphore:
            for attempt in range(self._max_retries + 1):
                attempts = attempt + 1
                try:
                    if self._backend == "transformers-local":
                        text = self._complete_transformers(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            temperature=temperature,
                            seed=seed,
                        )
                    else:
                        text = self._complete_vllm(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            temperature=temperature,
                            seed=seed,
                        )
                    self._audit_buffer.append(
                        {
                            "provider": "local",
                            "backend": self._backend,
                            "model_name": self.model_name,
                            "model_profile": self._effective_profile,
                            "request_id": request_id,
                            "status": "ok",
                            "attempts": attempts,
                            "retry_count": attempt,
                            "latency_ms": round((time.perf_counter() - start_ts) * 1000, 2),
                            "system_prompt_chars": len(system_prompt),
                            "user_prompt_chars": len(user_prompt),
                            "allow_download": self._allow_download,
                            "gpu_devices": self._gpu_devices,
                            "max_gpu_mem_gb": self._max_gpu_mem_gb,
                        }
                    )
                    return text
                except LocalProviderInvocationError as exc:
                    final_error = exc
                    if exc.code == "cuda_oom" and self._try_downgrade_profile():
                        continue
                    if exc.code == "cuda_runtime_error" and self._try_fallback_to_cpu():
                        continue
                    if exc.retryable and attempt < self._max_retries:
                        time.sleep(self._retry_backoff_base * (2**attempt))
                        continue
                    break

        assert final_error is not None
        self._audit_buffer.append(
            {
                "provider": "local",
                "backend": self._backend,
                "model_name": self.model_name,
                "model_profile": self._effective_profile,
                "request_id": request_id,
                "status": "error",
                "attempts": attempts,
                "retry_count": max(0, attempts - 1),
                "error_code": final_error.code,
                "error_retryable": final_error.retryable,
                "latency_ms": round((time.perf_counter() - start_ts) * 1000, 2),
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
                "allow_download": self._allow_download,
                "gpu_devices": self._gpu_devices,
                "max_gpu_mem_gb": self._max_gpu_mem_gb,
            }
        )
        raise final_error

    def pop_audit_events(self) -> list[dict[str, Any]]:
        events = list(self._audit_buffer)
        self._audit_buffer.clear()
        return events

    def _complete_transformers(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        seed: int,
    ) -> str:
        pipeline = self._ensure_transformers_pipeline()
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        try:
            try:
                import torch  # type: ignore[import-not-found]

                torch.manual_seed(seed)
            except Exception:
                pass
            outputs = pipeline(
                prompt,
                max_new_tokens=128,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1e-4),
                top_p=0.95,
                num_return_sequences=1,
            )
            generated = str(outputs[0].get("generated_text", ""))
            text = generated[len(prompt) :].strip() if generated.startswith(prompt) else generated.strip()
            return text or "local:empty_completion"
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                raise LocalProviderInvocationError(
                    "Local transformers backend OOM",
                    code="cuda_oom",
                    retryable=False,
                ) from exc
            kind = self._classify_error(exc)
            raise LocalProviderInvocationError(
                f"Local transformers call failed: {kind.code}",
                code=kind.code,
                retryable=kind.retryable,
            ) from exc
        except Exception as exc:
            kind = self._classify_error(exc)
            raise LocalProviderInvocationError(
                f"Local transformers call failed: {kind.code}",
                code=kind.code,
                retryable=kind.retryable,
            ) from exc

    def _complete_vllm(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        seed: int,
    ) -> str:
        model = self._resolve_model_identifier()
        endpoint = f"{self._endpoint}/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:  # noqa: S310
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            choices = parsed.get("choices", [])
            if not choices:
                return "local:vllm_empty_choice"
            message = choices[0].get("message", {})
            return str(message.get("content", "")).strip() or "local:vllm_empty_content"
        except urllib.error.HTTPError as exc:
            retryable = int(exc.code) >= 500
            code = f"vllm_http_{exc.code}"
            raise LocalProviderInvocationError(
                f"Local vLLM request failed: {code}",
                code=code,
                retryable=retryable,
            ) from exc
        except urllib.error.URLError as exc:
            raise LocalProviderInvocationError(
                "Local vLLM endpoint unavailable",
                code="vllm_connection_error",
                retryable=True,
            ) from exc
        except TimeoutError as exc:
            raise LocalProviderInvocationError(
                "Local vLLM request timeout",
                code="timeout",
                retryable=True,
            ) from exc
        except Exception as exc:
            kind = self._classify_error(exc)
            raise LocalProviderInvocationError(
                f"Local vLLM call failed: {kind.code}",
                code=kind.code,
                retryable=kind.retryable,
            ) from exc

    def _ensure_transformers_pipeline(self) -> Any:
        if self._transformers_pipeline is not None:
            return self._transformers_pipeline
        model_id = self._resolve_model_identifier()
        cache_key = (self._backend, model_id, self._gpu_devices or "auto")
        with _TRANSFORMERS_PIPELINE_CACHE_LOCK:
            cached = _TRANSFORMERS_PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            self._transformers_pipeline = cached
            return cached
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
        except Exception as exc:
            raise ExternalDependencyError(
                "Transformers backend requires `transformers` and `torch` packages.",
                code="transformers_package_missing",
                retryable=False,
            ) from exc
        try:
            local_only = not self._allow_download
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=local_only,
                trust_remote_code=True,
            )
            model_load_kwargs = self._model_load_options(local_only)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **model_load_kwargs,
                )
            except ImportError as exc:
                # Some offline environments do not ship `accelerate`; keep
                # torch_dtype auto while retrying without low_cpu_mem_usage.
                if "requires Accelerate" not in str(exc):
                    raise
                model_load_kwargs.pop("low_cpu_mem_usage", None)
                model_load_kwargs.pop("max_memory", None)
                if self._uses_multi_gpu():
                    raise ExternalDependencyError(
                        "Multi-GPU transformers loading requires `accelerate` in the local environment.",
                        code="accelerate_missing_for_multi_gpu",
                        retryable=False,
                    ) from exc
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **model_load_kwargs,
                )
            device = self._pipeline_device()
            pipeline_kwargs: dict[str, Any] = {
                "task": "text-generation",
                "model": model,
                "tokenizer": tokenizer,
            }
            if not self._uses_multi_gpu() and not getattr(model, "hf_device_map", None):
                pipeline_kwargs["device"] = device
            self._transformers_pipeline = pipeline(**pipeline_kwargs)
            with _TRANSFORMERS_PIPELINE_CACHE_LOCK:
                _TRANSFORMERS_PIPELINE_CACHE[cache_key] = self._transformers_pipeline
            return self._transformers_pipeline
        except Exception as exc:
            message = str(exc).lower()
            if not self._allow_download and ("not found" in message or "no such file" in message):
                raise ExternalDependencyError(
                    f"Local model not found on disk: {model_id}. "
                    "Pass a local model path or enable downloads explicitly.",
                    code="local_model_missing",
                    retryable=False,
                ) from exc
            if "out of memory" in message:
                raise LocalProviderInvocationError(
                    "Local transformers backend OOM during model load",
                    code="cuda_oom",
                    retryable=False,
                ) from exc
            raise ExternalDependencyError(
                f"Failed to initialize local transformers backend: {exc}",
                code="local_backend_init_failed",
                retryable=False,
            ) from exc

    def _pipeline_device(self) -> int:
        force_cpu = self._gpu_devices.lower() in {"cpu", "none", "-1"}
        device = -1
        try:
            import torch

            if torch.cuda.is_available() and not force_cpu:
                device = 0
        except Exception:
            device = -1
        return device

    def _model_load_options(self, local_only: bool) -> dict[str, Any]:
        model_load_kwargs: dict[str, Any] = {
            "local_files_only": local_only,
            "trust_remote_code": True,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        if self._uses_multi_gpu():
            model_load_kwargs["device_map"] = "auto"
            max_memory = self._build_max_memory_map()
            if max_memory:
                model_load_kwargs["max_memory"] = max_memory
        return model_load_kwargs

    def _resolve_model_identifier(self) -> str:
        if self._model_path:
            return self._model_path
        model_name = self.model_name.strip()
        if model_name and not model_name.startswith("mock"):
            return model_name
        profile = self._resolve_profile_with_mem_guard(self._model_profile)
        return _PROFILE_DEFAULT_MODEL[profile]

    def _resolve_profile_with_mem_guard(self, profile: LocalProfile) -> LocalProfile:
        if self._model_path:
            return profile
        if self._max_gpu_mem_gb <= 0:
            self._effective_profile = profile
            return profile
        idx = _PROFILE_DESC_ORDER.index(profile)
        current = profile
        while _PROFILE_MIN_MEM_GB[current] > self._max_gpu_mem_gb and idx < len(_PROFILE_DESC_ORDER) - 1:
            idx += 1
            current = _PROFILE_DESC_ORDER[idx]
        self._effective_profile = current
        return current

    def _try_downgrade_profile(self) -> bool:
        if self._model_path:
            return False
        idx = _PROFILE_DESC_ORDER.index(self._effective_profile)
        if idx >= len(_PROFILE_DESC_ORDER) - 1:
            return False
        self._effective_profile = _PROFILE_DESC_ORDER[idx + 1]
        self.model_name = _PROFILE_DEFAULT_MODEL[self._effective_profile]
        self._transformers_pipeline = None
        return True

    def _try_fallback_to_cpu(self) -> bool:
        if self._gpu_devices.lower() in {"cpu", "none", "-1"}:
            return False
        self._gpu_devices = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self._transformers_pipeline = None
        return True

    def _classify_error(self, exc: Exception) -> _ErrorKind:
        text = str(exc).lower()
        if "timeout" in text:
            return _ErrorKind(code="timeout", retryable=True)
        if "connection" in text:
            return _ErrorKind(code="connection_error", retryable=True)
        if "out of memory" in text:
            return _ErrorKind(code="cuda_oom", retryable=False)
        if "device-side assert" in text or "cuda error" in text:
            return _ErrorKind(code="cuda_runtime_error", retryable=False)
        return _ErrorKind(code=f"unknown_error_{exc.__class__.__name__}", retryable=False)

    def _requested_gpu_slots(self) -> list[str]:
        if not self._gpu_devices or self._gpu_devices.lower() in {"cpu", "none", "-1"}:
            return []
        return [item.strip() for item in self._gpu_devices.split(",") if item.strip()]

    def _uses_multi_gpu(self) -> bool:
        return len(self._requested_gpu_slots()) > 1

    def _build_max_memory_map(self) -> dict[Any, str] | None:
        if not self._uses_multi_gpu() or self._max_gpu_mem_gb <= 0:
            return None
        per_gpu = f"{max(1, int(self._max_gpu_mem_gb))}GiB"
        max_memory: dict[Any, str] = {idx: per_gpu for idx, _ in enumerate(self._requested_gpu_slots())}
        max_memory["cpu"] = "96GiB"
        return max_memory
