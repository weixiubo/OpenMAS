"""Multimodal prompt and tensor preparation, independent of provider lifecycle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openagora_experiments.egor.provider_errors import EgorLocalProviderError
from openagora_experiments.egor.runtime_types import ArtifactSpec


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

    instruction = "\n\n".join(item for item in (system_prompt.strip(), user_prompt.strip()) if item)
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
        return {key: _move_value_to_device(value, device=device) for key, value in candidate.items()}
    return _move_value_to_device(candidate, device=device)


def _move_value_to_device(value: Any, *, device: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _move_value_to_device(item, device=device) for key, item in value.items()}
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
        model._egor_input_device = device
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
