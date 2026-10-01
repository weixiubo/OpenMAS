from __future__ import annotations

import sys
import types
from typing import Any

from openagora_experiments.egor.local_provider import _prepare_qwen_vl_inputs
from openagora_experiments.egor.runtime_types import ArtifactSpec


class RecordingProcessor:
    def __init__(self) -> None:
        self.template_kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: list[dict], **kwargs: Any) -> str:
        del messages
        self.template_kwargs = kwargs
        return "prepared"

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs


def test_qwen35_structured_generation_disables_thinking(monkeypatch) -> None:
    processor = RecordingProcessor()
    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        types.SimpleNamespace(
            process_vision_info=lambda messages: (["processed-image"], [])
        ),
    )
    prepared = _prepare_qwen_vl_inputs(
        processor=processor,
        system_prompt="return JSON",
        user_prompt="inspect",
        image_artifact=ArtifactSpec(
            artifact_id="image",
            kind="image",
            path="/data/image.png",
            mime_type="image/png",
        ),
        enable_thinking=False,
    )
    assert processor.template_kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert prepared["text"] == ["prepared"]
