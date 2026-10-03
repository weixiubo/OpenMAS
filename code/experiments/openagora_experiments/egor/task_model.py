"""Answer-blind prompts and strict structured-output parsing for EGOR stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_IMAGE_MIME_BY_SUFFIX = {
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class PilotTask:
    """One multimodal task; reference fields never enter a generation prompt."""

    task_id: str
    source_id: str
    question: str
    image_path: str
    mime_type: str
    metric: str
    answer: str
    reference_answers: tuple[str, ...]
    options: tuple[tuple[str, str], ...]
    metadata: Mapping[str, Any]

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> PilotTask:
        image_path = str(payload.get("image_path", "")).strip()
        mime_type = _infer_image_mime_type(image_path)
        raw_references = payload.get("reference_answers", [])
        references = (
            tuple(str(item).strip() for item in raw_references if str(item).strip())
            if isinstance(raw_references, list)
            else ()
        )
        options: list[tuple[str, str]] = []
        raw_options = payload.get("options", [])
        if isinstance(raw_options, list):
            for item in raw_options:
                if not isinstance(item, Mapping):
                    continue
                option_id = str(item.get("id", "")).strip().upper()
                text = str(item.get("text", "")).strip()
                if option_id and text:
                    options.append((option_id, text))
        metadata = payload.get("metadata", {})
        task = cls(
            task_id=str(payload.get("task_id", "")).strip(),
            source_id=str(payload.get("benchmark", payload.get("source_id", ""))).strip(),
            question=str(payload.get("question", "")).strip(),
            image_path=image_path,
            mime_type=mime_type,
            metric=str(payload.get("metric", "accuracy")).strip(),
            answer=str(payload.get("answer", "")).strip(),
            reference_answers=references,
            options=tuple(options),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )
        if not task.task_id or not task.source_id or not task.question or not task.image_path:
            raise ValueError("task_id, benchmark/source_id, question, and image_path are required")
        return task

    def visible_question(self) -> str:
        if not self.options:
            return self.question
        rendered_options = "\n".join(f"{option_id}. {text}" for option_id, text in self.options)
        return f"{self.question}\n\nOptions:\n{rendered_options}"


def _infer_image_mime_type(image_path: str) -> str:
    """Infer an honest image MIME type without assuming every non-JPEG is PNG."""

    path = Path(image_path)
    suffix_type = _IMAGE_MIME_BY_SUFFIX.get(path.suffix.lower())
    if suffix_type is not None:
        return suffix_type
    try:
        header = path.read_bytes()[:16]
    except OSError:
        return "application/octet-stream"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"BM"):
        return "image/bmp"
    if header[:4] in {b"II*\x00", b"MM\x00*"}:
        return "image/tiff"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"
