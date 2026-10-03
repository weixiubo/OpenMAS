"""Small runtime types owned by the self-contained EGOR experiment package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArtifactSpec:
    """One external artifact attached to a multimodal model request."""

    artifact_id: str
    kind: str
    path: str
    mime_type: str = "application/octet-stream"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id is required")
        if not self.kind.strip():
            raise ValueError("artifact kind is required")
        if not self.path.strip():
            raise ValueError("artifact path is required")
        if not self.mime_type.strip():
            raise ValueError("artifact MIME type is required")
