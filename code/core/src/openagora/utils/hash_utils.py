"""Hash helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_text(value: str) -> str:
    """Hash text using SHA-256."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(payload: dict[str, Any]) -> str:
    """Hash canonicalized JSON mapping."""
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(canonical)


def sha256_file(path: str) -> str:
    """Hash file contents using SHA-256."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as fp:
        while True:
            chunk = fp.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
