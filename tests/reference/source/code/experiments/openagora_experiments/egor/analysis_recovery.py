"""Integrity primitives for isolated EGOR analysis recovery."""

from __future__ import annotations

import hashlib
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_paths(paths: list[Path]) -> dict[str, Any]:
    roots = [path.resolve() for path in paths]
    entries: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"protected path does not exist: {root}")
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if path.is_symlink():
                raise ValueError(f"protected path must not be a symlink: {path}")
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            metadata = resolved.stat()
            entries.append(
                {
                    "path": str(resolved),
                    "sha256": sha256(resolved),
                    "size": metadata.st_size,
                    "mtime_ns": metadata.st_mtime_ns,
                    "mode": stat.S_IMODE(metadata.st_mode),
                }
            )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protected_roots": [str(path) for path in roots],
        "entries": sorted(entries, key=lambda item: item["path"]),
    }


def verify_snapshot(snapshot: dict[str, Any]) -> None:
    if snapshot.get("schema_version") != 1:
        raise ValueError("unsupported protected-artifact snapshot schema")
    raw_roots = snapshot.get("protected_roots")
    raw_entries = snapshot.get("entries")
    if not isinstance(raw_roots, list) or not raw_roots:
        raise ValueError("snapshot has no protected roots")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("snapshot has no protected files")
    current = snapshot_paths([Path(str(path)) for path in raw_roots])
    expected = sorted(raw_entries, key=lambda item: str(item.get("path", "")))
    if current["entries"] != expected:
        raise ValueError("protected EGOR artifacts changed after the snapshot")
