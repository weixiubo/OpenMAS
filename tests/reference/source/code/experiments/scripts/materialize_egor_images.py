#!/usr/bin/env python3
"""Materialize an answer-blind, pixel-capped EGOR task image set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def main() -> int:
    args = _parse_args()
    source_path = Path(args.tasks).resolve()
    output_path = Path(args.out_tasks).resolve()
    image_root = Path(args.out_images).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else output_path.with_suffix(output_path.suffix + ".manifest.json")
    )
    if args.max_pixels <= 0:
        raise ValueError("max-pixels must be positive")
    image_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(source_path)
    materialized: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        source_image = Path(str(row.get("image_path", ""))).expanduser().resolve()
        if not task_id or task_id in seen_ids or not source_image.is_file():
            raise ValueError(f"invalid task/image input: {task_id} {source_image}")
        seen_ids.add(task_id)
        destination = image_root / f"{_safe_name(task_id)}.png"
        original_size, output_size, resized = _materialize_image(
            source_image,
            destination,
            max_pixels=args.max_pixels,
        )
        output_row = dict(row)
        output_row["image_path"] = str(destination)
        output_row["mime_type"] = "image/png"
        output_rows.append(output_row)
        materialized.append(
            {
                "task_id": task_id,
                "source_path": str(source_image),
                "source_sha256": _sha256(source_image),
                "source_width": original_size[0],
                "source_height": original_size[1],
                "output_path": str(destination),
                "output_sha256": _sha256(destination),
                "output_width": output_size[0],
                "output_height": output_size[1],
                "output_pixels": output_size[0] * output_size[1],
                "resized": resized,
            }
        )
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_policy": (
            "deterministic image geometry only; questions, answers, references, "
            "metadata, and model outcomes are not read by the resize decision"
        ),
        "resampling": "Pillow LANCZOS with aspect ratio preserved; PNG output",
        "max_pixels": args.max_pixels,
        "tasks": len(output_rows),
        "resized_tasks": sum(item["resized"] for item in materialized),
        "source_tasks": str(source_path),
        "source_tasks_sha256": _sha256(source_path),
        "output_tasks": str(output_path),
        "output_tasks_sha256": _sha256(output_path),
        "images": materialized,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"materialized {len(output_rows)} images ({manifest['resized_tasks']} resized) "
        f"with max_pixels={args.max_pixels}"
    )
    return 0


def _materialize_image(
    source: Path,
    destination: Path,
    *,
    max_pixels: int,
) -> tuple[tuple[int, int], tuple[int, int], bool]:
    with Image.open(source) as handle:
        image = ImageOps.exif_transpose(handle)
        image.load()
    original = (int(image.width), int(image.height))
    resized = original[0] * original[1] > max_pixels
    if resized:
        scale = math.sqrt(max_pixels / (original[0] * original[1]))
        width = max(1, int(math.floor(original[0] * scale)))
        height = max(1, int(math.floor(original[1] * scale)))
        while width * height > max_pixels:
            if width >= height:
                width -= 1
            else:
                height -= 1
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
    if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I", "F"}:
        image = image.convert("RGB")
    image.save(destination, format="PNG", optimize=False)
    return original, (int(image.width), int(image.height)), resized


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("task JSONL rows must be objects")
        rows.append(payload)
    if not rows:
        raise ValueError("task JSONL is empty")
    return rows


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or hashlib.sha256(value.encode()).hexdigest()[:16]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out-tasks", required=True)
    parser.add_argument("--out-images", required=True)
    parser.add_argument("--max-pixels", type=int, required=True)
    parser.add_argument("--manifest", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
