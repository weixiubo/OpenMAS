"""Answer-blind, image-disjoint ChartQA split construction for EGOR."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChartQASelection:
    source_index: int
    task_id: str
    image_name: str
    row: Mapping[str, Any]


def select_chartqa_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_task_ids: set[str],
    excluded_images: set[str],
    development_size: int,
    heldout_size: int,
    salt: str,
) -> tuple[list[ChartQASelection], list[ChartQASelection], dict[str, int]]:
    """Select one question per image using visible identifiers only."""

    if development_size <= 0 or heldout_size <= 0:
        raise ValueError("development and heldout sizes must be positive")
    if not salt.strip():
        raise ValueError("salt must be non-empty")
    by_image: dict[str, list[ChartQASelection]] = defaultdict(list)
    excluded_by_task = 0
    excluded_by_image = 0
    for source_index, row in enumerate(rows):
        image_name = Path(str(row.get("imgname", "")).strip()).name
        question = str(row.get("query", "")).strip()
        task_id = f"chartqa-hf-test-{source_index + 1:05d}"
        if not image_name or not question:
            raise ValueError(f"ChartQA row {source_index} lacks imgname/query")
        if task_id in excluded_task_ids:
            excluded_by_task += 1
            continue
        if image_name in excluded_images:
            excluded_by_image += 1
            continue
        by_image[image_name].append(
            ChartQASelection(
                source_index=source_index,
                task_id=task_id,
                image_name=image_name,
                row=row,
            )
        )

    candidates: list[ChartQASelection] = []
    for image_name in sorted(
        by_image,
        key=lambda value: (_rank(salt, "image", value), value),
    ):
        selected = min(
            by_image[image_name],
            key=lambda item: (
                _rank(salt, "task", f"{item.task_id}\0{item.row.get('query', '')}"),
                item.task_id,
            ),
        )
        candidates.append(selected)

    required = development_size + heldout_size
    if len(candidates) < required:
        raise ValueError(f"only {len(candidates)} eligible images for {required} tasks")
    development = candidates[:development_size]
    heldout = candidates[development_size:required]
    if {item.image_name for item in development}.intersection(
        item.image_name for item in heldout
    ):
        raise AssertionError("development and heldout images overlap")
    return (
        development,
        heldout,
        {
            "source_rows": len(rows),
            "source_unique_images": len(
                {Path(str(row.get("imgname", ""))).name for row in rows}
            ),
            "excluded_task_ids_declared": len(excluded_task_ids),
            "excluded_images_declared": len(excluded_images),
            "rows_excluded_by_task_id": excluded_by_task,
            "rows_excluded_by_image": excluded_by_image,
            "eligible_images": len(candidates),
            "image_overlap_between_splits": 0,
        },
    )


def materialize_chartqa_rows(
    selected: Sequence[ChartQASelection],
    *,
    images_root: Path,
) -> list[dict[str, Any]]:
    """Write selected encoded images and normalize their QA records."""

    images_root.mkdir(parents=True, exist_ok=True)
    normalized: list[dict[str, Any]] = []
    for item in selected:
        answer = str(item.row.get("label", "")).strip()
        image_payload = item.row.get("image")
        if not answer:
            raise ValueError(f"selected ChartQA task {item.task_id} has no label")
        if not isinstance(image_payload, (bytes, bytearray, memoryview)):
            raise ValueError(
                f"selected ChartQA task {item.task_id} has no encoded image"
            )
        encoded = bytes(image_payload)
        image_path = images_root / item.image_name
        if image_path.exists() and image_path.read_bytes() != encoded:
            raise ValueError(f"existing image differs from Arrow payload: {image_path}")
        if not image_path.exists():
            image_path.write_bytes(encoded)
        normalized.append(
            {
                "task_id": item.task_id,
                "benchmark": "ChartQA-EGOR-Fresh",
                "question": str(item.row.get("query", "")).strip(),
                "answer": answer,
                "reference_answers": [answer],
                "metric": "relaxed_accuracy",
                "question_type": "qa",
                "image_path": str(image_path.resolve()),
                "mime_type": "image/png",
                "metadata": {
                    "primary_type": "chartqa",
                    "source_split": "test",
                    "source_subset": str(item.row.get("type", "unknown")),
                    "source_index": item.source_index,
                    "source_image": item.image_name,
                    "uncertainty": "medium",
                    "complexity": "high",
                    "interdependence": "reciprocal",
                    "egor_fresh_split": True,
                },
            }
        )
    return normalized


def load_exclusions(paths: Sequence[Path]) -> tuple[set[str], set[str]]:
    task_ids: set[str] = set()
    images: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = str(row.get("task_id", "")).strip()
                metadata = row.get("metadata", {})
                source_image = (
                    str(metadata.get("source_image", "")).strip()
                    if isinstance(metadata, Mapping)
                    else ""
                )
                image_path = str(row.get("image_path", "")).strip()
                if task_id:
                    task_ids.add(task_id)
                if source_image or image_path:
                    images.add(Path(source_image or image_path).name)
    return task_ids, images


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def identifier_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(str(row.get("task_id", "")) for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def selected_image_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item.get("task_id", ""))):
        path = Path(str(row.get("image_path", "")))
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(salt: str, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{salt}\0{namespace}\0{value}".encode()).hexdigest()
