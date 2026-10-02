"""GAIA mini adapter: task loading, manifest selection and deterministic parsing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openagora.types import TaskSpec
from openagora.utils.hash_utils import sha256_file


def load_gaia_mini_tasks(
    path: str,
    *,
    manifest_path: str,
    limit: int | None = None,
) -> list[TaskSpec]:
    """Load GAIA mini tasks using fixed manifest order."""
    rows_by_id = _load_jsonl_as_map(path)
    manifest = load_gaia_manifest(manifest_path)
    _validate_manifest_rows(manifest, rows_by_id, dataset_path=path)

    selected_ids: list[str] = [str(task_id) for task_id in manifest.get("task_ids", [])]

    tasks: list[TaskSpec] = []
    for task_id in selected_ids:
        raw = rows_by_id.get(task_id)
        if raw is None:
            continue
        tasks.append(_to_task_spec(raw))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_gaia_manifest(path: str) -> dict[str, Any]:
    """Load GAIA-40 manifest metadata."""
    payload = json.loads(Path(path).read_text(encoding="utf-8").lstrip("\ufeff"))
    if not isinstance(payload, dict):
        raise ValueError("GAIA manifest must be an object")
    task_ids = payload.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError("GAIA manifest task_ids must be a non-empty list")
    return payload


def _load_jsonl_as_map(path: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip().lstrip("\ufeff")
            if not text:
                continue
            raw = json.loads(text)
            if not isinstance(raw, dict):
                continue
            task_id = str(raw.get("id") or raw.get("task_id") or "")
            if task_id:
                out[task_id] = raw
    return out


def _to_task_spec(raw: dict[str, Any]) -> TaskSpec:
    # Compatibility: supports direct TaskSpec rows and GAIA-style rows.
    if "task_id" in raw and "prompt" in raw:
        task_payload = dict(raw)
        metadata = dict(task_payload.get("metadata", {}))
        if "answer" in task_payload and "reference_answer" not in metadata:
            metadata["reference_answer"] = str(task_payload["answer"])
        if "allowed_tools" in task_payload and "allowed_tools" not in metadata:
            metadata["allowed_tools"] = task_payload["allowed_tools"]
        if "answer_aliases" in task_payload and "answer_aliases" not in metadata:
            metadata["answer_aliases"] = task_payload["answer_aliases"]
        task_payload["metadata"] = metadata
        return TaskSpec.model_validate(task_payload)

    return TaskSpec.model_validate(
        {
            "task_id": str(raw.get("id", "gaia-unknown")),
            "prompt": str(raw.get("question", "")),
            "metadata": {
                "reference_answer": str(raw.get("answer", "")),
                "answer_aliases": raw.get("answer_aliases", []),
                "allowed_tools": raw.get("allowed_tools", ["search", "calculator", "synthesize"]),
                "primary_type": str(raw.get("primary_type", "gaia_mini")),
                "uncertainty": str(raw.get("uncertainty", "high")),
                "complexity": str(raw.get("complexity", "high")),
                "interdependence": str(raw.get("interdependence", "reciprocal")),
            },
        }
    )


def _validate_manifest_rows(
    manifest: dict[str, Any],
    rows_by_id: dict[str, dict[str, Any]],
    *,
    dataset_path: str,
) -> None:
    expected_rows = int(manifest.get("expected_rows", len(manifest.get("task_ids", []))))
    if expected_rows <= 0:
        raise ValueError("GAIA manifest expected_rows must be > 0")

    task_ids = [str(task_id) for task_id in manifest.get("task_ids", [])]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("GAIA manifest contains duplicate task_ids")

    missing = [task_id for task_id in task_ids if task_id not in rows_by_id]
    if missing:
        raise ValueError(f"GAIA dataset missing task_ids from manifest: {missing[:5]}")

    if len(task_ids) != expected_rows:
        raise ValueError("GAIA manifest expected_rows mismatch with task_ids length")

    expected_hash = str(manifest.get("dataset_sha256", "")).strip().lower()
    if expected_hash:
        actual_hash = sha256_file(dataset_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"GAIA dataset hash mismatch: expected {expected_hash}, got {actual_hash}"
            )


def normalize_text(text: str) -> str:
    """Normalize free-form text for rule matching."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\-\+]", "", text)
    return text
