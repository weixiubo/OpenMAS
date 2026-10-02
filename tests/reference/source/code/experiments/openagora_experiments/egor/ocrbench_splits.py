"""Answer-blind OCRBench-v2 transfer selection and materialization helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASIC_VQA_TYPES = (
    "APP agent en",
    "ASCII art classification en",
    "cognition VQA en",
    "diagram QA en",
    "document classification en",
    "reasoning VQA en",
    "science QA en",
    "text recognition en",
)


CAPABILITY_BY_TYPE = {
    "APP agent en": "knowledge_reasoning",
    "ASCII art classification en": "knowledge_reasoning",
    "cognition VQA en": "visual_text_understanding",
    "diagram QA en": "visual_text_understanding",
    "document classification en": "visual_text_understanding",
    "math QA en": "knowledge_reasoning",
    "reasoning VQA en": "knowledge_reasoning",
    "science QA en": "knowledge_reasoning",
    "text recognition en": "text_recognition",
}


@dataclass(frozen=True)
class OCRBenchCandidate:
    """A source row indexed without using its answer for selection."""

    source_id: int
    question: str
    answers: tuple[str, ...]
    dataset_name: str
    task_type: str
    image_sha256: str
    shard_path: str = ""
    row_index: int = -1

    @property
    def task_id(self) -> str:
        return f"ocrbench-v2-{self.source_id:05d}"


def select_transfer_rows(
    candidates: Sequence[OCRBenchCandidate],
    *,
    excluded_source_ids: set[int],
    excluded_image_hashes: set[str],
    task_types: Sequence[str] = DEFAULT_BASIC_VQA_TYPES,
    per_type: int = 30,
    salt: str = "egor-aaai27-ocrbench-transfer-v1",
) -> tuple[list[OCRBenchCandidate], dict[str, Any]]:
    """Select an image-unique, type-balanced transfer set.

    Candidate ranking uses only visible inputs and stable identifiers. Answers
    are carried through for later scoring but are never supplied to ``_rank``.
    """

    if per_type <= 0:
        raise ValueError("per_type must be positive")
    requested_types = tuple(dict.fromkeys(str(value).strip() for value in task_types))
    return select_transfer_rows_by_type(
        candidates,
        excluded_source_ids=excluded_source_ids,
        excluded_image_hashes=excluded_image_hashes,
        quota_by_type={task_type: per_type for task_type in requested_types},
        salt=salt,
    )


def select_transfer_rows_by_type(
    candidates: Sequence[OCRBenchCandidate],
    *,
    excluded_source_ids: set[int],
    excluded_image_hashes: set[str],
    quota_by_type: Mapping[str, int],
    salt: str = "egor-aaai27-ocrbench-transfer-v1",
) -> tuple[list[OCRBenchCandidate], dict[str, Any]]:
    """Select image-unique rows under frozen per-type quotas.

    The mapping order is part of the deterministic selection contract. Candidate
    answers are carried to the output but never supplied to the rank function.
    """

    quota_items = [
        (str(key).strip(), int(value)) for key, value in quota_by_type.items()
    ]
    requested_types = tuple(key for key, _ in quota_items)
    if (
        not requested_types
        or any(not key or value <= 0 for key, value in quota_items)
        or len(set(requested_types)) != len(requested_types)
    ):
        raise ValueError("task types must be unique and quotas must be positive")
    quotas = dict(quota_items)
    if not salt.strip():
        raise ValueError("a non-empty salt is required")

    seen_source_ids: set[int] = set()
    groups: dict[str, list[OCRBenchCandidate]] = {
        task_type: [] for task_type in requested_types
    }
    excluded_by_source_id = 0
    excluded_by_image = 0
    for candidate in candidates:
        if candidate.source_id in seen_source_ids:
            raise ValueError(f"duplicate OCRBench-v2 source id: {candidate.source_id}")
        seen_source_ids.add(candidate.source_id)
        if candidate.task_type not in groups:
            continue
        if candidate.source_id in excluded_source_ids:
            excluded_by_source_id += 1
            continue
        if candidate.image_sha256 in excluded_image_hashes:
            excluded_by_image += 1
            continue
        groups[candidate.task_type].append(candidate)

    for task_type, rows in groups.items():
        rows.sort(
            key=lambda row: (
                _rank(
                    salt,
                    task_type,
                    row.image_sha256,
                    str(row.source_id),
                    row.dataset_name,
                    row.question,
                ),
                row.source_id,
            )
        )

    selected: list[OCRBenchCandidate] = []
    used_images: set[str] = set()
    selected_by_type = {task_type: 0 for task_type in requested_types}
    cursors = {task_type: 0 for task_type in requested_types}
    # Interleaving prevents an earlier type from monopolizing images that are
    # shared by rows belonging to different official task labels.
    for _ in range(max(quotas.values())):
        for task_type in requested_types:
            if selected_by_type[task_type] >= quotas[task_type]:
                continue
            rows = groups[task_type]
            while cursors[task_type] < len(rows):
                candidate = rows[cursors[task_type]]
                cursors[task_type] += 1
                if candidate.image_sha256 in used_images:
                    continue
                selected.append(candidate)
                used_images.add(candidate.image_sha256)
                selected_by_type[task_type] += 1
                break
            else:
                raise ValueError(
                    f"task type {task_type!r} has fewer than {quotas[task_type]} "
                    "eligible image-unique rows"
                )

    return selected, {
        "source_rows": len(candidates),
        "source_unique_images": len({row.image_sha256 for row in candidates}),
        "requested_task_types": list(requested_types),
        "requested_rows_by_type": quotas,
        "eligible_rows_by_type": {
            task_type: len(groups[task_type]) for task_type in requested_types
        },
        "selected_rows_by_type": selected_by_type,
        "excluded_source_ids_declared": len(excluded_source_ids),
        "excluded_image_hashes_declared": len(excluded_image_hashes),
        "rows_excluded_by_source_id": excluded_by_source_id,
        "rows_excluded_by_image": excluded_by_image,
        "selected_unique_images": len(used_images),
        "selected_duplicate_images": len(selected) - len(used_images),
    }


def normalize_transfer_row(
    candidate: OCRBenchCandidate,
    *,
    image_path: Path,
) -> dict[str, Any]:
    if not candidate.answers:
        raise ValueError(f"selected task {candidate.task_id} has no reference answer")
    if candidate.task_type not in CAPABILITY_BY_TYPE:
        raise ValueError(f"unsupported basic-VQA task type: {candidate.task_type}")
    return {
        "task_id": candidate.task_id,
        "benchmark": "OCRBench-v2-BasicVQA-EGOR-Transfer",
        "question": candidate.question,
        "answer": candidate.answers[0],
        "reference_answers": list(candidate.answers),
        "metric": "ocrbench_v2_vqa",
        "question_type": candidate.task_type,
        "image_path": str(image_path.resolve()),
        "metadata": {
            "primary_type": "ocrbench_v2_basic_vqa",
            "source_split": "test",
            "source_id": candidate.source_id,
            "source_dataset_name": candidate.dataset_name,
            "source_image_sha256": candidate.image_sha256,
            "official_type": candidate.task_type,
            "official_capability": CAPABILITY_BY_TYPE[candidate.task_type],
            "language": "en",
            "uncertainty": "medium",
            "complexity": "high",
            "interdependence": "sequential",
            "egor_zero_adaptation_transfer": True,
        },
    }


def load_exclusions(paths: Sequence[Path]) -> tuple[set[int], set[str]]:
    """Read prior source ids and image hashes from JSON/JSONL artifacts."""

    source_ids: set[int] = set()
    image_hashes: set[str] = set()
    for path in paths:
        for row in _json_rows(path):
            metadata = row.get("metadata", {})
            metadata = metadata if isinstance(metadata, Mapping) else {}
            benchmark = str(row.get("benchmark", "")).lower()
            primary_type = str(metadata.get("primary_type", "")).lower()
            if "ocrbench" in benchmark or "ocrbench" in primary_type or "id" in row:
                raw_id = row.get("id", metadata.get("source_id"))
                parsed_id = _integer(raw_id)
                if parsed_id is not None:
                    source_ids.add(parsed_id)
                task_id = str(row.get("task_id", "")).strip()
                parsed_task_id = _ocrbench_task_source_id(task_id)
                if parsed_task_id is not None:
                    source_ids.add(parsed_task_id)
            recorded_hash = str(metadata.get("source_image_sha256", "")).strip().lower()
            if len(recorded_hash) == 64:
                image_hashes.add(recorded_hash)
            for key in ("image_path", "image"):
                raw_image = row.get(key)
                if not isinstance(raw_image, str) or not raw_image.strip():
                    continue
                image_path = Path(raw_image)
                if not image_path.is_absolute():
                    image_path = path.parent / image_path
                if image_path.is_file():
                    image_hashes.add(sha256(image_path))
    return source_ids, image_hashes


def selected_identifier_hash(rows: Sequence[OCRBenchCandidate]) -> str:
    payload = "\n".join(row.task_id for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def selected_image_hash(rows: Sequence[OCRBenchCandidate]) -> str:
    payload = "\n".join(row.image_sha256 for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _rank(salt: str, *visible_values: str) -> str:
    payload = "\0".join((salt, *visible_values)).encode()
    return hashlib.sha256(payload).hexdigest()


def _json_rows(path: Path) -> list[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        payloads = [json.loads(line) for line in text.splitlines() if line.strip()]
        return [row for row in payloads if isinstance(row, Mapping)]
    payload = json.loads(text)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("data", "rows", "examples", "items"):
            values = payload.get(key)
            if isinstance(values, list):
                return [row for row in values if isinstance(row, Mapping)]
    return []


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ocrbench_task_source_id(task_id: str) -> int | None:
    prefix = "ocrbench-v2-"
    lowered = task_id.lower()
    return _integer(lowered[len(prefix) :]) if lowered.startswith(prefix) else None
