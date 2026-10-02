"""Load and audit complete counterfactual organization outcome matrices."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openagora_experiments.cago.scoring import score_prediction


class MatrixValidationError(ValueError):
    """Raised when row-level evidence is incomplete or internally inconsistent."""


@dataclass(frozen=True, order=True)
class TaskKey:
    """Globally unique task identity without relying on benchmark-local IDs."""

    source_id: str
    task_id: str

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.task_id.strip():
            raise MatrixValidationError("source_id and task_id must be non-empty")

    @property
    def stable_id(self) -> str:
        return f"{self.source_id}::{self.task_id}"


@dataclass(frozen=True)
class TaskRecord:
    """Authoritative task data; only `visible_features` may reach a policy."""

    key: TaskKey
    question: str
    answer: str
    reference_answers: tuple[str, ...]
    metric: str
    question_type: str
    options: tuple[tuple[str, str], ...]
    metadata: Mapping[str, Any]

    @classmethod
    def from_json(cls, *, source_id: str, payload: Mapping[str, Any]) -> TaskRecord:
        task_id = str(payload.get("task_id", "")).strip()
        options: list[tuple[str, str]] = []
        raw_options = payload.get("options", [])
        if isinstance(raw_options, list):
            for item in raw_options:
                if not isinstance(item, Mapping):
                    continue
                option_id = str(item.get("id", "")).strip().upper()
                option_text = str(item.get("text", "")).strip()
                if option_id and option_text:
                    options.append((option_id, option_text))
        raw_references = payload.get("reference_answers", [])
        references = (
            tuple(str(value).strip() for value in raw_references if str(value).strip())
            if isinstance(raw_references, list)
            else ()
        )
        metadata = payload.get("metadata", {})
        return cls(
            key=TaskKey(source_id=source_id, task_id=task_id),
            question=str(payload.get("question", "")),
            answer=str(payload.get("answer", "")),
            reference_answers=references,
            metric=str(payload.get("metric", "accuracy")),
            question_type=str(payload.get("question_type", "")),
            options=tuple(options),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )

    def score(self, prediction: str) -> float:
        return score_prediction(
            metric=self.metric,
            predicted=prediction,
            answer=self.answer,
            reference_answers=self.reference_answers,
            options=tuple({"id": option_id, "text": text} for option_id, text in self.options),
        )


@dataclass(frozen=True)
class OutcomeRecord:
    """One organization's observed outcome for one task."""

    key: TaskKey
    organization_id: str
    predicted_answer: str
    quality: float
    reported_quality: float
    tokens: float
    latency_seconds: float
    cost_usd: float
    source_path: str
    run_id: str = ""


@dataclass(frozen=True)
class MatrixAudit:
    """Machine-checkable evidence about matrix completeness and rescoring."""

    task_count: int
    organization_count: int
    outcome_count: int
    expected_outcome_count: int
    duplicate_rows: int
    missing_cells: tuple[str, ...]
    extra_cells: tuple[str, ...]
    reported_score_disagreements: int
    max_reported_score_difference: float
    source_task_counts: Mapping[str, int]
    organization_row_counts: Mapping[str, int]
    input_sha256: Mapping[str, str]

    @property
    def complete(self) -> bool:
        return (
            self.outcome_count == self.expected_outcome_count
            and self.duplicate_rows == 0
            and not self.missing_cells
            and not self.extra_cells
        )


@dataclass(frozen=True)
class CounterfactualMatrix:
    """Complete paired outcomes over one fixed organization candidate set."""

    tasks: Mapping[TaskKey, TaskRecord]
    outcomes: Mapping[tuple[TaskKey, str], OutcomeRecord]
    organization_ids: tuple[str, ...]
    audit: MatrixAudit

    def __post_init__(self) -> None:
        if not self.audit.complete:
            raise MatrixValidationError("counterfactual matrix is incomplete")

    @property
    def task_keys(self) -> tuple[TaskKey, ...]:
        return tuple(sorted(self.tasks))

    def outcome(self, key: TaskKey, organization_id: str) -> OutcomeRecord:
        try:
            return self.outcomes[(key, organization_id)]
        except KeyError as exc:
            raise MatrixValidationError(f"missing counterfactual cell {key.stable_id}::{organization_id}") from exc


def load_counterfactual_matrix(
    *,
    task_paths: Mapping[str, Path],
    outcome_paths: Mapping[str, Mapping[str, Path]],
    score_tolerance: float = 1e-9,
) -> CounterfactualMatrix:
    """Load tasks and all organizations, rescore uniformly, then require completeness."""

    if not task_paths:
        raise MatrixValidationError("at least one task source is required")
    if not outcome_paths:
        raise MatrixValidationError("at least one organization is required")

    tasks, file_hashes, source_task_counts = _load_task_sources(task_paths)

    organizations = tuple(sorted(outcome_paths))
    outcomes: dict[tuple[TaskKey, str], OutcomeRecord] = {}
    duplicate_rows = 0
    extra_cells: list[str] = []
    score_disagreements = 0
    max_score_difference = 0.0
    organization_row_counts: dict[str, int] = {organization_id: 0 for organization_id in organizations}

    for organization_id, source_paths in sorted(outcome_paths.items()):
        _check_organization_sources(organization_id, task_paths, source_paths)
        for source_id, path in sorted(source_paths.items()):
            resolved = Path(path).resolve()
            file_hashes[str(resolved)] = _sha256(resolved)
            with resolved.open(newline="", encoding="utf-8") as handle:
                for line_number, row in enumerate(csv.DictReader(handle), start=2):
                    key = TaskKey(source_id=source_id, task_id=str(row.get("task_id", "")).strip())
                    task_record = tasks.get(key)
                    if task_record is None:
                        extra_cells.append(f"{key.stable_id}::{organization_id}@{resolved}:{line_number}")
                        continue
                    cell = (key, organization_id)
                    if cell in outcomes:
                        duplicate_rows += 1
                        continue
                    predicted = str(row.get("predicted_answer", ""))
                    rescored = task_record.score(predicted)
                    reported = _as_float(row.get("quality_score", row.get("final_score", 0.0)))
                    difference = abs(reported - rescored)
                    if difference > score_tolerance:
                        score_disagreements += 1
                        max_score_difference = max(max_score_difference, difference)
                    outcomes[cell] = OutcomeRecord(
                        key=key,
                        organization_id=organization_id,
                        predicted_answer=predicted,
                        quality=rescored,
                        reported_quality=reported,
                        tokens=_as_float(row.get("token_consumption", 0.0)),
                        latency_seconds=_as_float(row.get("latency_seconds", 0.0)),
                        cost_usd=_as_float(row.get("cost_usd", 0.0)),
                        source_path=str(resolved),
                        run_id=str(row.get("run_id", "")),
                    )
                    organization_row_counts[organization_id] += 1

    expected_cells = {(key, organization_id) for key in tasks for organization_id in organizations}
    missing_cells = sorted(
        f"{key.stable_id}::{organization_id}" for key, organization_id in expected_cells.difference(outcomes)
    )
    audit = MatrixAudit(
        task_count=len(tasks),
        organization_count=len(organizations),
        outcome_count=len(outcomes),
        expected_outcome_count=len(expected_cells),
        duplicate_rows=duplicate_rows,
        missing_cells=tuple(missing_cells),
        extra_cells=tuple(sorted(extra_cells)),
        reported_score_disagreements=score_disagreements,
        max_reported_score_difference=max_score_difference,
        source_task_counts=dict(sorted(source_task_counts.items())),
        organization_row_counts=dict(sorted(organization_row_counts.items())),
        input_sha256=dict(sorted(file_hashes.items())),
    )
    if not audit.complete:
        preview = [*audit.missing_cells[:3], *audit.extra_cells[:3]]
        raise MatrixValidationError("counterfactual evidence failed completeness audit: " + ", ".join(preview))
    return CounterfactualMatrix(
        tasks=tasks,
        outcomes=outcomes,
        organization_ids=organizations,
        audit=audit,
    )


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_task_sources(
    task_paths: Mapping[str, Path],
) -> tuple[dict[TaskKey, TaskRecord], dict[str, str], dict[str, int]]:
    tasks: dict[TaskKey, TaskRecord] = {}
    file_hashes: dict[str, str] = {}
    source_task_counts: dict[str, int] = {}
    for source_id, path in sorted(task_paths.items()):
        resolved = Path(path).resolve()
        file_hashes[str(resolved)] = _sha256(resolved)
        source_count = 0
        with resolved.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                loaded_task = TaskRecord.from_json(source_id=source_id, payload=payload)
                if loaded_task.key in tasks:
                    raise MatrixValidationError(
                        f"duplicate task {loaded_task.key.stable_id} at {resolved}:{line_number}"
                    )
                tasks[loaded_task.key] = loaded_task
                source_count += 1
        source_task_counts[source_id] = source_count

    return tasks, file_hashes, source_task_counts


def _check_organization_sources(
    organization_id: str, task_paths: Mapping[str, Path], source_paths: Mapping[str, Path]
) -> None:
    missing_sources = sorted(set(task_paths).difference(source_paths))
    extra_sources = sorted(set(source_paths).difference(task_paths))
    if missing_sources or extra_sources:
        raise MatrixValidationError(
            f"organization {organization_id} source mismatch: missing={missing_sources}, extra={extra_sources}"
        )
    return None
