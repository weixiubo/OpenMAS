"""Resumable stage runners that keep one local model loaded at a time."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from openagora.mechanisms.evidence_gap_repair import EvidenceType
from openagora_experiments.egor.protocol import (
    PilotTask,
    build_answer_only_verifier_prompt,
    build_baseline_prompt,
    build_scheduler_prompt,
    build_specialist_prompt,
    build_synthesis_prompt,
    build_verifier_prompt,
    parse_answer_only_verifier_output,
    parse_baseline_output,
    parse_scheduler_output,
    parse_specialist_output,
    parse_synthesis_output,
    parse_verifier_output,
    parse_verifier_output_v6,
    parse_verifier_output_v7,
)
from openagora_experiments.egor.runtime_types import ArtifactSpec


class MultimodalProvider(Protocol):
    model_name: str

    def complete_multimodal(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        artifacts: list[ArtifactSpec],
        temperature: float,
        seed: int,
        request_id: str | None = None,
    ) -> str: ...

    def pop_audit_events(self) -> list[dict[str, Any]]: ...


def load_tasks(
    path: Path, *, limit: int = 0, allowed_task_ids: set[str] | None = None
) -> list[PilotTask]:
    tasks: list[PilotTask] = []
    seen: set[tuple[str, str]] = set()
    with Path(path).resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            task = PilotTask.from_json(json.loads(line))
            key = (task.source_id, task.task_id)
            if key in seen:
                raise ValueError(
                    f"duplicate task at line {line_number}: {task.source_id}::{task.task_id}"
                )
            seen.add(key)
            if allowed_task_ids is not None and task.task_id not in allowed_task_ids:
                continue
            tasks.append(task)
            if limit > 0 and len(tasks) >= limit:
                break
    if not tasks:
        raise ValueError("no tasks selected")
    return tasks


def load_successful_stage_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with Path(path).resolve().open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("status", "")) != "ok":
                continue
            key = (str(row.get("source_id", "")), str(row.get("task_id", "")))
            records[key] = row
    return records


def load_latest_stage_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the final append-only attempt per task, including failed attempts."""

    records: dict[tuple[str, str], dict[str, Any]] = {}
    with Path(path).resolve().open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("source_id", "")), str(row.get("task_id", "")))
            if not all(key):
                raise ValueError(f"stage row lacks source_id/task_id: {path}")
            records[key] = row
    return records


def run_baseline_stage(
    *,
    provider: MultimodalProvider,
    tasks: Sequence[PilotTask],
    seed: int,
    task_indices: Mapping[tuple[str, str], int] | None = None,
) -> Iterable[dict[str, Any]]:
    for index, task in enumerate(tasks):
        system_prompt, user_prompt = build_baseline_prompt(task)
        yield _invoke(
            provider=provider,
            task=task,
            stage="baseline",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            seed=_stable_task_seed(
                seed, task=task, fallback_index=index, task_indices=task_indices
            ),
            # "proposal" selects the runtime's frozen 384-token structured-output budget.
            request_id=f"egor::proposal::baseline::{task.source_id}::{task.task_id}",
            parser=parse_baseline_output,
            extra={},
        )


def run_verifier_stage(
    *,
    provider: MultimodalProvider,
    tasks: Sequence[PilotTask],
    seed: int,
    task_indices: Mapping[tuple[str, str], int] | None = None,
) -> Iterable[dict[str, Any]]:
    """Run one independent compact verifier without exposing any draft."""

    for index, task in enumerate(tasks):
        system_prompt, user_prompt = build_verifier_prompt(task)
        yield _invoke(
            provider=provider,
            task=task,
            stage="verifier",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            seed=_stable_task_seed(
                seed, task=task, fallback_index=index, task_indices=task_indices
            ),
            request_id=(
                f"egor::proposal::independent_verifier::{task.source_id}::{task.task_id}"
            ),
            parser=parse_verifier_output,
            extra={"information_boundary": "question_options_image_only"},
        )


def run_verifier_v6_stage(
    *,
    provider: MultimodalProvider,
    tasks: Sequence[PilotTask],
    seed: int,
    task_indices: Mapping[tuple[str, str], int] | None = None,
) -> Iterable[dict[str, Any]]:
    """Run the v6 compact verifier with its frozen transport contract."""

    for index, task in enumerate(tasks):
        system_prompt, user_prompt = build_verifier_prompt(task)
        yield _invoke(
            provider=provider,
            task=task,
            stage="verifier_v6",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            seed=_stable_task_seed(
                seed, task=task, fallback_index=index, task_indices=task_indices
            ),
            request_id=(
                f"egor::proposal::independent_verifier_v6::{task.source_id}::{task.task_id}"
            ),
            parser=parse_verifier_output_v6,
            extra={
                "information_boundary": "question_options_image_only",
                "parser_contract": "v6_bounded_transport_and_safe_abstention",
            },
        )


def run_verifier_v7_stage(
    *,
    provider: MultimodalProvider,
    tasks: Sequence[PilotTask],
    seed: int,
    task_indices: Mapping[tuple[str, str], int] | None = None,
) -> Iterable[dict[str, Any]]:
    """Run the v7 verifier with bounded trailing-delimiter normalization."""

    for index, task in enumerate(tasks):
        system_prompt, user_prompt = build_verifier_prompt(task)
        yield _invoke(
            provider=provider,
            task=task,
            stage="verifier_v7",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            seed=_stable_task_seed(
                seed, task=task, fallback_index=index, task_indices=task_indices
            ),
            request_id=(
                f"egor::proposal::independent_verifier_v7::{task.source_id}::{task.task_id}"
            ),
            parser=parse_verifier_output_v7,
            extra={
                "information_boundary": "question_options_image_only",
                "parser_contract": "v7_bounded_trailing_delimiter_and_safe_abstention",
            },
        )


def run_verifier_v8_stage(
    *,
    provider: MultimodalProvider,
    tasks: Sequence[PilotTask],
    seed: int,
    task_indices: Mapping[tuple[str, str], int] | None = None,
) -> Iterable[dict[str, Any]]:
    """Run the blind resolver stage described in ``sec:blind-support``."""

    for index, task in enumerate(tasks):
        system_prompt, user_prompt = build_answer_only_verifier_prompt(task)
        yield _invoke(
            provider=provider,
            task=task,
            stage="verifier_v8",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            seed=_stable_task_seed(
                seed, task=task, fallback_index=index, task_indices=task_indices
            ),
            request_id=(
                f"egor::answer_only_verifier_v8::{task.source_id}::{task.task_id}"
            ),
            parser=parse_answer_only_verifier_output,
            extra={
                "information_boundary": "question_options_image_only",
                "parser_contract": "v10_answer_only_with_fail_closed_abstention",
                "max_new_tokens_contract": 64,
            },
        )


def run_specialist_stage(
    *,
    provider: MultimodalProvider,
    tasks: Sequence[PilotTask],
    baseline_records: Mapping[tuple[str, str], Mapping[str, Any]],
    specialist_type: EvidenceType,
    specialist_id: str,
    seed: int,
    task_indices: Mapping[tuple[str, str], int] | None = None,
) -> Iterable[dict[str, Any]]:
    for index, task in enumerate(tasks):
        key = (task.source_id, task.task_id)
        baseline = baseline_records.get(key)
        if baseline is None:
            yield _missing_dependency_record(
                task, stage="specialist", dependency="baseline"
            )
            continue
        baseline_parsed = _parsed(baseline)
        system_prompt, user_prompt = build_specialist_prompt(
            task,
            specialist_type=specialist_type,
            baseline=baseline_parsed,
        )
        yield _invoke(
            provider=provider,
            task=task,
            stage="specialist",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            seed=_stable_task_seed(
                seed, task=task, fallback_index=index, task_indices=task_indices
            ),
            request_id=f"egor::proposal::{specialist_id}::{task.source_id}::{task.task_id}",
            parser=lambda raw: parse_specialist_output(
                raw,
                specialist_type=specialist_type,
                producer_id=specialist_id,
            ),
            extra={
                "specialist_id": specialist_id,
                "specialist_type": specialist_type.value,
            },
        )


def run_synthesis_stage(
    *,
    provider: MultimodalProvider,
    tasks: Sequence[PilotTask],
    baseline_records: Mapping[tuple[str, str], Mapping[str, Any]],
    specialist_record_sets: Sequence[Mapping[tuple[str, str], Mapping[str, Any]]],
    candidate_id: str,
    seed: int,
    task_indices: Mapping[tuple[str, str], int] | None = None,
) -> Iterable[dict[str, Any]]:
    if not candidate_id.strip():
        raise ValueError("candidate_id is required")
    for index, task in enumerate(tasks):
        key = (task.source_id, task.task_id)
        baseline = baseline_records.get(key)
        specialists = [records.get(key) for records in specialist_record_sets]
        if baseline is None or any(item is None for item in specialists):
            yield _missing_dependency_record(
                task, stage="synthesis", dependency="baseline_or_specialist"
            )
            continue
        specialist_parsed = [_parsed(item) for item in specialists if item is not None]
        system_prompt, user_prompt = build_synthesis_prompt(
            task,
            baseline=_parsed(baseline),
            specialist_records=specialist_parsed,
        )
        yield _invoke(
            provider=provider,
            task=task,
            stage="synthesis",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            seed=_stable_task_seed(
                seed, task=task, fallback_index=index, task_indices=task_indices
            ),
            request_id=f"egor::finalize::{candidate_id}::{task.source_id}::{task.task_id}",
            parser=parse_synthesis_output,
            extra={"candidate_id": candidate_id},
        )


def run_scheduler_stage(
    *,
    provider: MultimodalProvider,
    tasks: Sequence[PilotTask],
    baseline_records: Mapping[tuple[str, str], Mapping[str, Any]],
    template_catalog: Sequence[Mapping[str, Any]],
    seed: int,
    task_indices: Mapping[tuple[str, str], int] | None = None,
) -> Iterable[dict[str, Any]]:
    """Run a strong-LLM routing baseline on the frozen repair catalog."""

    allowed = tuple(
        str(item.get("template_id", "")).strip() for item in template_catalog
    )
    if (
        not allowed
        or any(not value for value in allowed)
        or len(set(allowed)) != len(allowed)
    ):
        raise ValueError("scheduler template catalog requires unique non-empty IDs")
    catalog_sha256 = hashlib.sha256(
        json.dumps(list(template_catalog), ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    for index, task in enumerate(tasks):
        key = (task.source_id, task.task_id)
        baseline = baseline_records.get(key)
        if baseline is None:
            yield _missing_dependency_record(
                task, stage="scheduler", dependency="baseline"
            )
            continue
        system_prompt, user_prompt = build_scheduler_prompt(
            task,
            baseline=_parsed(baseline),
            template_catalog=template_catalog,
        )
        yield _invoke(
            provider=provider,
            task=task,
            stage="scheduler",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            seed=_stable_task_seed(
                seed, task=task, fallback_index=index, task_indices=task_indices
            ),
            request_id=f"egor::route::llm_scheduler::{task.source_id}::{task.task_id}",
            parser=lambda raw: parse_scheduler_output(
                raw,
                allowed_template_ids=allowed,
            ),
            extra={"template_catalog_sha256": catalog_sha256},
        )


def _stable_task_seed(
    base_seed: int,
    *,
    task: PilotTask,
    fallback_index: int,
    task_indices: Mapping[tuple[str, str], int] | None,
) -> int:
    """Bind a stage seed to the task's index in the original requested set.

    A resumable caller may pass only the unfinished subset.  Using that subset's
    enumerate index would silently change seeds after interruption, so callers
    that filter tasks first should retain and pass the original indices.
    """

    if task_indices is None:
        return base_seed + fallback_index
    key = (task.source_id, task.task_id)
    if key not in task_indices:
        raise KeyError(f"missing original task index for {task.source_id}::{task.task_id}")
    return base_seed + int(task_indices[key])


def _invoke(
    *,
    provider: MultimodalProvider,
    task: PilotTask,
    stage: str,
    system_prompt: str,
    user_prompt: str,
    seed: int,
    request_id: str,
    parser: Any,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "source_id": task.source_id,
        "task_id": task.task_id,
        "model_name": provider.model_name,
        "seed": seed,
        "request_id": request_id,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(user_prompt.encode()).hexdigest(),
        "image_path": task.image_path,
        **extra,
    }
    try:
        raw = provider.complete_multimodal(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            artifacts=[
                ArtifactSpec(
                    artifact_id=f"{task.source_id}-{task.task_id}-image",
                    kind="image",
                    path=task.image_path,
                    mime_type=task.mime_type,
                    metadata={},
                )
            ],
            temperature=0.0,
            seed=seed,
            request_id=request_id,
        )
        if not raw.strip() or raw.strip() in {
            "local:empty_completion",
            "local:mm_empty_completion",
        }:
            raise RuntimeError("provider returned an empty completion")
        parsed, parse_issues = parser(raw)
        events = provider.pop_audit_events()
        return {
            **common,
            "status": "ok",
            "raw_output": raw,
            "parsed": parsed,
            "parse_issues": parse_issues,
            "usage": _summarize_usage(events),
            "audit_events": events,
        }
    except Exception as exc:  # pragma: no cover - exercised with real provider failures
        events = provider.pop_audit_events()
        return {
            **common,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "usage": _summarize_usage(events),
            "audit_events": events,
        }


def _summarize_usage(events: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        "prompt_tokens": sum(
            _float(event.get("prompt_tokens", 0.0)) for event in events
        ),
        "completion_tokens": sum(
            _float(event.get("completion_tokens", 0.0)) for event in events
        ),
        "total_tokens": sum(_float(event.get("total_tokens", 0.0)) for event in events),
        "latency_seconds": sum(_float(event.get("latency_ms", 0.0)) for event in events)
        / 1000.0,
        "attempts": sum(_float(event.get("attempts", 0.0)) for event in events),
    }


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parsed(record: Mapping[str, Any]) -> Mapping[str, Any]:
    parsed = record.get("parsed", {})
    return parsed if isinstance(parsed, Mapping) else {}


def _missing_dependency_record(
    task: PilotTask, *, stage: str, dependency: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": stage,
        "source_id": task.source_id,
        "task_id": task.task_id,
        "status": "error",
        "error_type": "MissingDependency",
        "error": dependency,
    }
