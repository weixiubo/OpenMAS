"""Persistence and audit lifecycle for a single engine execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from openagora.config.models import RunConfig
from openagora.core.errors import normalize_error
from openagora.core.interfaces import Observer
from openagora.core.observers import CompositeObserver, JsonlObserver, RedactingObserver, StorageObserver
from openagora.storage.sqlite_store import SQLiteStore
from openagora.types import RunEvent, RunResult, TaskSpec
from openagora.utils.artifacts import build_file_hashes, write_hash_manifest
from openagora.utils.audit import assert_event_integrity
from openagora.utils.repro import build_manifest, write_manifest


@dataclass(frozen=True)
class RunRecording:
    run_id: str
    manifest_path: Path
    events_path: Path
    result_path: Path
    hashes_path: Path
    observer: Observer
    store: SQLiteStore
    schema: str

    @classmethod
    def start(cls, task: TaskSpec, cfg: RunConfig, store: SQLiteStore, extra: list[Observer]) -> RunRecording:
        run_id = str(uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        directory = Path(cfg.paths.artifacts_dir).resolve() / run_id
        directory.mkdir(parents=True, exist_ok=True)
        events, manifest_file, result, hashes = (
            directory / name for name in ("events.jsonl", "manifest.json", "result.json", "hashes.json")
        )
        locks = [cfg.provenance.dependency_lock_path] if cfg.provenance.dependency_lock_path else []
        manifest = build_manifest(run_id, cfg, locks, event_schema_version=cfg.audit.event_schema_version)
        write_manifest(manifest, str(manifest_file))
        store.start_run(
            run_id=run_id,
            task_id=task.task_id,
            started_at=started_at,
            config_hash=manifest.config_hash,
            manifest_path=str(manifest_file),
            events_path=str(events),
            artifact_hashes_path=str(hashes),
        )
        observer = RedactingObserver(
            CompositeObserver([JsonlObserver(str(events)), StorageObserver(store, run_id), *extra]),
            max_prompt_chars=cfg.audit.max_prompt_chars,
        )
        return cls(run_id, manifest_file, events, result, hashes, observer, store, cfg.audit.event_schema_version)

    def complete(self, result: RunResult) -> None:
        with self.result_path.open("w", encoding="utf-8") as handle:
            json.dump(result.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
        self.store.save_metrics(self.run_id, result.metrics)
        self.observer.on_event(
            RunEvent(
                event_type="run_completed",
                schema_version=self.schema,
                payload={
                    "run_id": self.run_id,
                    "quality_score": result.metrics["quality_score"],
                    "ot": result.ot_alignment.score,
                },
            )
        )
        integrity = assert_event_integrity(events_path=self.events_path, schema_version=self.schema)
        self.observer.on_event(
            RunEvent(
                event_type="audit_integrity",
                schema_version=self.schema,
                payload={"run_id": self.run_id, **integrity},
            )
        )
        write_hash_manifest(
            self.hashes_path, build_file_hashes([self.manifest_path, self.events_path, self.result_path])
        )
        self.store.finish_run(
            run_id=self.run_id,
            ended_at=datetime.now(timezone.utc).isoformat(),
            status="completed",
            result_path=str(self.result_path),
        )

    def fail(self, exc: Exception) -> None:
        info = normalize_error(exc)
        self.observer.on_event(
            RunEvent(
                event_type="run_failed",
                schema_version=self.schema,
                payload={
                    "run_id": self.run_id,
                    "error_type": info.error_type,
                    "error": info.message,
                    "error_category": info.category,
                    "error_code": info.code,
                    "error_retryable": info.retryable,
                },
            )
        )
        self.store.finish_run(
            run_id=self.run_id,
            ended_at=datetime.now(timezone.utc).isoformat(),
            status="failed",
            result_path=str(self.result_path),
        )
