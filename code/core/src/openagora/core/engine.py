"""OpenAGORA engine."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from openagora.config.models import RunConfig
from openagora.core.errors import normalize_error
from openagora.core.interfaces import ModelProvider, Observer, TaskAdapter, ToolAdapter
from openagora.core.observers import CompositeObserver, JsonlObserver, RedactingObserver, StorageObserver
from openagora.mechanisms.caching import CacheLifecycleManager
from openagora.mechanisms.negotiation import NegotiationLoop
from openagora.mechanisms.ot_alignment import FixedPrototypeScorer, OTAlignmentScorer
from openagora.mechanisms.runtime import RuntimeExecutor
from openagora.storage.sqlite_store import SQLiteStore
from openagora.templates.runtime import build_template_runtime
from openagora.types import RunEvent, RunResult, TaskSpec
from openagora.utils.artifacts import build_file_hashes, write_hash_manifest
from openagora.utils.audit import assert_event_integrity
from openagora.utils.repro import build_manifest, write_manifest


class OpenAGORAEngine:
    """Main API for task execution and replay."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        task_adapter: TaskAdapter,
        tools: ToolAdapter,
        store: SQLiteStore,
        observers: list[Observer] | None = None,
    ) -> None:
        self._provider = provider
        self._task_adapter = task_adapter
        self._tools = tools
        self._store = store
        self._extra_observers = observers or []

    def run(self, task: TaskSpec, run_config: RunConfig) -> RunResult:
        """Run one task through full AGORA lifecycle."""
        run_id = str(uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        artifacts_dir = Path(run_config.paths.artifacts_dir).resolve() / run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        events_path = artifacts_dir / "events.jsonl"
        manifest_path = artifacts_dir / "manifest.json"
        result_path = artifacts_dir / "result.json"
        hashes_path = artifacts_dir / "hashes.json"

        lock_candidates = [
            str(Path(run_config.paths.artifacts_dir).resolve().parent / "uv.lock"),
            str(Path(run_config.paths.artifacts_dir).resolve().parent / "pyproject.toml"),
            "uv.lock",
            "pyproject.toml",
        ]
        manifest = build_manifest(
            run_id,
            run_config,
            lock_candidates,
            event_schema_version=run_config.audit.event_schema_version,
        )
        write_manifest(manifest, str(manifest_path))
        self._store.start_run(
            run_id=run_id,
            task_id=task.task_id,
            started_at=started_at,
            config_hash=manifest.config_hash,
            manifest_path=str(manifest_path),
            events_path=str(events_path),
            artifact_hashes_path=str(hashes_path),
        )

        raw_observer = CompositeObserver(
            [JsonlObserver(str(events_path)), StorageObserver(self._store, run_id), *self._extra_observers]
        )
        observer: Observer = RedactingObserver(raw_observer, max_prompt_chars=run_config.audit.max_prompt_chars)
        template_runtime = build_template_runtime(run_config)
        template_runtime.set_observer(observer)

        scorer = self._build_alignment_scorer(run_config)
        negotiation = NegotiationLoop(
            self._store,
            scorer,
            observer,
            provider=self._provider,
            template_runtime=template_runtime,
        )
        runtime = RuntimeExecutor(
            self._task_adapter,
            self._tools,
            scorer,
            observer,
            provider=self._provider,
            template_runtime=template_runtime,
        )
        cache_manager = CacheLifecycleManager(self._store, observer)

        try:
            signature = task.signature if task.signature is not None else self._task_adapter.analyze_task(task)
            if task.signature is None:
                task = task.model_copy(update={"signature": signature})
            observer.on_event(
                RunEvent(
                    event_type="run_started",
                    schema_version=run_config.audit.event_schema_version,
                    payload={
                        "run_id": run_id,
                        "task_id": task.task_id,
                        "provider": self._provider.model_name,
                        "provider_mode": run_config.provider.provider_mode,
                        "template_id": template_runtime.template_id,
                        "template_version": template_runtime.version,
                    },
                )
            )

            phase1 = negotiation.negotiate(task, signature, run_config)
            execution = runtime.execute(
                task=task,
                signature=signature,
                architecture=phase1.architecture,
                cfg=run_config,
            )
            result_payload = self._task_adapter.evaluate_result(
                task=task,
                step_results=execution.step_results,
                architecture=execution.final_architecture,
            )
            quality_score = float(result_payload.get("quality_score", 0.0))
            ot_alignment = scorer.compute(signature, execution.final_architecture)

            cache_manager.write_if_eligible(
                signature=signature,
                architecture=execution.final_architecture,
                performance=quality_score,
                ot_alignment=ot_alignment.score,
                cfg=run_config,
                source="run",
            )
            cache_manager.post_reuse_validation(
                signature=signature,
                reused_cache=phase1.used_cache,
                performance=quality_score,
                cfg=run_config,
            )

            metrics = self._compute_metrics(
                quality_score=quality_score,
                negotiation_rounds=phase1.rounds,
                step_count=len(execution.step_results),
                reconfig_count=execution.reconfig_count,
                cache_used=phase1.used_cache,
                negotiation_enabled=run_config.toggles.enable_negotiation,
            )
            run_result = RunResult(
                run_id=run_id,
                task_id=task.task_id,
                final_architecture=execution.final_architecture,
                result=result_payload,
                ot_alignment=ot_alignment,
                metrics=metrics,
                cache_used=phase1.used_cache,
                negotiation_rounds=phase1.rounds,
                reconfig_count=execution.reconfig_count,
                override_used=phase1.override_used,
                deadlock_breaker=phase1.deadlock_breaker,
                manifest_path=str(manifest_path),
                events_path=str(events_path),
            )
            with result_path.open("w", encoding="utf-8") as fp:
                json.dump(run_result.model_dump(mode="json"), fp, ensure_ascii=False, indent=2)

            self._store.save_metrics(run_id, metrics)
            observer.on_event(
                RunEvent(
                    event_type="run_completed",
                    schema_version=run_config.audit.event_schema_version,
                    payload={
                        "run_id": run_id,
                        "quality_score": quality_score,
                        "ot": ot_alignment.score,
                    },
                )
            )

            integrity = assert_event_integrity(
                events_path=events_path,
                schema_version=run_config.audit.event_schema_version,
            )
            observer.on_event(
                RunEvent(
                    event_type="audit_integrity",
                    schema_version=run_config.audit.event_schema_version,
                    payload={"run_id": run_id, **integrity},
                )
            )

            hashes = build_file_hashes([manifest_path, events_path, result_path])
            write_hash_manifest(hashes_path, hashes)

            self._store.finish_run(
                run_id=run_id,
                ended_at=datetime.now(timezone.utc).isoformat(),
                status="completed",
                result_path=str(result_path),
            )
            return run_result
        except Exception as exc:
            info = normalize_error(exc)
            observer.on_event(
                RunEvent(
                    event_type="run_failed",
                    schema_version=run_config.audit.event_schema_version,
                    payload={
                        "run_id": run_id,
                        "error_type": info.error_type,
                        "error": info.message,
                        "error_category": info.category,
                        "error_code": info.code,
                        "error_retryable": info.retryable,
                    },
                )
            )
            self._store.finish_run(
                run_id=run_id,
                ended_at=datetime.now(timezone.utc).isoformat(),
                status="failed",
                result_path=str(result_path),
            )
            raise

    def _build_alignment_scorer(self, run_config: RunConfig) -> OTAlignmentScorer:
        """Build the scorer used by the runtime.

        Public config remains unchanged; internal experiment variants may attach
        `_ot_mode` to the config object to swap in ablation scorers.
        """
        ot_mode = str(getattr(run_config, "_ot_mode", "default")).strip().lower()
        if ot_mode == "fixed_ss":
            return FixedPrototypeScorer("SS")
        return OTAlignmentScorer()

    def replay(self, run_id: str) -> RunResult:
        """Replay a completed run from persisted artifact only."""
        run = self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"run_id not found: {run_id}")
        result_path = run.get("result_path")
        if not result_path:
            raise KeyError(f"run_id has no result_path: {run_id}")
        with Path(result_path).open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return RunResult.model_validate(payload)

    def _compute_metrics(
        self,
        *,
        quality_score: float,
        negotiation_rounds: int,
        step_count: int,
        reconfig_count: int,
        cache_used: bool,
        negotiation_enabled: bool,
    ) -> dict[str, float]:
        tokens = 700 + 220 * step_count + 180 * negotiation_rounds + 140 * reconfig_count
        if cache_used:
            tokens -= 220
        if not negotiation_enabled:
            tokens -= 150
        tokens = max(150, tokens)
        latency = 4.0 + 2.4 * step_count + 1.3 * negotiation_rounds + 1.8 * reconfig_count
        cost = tokens / 1000.0 * 0.015
        perf_cost = quality_score / max(cost, 1e-6)
        return {
            "quality_score": quality_score,
            "token_consumption": float(tokens),
            "latency_seconds": float(latency),
            "cost_usd": float(cost),
            "perf_cost_ratio": float(perf_cost),
            "reconfig_count": float(reconfig_count),
            "negotiation_rounds": float(negotiation_rounds),
        }
