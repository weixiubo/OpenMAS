"""OpenAGORA engine."""

from __future__ import annotations

import json
from pathlib import Path

from openagora.config.models import RunConfig
from openagora.core.interfaces import ModelProvider, Observer, TaskAdapter, ToolAdapter
from openagora.core.recording import RunRecording
from openagora.mechanisms.caching import CacheLifecycleManager
from openagora.mechanisms.negotiation import NegotiationLoop
from openagora.mechanisms.ot_alignment import FixedPrototypeScorer, OTAlignmentScorer
from openagora.mechanisms.runtime import RuntimeExecutor
from openagora.storage.sqlite_store import SQLiteStore
from openagora.templates.runtime import build_template_runtime
from openagora.types import RunEvent, RunResult, TaskSpec


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
        """Coordinate negotiation, execution, cache policy and durable recording."""
        recording = RunRecording.start(task, run_config, self._store, self._extra_observers)
        try:
            result = self._execute(task, run_config, recording)
            recording.complete(result)
            return result
        except Exception as exc:
            recording.fail(exc)
            raise

    def _execute(self, task: TaskSpec, cfg: RunConfig, recording: RunRecording) -> RunResult:
        observer = recording.observer
        templates = build_template_runtime(cfg)
        templates.set_observer(observer)
        scorer = self._build_alignment_scorer(cfg)
        negotiation = NegotiationLoop(
            self._store, scorer, observer, provider=self._provider, template_runtime=templates
        )
        runtime = RuntimeExecutor(
            self._task_adapter, self._tools, scorer, observer, provider=self._provider, template_runtime=templates
        )
        signature = task.signature if task.signature is not None else self._task_adapter.analyze_task(task)
        if task.signature is None:
            task = task.model_copy(update={"signature": signature})
        observer.on_event(
            RunEvent(
                event_type="run_started",
                schema_version=cfg.audit.event_schema_version,
                payload={
                    "run_id": recording.run_id,
                    "task_id": task.task_id,
                    "provider": self._provider.model_name,
                    "provider_mode": cfg.provider.provider_mode,
                    "template_id": templates.template_id,
                    "template_version": templates.version,
                },
            )
        )
        phase1 = negotiation.negotiate(task, signature, cfg)
        execution = runtime.execute(task=task, signature=signature, architecture=phase1.architecture, cfg=cfg)
        result_payload = self._task_adapter.evaluate_result(
            task=task, step_results=execution.step_results, architecture=execution.final_architecture
        )
        quality = float(result_payload.get("quality_score", 0.0))
        alignment = scorer.compute(signature, execution.final_architecture)
        cache = CacheLifecycleManager(self._store, observer)
        cache.write_if_eligible(
            signature=signature,
            architecture=execution.final_architecture,
            performance=quality,
            ot_alignment=alignment.score,
            cfg=cfg,
            source="run",
        )
        cache.post_reuse_validation(signature=signature, reused_cache=phase1.used_cache, performance=quality, cfg=cfg)
        metrics = self._compute_metrics(
            quality_score=quality,
            negotiation_rounds=phase1.rounds,
            step_count=len(execution.step_results),
            reconfig_count=execution.reconfig_count,
            cache_used=phase1.used_cache,
            negotiation_enabled=cfg.toggles.enable_negotiation,
        )
        return RunResult(
            run_id=recording.run_id,
            task_id=task.task_id,
            final_architecture=execution.final_architecture,
            result=result_payload,
            ot_alignment=alignment,
            metrics=metrics,
            cache_used=phase1.used_cache,
            negotiation_rounds=phase1.rounds,
            reconfig_count=execution.reconfig_count,
            override_used=phase1.override_used,
            deadlock_breaker=phase1.deadlock_breaker,
            manifest_path=str(recording.manifest_path),
            events_path=str(recording.events_path),
        )

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
