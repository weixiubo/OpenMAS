"""Runtime execution and reconfiguration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openagora.config.models import RunConfig
from openagora.core.interfaces import ModelProvider, Observer, TaskAdapter, ToolAdapter
from openagora.mechanisms.ot_alignment import OTAlignmentScorer
from openagora.providers.audit import emit_provider_audit
from openagora.templates.runtime import TemplateRuntime, render_json
from openagora.types import ArchitectureSpec, ProgressEvaluation, RunEvent, TaskSignature, TaskSpec


@dataclass(frozen=True)
class RuntimeExecutionOutcome:
    """Phase-2/3 execution result."""

    final_architecture: ArchitectureSpec
    step_results: list[dict[str, Any]]
    reconfig_count: int


class RuntimeExecutor:
    """Execution loop with monitoring and bounded runtime reconfiguration."""

    def __init__(
        self,
        task_adapter: TaskAdapter,
        tools: ToolAdapter,
        scorer: OTAlignmentScorer,
        observer: Observer,
        *,
        provider: ModelProvider | None = None,
        template_runtime: TemplateRuntime | None = None,
    ) -> None:
        self._task_adapter = task_adapter
        self._tools = tools
        self._scorer = scorer
        self._observer = observer
        self._provider = provider
        self._template_runtime = template_runtime

    def execute(
        self,
        *,
        task: TaskSpec,
        signature: TaskSignature,
        architecture: ArchitectureSpec,
        cfg: RunConfig,
    ) -> RuntimeExecutionOutcome:
        """Run monitored execution and trigger runtime reconfiguration when needed."""
        current = architecture
        reconfig_count = 0
        steps_since_reconfig = cfg.thresholds.cooldown_steps
        always_on_rt = bool(getattr(cfg, "_always_on_rt", False))
        all_step_results: list[dict[str, Any]] = []
        for step_idx in range(cfg.runtime.max_steps):
            subtasks = self._task_adapter.decompose_task(task, current)
            subtask = subtasks[step_idx % len(subtasks)]
            step_result = self._task_adapter.execute_subtask(
                task=task,
                subtask=subtask,
                architecture=current,
                tools=self._tools,
                seed=cfg.provider.seed + step_idx,
            )
            all_step_results.append(step_result)
            steps_since_reconfig += 1
            evaluation = self.evaluate_progress(
                signature=signature,
                architecture=current,
                step_results=all_step_results,
            )
            self._emit_feedback_summary(
                task=task,
                step_results=all_step_results,
                cfg=cfg,
                step_idx=step_idx,
            )
            self._observer.on_event(
                RunEvent(
                    event_type="runtime_step",
                    payload={
                        "step_index": step_idx,
                        "status": evaluation.status,
                        "diagnosis": evaluation.diagnosis,
                    },
                )
            )
            should_reconfig = evaluation.status == "architecture_mismatch" or always_on_rt
            if (
                should_reconfig
                and cfg.toggles.enable_runtime_reconfig
                and reconfig_count < cfg.thresholds.max_reconfigs
                and steps_since_reconfig >= cfg.thresholds.cooldown_steps
            ):
                current = self.runtime_reconfig(
                    task=task,
                    signature=signature,
                    old_architecture=current,
                    max_rounds=cfg.thresholds.max_reconfig_rounds,
                    worker_count=cfg.runtime.worker_count,
                    cfg=cfg,
                )
                reconfig_count += 1
                steps_since_reconfig = 0
                self._observer.on_event(
                    RunEvent(
                        event_type="runtime_reconfig",
                        payload={
                            "step_index": step_idx,
                            "reconfig_count": reconfig_count,
                            "new_config": self._scorer.classify_architecture(current),
                            "trigger": (
                                "always_on_rt"
                                if always_on_rt and evaluation.status != "architecture_mismatch"
                                else evaluation.status
                            ),
                        },
                    )
                )
        return RuntimeExecutionOutcome(
            final_architecture=current,
            step_results=all_step_results,
            reconfig_count=reconfig_count,
        )

    def evaluate_progress(
        self,
        *,
        signature: TaskSignature,
        architecture: ArchitectureSpec,
        step_results: list[dict[str, Any]],
    ) -> ProgressEvaluation:
        """Evaluate whether current architecture is still adequate."""
        alignment = self._scorer.compute(signature, architecture)
        if alignment.score >= 0.8:
            return ProgressEvaluation(status="on_track", diagnosis={"ot_alignment": alignment.score})
        if alignment.score >= 0.6:
            return ProgressEvaluation(
                status="minor_issue",
                diagnosis={"ot_alignment": alignment.score},
                recommended_action="adjust_subtask_assignment",
            )
        recent_steps = step_results[-2:]
        return ProgressEvaluation(
            status="architecture_mismatch",
            diagnosis={
                "ot_alignment": alignment.score,
                "recent_steps": len(recent_steps),
                "issue": "coordination mismatch",
            },
            recommended_action="runtime_reconfig",
        )

    def runtime_reconfig(
        self,
        *,
        task: TaskSpec,
        signature: TaskSignature,
        old_architecture: ArchitectureSpec,
        max_rounds: int,
        worker_count: int,
        cfg: RunConfig,
    ) -> ArchitectureSpec:
        """Construct an incrementally adjusted architecture."""
        del max_rounds
        predicted = self._scorer.predict_config(signature)
        candidate = self._scorer.prototype_architecture(predicted, worker_count)
        if self._provider is None or self._template_runtime is None:
            return candidate
        rendered = self._template_runtime.render(
            "override",
            {
                "task_id": task.task_id,
                "task_prompt": task.prompt,
                "signature_json": signature.model_dump_json(),
                "architecture_json": old_architecture.model_dump_json(),
                "feedback_json": render_json([]),
                "persona": "runtime_reconfig_controller",
            },
        )
        suggestion = self._provider.complete(
            system_prompt=rendered.system_prompt,
            user_prompt=rendered.user_prompt,
            temperature=cfg.provider.temperature,
            seed=cfg.provider.seed + 77,
            request_id=f"{task.task_id}-runtime-reconfig",
        )
        emit_provider_audit(self._provider, self._observer, stage="runtime_reconfig")
        candidate_copy = candidate.model_copy(deep=True)
        candidate_copy.rationale = f"{candidate.rationale} | runtime:{suggestion[:120]}"
        return candidate_copy

    def _emit_feedback_summary(
        self,
        *,
        task: TaskSpec,
        step_results: list[dict[str, Any]],
        cfg: RunConfig,
        step_idx: int,
    ) -> None:
        if self._provider is None or self._template_runtime is None:
            return
        rendered = self._template_runtime.render(
            "feedback_summary",
            {
                "task_id": task.task_id,
                "feedback_json": render_json(step_results[-2:]),
                "persona": "feedback_summarizer",
            },
        )
        summary = self._provider.complete(
            system_prompt=rendered.system_prompt,
            user_prompt=rendered.user_prompt,
            temperature=cfg.provider.temperature,
            seed=cfg.provider.seed + step_idx + 13,
            request_id=f"{task.task_id}-feedback-summary-{step_idx}",
        )
        emit_provider_audit(self._provider, self._observer, stage="runtime_feedback")
        self._observer.on_event(
            RunEvent(
                event_type="feedback_summary",
                payload={"step_index": step_idx, "summary": summary[:180]},
            )
        )
