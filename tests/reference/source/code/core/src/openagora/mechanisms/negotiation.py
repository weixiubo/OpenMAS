"""Negotiation loop implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from openagora.config.models import RunConfig
from openagora.core.interfaces import CacheStore, ModelProvider, Observer
from openagora.mechanisms.ot_alignment import OTAlignmentScorer
from openagora.providers.audit import emit_provider_audit
from openagora.templates.runtime import TemplateRuntime, render_json
from openagora.types import ArchitectureSpec, RunEvent, TaskSignature, TaskSpec, WorkerFeedback


@dataclass(frozen=True)
class NegotiationOutcome:
    """Result of phase-1 negotiation."""

    architecture: ArchitectureSpec
    used_cache: bool
    rounds: int
    override_used: bool
    deadlock_breaker: bool


@dataclass
class _NegotiationContext:
    task: TaskSpec
    cfg: RunConfig
    signature: TaskSignature
    personas: dict[int, str]
    round_idx: int


class NegotiationLoop:
    """Negotiation loop with cache lookup, consensus, override and deadlock breaker."""

    def __init__(
        self,
        store: CacheStore,
        scorer: OTAlignmentScorer,
        observer: Observer,
        *,
        provider: ModelProvider | None = None,
        template_runtime: TemplateRuntime | None = None,
    ) -> None:
        self._store = store
        self._scorer = scorer
        self._observer = observer
        self._provider = provider
        self._template_runtime = template_runtime
        self._active_context: _NegotiationContext | None = None

    def negotiate(self, task: TaskSpec, signature: TaskSignature, cfg: RunConfig) -> NegotiationOutcome:
        """Run phase-1 negotiation and return architecture outcome."""
        thresholds = cfg.thresholds
        if cfg.toggles.enable_cache:
            cached = self._store.lookup(signature, thresholds.cache_hit_threshold, thresholds.type_weight)
            if cached is not None:
                self._observer.on_event(
                    RunEvent(
                        event_type="cache_hit",
                        payload={
                            "task_id": task.task_id,
                            "task_type": signature.primary_type,
                            "version_id": cached.version_id,
                        },
                    )
                )
                return NegotiationOutcome(
                    architecture=cached.architecture,
                    used_cache=True,
                    rounds=0,
                    override_used=False,
                    deadlock_breaker=False,
                )
        personas = self._assign_personas(cfg)
        if not cfg.toggles.enable_negotiation:
            prototype = self._scorer.prototype_architecture(
                self._scorer.predict_config(signature), cfg.runtime.worker_count
            )
            prototype = self._decorate_proposal(
                proposal=prototype,
                task=task,
                signature=signature,
                round_idx=0,
                personas=personas,
                use_override_template=False,
            )
            return NegotiationOutcome(
                architecture=prototype,
                used_cache=False,
                rounds=0,
                override_used=False,
                deadlock_breaker=False,
            )

        proposal = self._scorer.prototype_architecture(
            self._scorer.predict_config(signature), cfg.runtime.worker_count
        )
        proposal = self._decorate_proposal(
            proposal=proposal,
            task=task,
            signature=signature,
            round_idx=0,
            personas=personas,
            use_override_template=False,
        )
        previous_feedback: list[WorkerFeedback] = []
        for round_idx in range(1, cfg.thresholds.max_negotiation_rounds + 1):
            self._active_context = _NegotiationContext(
                task=task,
                cfg=cfg,
                signature=signature,
                personas=personas,
                round_idx=round_idx,
            )
            feedback = self._collect_feedback(proposal, signature)
            self._observer.on_event(
                RunEvent(
                    event_type="negotiation_round",
                    payload={
                        "round": round_idx,
                        "feedback": [item.model_dump(mode="json") for item in feedback],
                    },
                )
            )
            if self.consensus(feedback):
                self._active_context = None
                return NegotiationOutcome(
                    architecture=proposal,
                    used_cache=False,
                    rounds=round_idx,
                    override_used=False,
                    deadlock_breaker=False,
                )
            if self.should_override(
                current_feedback=feedback,
                previous_feedback=previous_feedback,
                round_idx=round_idx,
                max_rounds=cfg.thresholds.max_negotiation_rounds,
                stable_threshold=cfg.thresholds.feedback_stable_threshold,
            ):
                adjusted = self._revise_proposal(
                    proposal=proposal,
                    feedback=feedback,
                    signature=signature,
                    worker_count=cfg.runtime.worker_count,
                    use_override_template=True,
                )
                self._active_context = None
                return NegotiationOutcome(
                    architecture=adjusted,
                    used_cache=False,
                    rounds=round_idx,
                    override_used=True,
                    deadlock_breaker=False,
                )
            proposal = self._revise_proposal(
                proposal=proposal,
                feedback=feedback,
                signature=signature,
                worker_count=cfg.runtime.worker_count,
                use_override_template=False,
            )
            previous_feedback = feedback

        self._active_context = None
        fallback = self._scorer.prototype_architecture("SS", cfg.runtime.worker_count)
        self._observer.on_event(
            RunEvent(event_type="deadlock_breaker", payload={"task_id": task.task_id})
        )
        return NegotiationOutcome(
            architecture=fallback,
            used_cache=False,
            rounds=cfg.thresholds.max_negotiation_rounds,
            override_used=False,
            deadlock_breaker=True,
        )

    def consensus(self, feedback: Sequence[WorkerFeedback]) -> bool:
        """Strict consensus: no explicit disagreement."""
        return all(item.overall_assessment != "disagree" for item in feedback)

    def should_override(
        self,
        *,
        current_feedback: Sequence[WorkerFeedback],
        previous_feedback: Sequence[WorkerFeedback],
        round_idx: int,
        max_rounds: int,
        stable_threshold: float,
    ) -> bool:
        """Override on near-timeout or stabilized feedback."""
        if round_idx >= max_rounds - 1:
            return True
        if not previous_feedback:
            return False
        similarity = self._feedback_similarity(current_feedback, previous_feedback)
        return similarity >= stable_threshold

    def _collect_feedback(
        self, proposal: ArchitectureSpec, signature: TaskSignature
    ) -> list[WorkerFeedback]:
        predicted = self._scorer.predict_config(signature)
        actual = self._scorer.classify_architecture(proposal)
        aligned = predicted == actual
        feedback: list[WorkerFeedback] = []
        for role in proposal.roles:
            assessment: Literal["agree", "conditional_agree", "disagree"]
            if aligned:
                assessment = "agree" if role.worker_id % 2 == 0 else "conditional_agree"
            else:
                assessment = "disagree" if role.worker_id % 3 == 0 else "conditional_agree"
            suggestion = self._render_worker_feedback(
                role_id=role.worker_id,
                proposal=proposal,
                predicted=predicted,
                actual=actual,
            )
            if assessment == "disagree":
                self._emit_minority_report(role_id=role.worker_id, suggestion=suggestion)
            feedback.append(
                WorkerFeedback(
                    worker_id=role.worker_id,
                    overall_assessment=assessment,
                    role_fit="acceptable" if assessment != "disagree" else "poor_fit",
                    concerns=[] if assessment != "disagree" else ["coordination mismatch"],
                    confidence="medium",
                    suggestion=suggestion,
                )
            )
        return feedback

    def _revise_proposal(
        self,
        *,
        proposal: ArchitectureSpec,
        feedback: Sequence[WorkerFeedback],
        signature: TaskSignature,
        worker_count: int,
        use_override_template: bool,
    ) -> ArchitectureSpec:
        disagree_count = sum(item.overall_assessment == "disagree" for item in feedback)
        if disagree_count >= max(1, len(feedback) // 3):
            predicted = self._scorer.predict_config(signature)
            adjusted = self._scorer.prototype_architecture(predicted, worker_count)
            return self._decorate_proposal(
                proposal=adjusted,
                task=self._active_context.task if self._active_context else TaskSpec(task_id="n/a", prompt=""),
                signature=signature,
                round_idx=self._active_context.round_idx if self._active_context else 0,
                personas=self._active_context.personas if self._active_context else {},
                use_override_template=use_override_template,
                feedback=feedback,
            )
        proposal_copy = proposal.model_copy(deep=True)
        proposal_copy.rationale = f"{proposal.rationale} | revised_by_feedback"
        return self._decorate_proposal(
            proposal=proposal_copy,
            task=self._active_context.task if self._active_context else TaskSpec(task_id="n/a", prompt=""),
            signature=signature,
            round_idx=self._active_context.round_idx if self._active_context else 0,
            personas=self._active_context.personas if self._active_context else {},
            use_override_template=use_override_template,
            feedback=feedback,
        )

    def _decorate_proposal(
        self,
        *,
        proposal: ArchitectureSpec,
        task: TaskSpec,
        signature: TaskSignature,
        round_idx: int,
        personas: dict[int, str],
        use_override_template: bool,
        feedback: Sequence[WorkerFeedback] | None = None,
    ) -> ArchitectureSpec:
        if self._provider is None or self._template_runtime is None:
            return proposal
        template_key = "override" if use_override_template else "leader_proposal"
        rendered = self._template_runtime.render(
            template_key,
            {
                "task_id": task.task_id,
                "task_prompt": task.prompt,
                "signature_json": signature.model_dump_json(),
                "architecture_json": proposal.model_dump_json(),
                "round_idx": round_idx,
                "feedback_json": render_json([item.model_dump(mode="json") for item in (feedback or [])]),
                "persona": personas.get(1, "planner"),
            },
        )
        message = self._provider.complete(
            system_prompt=rendered.system_prompt,
            user_prompt=rendered.user_prompt,
            temperature=0.0,
            seed=(self._active_context.cfg.provider.seed if self._active_context else 0) + round_idx,
            request_id=f"{task.task_id}-proposal-{round_idx}",
        )
        emit_provider_audit(self._provider, self._observer, stage="negotiation_proposal")
        proposal_copy = proposal.model_copy(deep=True)
        proposal_copy.rationale = f"{proposal.rationale} | template:{message[:120]}"
        return proposal_copy

    def _render_worker_feedback(
        self,
        *,
        role_id: int,
        proposal: ArchitectureSpec,
        predicted: str,
        actual: str,
    ) -> str:
        context = self._active_context
        if self._provider is None or self._template_runtime is None or context is None:
            return "shift_towards_predicted" if predicted != actual else ""
        persona = context.personas.get(role_id, f"worker_{role_id}")
        rendered = self._template_runtime.render(
            "worker_feedback",
            {
                "task_id": context.task.task_id,
                "task_prompt": context.task.prompt,
                "signature_json": context.signature.model_dump_json(),
                "architecture_json": proposal.model_dump_json(),
                "worker_id": role_id,
                "persona": persona,
            },
        )
        response = self._provider.complete(
            system_prompt=rendered.system_prompt,
            user_prompt=rendered.user_prompt,
            temperature=context.cfg.provider.temperature,
            seed=context.cfg.provider.seed + role_id + context.round_idx,
            request_id=f"{context.task.task_id}-feedback-{context.round_idx}-{role_id}",
        )
        emit_provider_audit(self._provider, self._observer, stage="negotiation_feedback")
        return response[:180]

    def _emit_minority_report(self, *, role_id: int, suggestion: str) -> None:
        context = self._active_context
        if self._provider is None or self._template_runtime is None or context is None:
            return
        persona = context.personas.get(role_id, f"worker_{role_id}")
        rendered = self._template_runtime.render(
            "minority_report",
            {
                "task_id": context.task.task_id,
                "worker_id": role_id,
                "persona": persona,
                "feedback": suggestion,
            },
        )
        report = self._provider.complete(
            system_prompt=rendered.system_prompt,
            user_prompt=rendered.user_prompt,
            temperature=context.cfg.provider.temperature,
            seed=context.cfg.provider.seed + role_id + context.round_idx + 99,
            request_id=f"{context.task.task_id}-minority-{context.round_idx}-{role_id}",
        )
        emit_provider_audit(self._provider, self._observer, stage="minority_report")
        self._observer.on_event(
            RunEvent(
                event_type="minority_report",
                payload={"worker_id": role_id, "report": report[:180]},
            )
        )

    def _assign_personas(self, cfg: RunConfig) -> dict[int, str]:
        if self._template_runtime is None:
            return {}
        return self._template_runtime.assign_personas(cfg=cfg, worker_count=cfg.runtime.worker_count)

    def _feedback_similarity(
        self, current_feedback: Sequence[WorkerFeedback], previous_feedback: Sequence[WorkerFeedback]
    ) -> float:
        if len(current_feedback) != len(previous_feedback) or not current_feedback:
            return 0.0
        matched = 0
        prev_by_worker = {item.worker_id: item for item in previous_feedback}
        for item in current_feedback:
            prev = prev_by_worker.get(item.worker_id)
            if prev is None:
                continue
            if (
                item.overall_assessment == prev.overall_assessment
                and item.role_fit == prev.role_fit
                and item.suggestion == prev.suggestion
            ):
                matched += 1
        return matched / len(current_feedback)
