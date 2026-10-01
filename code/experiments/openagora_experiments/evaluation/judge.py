"""Hybrid LLM judge wrapper with deterministic fallback."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from openagora.config.models import RunConfig
from openagora.providers.openai_provider import OpenAIModelProvider, ProviderInvocationError

from openagora_experiments.evaluation.protocol import FinalEvaluation, RuleEvaluation, should_trigger_judge, verdict_from_score


@dataclass(frozen=True)
class JudgeDecision:
    """Structured decision emitted by LLM judge."""

    verdict: str
    score: float
    rationale: str


class HybridJudge:
    """Rule-first judge with optional LLM review for borderline/conflict samples."""

    def __init__(
        self,
        *,
        mode: str,
        model_name: str,
        provider: OpenAIModelProvider | None,
        borderline_low: float,
        borderline_high: float,
        seed: int,
    ) -> None:
        self._mode = mode
        self._model_name = model_name
        self._provider = provider
        self._borderline_low = borderline_low
        self._borderline_high = borderline_high
        self._seed = seed

    def evaluate(
        self,
        *,
        task_id: str,
        variant: str,
        predicted: str,
        expected: str,
        rule_eval: RuleEvaluation,
    ) -> tuple[FinalEvaluation, dict[str, Any] | None]:
        """Return final evaluation and optional judge audit record."""
        triggered = should_trigger_judge(
            result=rule_eval,
            borderline_low=self._borderline_low,
            borderline_high=self._borderline_high,
        )

        # Rule mode has no judge audit by design.
        if self._mode == "rule":
            return _from_rule(rule_eval, self._borderline_high), None

        audit: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "variant": variant,
            "mode_requested": self._mode,
            "mode_effective": "hybrid",
            "judge_model": self._model_name,
            "triggered": triggered,
            "rule_score": rule_eval.rule_score,
            "rule_verdict": rule_eval.rule_verdict,
            "rule_conflict": rule_eval.conflict,
            "request_id": f"judge-{task_id}-{variant}",
        }

        if not triggered:
            final = _from_rule(rule_eval, self._borderline_high)
            audit["status"] = "skipped_not_borderline"
            audit["final_source"] = final.source
            audit["final_verdict"] = final.final_verdict
            audit["final_score"] = final.final_score
            return final, audit

        if self._provider is None:
            final = _from_rule(rule_eval, self._borderline_high)
            audit["status"] = "skipped_no_api_key"
            audit["mode_effective"] = "rule"
            audit["final_source"] = final.source
            audit["final_verdict"] = final.final_verdict
            audit["final_score"] = final.final_score
            return final, audit

        system_prompt = (
            "You are a strict evaluator. Return only JSON with keys: verdict, score, rationale. "
            "verdict must be one of correct/incorrect. score must be float in [0,1]."
        )
        user_prompt = (
            f"Task ID: {task_id}\n"
            f"Expected answer: {expected}\n"
            f"Predicted answer: {predicted}\n"
            f"Rule score: {rule_eval.rule_score}\n"
            "Decide correctness and confidence score."
        )

        try:
            response = self._provider.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                seed=self._seed,
                request_id=str(audit["request_id"]),
            )
            audit["response_preview"] = response[:400]
            decision = _parse_judge_response(response)
            final = FinalEvaluation(
                final_score=decision.score,
                final_verdict=decision.verdict,
                is_correct=decision.verdict == "correct",
                source="judge",
                error_type="none" if decision.verdict == "correct" else rule_eval.error_type,
            )
            audit["status"] = "ok"
            audit["parsed_verdict"] = decision.verdict
            audit["parsed_score"] = decision.score
            audit["final_source"] = final.source
            audit["final_verdict"] = final.final_verdict
            audit["final_score"] = final.final_score
            return final, audit
        except (ProviderInvocationError, ValueError, KeyError, json.JSONDecodeError) as exc:
            final = _from_rule(rule_eval, self._borderline_high)
            audit["status"] = "fallback_rule"
            audit["error_type"] = exc.__class__.__name__
            audit["error"] = str(exc)
            audit["final_source"] = final.source
            audit["final_verdict"] = final.final_verdict
            audit["final_score"] = final.final_score
            return final, audit


def build_hybrid_judge(cfg: RunConfig, *, seed: int) -> HybridJudge:
    """Build judge from runtime config; auto-downgrade to rule when API key is absent."""
    mode = cfg.eval.judge_mode
    provider: OpenAIModelProvider | None = None
    if mode == "hybrid" and os.getenv("OPENAI_API_KEY"):
        provider = OpenAIModelProvider(
            cfg.eval.judge_model,
            timeout_s=cfg.provider.timeout_s,
            max_retries=cfg.provider.max_retries,
            retry_backoff_base=cfg.provider.retry_backoff_base,
            max_concurrency=cfg.provider.max_concurrency,
            enable_idempotency_key=cfg.provider.enable_idempotency_key,
        )
    return HybridJudge(
        mode=mode,
        model_name=cfg.eval.judge_model,
        provider=provider,
        borderline_low=cfg.eval.borderline_low,
        borderline_high=cfg.eval.borderline_high,
        seed=seed,
    )


def _from_rule(rule_eval: RuleEvaluation, borderline_high: float) -> FinalEvaluation:
    score = rule_eval.rule_score
    verdict = verdict_from_score(score, borderline_high=borderline_high)
    return FinalEvaluation(
        final_score=score,
        final_verdict=verdict,
        is_correct=verdict == "correct",
        source="rule",
        error_type="none" if verdict == "correct" else rule_eval.error_type,
    )


def _parse_judge_response(text: str) -> JudgeDecision:
    json_blob = _extract_json_object(text)
    payload = json.loads(json_blob)
    verdict = str(payload["verdict"]).strip().lower()
    if verdict not in {"correct", "incorrect"}:
        raise ValueError("Judge verdict must be correct|incorrect")
    score = float(payload["score"])
    if score < 0.0 or score > 1.0:
        raise ValueError("Judge score out of range")
    rationale = str(payload.get("rationale", ""))
    return JudgeDecision(verdict=verdict, score=score, rationale=rationale)


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match is None:
        raise ValueError("No JSON object in judge response")
    return match.group(0)
