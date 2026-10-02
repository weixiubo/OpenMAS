from __future__ import annotations

from openagora_experiments.evaluation.judge import HybridJudge
from openagora_experiments.evaluation.protocol import RuleEvaluation


class _StubProvider:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, **kwargs):  # type: ignore[no-untyped-def]
        return self._response


def _rule_borderline() -> RuleEvaluation:
    return RuleEvaluation(
        rule_score=0.6,
        rule_verdict="borderline",
        exact_match=False,
        normalized_match=False,
        numeric_match=False,
        alias_match=False,
        conflict=False,
        overlap=0.5,
        error_type="partial_match",
    )


def test_hybrid_judge_uses_llm_when_triggered() -> None:
    judge = HybridJudge(
        mode="hybrid",
        model_name="gpt-4o-mini",
        provider=_StubProvider('{"verdict":"correct","score":0.91,"rationale":"ok"}'),
        borderline_low=0.4,
        borderline_high=0.8,
        seed=2026,
    )
    final, audit = judge.evaluate(
        task_id="t1",
        variant="OpenAGORA_Full",
        predicted="41",
        expected="42",
        rule_eval=_rule_borderline(),
    )
    assert final.source == "judge"
    assert final.is_correct
    assert audit is not None
    assert audit["status"] == "ok"


def test_hybrid_judge_fallback_on_invalid_json() -> None:
    judge = HybridJudge(
        mode="hybrid",
        model_name="gpt-4o-mini",
        provider=_StubProvider("not-json"),
        borderline_low=0.4,
        borderline_high=0.8,
        seed=2026,
    )
    final, audit = judge.evaluate(
        task_id="t2",
        variant="OpenAGORA_Full",
        predicted="41",
        expected="42",
        rule_eval=_rule_borderline(),
    )
    assert final.source == "rule"
    assert not final.is_correct
    assert audit is not None
    assert audit["status"] == "fallback_rule"
