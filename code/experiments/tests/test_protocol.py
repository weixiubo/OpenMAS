from __future__ import annotations

from openagora_experiments.evaluation.protocol import evaluate_rule, should_trigger_judge


def test_rule_exact_numeric_alias_paths() -> None:
    exact = evaluate_rule(
        predicted="42",
        expected="42",
        answer_aliases=None,
        borderline_low=0.4,
        borderline_high=0.8,
    )
    assert exact.rule_verdict == "correct"
    assert exact.rule_score == 1.0

    numeric = evaluate_rule(
        predicted="answer is 42",
        expected="42",
        answer_aliases=None,
        borderline_low=0.4,
        borderline_high=0.8,
    )
    assert numeric.numeric_match
    assert numeric.rule_score >= 0.8

    alias = evaluate_rule(
        predicted="forty two",
        expected="42",
        answer_aliases=["forty two"],
        borderline_low=0.4,
        borderline_high=0.8,
    )
    assert alias.alias_match
    assert should_trigger_judge(result=alias, borderline_low=0.4, borderline_high=0.8)
