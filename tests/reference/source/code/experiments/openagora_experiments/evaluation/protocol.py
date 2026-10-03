"""Rule-first evaluation protocol for GAIA mini outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RuleEvaluation:
    """Outcome from deterministic rule evaluation."""

    rule_score: float
    rule_verdict: str
    exact_match: bool
    normalized_match: bool
    numeric_match: bool
    alias_match: bool
    conflict: bool
    overlap: float
    error_type: str


@dataclass(frozen=True)
class FinalEvaluation:
    """Final decision after optional judge review."""

    final_score: float
    final_verdict: str
    is_correct: bool
    source: str
    error_type: str


def evaluate_rule(
    *,
    predicted: str,
    expected: str,
    answer_aliases: list[str] | None,
    borderline_low: float,
    borderline_high: float,
) -> RuleEvaluation:
    """Evaluate with exact/normalized/numeric/alias rules."""
    raw_pred = predicted.strip()
    raw_exp = expected.strip()
    norm_pred = _normalize(raw_pred)
    norm_exp = _normalize(raw_exp)

    exact_match = bool(raw_pred and raw_exp and raw_pred == raw_exp)
    normalized_match = bool(norm_pred and norm_exp and norm_pred == norm_exp)
    numeric_match = _numeric_match(norm_pred, norm_exp)
    alias_match = _alias_match(norm_pred, answer_aliases)
    overlap = _token_overlap(norm_pred, norm_exp)
    conflict = _is_conflict(norm_pred, norm_exp, numeric_match, alias_match)

    if exact_match:
        score = 1.0
        error_type = "none"
    elif normalized_match:
        score = 0.9
        error_type = "none"
    elif numeric_match:
        score = 0.85
        error_type = "none"
    elif alias_match:
        score = 0.75
        error_type = "alias_only"
    elif overlap >= 0.6:
        score = 0.6
        error_type = "partial_match"
    elif overlap >= 0.4:
        score = 0.45
        error_type = "weak_partial_match"
    else:
        score = 0.0
        if _has_number(norm_exp) and _has_number(norm_pred):
            error_type = "numeric_mismatch"
        elif not norm_pred:
            error_type = "empty_prediction"
        else:
            error_type = "no_match"

    if conflict:
        score = min(score, 0.6)
        error_type = "rule_conflict"

    verdict = _score_to_verdict(score, borderline_low, borderline_high)
    if verdict == "correct":
        error_type = "none"

    return RuleEvaluation(
        rule_score=score,
        rule_verdict=verdict,
        exact_match=exact_match,
        normalized_match=normalized_match,
        numeric_match=numeric_match,
        alias_match=alias_match,
        conflict=conflict,
        overlap=overlap,
        error_type=error_type,
    )


def should_trigger_judge(*, result: RuleEvaluation, borderline_low: float, borderline_high: float) -> bool:
    """Trigger judge only for ambiguous cases or conflicting rules."""
    return result.conflict or (borderline_low <= result.rule_score < borderline_high)


def verdict_from_score(score: float, *, borderline_high: float) -> str:
    """Map score to final binary verdict."""
    return "correct" if score >= borderline_high else "incorrect"


def _score_to_verdict(score: float, low: float, high: float) -> str:
    if score >= high:
        return "correct"
    if score < low:
        return "incorrect"
    return "borderline"


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\-\+]", "", text)
    return text


def _numeric_match(predicted: str, expected: str) -> bool:
    exp_num = _extract_first_number(expected)
    if exp_num is None:
        return False
    pred_nums = _extract_numbers(predicted)
    return any(abs(num - exp_num) <= 1e-9 for num in pred_nums)


def _has_number(text: str) -> bool:
    return _extract_first_number(text) is not None


def _extract_first_number(text: str) -> float | None:
    numbers = _extract_numbers(text)
    return None if not numbers else numbers[0]


def _extract_numbers(text: str) -> list[float]:
    numbers: list[float] = []
    for match in re.finditer(r"-?\d+(?:\.\d+)?", text):
        try:
            numbers.append(float(match.group(0)))
        except ValueError:
            continue
    return numbers


def _alias_match(predicted: str, answer_aliases: list[str] | None) -> bool:
    if not predicted or not answer_aliases:
        return False
    normalized_aliases = {_normalize(alias) for alias in answer_aliases if alias.strip()}
    return predicted in normalized_aliases


def _token_overlap(predicted: str, expected: str) -> float:
    if not predicted or not expected:
        return 0.0
    p_tokens = set(token for token in predicted.split(" ") if token)
    e_tokens = set(token for token in expected.split(" ") if token)
    if not p_tokens or not e_tokens:
        return 0.0
    return len(p_tokens & e_tokens) / len(e_tokens)


def _is_conflict(predicted: str, expected: str, numeric_match: bool, alias_match: bool) -> bool:
    if not predicted or not expected:
        return False
    if _has_number(predicted) and _has_number(expected) and not numeric_match and alias_match:
        return True
    return False
