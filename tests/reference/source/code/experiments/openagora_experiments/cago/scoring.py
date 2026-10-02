"""One common scorer for every organization in a counterfactual matrix."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


_ANSWER_PREFIX = re.compile(
    r"^\s*(?:final\s+answer|answer|prediction|result)\s*(?::|is\b)\s*",
    flags=re.IGNORECASE,
)
_NUMERIC = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_NUMERIC_WITH_PERCENT = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")


def score_prediction(
    *,
    metric: str,
    predicted: str,
    answer: str,
    reference_answers: Sequence[str] = (),
    options: Sequence[Mapping[str, str]] = (),
) -> float:
    """Score one prediction with a source-independent implementation.

    All organizations in a matrix pass through this function. In particular,
    ANLS takes the maximum over every official reference instead of silently
    scoring only the first reference.
    """

    references = _unique_non_empty([answer, *reference_answers])
    metric_name = metric.strip().lower()
    if metric_name == "anls":
        return max(
            (_anls(predicted, reference) for reference in references), default=0.0
        )
    if metric_name == "choice_accuracy":
        return _choice_accuracy(predicted, references=references, options=options)
    if metric_name == "relaxed_accuracy":
        return max(
            (_relaxed_accuracy(predicted, reference) for reference in references),
            default=0.0,
        )
    if metric_name == "multi_ref_accuracy":
        return _multi_reference_accuracy(predicted, references)
    if metric_name == "ocrbench_v2_vqa":
        return max(
            (_ocrbench_v2_vqa(predicted, reference) for reference in references),
            default=0.0,
        )
    return _multi_reference_accuracy(predicted, references)


def _multi_reference_accuracy(predicted: str, references: Sequence[str]) -> float:
    normalized_predicted = _normalize(predicted)
    if not normalized_predicted:
        return 0.0
    leading_number = _leading_number(predicted)
    for reference in references:
        normalized_reference = _normalize(reference)
        if normalized_predicted == normalized_reference:
            return 1.0
        if _is_numeric(normalized_reference) and leading_number is not None:
            if _normalize_number(leading_number) == normalized_reference:
                return 1.0
        if normalized_reference in {"yes", "no", "true", "false", "on", "off"}:
            if normalized_predicted.startswith(normalized_reference):
                return 1.0
    return 0.0


def _relaxed_accuracy(
    predicted: str, reference: str, *, tolerance: float = 0.05
) -> float:
    """ChartQA-style relaxed accuracy with relative tolerance for numbers."""

    predicted_number = _to_float_or_embedded(_strip_wrapper(predicted))
    reference_number = _to_float_or_embedded(_strip_wrapper(reference))
    if predicted_number is not None and reference_number is not None:
        if reference_number == 0.0:
            return float(predicted_number == reference_number)
        relative_error = abs(predicted_number - reference_number) / abs(
            reference_number
        )
        return float(relative_error <= tolerance)
    return float(_normalize(predicted) == _normalize(reference))


def _choice_accuracy(
    predicted: str,
    *,
    references: Sequence[str],
    options: Sequence[Mapping[str, str]],
) -> float:
    if not options:
        return _multi_reference_accuracy(predicted, references)
    predicted_keys = _choice_keys(predicted, options)
    reference_keys: set[str] = set()
    for reference in references:
        reference_keys.update(_choice_keys(reference, options))
    return 1.0 if predicted_keys.intersection(reference_keys) else 0.0


def _choice_keys(text: str, options: Sequence[Mapping[str, str]]) -> set[str]:
    cleaned = _strip_wrapper(text)
    normalized = _normalize(cleaned)
    keys: set[str] = set()
    option_by_id = {
        str(option.get("id", "")).strip().upper(): _normalize(
            str(option.get("text", ""))
        )
        for option in options
        if str(option.get("id", "")).strip()
    }
    for option_id, option_text in option_by_id.items():
        starts_with_id = re.match(
            rf"^\s*[\(\[]?{re.escape(option_id)}[\)\].):-]",
            cleaned,
            re.IGNORECASE,
        )
        names_id = re.search(
            rf"\b(?:answer|option|choice)\s*(?:is\s*|[:=-]\s*){re.escape(option_id)}\b",
            cleaned,
            re.IGNORECASE,
        )
        if starts_with_id or names_id:
            keys.add(option_id)
        if option_text and normalized == option_text:
            keys.add(option_id)
    exact_id = cleaned.strip().strip("()[]{}.:,- ").upper()
    if exact_id in option_by_id:
        keys.add(exact_id)
    return keys


def _anls(predicted: str, reference: str) -> float:
    left = _normalize(predicted)
    right = _normalize(reference)
    if not right:
        return 0.0
    if left == right:
        return 1.0
    distance = _levenshtein(left, right)
    score = 1.0 - distance / max(len(left), len(right), 1)
    return score if score >= 0.5 else 0.0


def _ocrbench_v2_vqa(predicted: str, reference: str) -> float:
    """Match the official OCRBench-v2 BasicVQA evaluation rule."""

    left = str(predicted).lower().strip().replace("\n", " ")
    right = str(reference).lower().strip().replace("\n", " ")
    if not right:
        return 0.0
    if len(right.split()) < 5 and right in left:
        return 1.0
    distance = _levenshtein(left, right)
    score = 1.0 - distance / max(len(left), len(right), 1)
    return score if score >= 0.5 else 0.0


def _normalize(text: str) -> str:
    cleaned = " ".join(_strip_wrapper(str(text)).strip().lower().split())
    return _normalize_number(cleaned) if _is_numeric(cleaned) else cleaned


def _strip_wrapper(text: str) -> str:
    cleaned = str(text).strip()
    cleaned = _ANSWER_PREFIX.sub("", cleaned)
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _is_numeric(text: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text.strip()))


def _normalize_number(text: str) -> str:
    cleaned = text.strip().replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return text
    return (
        str(int(value))
        if value.is_integer()
        else f"{value:.8f}".rstrip("0").rstrip(".")
    )


def _leading_number(text: str) -> str | None:
    match = _NUMERIC.search(_strip_wrapper(text))
    return match.group(0) if match else None


def _to_float(text: str) -> float | None:
    cleaned = text.strip().replace(",", "")
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1].strip()
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value / 100.0 if is_percent else value


def _to_float_or_embedded(text: str) -> float | None:
    direct = _to_float(text)
    if direct is not None:
        return direct
    match = _NUMERIC_WITH_PERCENT.search(text)
    return _to_float(match.group(0)) if match else None


def _unique_non_empty(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return tuple(unique)


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_char in enumerate(right, start=1):
            substitution = 0 if left_char == right_char else 1
            current.append(
                min(
                    current[column_index - 1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + substitution,
                )
            )
        previous = current
    return previous[-1]
