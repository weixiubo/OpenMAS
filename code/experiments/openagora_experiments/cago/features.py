"""Answer-blind, pre-execution task features for CAGO diagnostics."""

from __future__ import annotations

import re

from openagora_experiments.cago.data import TaskRecord

FEATURE_NAMES = (
    "question_chars",
    "question_words",
    "digit_tokens",
    "math_symbols",
    "image_markers",
    "option_count",
    "option_text_chars",
    "is_multiple_choice",
    "has_percent",
    "has_units",
    "ocr_term_hits",
    "chart_term_hits",
    "math_term_hits",
    "spatial_term_hits",
    "yes_no_form",
)

_OCR_TERMS = ("text", "word", "name", "written", "read", "label", "title", "number")
_CHART_TERMS = ("chart", "graph", "plot", "trend", "axis", "table", "figure", "value")
_MATH_TERMS = ("calculate", "equation", "sum", "difference", "ratio", "probability", "volume", "mass")
_SPATIAL_TERMS = ("above", "below", "left", "right", "behind", "front", "near", "between")
_UNIT_PATTERN = re.compile(
    r"(?:\bkg\b|\bkm\b|\bcm\b|\bmm\b|\bmeters?\b|\bseconds?\b|\bdollars?\b|\$)",
    flags=re.IGNORECASE,
)


def visible_feature_vector(task: TaskRecord) -> tuple[float, ...]:
    """Extract policy features without reading answers, references, or scores."""

    question = task.question
    lowered = question.lower()
    words = re.findall(r"[a-z0-9]+", lowered)
    option_text_chars = sum(len(text) for _, text in task.options)
    values = (
        min(len(question), 2000) / 500.0,
        min(len(words), 300) / 100.0,
        min(len(re.findall(r"\d+(?:\.\d+)?", question)), 20) / 10.0,
        min(sum(question.count(symbol) for symbol in "+-*/=<>^"), 20) / 10.0,
        min(lowered.count("<image"), 10) / 5.0,
        min(len(task.options), 20) / 10.0,
        min(option_text_chars, 3000) / 1000.0,
        float(bool(task.options) or task.question_type.strip().lower() == "multiple_choice"),
        float("%" in question or "percent" in lowered or "percentage" in lowered),
        float(bool(_UNIT_PATTERN.search(question))),
        _term_density(words, _OCR_TERMS),
        _term_density(words, _CHART_TERMS),
        _term_density(words, _MATH_TERMS),
        _term_density(words, _SPATIAL_TERMS),
        float(lowered.startswith(("is ", "are ", "does ", "do ", "can ", "was ", "were "))),
    )
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError("feature schema and vector length diverged")
    return values


def _term_density(words: list[str], terms: tuple[str, ...]) -> float:
    hits = sum(1 for word in words if word in terms)
    return min(hits, 10) / 5.0
