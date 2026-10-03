"""Gold-free answer comparison and structural evidence validation.

These functions accept generated answers and provenance-bearing evidence only;
neither a reference answer nor an observed experiment outcome is an input.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from openagora_experiments.cago.scoring import score_prediction

_ANSWER_PREFIX = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+)?(?:answer|output|prediction|result)"
    r"(?:\s+produced)?\s*(?::|is\b)\s*",
    flags=re.IGNORECASE,
)


def answers_agree(
    left: Any,
    right: Any,
    *,
    metric: str,
    options: Sequence[Mapping[str, str]] = (),
    threshold: float = 0.95,
) -> bool:
    """Compare generated answers without gold (eq:pairwise-agreement)."""

    first = canonical_answer(left)
    second = canonical_answer(right)
    if not first or not second:
        return False
    if first == second:
        return True
    forward = score_prediction(
        metric=metric,
        predicted=first,
        answer=second,
        reference_answers=(second,),
        options=options,
    )
    reverse = score_prediction(
        metric=metric,
        predicted=second,
        answer=first,
        reference_answers=(first,),
        options=options,
    )
    return max(forward, reverse) >= threshold


def canonical_answer(value: Any) -> str:
    """Remove common wrappers while preserving answer-bearing content."""

    text = str(value).strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, Mapping):
        text = str(payload.get("answer", "")).strip()
    text = _ANSWER_PREFIX.sub("", text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return " ".join(text.lower().split()).strip(" .")


def valid_evidence_cards(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return structurally valid cards for ``sec:blind-support``.

    Card validity is one conjunct in ``eq:overwrite-authorization``.
    """

    raw = record.get("evidence", [])
    if not isinstance(raw, list):
        return ()
    return tuple(
        card
        for card in raw
        if isinstance(card, Mapping)
        and str(card.get("evidence_id", "")).strip()
        and str(card.get("claim", "")).strip()
        and str(card.get("support", "")).strip()
        and str(card.get("provenance", "")).strip()
    )


def _corroborating_specialists(
    *,
    specialists: Mapping[str, Mapping[str, Any]],
    core_answer: Any,
    metric: str,
    options: Sequence[Mapping[str, str]],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            specialist_id
            for specialist_id, record in specialists.items()
            if valid_evidence_cards(record)
            and answers_agree(
                _candidate_answer(record),
                core_answer,
                metric=metric,
                options=options,
            )
        )
    )


def _candidate_answer(record: Mapping[str, Any]) -> str:
    return str(record.get("candidate_answer", record.get("answer", ""))).strip()
