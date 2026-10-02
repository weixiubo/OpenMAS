"""Outcome-blind overwrite authorization shared by EGOR v9 and v10."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openagora_experiments.egor.evidence import _candidate_answer
from openagora_experiments.egor.verification import (
    VerifiedOverwriteDecision,
    answers_agree,
    canonical_answer,
    valid_evidence_cards,
)


def decide_supported_blind_resolver_overwrite(
    *,
    baseline_answer: Any,
    resolver: Mapping[str, Any],
    specialists: Mapping[str, Mapping[str, Any]],
    metric: str,
    options: Sequence[Mapping[str, str]] = (),
    minimum_specialist_corroborators: int = 1,
) -> VerifiedOverwriteDecision:
    """Release the blind resolver answer only with specialist-held evidence.

    The resolver receives only the task input. It never receives the baseline,
    specialist answers, a reference answer, a score, or a realized outcome.
    With two required corroborators, this implements ``sec:authorization`` and
    ``eq:overwrite-authorization``; generated-answer comparison follows
    ``eq:pairwise-agreement``.
    """

    if minimum_specialist_corroborators < 1:
        raise ValueError("at least one specialist corroborator is required")
    transport = resolver.get("transport", {})
    operational = isinstance(transport, Mapping) and bool(transport.get("operational_complete"))
    if not operational:
        return _decision(False, "blind_resolver_incomplete")

    resolver_answer = _candidate_answer(resolver)
    if not canonical_answer(resolver_answer):
        return _decision(False, "blind_resolver_abstained")
    if answers_agree(baseline_answer, resolver_answer, metric=metric, options=options):
        return _decision(False, "no_baseline_overwrite")

    corroborators = tuple(
        sorted(
            specialist_id
            for specialist_id, record in specialists.items()
            if valid_evidence_cards(record)
            and answers_agree(
                _candidate_answer(record),
                resolver_answer,
                metric=metric,
                options=options,
            )
        )
    )
    if len(corroborators) < minimum_specialist_corroborators:
        return VerifiedOverwriteDecision(
            accept_overwrite=False,
            reason="no_heterogeneous_corroborator",
            corroborating_specialists=corroborators,
            verifier_evidence_count=0,
        )
    return VerifiedOverwriteDecision(
        accept_overwrite=True,
        reason="supported_blind_resolver_overwrite",
        corroborating_specialists=corroborators,
        verifier_evidence_count=0,
    )


def _decision(accept: bool, reason: str) -> VerifiedOverwriteDecision:
    return VerifiedOverwriteDecision(
        accept_overwrite=accept,
        reason=reason,
        corroborating_specialists=(),
        verifier_evidence_count=0,
    )
