"""Gold-free independent verification for baseline-overwrite decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openagora_experiments.egor.evidence import (
    _ANSWER_PREFIX as _ANSWER_PREFIX,
)
from openagora_experiments.egor.evidence import (
    _candidate_answer as _candidate_answer,
)
from openagora_experiments.egor.evidence import (
    _corroborating_specialists as _corroborating_specialists,
)
from openagora_experiments.egor.evidence import (
    answers_agree as answers_agree,
)
from openagora_experiments.egor.evidence import (
    canonical_answer as canonical_answer,
)
from openagora_experiments.egor.evidence import (
    valid_evidence_cards as valid_evidence_cards,
)


@dataclass(frozen=True)
class VerifiedOverwriteDecision:
    """One outcome-blind decision about replacing the baseline answer."""

    accept_overwrite: bool
    reason: str
    corroborating_specialists: tuple[str, ...]
    verifier_evidence_count: int


def decide_verified_overwrite(
    *,
    baseline_answer: Any,
    core_answer: Any,
    verifier: Mapping[str, Any],
    specialists: Mapping[str, Mapping[str, Any]],
    metric: str,
    options: Sequence[Mapping[str, str]] = (),
    minimum_specialist_corroborators: int = 1,
) -> VerifiedOverwriteDecision:
    """Require independent evidence before a core answer may overwrite baseline.

    The verifier is an independently executed solver that never receives the
    baseline or core answer.  At least one heterogeneous core specialist must
    also independently support the proposed answer.  The function accepts no
    source ID, reference answer, gold score, or realized outcome.
    """

    if minimum_specialist_corroborators < 1:
        raise ValueError("at least one specialist corroborator is required")
    if not canonical_answer(core_answer):
        return _decision(False, "empty_core_answer")
    if answers_agree(baseline_answer, core_answer, metric=metric, options=options):
        return _decision(False, "no_baseline_overwrite")

    verifier_answer = _candidate_answer(verifier)
    verifier_cards = valid_evidence_cards(verifier)
    if not verifier_answer or not verifier_cards:
        return VerifiedOverwriteDecision(
            accept_overwrite=False,
            reason="verifier_incomplete",
            corroborating_specialists=(),
            verifier_evidence_count=len(verifier_cards),
        )
    if not answers_agree(verifier_answer, core_answer, metric=metric, options=options):
        return VerifiedOverwriteDecision(
            accept_overwrite=False,
            reason="verifier_disagrees",
            corroborating_specialists=(),
            verifier_evidence_count=len(verifier_cards),
        )

    corroborators = _corroborating_specialists(
        specialists=specialists,
        core_answer=core_answer,
        metric=metric,
        options=options,
    )
    if len(corroborators) < minimum_specialist_corroborators:
        return VerifiedOverwriteDecision(
            accept_overwrite=False,
            reason="no_heterogeneous_corroborator",
            corroborating_specialists=corroborators,
            verifier_evidence_count=len(verifier_cards),
        )
    return VerifiedOverwriteDecision(
        accept_overwrite=True,
        reason="independently_verified_overwrite",
        corroborating_specialists=corroborators,
        verifier_evidence_count=len(verifier_cards),
    )


def decide_answer_only_verified_overwrite(
    *,
    baseline_answer: Any,
    core_answer: Any,
    verifier: Mapping[str, Any],
    specialists: Mapping[str, Mapping[str, Any]],
    metric: str,
    options: Sequence[Mapping[str, str]] = (),
    minimum_specialist_corroborators: int = 1,
) -> VerifiedOverwriteDecision:
    """Authorize with a blind short answer and specialist-held evidence.

    V8 separates responsibilities: the candidate-blind verifier contributes
    an independent answer, while structurally valid evidence must come from at
    least one heterogeneous core specialist. The function accepts no source
    identity, reference answer, score, or realized outcome.
    Paper: sec:verified-overwrite, eq:overwrite-authorization,
    eq:verified-overwrite.
    """

    if minimum_specialist_corroborators < 1:
        raise ValueError("at least one specialist corroborator is required")
    if not canonical_answer(core_answer):
        return _decision(False, "empty_core_answer")
    if answers_agree(baseline_answer, core_answer, metric=metric, options=options):
        return _decision(False, "no_baseline_overwrite")

    transport = verifier.get("transport", {})
    operational = isinstance(transport, Mapping) and bool(transport.get("operational_complete"))
    verifier_answer = _candidate_answer(verifier)
    if not operational:
        return _decision(False, "answer_only_verifier_incomplete")
    if not verifier_answer:
        return _decision(False, "answer_only_verifier_abstained")
    if not answers_agree(verifier_answer, core_answer, metric=metric, options=options):
        return _decision(False, "answer_only_verifier_disagrees")

    corroborators = _corroborating_specialists(
        specialists=specialists,
        core_answer=core_answer,
        metric=metric,
        options=options,
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
        reason="answer_only_independently_verified_overwrite",
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
