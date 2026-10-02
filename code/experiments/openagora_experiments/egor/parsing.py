"""Answer-blind prompts and strict structured-output parsing for EGOR stages."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from openagora.mechanisms.evidence_gap_repair import (
    EvidenceCard,
    EvidenceDemand,
    EvidenceType,
)

from openagora_experiments.egor.values import sequence_value

_EVIDENCE_TYPES = {item.value: item for item in EvidenceType}


def parse_baseline_output(raw: str) -> tuple[dict[str, Any], list[str]]:
    payload, issues = _extract_payload(raw)
    answer = _answer_or_raw(payload, raw)
    demands = _parse_demands(payload.get("demands", []), issues)
    evidence = _parse_evidence(payload.get("evidence", []), producer_id="baseline", issues=issues)
    open_gaps = _parse_types(payload.get("open_gap_types", []), issues, field_name="open_gap_types")
    return (
        {
            "answer": answer,
            "confidence": _probability(payload.get("confidence", 0.0), issues, "confidence"),
            "demands": demands,
            "evidence": evidence,
            "open_gap_types": open_gaps,
        },
        issues,
    )


def parse_verifier_output(raw: str) -> tuple[dict[str, Any], list[str]]:
    """Parse a compact verifier response without raw-text answer fallback."""

    payload, issues = _extract_payload(raw)
    extra_keys = sorted(set(payload) - {"answer", "evidence"})
    if extra_keys:
        issues.append(f"discarded unexpected verifier keys: {extra_keys}")
    answer = str(payload.get("answer", "")).strip()
    evidence = _parse_evidence(
        payload.get("evidence", []),
        producer_id="independent_verifier",
        issues=issues,
    )
    if len(evidence) > 1:
        evidence = evidence[:1]
        issues.append("discarded verifier evidence beyond the first card")
    return ({"answer": answer, "evidence": evidence}, issues)


def parse_verifier_output_v6(raw: str) -> tuple[dict[str, Any], list[str]]:
    """Parse the v6 verifier contract with bounded transport normalization.

    A valid explicit abstention is operationally complete. Invalid inner
    evidence is converted to a safe abstention, while malformed top-level
    transport is not allowed to influence an overwrite decision.
    """

    payload, mode = _extract_verifier_payload_v6(raw)
    issues: list[str] = []
    transport = {
        "mode": mode,
        "operational_complete": False,
        "sanitized_abstention": False,
        "decisive": False,
    }
    if payload is None:
        issues.append("invalid v6 verifier transport; retained baseline")
        return {"answer": "", "evidence": [], "transport": transport}, issues

    keys = set(payload)
    if keys != {"answer", "evidence"}:
        issues.append("v6 verifier schema requires exactly answer and evidence keys")
        return {"answer": "", "evidence": [], "transport": transport}, issues
    if not isinstance(payload.get("answer"), str) or not isinstance(payload.get("evidence"), list):
        issues.append("v6 verifier answer/evidence have invalid top-level types")
        return {"answer": "", "evidence": [], "transport": transport}, issues

    transport["operational_complete"] = True
    answer = str(payload["answer"]).strip()
    raw_evidence = list(payload["evidence"])
    evidence_issues: list[str] = []
    evidence = _parse_evidence(
        raw_evidence,
        producer_id="independent_verifier",
        issues=evidence_issues,
    )
    if len(raw_evidence) > 1:
        evidence_issues.append("v6 verifier emitted more than one evidence card")
    if evidence_issues or len(evidence) != len(raw_evidence):
        issues.extend(evidence_issues)
        issues.append("invalid v6 evidence sanitized to explicit abstention")
        transport["sanitized_abstention"] = True
        return {"answer": "", "evidence": [], "transport": transport}, issues

    if not answer and not evidence:
        transport["sanitized_abstention"] = True
        return {"answer": "", "evidence": [], "transport": transport}, issues
    if not answer or len(evidence) != 1:
        issues.append("incomplete v6 answer/evidence pair sanitized to abstention")
        transport["sanitized_abstention"] = True
        return {"answer": "", "evidence": [], "transport": transport}, issues

    transport["decisive"] = True
    return {"answer": answer, "evidence": evidence, "transport": transport}, issues


def parse_verifier_output_v7(raw: str) -> tuple[dict[str, Any], list[str]]:
    """Extend v6 by removing one proven stray trailing backtick only."""

    parsed, issues = parse_verifier_output_v6(raw)
    transport = parsed.get("transport", {})
    if transport.get("mode") != "invalid":
        return parsed, issues
    text = raw.strip()
    if not text.endswith("`"):
        return parsed, issues
    canonical = text[:-1].rstrip()
    if _load_single_json_object(canonical) is None:
        return parsed, issues
    repaired, repaired_issues = parse_verifier_output_v6(canonical)
    if repaired.get("transport", {}).get("operational_complete"):
        repaired["transport"]["mode"] = "stripped_single_trailing_backtick"
        return repaired, repaired_issues
    return parsed, issues


def parse_answer_only_verifier_output(
    raw: str,
) -> tuple[dict[str, Any], list[str]]:
    """Parse fail-closed answer-only transport for ``sec:blind-support``.

    Sanitized abstentions cannot authorize ``sec:authorization``.
    """

    text = str(raw).strip()
    transport = {
        "mode": "invalid",
        "operational_complete": False,
        "sanitized_abstention": False,
        "decisive": False,
    }

    def safe_abstention(issue: str) -> tuple[dict[str, Any], list[str]]:
        transport.update(
            mode="sanitized_transport_abstention",
            operational_complete=True,
            sanitized_abstention=True,
        )
        return {"answer": "", "evidence": [], "transport": transport}, [issue]

    if not text:
        return safe_abstention("empty v8 answer-only verifier response")
    if text.upper() == "<ABSTAIN>":
        transport.update(
            mode="explicit_abstention",
            operational_complete=True,
            sanitized_abstention=True,
        )
        return {"answer": "", "evidence": [], "transport": transport}, []
    if "\n" in text or "\r" in text:
        return safe_abstention("v8 answer-only verifier emitted more than one line")
    if len(text) > 256:
        return safe_abstention("v8 answer-only verifier exceeded 256 characters")
    if text.startswith("```") or text.endswith("```"):
        return safe_abstention("v8 answer-only verifier emitted markdown")
    transport.update(mode="plain_single_line", operational_complete=True, decisive=True)
    return {"answer": text, "evidence": [], "transport": transport}, []


def _extract_verifier_payload_v6(raw: str) -> tuple[dict[str, Any] | None, str]:
    """Accept exact JSON, one full JSON fence, or one missing outer brace.

    The final-brace repair is accepted only when appending exactly one ``}``
    makes the entire response a single JSON object. Nested-object recovery,
    raw-answer fallback, arbitrary suffix removal, and multi-character repair
    are intentionally forbidden.
    """

    text = raw.strip()
    exact = _load_single_json_object(text)
    if exact is not None:
        return exact, "exact_json"

    fenced = re.fullmatch(
        r"```(?:json)?\s*(\{.*\})\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        payload = _load_single_json_object(fenced.group(1).strip())
        if payload is not None:
            return payload, "single_json_fence"

    if text.startswith("{") and not text.endswith("}") and "```" not in text:
        payload = _load_single_json_object(text + "}")
        if payload is not None:
            return payload, "appended_missing_outer_brace"
    return None, "invalid"


def _load_single_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {str(key): value for key, value in payload.items()}


def parse_specialist_output(
    raw: str,
    *,
    specialist_type: EvidenceType,
    producer_id: str,
) -> tuple[dict[str, Any], list[str]]:
    payload, issues = _extract_payload(raw)
    evidence = _parse_evidence(payload.get("evidence", []), producer_id=producer_id, issues=issues)
    evidence = [item for item in evidence if item["type"] == specialist_type.value]
    if len(evidence) < len(sequence_value(payload.get("evidence", []))):
        issues.append("discarded evidence with invalid or mismatched specialist type")
    remaining = _parse_types(payload.get("remaining_gap_types", []), issues, field_name="remaining_gap_types")
    if not evidence and specialist_type.value not in remaining:
        remaining.append(specialist_type.value)
        issues.append("no valid specialist evidence; assigned gap remains open")
    return (
        {
            "specialist_type": specialist_type.value,
            "candidate_answer": str(payload.get("candidate_answer", "")).strip(),
            "evidence": evidence,
            "remaining_gap_types": remaining,
        },
        issues,
    )


def parse_synthesis_output(raw: str) -> tuple[dict[str, Any], list[str]]:
    payload, issues = _extract_payload(raw)
    return (
        {
            "answer": _answer_or_raw(payload, raw),
            "used_evidence_ids": _strings(payload.get("used_evidence_ids", [])),
            "rejected_evidence_ids": _strings(payload.get("rejected_evidence_ids", [])),
            "remaining_gap_types": _parse_types(
                payload.get("remaining_gap_types", []),
                issues,
                field_name="remaining_gap_types",
            ),
            "changed_baseline": bool(payload.get("changed_baseline", False)),
            "confidence": _probability(payload.get("confidence", 0.0), issues, "confidence"),
        },
        issues,
    )


def parse_scheduler_output(
    raw: str,
    *,
    allowed_template_ids: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    payload, issues = _extract_payload(raw)
    allowed = {"baseline", *(str(value).strip() for value in allowed_template_ids)}
    selected = str(payload.get("selected_template_id", "baseline")).strip()
    if selected not in allowed:
        issues.append(f"unknown scheduler template {selected!r}; fell back to baseline")
        selected = "baseline"
    return (
        {
            "selected_template_id": selected,
            "target_gap_types": _parse_types(
                payload.get("target_gap_types", []),
                issues,
                field_name="target_gap_types",
            ),
            "confidence": _probability(payload.get("confidence", 0.0), issues, "confidence"),
            "rationale": str(payload.get("rationale", "")).strip(),
        },
        issues,
    )


def _extract_payload(raw: str) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        index = candidate.find("{")
        if index < 0:
            continue
        try:
            payload, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return {str(key): value for key, value in payload.items()}, issues
    issues.append("no valid JSON object; used raw text as answer fallback")
    return {}, issues


def _parse_demands(value: Any, issues: list[str]) -> list[dict[str, Any]]:
    demands: list[dict[str, Any]] = []
    for index, item in enumerate(sequence_value(value), start=1):
        if not isinstance(item, Mapping):
            issues.append("discarded non-object demand")
            continue
        requirement = str(item.get("requirement", "")).strip()
        demand_id = str(item.get("demand_id", f"d{index}")).strip() or f"d{index}"
        try:
            demand = EvidenceDemand(
                demand_id=demand_id,
                evidence_type=_type(str(item.get("type", ""))),
                requirement=requirement,
                weight=_probability(item.get("weight", 1.0), issues, "demand.weight"),
            )
        except (KeyError, ValueError) as exc:
            issues.append(f"discarded invalid demand: {exc}")
            continue
        demands.append(
            {
                "demand_id": demand.demand_id,
                "type": demand.evidence_type.value,
                "requirement": demand.requirement,
                "weight": demand.weight,
            }
        )
    return demands


def _parse_evidence(value: Any, *, producer_id: str, issues: list[str]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index, item in enumerate(sequence_value(value), start=1):
        if not isinstance(item, Mapping):
            issues.append("discarded non-object evidence card")
            continue
        local_evidence_id = str(item.get("evidence_id", f"e{index}")).strip() or f"e{index}"
        evidence_id = (
            local_evidence_id
            if local_evidence_id.startswith(f"{producer_id}::")
            else f"{producer_id}::{local_evidence_id}"
        )
        try:
            card = EvidenceCard(
                evidence_id=evidence_id,
                evidence_type=_type(str(item.get("type", ""))),
                claim=str(item.get("claim", "")).strip(),
                support=str(item.get("support", "")).strip(),
                provenance=str(item.get("provenance", "")).strip(),
                producer_id=producer_id,
                confidence=_probability(item.get("confidence", 0.0), issues, "evidence.confidence"),
            )
        except (KeyError, ValueError) as exc:
            issues.append(f"discarded invalid evidence card: {exc}")
            continue
        evidence.append(
            {
                "evidence_id": card.evidence_id,
                "type": card.evidence_type.value,
                "claim": card.claim,
                "support": card.support,
                "provenance": card.provenance,
                "producer_id": card.producer_id,
                "confidence": card.confidence,
            }
        )
    return evidence


def _parse_types(value: Any, issues: list[str], *, field_name: str) -> list[str]:
    parsed: list[str] = []
    for item in sequence_value(value):
        try:
            parsed.append(_type(str(item)).value)
        except KeyError:
            issues.append(f"discarded unknown type in {field_name}: {item}")
    return list(dict.fromkeys(parsed))


def _type(value: str) -> EvidenceType:
    normalized = value.strip().lower()
    if normalized not in _EVIDENCE_TYPES:
        raise KeyError(f"unknown evidence type: {value}")
    return _EVIDENCE_TYPES[normalized]


def _probability(value: Any, issues: list[str], field_name: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        issues.append(f"invalid {field_name}; replaced with 0")
        return 0.0
    if probability < 0.0 or probability > 1.0:
        issues.append(f"out-of-range {field_name}; clipped to [0,1]")
    return min(1.0, max(0.0, probability))


def _answer_or_raw(payload: Mapping[str, Any], raw: str) -> str:
    answer = str(payload.get("answer", "")).strip()
    return answer or raw.strip()


def _strings(value: Any) -> list[str]:
    return [str(item).strip() for item in sequence_value(value) if str(item).strip()]
