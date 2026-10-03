"""Answer-blind prompts and strict structured-output parsing for EGOR stages."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openagora.mechanisms.evidence_gap_repair import (
    EvidenceCard,
    EvidenceDemand,
    EvidenceType,
)


_EVIDENCE_TYPES = {item.value: item for item in EvidenceType}

_IMAGE_MIME_BY_SUFFIX = {
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class PilotTask:
    """One multimodal task; reference fields never enter a generation prompt."""

    task_id: str
    source_id: str
    question: str
    image_path: str
    mime_type: str
    metric: str
    answer: str
    reference_answers: tuple[str, ...]
    options: tuple[tuple[str, str], ...]
    metadata: Mapping[str, Any]

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> PilotTask:
        image_path = str(payload.get("image_path", "")).strip()
        mime_type = _infer_image_mime_type(image_path)
        raw_references = payload.get("reference_answers", [])
        references = (
            tuple(str(item).strip() for item in raw_references if str(item).strip())
            if isinstance(raw_references, list)
            else ()
        )
        options: list[tuple[str, str]] = []
        raw_options = payload.get("options", [])
        if isinstance(raw_options, list):
            for item in raw_options:
                if not isinstance(item, Mapping):
                    continue
                option_id = str(item.get("id", "")).strip().upper()
                text = str(item.get("text", "")).strip()
                if option_id and text:
                    options.append((option_id, text))
        metadata = payload.get("metadata", {})
        task = cls(
            task_id=str(payload.get("task_id", "")).strip(),
            source_id=str(
                payload.get("benchmark", payload.get("source_id", ""))
            ).strip(),
            question=str(payload.get("question", "")).strip(),
            image_path=image_path,
            mime_type=mime_type,
            metric=str(payload.get("metric", "accuracy")).strip(),
            answer=str(payload.get("answer", "")).strip(),
            reference_answers=references,
            options=tuple(options),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )
        if (
            not task.task_id
            or not task.source_id
            or not task.question
            or not task.image_path
        ):
            raise ValueError(
                "task_id, benchmark/source_id, question, and image_path are required"
            )
        return task

    def visible_question(self) -> str:
        if not self.options:
            return self.question
        rendered_options = "\n".join(
            f"{option_id}. {text}" for option_id, text in self.options
        )
        return f"{self.question}\n\nOptions:\n{rendered_options}"


def _infer_image_mime_type(image_path: str) -> str:
    """Infer an honest image MIME type without assuming every non-JPEG is PNG."""

    path = Path(image_path)
    suffix_type = _IMAGE_MIME_BY_SUFFIX.get(path.suffix.lower())
    if suffix_type is not None:
        return suffix_type
    try:
        header = path.read_bytes()[:16]
    except OSError:
        return "application/octet-stream"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"BM"):
        return "image/bmp"
    if header[:4] in {b"II*\x00", b"MM\x00*"}:
        return "image/tiff"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def build_baseline_prompt(task: PilotTask) -> tuple[str, str]:
    """Ask a strong baseline for an answer and a provenance-bearing evidence audit."""

    system = (
        "You are a strong multimodal solver and a strict evidence auditor. Solve the task, then identify "
        "what evidence the answer requires and which requirements remain unsupported. Never invent a quote, "
        "coordinate, calculation, or source. Return exactly one JSON object and no markdown."
    )
    user = f"""Task:
{task.visible_question()}

Allowed evidence types:
exact_span, visual_relation, numerical_derivation, external_fact, constraint_check, independent_verification

Return this schema:
{{
  "answer": "short final answer",
  "confidence": 0.0,
  "demands": [
    {{"demand_id": "d1", "type": "one allowed type", "requirement": "specific evidence needed", "weight": 1.0}}
  ],
  "evidence": [
    {{"evidence_id": "b1", "type": "one allowed type", "claim": "claim supported", "support": "visible span, relation, or derivation", "provenance": "image region description or tool/trace source", "confidence": 0.0}}
  ],
  "open_gap_types": ["allowed type still unsupported"]
}}

Use confidence values in [0,1]. Keep the final answer concise. If evidence is not visible, leave support empty and mark its type open instead of guessing."""
    return system, user


def build_verifier_prompt(task: PilotTask) -> tuple[str, str]:
    """Ask one answer-blind verifier for a compact, decisive evidence card."""

    system = (
        "You are an independent multimodal verification agent. Solve the visible task "
        "from the question and image only. You are blind to every other agent answer, "
        "draft, and score. Return exactly one compact JSON object and no markdown. "
        "Never invent support or provenance."
    )
    user = f"""Task:
{task.visible_question()}

Return exactly this compact schema:
Allowed evidence types: exact_span, visual_relation, numerical_derivation,
external_fact, constraint_check, independent_verification.
{{
  "answer": "short final answer; for multiple choice use only the option letter",
  "evidence": [
    {{"evidence_id": "v1", "type": "one allowed evidence type", "claim": "at most 20 words", "support": "at most 40 words of decisive visible evidence or derivation", "provenance": "at most 12 words naming the image region or calculation", "confidence": 0.0}}
  ]
}}

Return at most one evidence card. If the answer cannot be independently verified,
return an empty answer and an empty evidence list. Do not output demands, gap types,
analysis, hidden-draft comparisons, or additional keys."""
    return system, user


def build_answer_only_verifier_prompt(task: PilotTask) -> tuple[str, str]:
    """Ask the baseline-blind resolver for one bounded answer (sec:blind-support)."""

    system = (
        "You are an independent multimodal verification agent. Solve the visible "
        "task from the question and image only. You are blind to every other agent "
        "answer, draft, evidence record, and score. Return only the final answer on "
        "one line. Never explain your reasoning and never use JSON or markdown."
    )
    user = f"""Task:
{task.visible_question()}

Return exactly one line containing only the final answer. For multiple choice,
return only the option letter. If the task cannot be solved, return exactly
<ABSTAIN>. Do not add a label, explanation, evidence, punctuation wrapper, or
second line."""
    return system, user


def build_specialist_prompt(
    task: PilotTask,
    *,
    specialist_type: EvidenceType,
    baseline: Mapping[str, Any],
) -> tuple[str, str]:
    """Request independent evidence without exposing the baseline draft or evidence."""

    descriptions = {
        EvidenceType.EXACT_SPAN: "Read and localize the exact answer-bearing text, label, number, or table cell.",
        EvidenceType.VISUAL_RELATION: "Independently inspect spatial, object, diagram, and chart relations.",
        EvidenceType.NUMERICAL_DERIVATION: "Derive the numerical result step by step and expose every operation.",
        EvidenceType.EXTERNAL_FACT: "Find only externally grounded facts with explicit source provenance.",
        EvidenceType.CONSTRAINT_CHECK: "Check options, units, schema, and answer-format constraints.",
        EvidenceType.INDEPENDENT_VERIFICATION: "Independently solve the task and expose evidence that can verify a hidden draft.",
    }
    system = (
        f"You are the {specialist_type.value} specialist in an evidence-repair team. "
        f"{descriptions[specialist_type]} Return exactly one JSON object and no markdown. "
        "You are intentionally blind to the baseline draft and its evidence. Solve independently and do not "
        "fabricate provenance."
    )
    demands = [
        {
            "demand_id": item.get("demand_id", ""),
            "type": item.get("type", ""),
            "requirement": item.get("requirement", ""),
            "weight": item.get("weight", 1.0),
        }
        for item in _sequence(baseline.get("demands", []))
        if isinstance(item, Mapping)
        and str(item.get("type", "")) == specialist_type.value
    ]
    user = f"""Task:
{task.visible_question()}

Assigned evidence requirements (possibly empty):
{json.dumps(demands, ensure_ascii=False, sort_keys=True)}

Return this schema:
{{
  "specialist_type": "{specialist_type.value}",
  "candidate_answer": "short answer or empty if unresolved",
  "evidence": [
    {{"evidence_id": "s1", "type": "{specialist_type.value}", "claim": "claim supported", "support": "exact observed support or derivation", "provenance": "image region description or tool/trace source", "confidence": 0.0}}
  ],
  "remaining_gap_types": []
}}

Only return evidence of type {specialist_type.value}. Do not infer or mention the hidden baseline draft. Empty support is preferable to an invented observation."""
    return system, user


def build_synthesis_prompt(
    task: PilotTask,
    *,
    baseline: Mapping[str, Any],
    specialist_records: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Fuse only cited evidence and disclose which cards changed the baseline."""

    system = (
        "You are the final evidence synthesizer. Compare the baseline with independently produced evidence cards. "
        "Use only supported cards, reject conflicts explicitly, and return exactly one JSON object with no markdown."
    )
    baseline_view = {
        "answer": str(baseline.get("answer", "")),
        "confidence": baseline.get("confidence", 0.0),
        "demands": baseline.get("demands", []),
        "evidence": baseline.get("evidence", []),
        "open_gap_types": baseline.get("open_gap_types", []),
    }
    specialist_view = [
        {
            "specialist_type": item.get("specialist_type", ""),
            "candidate_answer": item.get("candidate_answer", ""),
            "evidence": item.get("evidence", []),
        }
        for item in specialist_records
    ]
    user = f"""Task:
{task.visible_question()}

Baseline:
{json.dumps(baseline_view, ensure_ascii=False, sort_keys=True)}

Specialist records:
{json.dumps(specialist_view, ensure_ascii=False, sort_keys=True)}

Return this schema:
{{
  "answer": "short final answer",
  "used_evidence_ids": ["ids actually used"],
  "rejected_evidence_ids": ["ids rejected as unsupported or conflicting"],
  "remaining_gap_types": [],
  "changed_baseline": false,
  "confidence": 0.0
}}

An evidence ID may appear in used_evidence_ids only if its support and provenance justify the answer."""
    return system, user


def build_scheduler_prompt(
    task: PilotTask,
    *,
    baseline: Mapping[str, Any],
    template_catalog: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Ask a strong LLM router to choose from the same frozen repair catalog."""

    system = (
        "You are a cost-aware runtime scheduler for a multimodal agent. A strong baseline has already "
        "run. Choose exactly one allowed repair template, or baseline when no repair is likely to improve "
        "the answer. Prefer the smallest sufficient repair. Do not invent tools or agents. Return exactly "
        "one JSON object and no markdown."
    )
    baseline_view = {
        "answer": str(baseline.get("answer", "")),
        "confidence": baseline.get("confidence", 0.0),
        "demands": baseline.get("demands", []),
        "evidence": baseline.get("evidence", []),
        "open_gap_types": baseline.get("open_gap_types", []),
    }
    catalog = [
        {
            "template_id": str(item.get("template_id", "")),
            "action_ids": item.get("action_ids", []),
            "evidence_types": item.get("evidence_types", []),
        }
        for item in template_catalog
    ]
    user = f"""Task:
{task.visible_question()}

Baseline evidence state:
{json.dumps(baseline_view, ensure_ascii=False, sort_keys=True)}

Allowed repair templates:
{json.dumps(catalog, ensure_ascii=False, sort_keys=True)}

The special template_id "baseline" makes no additional specialist or synthesis calls.

Return this schema:
{{
  "selected_template_id": "baseline or one exact allowed template_id",
  "target_gap_types": ["evidence types this repair should close"],
  "confidence": 0.0,
  "rationale": "one concise scheduling reason"
}}

Select using only the task, image, baseline state, and catalog. Do not claim to know whether the hidden gold answer is correct."""
    return system, user


def parse_baseline_output(raw: str) -> tuple[dict[str, Any], list[str]]:
    payload, issues = _extract_payload(raw)
    answer = _answer_or_raw(payload, raw)
    demands = _parse_demands(payload.get("demands", []), issues)
    evidence = _parse_evidence(
        payload.get("evidence", []), producer_id="baseline", issues=issues
    )
    open_gaps = _parse_types(
        payload.get("open_gap_types", []), issues, field_name="open_gap_types"
    )
    return (
        {
            "answer": answer,
            "confidence": _probability(
                payload.get("confidence", 0.0), issues, "confidence"
            ),
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
        issues.append(
            "v6 verifier schema requires exactly answer and evidence keys"
        )
        return {"answer": "", "evidence": [], "transport": transport}, issues
    if not isinstance(payload.get("answer"), str) or not isinstance(
        payload.get("evidence"), list
    ):
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
        return safe_abstention(
            "v8 answer-only verifier emitted more than one line"
        )
    if len(text) > 256:
        return safe_abstention("v8 answer-only verifier exceeded 256 characters")
    if text.startswith("```") or text.endswith("```"):
        return safe_abstention("v8 answer-only verifier emitted markdown")
    transport.update(
        mode="plain_single_line", operational_complete=True, decisive=True
    )
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
    evidence = _parse_evidence(
        payload.get("evidence", []), producer_id=producer_id, issues=issues
    )
    evidence = [item for item in evidence if item["type"] == specialist_type.value]
    if len(evidence) < len(_sequence(payload.get("evidence", []))):
        issues.append("discarded evidence with invalid or mismatched specialist type")
    remaining = _parse_types(
        payload.get("remaining_gap_types", []), issues, field_name="remaining_gap_types"
    )
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
            "confidence": _probability(
                payload.get("confidence", 0.0), issues, "confidence"
            ),
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
            "confidence": _probability(
                payload.get("confidence", 0.0), issues, "confidence"
            ),
            "rationale": str(payload.get("rationale", "")).strip(),
        },
        issues,
    )


def _extract_payload(raw: str) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    text = raw.strip()
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL
    )
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
    for index, item in enumerate(_sequence(value), start=1):
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


def _parse_evidence(
    value: Any, *, producer_id: str, issues: list[str]
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index, item in enumerate(_sequence(value), start=1):
        if not isinstance(item, Mapping):
            issues.append("discarded non-object evidence card")
            continue
        local_evidence_id = (
            str(item.get("evidence_id", f"e{index}")).strip() or f"e{index}"
        )
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
                confidence=_probability(
                    item.get("confidence", 0.0), issues, "evidence.confidence"
                ),
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
    for item in _sequence(value):
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


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    return [str(item).strip() for item in _sequence(value) if str(item).strip()]
