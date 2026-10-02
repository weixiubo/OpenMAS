"""Answer-blind prompts and strict structured-output parsing for EGOR stages."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from openagora.mechanisms.evidence_gap_repair import (
    EvidenceType,
)

from openagora_experiments.egor.task_model import PilotTask
from openagora_experiments.egor.values import sequence_value


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
        for item in sequence_value(baseline.get("demands", []))
        if isinstance(item, Mapping) and str(item.get("type", "")) == specialist_type.value
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
