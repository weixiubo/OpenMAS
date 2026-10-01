"""Self-contained qualitative packets for the blinded EGOR evidence audit."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def build_trace_candidate(
    *,
    blind_id: str,
    source_id: str,
    task_id: str,
    visible_question: str,
    image_file: str,
    image_sha256: str,
    baseline_record: Mapping[str, Any],
    baseline: Mapping[str, Any],
    plan: Mapping[str, Any],
    policy: Mapping[str, Any],
    selected: Sequence[str],
    specialist_records: Sequence[Mapping[str, Any]],
    specialists: Sequence[Mapping[str, Any]],
    synthesis_record: Mapping[str, Any],
    synthesis: Mapping[str, Any],
    structural: Any,
    provenance_rejected: bool,
    cited_cards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one non-rater-facing end-to-end execution trace."""

    specialist_rows = []
    for action_id, record, parsed in zip(
        selected, specialist_records, specialists, strict=True
    ):
        specialist_rows.append(
            {
                "action_id": action_id,
                "answer": str(
                    parsed.get("answer", parsed.get("candidate_answer", ""))
                ).strip(),
                "evidence": _object_list(parsed.get("evidence", [])),
                "usage": _usage(record),
            }
        )
    candidate_answer = str(synthesis.get("answer", "")).strip()
    baseline_answer = str(baseline.get("answer", "")).strip()
    return {
        "schema_version": 1,
        "blind_item_id": blind_id,
        "source_id": source_id,
        "task_id": task_id,
        "visible_question": visible_question,
        "image_file": image_file,
        "image_sha256": image_sha256,
        "baseline_state": {
            "answer": baseline_answer,
            "confidence": _optional_number(baseline.get("confidence")),
            "demands": _object_list(baseline.get("demands", [])),
            "evidence": _object_list(baseline.get("evidence", [])),
            "open_gap_types": _strings(baseline.get("open_gap_types", [])),
            "usage": _usage(baseline_record),
        },
        "selector_decision": {
            "selected_template_id": str(plan.get("selected_template_id", "")),
            "selected_action_ids": list(selected),
            "combination_id": str(plan.get("combination_id", "")),
            "reason": str(plan.get("reason", "")).strip(),
            "gap_max": _optional_number(plan.get("gap_max")),
            "residual_gaps": dict(plan.get("residual_gaps", {}))
            if isinstance(plan.get("residual_gaps"), Mapping)
            else {},
            "predicted_advantages": dict(plan.get("predicted_advantages", {}))
            if isinstance(plan.get("predicted_advantages"), Mapping)
            else {},
            "net_advantage_scores": dict(plan.get("net_advantage_scores", {}))
            if isinstance(plan.get("net_advantage_scores"), Mapping)
            else {},
        },
        "specialists": specialist_rows,
        "synthesis": {
            "candidate_answer": candidate_answer,
            "used_evidence_ids": _strings(synthesis.get("used_evidence_ids", [])),
            "cited_cards": [dict(card) for card in cited_cards],
            "usage": _usage(synthesis_record),
        },
        "certificate": {
            "actual_changed": bool(structural.actual_changed),
            "reported_changed": bool(structural.reported_changed),
            "change_flag_agrees": bool(structural.change_flag_agrees),
            "used_valid_specialist_ids": list(
                structural.used_valid_specialist_ids
            ),
            "unknown_used_ids": list(structural.unknown_used_ids),
            "evidence_supported_change": bool(
                structural.evidence_supported_change
            ),
            "decision": "fallback_to_baseline"
            if provenance_rejected
            else "accept_candidate",
            "final_answer": baseline_answer
            if provenance_rejected
            else candidate_answer,
        },
        "automatic_outcome": {
            "baseline_quality": _optional_number(policy.get("baseline_quality")),
            "policy_quality": _optional_number(policy.get("policy_quality")),
            "quality_delta": _optional_number(policy.get("quality_delta")),
            "useful_repair": _boolean(policy.get("useful_repair")),
            "strict_harm": _boolean(policy.get("strict_harm")),
            "provenance_rejected": provenance_rejected,
        },
        "total_resources": {
            "calls": _optional_number(policy.get("calls")),
            "total_tokens": _optional_number(policy.get("total_tokens")),
            "latency_seconds": _optional_number(policy.get("latency_seconds")),
            "failed_stages": str(policy.get("failed_stages", "")).strip(),
        },
    }


def build_selected_trace_packet(
    selection: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve each frozen trace stratum to its full raw trace or null."""

    roles = {
        "useful_accepted": "useful_accepted_blind_item_id",
        "certificate_rejected": "certificate_rejected_blind_item_id",
        "strict_harm_accepted": "strict_harm_accepted_blind_item_id",
    }
    traces: dict[str, Any] = {}
    for role, key in roles.items():
        blind_id = selection.get(key)
        if blind_id is None:
            traces[role] = None
            continue
        if str(blind_id) not in candidates:
            raise ValueError(f"selected trace lacks raw candidate: {blind_id}")
        traces[role] = dict(candidates[str(blind_id)])
    return {
        "schema_version": 1,
        "audience": "non-rater-facing qualitative trace packet",
        "selection_rule": selection["selection_rule"],
        "traces": traces,
    }


def render_trace_markdown(packet: Mapping[str, Any]) -> str:
    """Render selected trace JSON as a deterministic author-facing document."""

    labels = {
        "useful_accepted": "Useful Accepted Change",
        "certificate_rejected": "Certificate-Rejected Change",
        "strict_harm_accepted": "Strict-Harm Accepted Change",
    }
    lines = [
        "# Deterministically Selected EGOR Traces",
        "",
        "> Non-rater-facing packet. Examples are selected by the frozen lowest-hash rule, not by author preference.",
        "",
        f"Selection rule: {packet['selection_rule']}",
        "",
    ]
    raw_traces = packet.get("traces", {})
    traces = raw_traces if isinstance(raw_traces, Mapping) else {}
    for role, label in labels.items():
        lines.extend((f"## {label}", ""))
        raw_trace = traces.get(role)
        if not isinstance(raw_trace, Mapping):
            lines.extend(("No item exists in this frozen stratum.", ""))
            continue
        lines.extend(
            (
                f"![Task image]({raw_trace['image_file']})",
                "",
                f"- Stable blind ID: `{raw_trace['blind_item_id']}`",
                f"- Source/task: `{raw_trace['source_id']}::{raw_trace['task_id']}`",
                "",
                "**Visible question**",
                "",
                _markdown_block(raw_trace.get("visible_question", "")),
                "",
                "**Baseline state**",
                "",
                _markdown_block(raw_trace.get("baseline_state", {})),
                "",
                "**Selector decision**",
                "",
                _markdown_block(raw_trace.get("selector_decision", {})),
                "",
                "**Recruited specialist outputs**",
                "",
                _markdown_block(raw_trace.get("specialists", [])),
                "",
                "**Synthesis and cited evidence**",
                "",
                _markdown_block(raw_trace.get("synthesis", {})),
                "",
                "**Certificate and final answer**",
                "",
                _markdown_block(raw_trace.get("certificate", {})),
                "",
                "**Resources and automatic outcome**",
                "",
                _markdown_block(
                    {
                        "resources": raw_trace.get("total_resources", {}),
                        "automatic_outcome": raw_trace.get(
                            "automatic_outcome", {}
                        ),
                    }
                ),
                "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def copy_audit_images(
    *,
    output_dir: Path,
    image_sources: Mapping[str, Path],
    blind_ids: set[str],
) -> list[Path]:
    """Copy exactly the rater-visible and selected-trace images by blind ID."""

    missing = sorted(blind_ids - set(image_sources))
    if missing:
        raise ValueError(f"missing source images for audit items: {missing}")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {
        Path(image_relative_path(blind_id, image_sources[blind_id])).name
        for blind_id in blind_ids
    }
    extras = sorted(
        path.name for path in images_dir.iterdir() if path.name not in expected_names
    )
    if extras:
        raise ValueError(
            "audit image directory contains stale unexpected entries: "
            + ", ".join(extras)
        )
    outputs: list[Path] = []
    for blind_id in sorted(blind_ids):
        source = image_sources[blind_id]
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output_dir / image_relative_path(blind_id, source)
        shutil.copyfile(source, target)
        outputs.append(target)
    return outputs


def image_relative_path(blind_id: str, source: Path) -> str:
    """Return a non-identifying, browser-readable packet image path."""

    suffix = source.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError(f"unsupported audit image suffix: {suffix or '<none>'}")
    return f"images/{blind_id}{suffix}"


def _markdown_block(value: Any) -> str:
    text = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        if isinstance(value, (Mapping, list))
        else str(value)
    )
    return "\n".join(f"    {line}" for line in text.splitlines() or [""])


def _object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _usage(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("usage", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_number(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
