#!/usr/bin/env python3
"""Build a gold- and outcome-blind evidence-alignment audit packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openagora_experiments.egor.audit_packet import (
    build_selected_trace_packet,
    build_trace_candidate,
    copy_audit_images,
    image_relative_path,
    render_trace_markdown,
)
from openagora_experiments.egor.evaluation import audit_synthesis
from openagora_experiments.egor.planning import combination_id
from openagora_experiments.egor.runner import load_latest_stage_records, load_tasks


ROOT = Path(__file__).resolve().parents[3]
LABEL_FIELDS = (
    "support_observable",
    "card_claim_supported",
    "change_supported",
    "change_material",
    "failure_mode",
    "notes",
)


def main() -> int:
    args = _parse_args()
    tasks_path = Path(args.tasks).resolve()
    baseline_path = Path(args.baseline).resolve()
    policy_rows_path = Path(args.policy_rows).resolve()
    plans_path = Path(args.plans).resolve()
    execution_config_path = Path(args.execution_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    execution_config = _load_object(execution_config_path)
    action_paths = _path_map(execution_config.get("actions", {}), execution_config_path)
    synthesis_paths = _path_map(
        execution_config.get("combinations", {}), execution_config_path
    )
    tasks = load_tasks(tasks_path)
    baselines = load_latest_stage_records(baseline_path)
    plans = _load_jsonl_by_key(plans_path)
    policy_rows = _load_policy_rows(policy_rows_path, expected_policy=args.policy_id)
    actions = {
        action_id: load_latest_stage_records(path)
        for action_id, path in action_paths.items()
    }
    syntheses = {
        combination: load_latest_stage_records(path)
        for combination, path in synthesis_paths.items()
    }

    items: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    trace_candidates: dict[str, dict[str, Any]] = {}
    image_sources: dict[str, Path] = {}
    seen_keys: set[tuple[str, str]] = set()
    for task in tasks:
        key = (task.source_id, task.task_id)
        seen_keys.add(key)
        baseline = _required(baselines, key, "baseline")
        plan = _required(plans, key, "plan")
        policy = _required(policy_rows, key, "policy row")
        selected = tuple(str(value) for value in plan.get("selected_action_ids", []))
        selected_combination = combination_id(selected)
        if str(plan.get("combination_id", "")) != selected_combination:
            raise ValueError(f"plan combination mismatch for {_render_key(key)}")
        triggered = _boolean(policy.get("triggered"))
        if triggered != bool(selected):
            raise ValueError(f"policy/plan trigger mismatch for {_render_key(key)}")
        blind_id = _blind_id(args.dataset_id, *key)
        if not selected:
            exclusions.append(_exclusion(blind_id, key, "not_triggered"))
            continue

        missing_actions = [
            action_id for action_id in selected if action_id not in actions
        ]
        if missing_actions or selected_combination not in syntheses:
            raise ValueError(
                f"unavailable stages for {_render_key(key)}: "
                f"actions={missing_actions}, combination={selected_combination}"
            )
        specialist_records = [actions[action_id].get(key) for action_id in selected]
        synthesis = syntheses[selected_combination].get(key)
        records = [baseline, *specialist_records, synthesis]
        if any(record is None or not _ok(record) for record in records):
            exclusions.append(_exclusion(blind_id, key, "incomplete_selected_stages"))
            continue

        baseline_parsed = _parsed(baseline)
        specialist_parsed = [_parsed(record or {}) for record in specialist_records]
        synthesis_parsed = _parsed(synthesis or {})
        structural = audit_synthesis(
            baseline=baseline_parsed,
            specialists=specialist_parsed,
            synthesis=synthesis_parsed,
        )
        if not structural.actual_changed:
            exclusions.append(_exclusion(blind_id, key, "answer_unchanged"))
            continue
        provenance_rejected = _boolean(policy.get("provenance_rejected"))
        expected_rejected = not structural.evidence_supported_change
        if provenance_rejected != expected_rejected:
            raise ValueError(
                f"policy/raw provenance decision mismatch for {_render_key(key)}"
            )
        population = (
            "certificate_rejected" if provenance_rejected else "accepted_changed"
        )
        image_source = Path(task.image_path).resolve()
        image_file = image_relative_path(blind_id, image_source)
        image_sha256 = _sha256(image_source)
        image_sources[blind_id] = image_source
        cards = _cited_cards(
            baseline=baseline_parsed,
            specialists=specialist_parsed,
            cited_ids=_strings(synthesis_parsed.get("used_evidence_ids", [])),
        )
        trace_candidates[blind_id] = build_trace_candidate(
            blind_id=blind_id,
            source_id=key[0],
            task_id=key[1],
            visible_question=task.visible_question(),
            image_file=image_file,
            image_sha256=image_sha256,
            baseline_record=baseline,
            baseline=baseline_parsed,
            plan=plan,
            policy=policy,
            selected=selected,
            specialist_records=[record or {} for record in specialist_records],
            specialists=specialist_parsed,
            synthesis_record=synthesis or {},
            synthesis=synthesis_parsed,
            structural=structural,
            provenance_rejected=provenance_rejected,
            cited_cards=cards,
        )
        index_rows.append(
            {
                "schema_version": 1,
                "blind_item_id": blind_id,
                "source_id": key[0],
                "task_id": key[1],
                "population": population,
                "selected_action_ids": list(selected),
                "combination_id": selected_combination,
                "useful_repair": _boolean(policy.get("useful_repair")),
                "strict_harm": _boolean(policy.get("strict_harm")),
            }
        )
        if provenance_rejected:
            continue
        items.append(
            {
                "schema_version": 1,
                "blind_item_id": blind_id,
                "visible_question": task.visible_question(),
                "image_file": image_file,
                "image_sha256": image_sha256,
                "baseline_answer": str(baseline_parsed.get("answer", "")).strip(),
                "candidate_answer": str(synthesis_parsed.get("answer", "")).strip(),
                "cited_evidence_ids": _strings(
                    synthesis_parsed.get("used_evidence_ids", [])
                ),
                "cited_cards": cards,
            }
        )

    extra_policy = sorted(set(policy_rows) - seen_keys)
    extra_plans = sorted(set(plans) - seen_keys)
    if extra_policy or extra_plans:
        raise ValueError(
            f"policy/plan rows outside task set: policy={extra_policy}, plans={extra_plans}"
        )

    items.sort(key=lambda row: row["blind_item_id"])
    index_rows.sort(key=lambda row: row["blind_item_id"])
    exclusions.sort(key=lambda row: row["blind_item_id"])
    blinded_path = output_dir / "blinded_items.jsonl"
    index_path = output_dir / "population_index.jsonl"
    exclusions_path = output_dir / "excluded_records.jsonl"
    trace_path = output_dir / "trace_selection.json"
    trace_packet_path = output_dir / "selected_trace_packet.json"
    trace_markdown_path = output_dir / "selected_traces.md"
    rater_a_path = output_dir / "annotation_template_rater_a.csv"
    rater_b_path = output_dir / "annotation_template_rater_b.csv"
    adjudication_path = output_dir / "adjudication_template.csv"
    _write_jsonl(blinded_path, items)
    _write_jsonl(index_path, index_rows)
    _write_jsonl(exclusions_path, exclusions)
    _write_annotation_template(rater_a_path, items)
    _write_annotation_template(rater_b_path, items)
    _write_annotation_template(adjudication_path, items)
    trace_selection = _trace_selection(index_rows)
    trace_path.write_text(
        json.dumps(trace_selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trace_packet = build_selected_trace_packet(trace_selection, trace_candidates)
    trace_packet_path.write_text(
        json.dumps(trace_packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trace_markdown_path.write_text(
        render_trace_markdown(trace_packet), encoding="utf-8"
    )
    selected_trace_ids = {
        str(value)
        for key, value in trace_selection.items()
        if key.endswith("_blind_item_id") and value is not None
    }
    image_ids = {str(item["blind_item_id"]) for item in items} | selected_trace_ids
    image_output_paths = copy_audit_images(
        output_dir=output_dir,
        image_sources=image_sources,
        blind_ids=image_ids,
    )

    input_paths = [
        tasks_path,
        baseline_path,
        policy_rows_path,
        plans_path,
        execution_config_path,
        *action_paths.values(),
        *synthesis_paths.values(),
        *(image_sources[blind_id] for blind_id in sorted(image_ids)),
    ]
    output_paths = [
        blinded_path,
        index_path,
        exclusions_path,
        trace_path,
        rater_a_path,
        rater_b_path,
        adjudication_path,
        trace_packet_path,
        trace_markdown_path,
        *image_output_paths,
    ]
    populations = Counter(row["population"] for row in index_rows)
    exclusion_counts = Counter(row["reason"] for row in exclusions)
    manifest = {
        "schema_version": 1,
        "source_commit": os.environ.get("EGOR_SOURCE_COMMIT", "").strip()
        or _git_output("rev-parse", "HEAD"),
        "analysis_commit": os.environ.get("EGOR_ANALYSIS_COMMIT", "").strip()
        or _git_output("rev-parse", "HEAD"),
        "dataset_id": args.dataset_id,
        "policy_id": args.policy_id,
        "construction_command": [str(value) for value in sys.argv],
        "task_count": len(tasks),
        "audit_item_count": len(items),
        "population_counts": dict(sorted(populations.items())),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "blinding_guarantee": {
            "blinded_items_hide": [
                "reference answers",
                "automatic scores and deltas",
                "policy and control identity",
                "model identity",
                "certificate decision",
                "useful-repair and strict-harm outcomes",
                "absolute source image paths",
            ],
            "population_index_is_not_rater_facing": True,
            "certificate_rejections_are_not_rater_facing": True,
            "images_are_hash_named_copies": True,
        },
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in dict.fromkeys(input_paths)
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in output_paths
        ],
    }
    (output_dir / "population_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"built {len(items)} blinded audit items "
        f"({populations['accepted_changed']} accepted, "
        f"{populations['certificate_rejected']} rejected) in {output_dir}"
    )
    return 0


def _cited_cards(
    *,
    baseline: Mapping[str, Any],
    specialists: Sequence[Mapping[str, Any]],
    cited_ids: Sequence[str],
) -> list[dict[str, Any]]:
    known: dict[str, dict[str, Any]] = {}
    for parsed in (baseline, *specialists):
        raw_cards = parsed.get("evidence", [])
        if not isinstance(raw_cards, list):
            continue
        for raw in raw_cards:
            if not isinstance(raw, Mapping):
                continue
            evidence_id = str(raw.get("evidence_id", "")).strip()
            if not evidence_id:
                continue
            card = {
                "evidence_id": evidence_id,
                "type": str(raw.get("type", "")).strip(),
                "claim": str(raw.get("claim", "")).strip(),
                "support": str(raw.get("support", "")).strip(),
                "provenance": str(raw.get("provenance", "")).strip(),
                "producer_id": str(raw.get("producer_id", "")).strip(),
                "confidence": float(raw.get("confidence", 0.0)),
            }
            if evidence_id in known and known[evidence_id] != card:
                raise ValueError(f"conflicting evidence card for {evidence_id}")
            known[evidence_id] = card
    return [known[evidence_id] for evidence_id in cited_ids if evidence_id in known]


def _trace_selection(index_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def first_id(predicate: Any) -> str | None:
        candidates = sorted(
            str(row["blind_item_id"]) for row in index_rows if predicate(row)
        )
        return candidates[0] if candidates else None

    return {
        "schema_version": 1,
        "selection_rule": "lexicographically lowest SHA-256 blind_item_id in stratum",
        "useful_accepted_blind_item_id": first_id(
            lambda row: (
                row["population"] == "accepted_changed" and bool(row["useful_repair"])
            )
        ),
        "certificate_rejected_blind_item_id": first_id(
            lambda row: row["population"] == "certificate_rejected"
        ),
        "strict_harm_accepted_blind_item_id": first_id(
            lambda row: (
                row["population"] == "accepted_changed" and bool(row["strict_harm"])
            )
        ),
    }


def _write_annotation_template(path: Path, items: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("blind_item_id", *LABEL_FIELDS))
        writer.writeheader()
        for item in items:
            writer.writerow({"blind_item_id": item["blind_item_id"]})


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _load_policy_rows(
    path: Path, *, expected_policy: str
) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            policy = str(row.get("policy_id", "")).strip()
            if policy != expected_policy:
                raise ValueError(f"unexpected policy_id {policy!r} in {path}")
            key = (str(row.get("source_id", "")), str(row.get("task_id", "")))
            if not all(key) or key in rows:
                raise ValueError(f"invalid or duplicate policy row: {key}")
            rows[key] = row
    return rows


def _load_jsonl_by_key(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("source_id", "")), str(row.get("task_id", "")))
            if not all(key) or key in rows:
                raise ValueError(f"invalid or duplicate JSONL row: {key}")
            rows[key] = row
    return rows


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _path_map(value: Any, config_path: Path) -> dict[str, Path]:
    if not isinstance(value, dict):
        raise ValueError("execution action/combination maps must be objects")
    paths: dict[str, Path] = {}
    for key, raw_path in value.items():
        path = Path(str(raw_path)).expanduser()
        paths[str(key)] = (
            path if path.is_absolute() else config_path.parent / path
        ).resolve()
    return paths


def _required(
    rows: Mapping[tuple[str, str], Any], key: tuple[str, str], label: str
) -> Any:
    if key not in rows:
        raise ValueError(f"missing {label} for {_render_key(key)}")
    return rows[key]


def _parsed(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("parsed", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _ok(record: Mapping[str, Any]) -> bool:
    return str(record.get("status", "")) == "ok"


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _blind_id(dataset_id: str, source_id: str, task_id: str) -> str:
    preimage = f"{dataset_id}::{source_id}::{task_id}".encode()
    return hashlib.sha256(preimage).hexdigest()


def _exclusion(blind_id: str, key: tuple[str, str], reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "blind_item_id": blind_id,
        "source_id": key[0],
        "task_id": key[1],
        "reason": reason,
    }


def _render_key(key: tuple[str, str]) -> str:
    return f"{key[0]}::{key[1]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--policy-rows", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--execution-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--policy-id", default="frozen_egor_selector")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
