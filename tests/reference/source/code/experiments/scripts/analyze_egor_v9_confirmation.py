#!/usr/bin/env python3
"""Score sealed EGOR-v9 confirmation only after both sources pass preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

import analyze_egor_v7_confirmation as shared
from openagora_experiments.egor.protocol import PilotTask
from openagora_experiments.egor.runner import load_latest_stage_records, load_tasks
from openagora_experiments.egor.verification import (
    answers_agree,
    canonical_answer,
    decide_answer_only_verified_overwrite,
    valid_evidence_cards,
)
from openagora_experiments.egor.verification_v9 import (
    decide_supported_blind_resolver_overwrite,
)


FILES = {
    "baseline": "baseline_qwen3.jsonl",
    "exact": "specialist_exact_qwen25.jsonl",
    "visual": "specialist_visual_internvl3.jsonl",
    "core": "core_pair_ablation.jsonl",
    "resolver": "resolver_v9_qwen35.jsonl",
    "repairability": "repairability.jsonl",
}
PRIMARY_POLICY = "v9"


def analyze_confirmation(
    *,
    sources: dict[str, tuple[Path, Path]],
    expected_commit: str,
    tasks_per_source: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preflight = preflight_confirmation(
        sources=sources,
        expected_commit=expected_commit,
        expected_tasks_per_source=tasks_per_source,
    )
    if not preflight["passed"]:
        raise ValueError("confirmation integrity preflight failed before scoring")

    rows_by_source = {
        name: analyze_source(
            source=name,
            tasks=load_tasks(tasks_path),
            root=root,
        )
        for name, (tasks_path, root) in sources.items()
    }
    rows = [row for name in ("chartqa", "ocrbench") for row in rows_by_source[name]]
    expected_total = tasks_per_source * len(sources)
    if len(rows) != expected_total:
        raise ValueError(f"confirmation denominator is {len(rows)}, expected {expected_total}")

    summaries = {
        name: summarize_source(
            source_rows,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index + 1,
        )
        for index, (name, source_rows) in enumerate(rows_by_source.items())
    }
    pooled = summarize_source(rows, bootstrap_samples=bootstrap_samples, seed=seed)
    source_deltas = {name: summary[PRIMARY_POLICY]["quality_delta"] for name, summary in summaries.items()}
    source_harm = {name: summary[PRIMARY_POLICY]["strict_harm_rate"] for name, summary in summaries.items()}
    completion = {name: summary["resolver_operational_completion"] for name, summary in summaries.items()}
    accepted = {name: summary["accepted_overwrites"] for name, summary in summaries.items()}
    criteria = [
        shared._criterion(
            "pooled_positive_with_strict_ci",
            pooled[PRIMARY_POLICY]["quality_delta"] > 0.0
            and pooled[PRIMARY_POLICY]["ci95_low"] > 0.0,
            {
                "delta": pooled[PRIMARY_POLICY]["quality_delta"],
                "ci95_low": pooled[PRIMARY_POLICY]["ci95_low"],
            },
            "pooled delta > 0 and paired-bootstrap 95% lower bound > 0",
        ),
        shared._criterion(
            "nonnegative_each_source",
            min(source_deltas.values()) >= 0.0,
            source_deltas,
            "each source point delta >= 0",
        ),
        shared._criterion(
            "bounded_strict_harm",
            pooled[PRIMARY_POLICY]["strict_harm_rate"] <= 0.01
            and max(source_harm.values()) <= 0.02,
            {"pooled": pooled[PRIMARY_POLICY]["strict_harm_rate"], "sources": source_harm},
            "pooled <= 0.01 and every source <= 0.02",
        ),
        shared._criterion(
            "resolver_operational_completion",
            min(completion.values()) >= 0.95,
            completion,
            "every source >= 0.95 on resolver-called rows",
        ),
        shared._criterion(
            "accepted_overwrite_breadth",
            sum(accepted.values()) >= 4 and min(accepted.values()) >= 1,
            {"total": sum(accepted.values()), "sources": accepted},
            "at least four total and at least one on each source",
        ),
        shared._criterion(
            "integrity_manifests",
            bool(preflight["passed"]),
            preflight["checks"],
            "all task, source, seed, model, prompt, code, and stage checks pass",
        ),
    ]
    direct_harms = {
        (row["source"], row["task_id"])
        for row in rows
        if row["direct_resolver_delta"] < -0.05
    }
    v9_harms = {
        (row["source"], row["task_id"])
        for row in rows
        if row["v9_delta"] < -0.05
    }
    direct_delta = pooled["direct_resolver"]["quality_delta"]
    report = {
        "schema_version": 1,
        "claim_boundary": "fresh_independent_sealed_v9_confirmation",
        "method": "egor_v9_supported_blind_resolver",
        "source_commit": expected_commit,
        "task_denominator": len(rows),
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "unit": "paired task with equal task weight",
        },
        "preflight": preflight,
        "pooled": pooled,
        "sources": summaries,
        "secondary_harm_suppression": {
            "direct_resolver_harms": len(direct_harms),
            "v9_harms": len(v9_harms),
            "direct_harms_blocked_by_v9": len(direct_harms - v9_harms),
            "gain_retention_ratio": (
                pooled["v9"]["quality_delta"] / direct_delta
                if direct_delta > 0
                else None
            ),
            "preregistered_as_descriptive_not_a_pass_gate": True,
        },
        "criteria": criteria,
        "passed": all(item["passed"] for item in criteria),
    }
    return report, rows


def preflight_confirmation(
    *,
    sources: dict[str, tuple[Path, Path]],
    expected_commit: str,
    expected_tasks_per_source: int,
) -> dict[str, Any]:
    source_reports: dict[str, Any] = {}
    visible: dict[str, list[dict[str, str]]] = {}
    for name, (tasks_path, root) in sources.items():
        audit_path = root / "audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        task_sha = _sha256(tasks_path)
        checks = {
            "completion_marker": (root / "SOURCE_GENERATION_COMPLETE").is_file(),
            "checksums": _verify_sha256sums(root / "SOURCE_GENERATION_SHA256SUMS", root),
            "source_commit": _read(root / "SOURCE_COMMIT") == expected_commit,
            "task_hash_marker": _read(root / "TASKS_SHA256") == task_sha,
            "audit_mode": audit.get("mode") == "confirmation",
            "audit_passed": audit.get("passed") is True,
            "audit_integrity": audit.get("integrity_passed") is True,
            "audit_outcome_blind": audit.get("outcome_fields_read") is False
            and audit.get("scores_computed") is False,
            "audit_commit": audit.get("source_commit") == expected_commit,
            "resolver_contract": audit.get("resolver_contract", {}).get("stage")
            == "verifier_v8"
            and audit.get("resolver_contract", {}).get("file")
            == FILES["resolver"],
        }
        source_reports[name] = {
            "passed": all(checks.values()),
            "checks": checks,
            "tasks_sha256": task_sha,
            "audit_sha256": _sha256(audit_path),
            "generation_sha256sums_sha256": _sha256(root / "SOURCE_GENERATION_SHA256SUMS"),
        }
        visible[name] = _load_visible_rows(tasks_path)

    all_keys = [(row["source_id"], row["task_id"]) for values in visible.values() for row in values]
    all_images = [row["image_path"] for values in visible.values() for row in values]
    cross_checks = {
        "two_sources": set(sources) == {"chartqa", "ocrbench"},
        "expected_tasks_each": all(len(values) == expected_tasks_per_source for values in visible.values()),
        "unique_source_task_keys": len(set(all_keys)) == len(all_keys),
        "unique_images": len(set(all_images)) == len(all_images),
        "disjoint_task_ids": not (
            {row["task_id"] for row in visible["chartqa"]}
            & {row["task_id"] for row in visible["ocrbench"]}
        ),
    }
    checks = {
        "sources": all(report["passed"] for report in source_reports.values()),
        "cross_source": all(cross_checks.values()),
    }
    return {
        "passed": all(checks.values()),
        "outcome_blind_before_scoring": True,
        "checks": checks,
        "cross_source_checks": cross_checks,
        "sources": source_reports,
    }


def analyze_source(*, source: str, tasks: list[PilotTask], root: Path) -> list[dict[str, Any]]:
    records = {
        name: load_latest_stage_records(root / filename)
        for name, filename in FILES.items()
        if name != "repairability"
    }
    repairability = shared._load_decisions(
        root / FILES["repairability"], "repairability_triggered"
    )
    rows: list[dict[str, Any]] = []
    for task in tasks:
        key = (task.source_id, task.task_id)
        triggered = bool(repairability[key]["repairability_triggered"])
        baseline_record = records["baseline"].get(key)
        exact_record = records["exact"].get(key)
        visual_record = records["visual"].get(key)
        core_record = records["core"].get(key)
        resolver_record = records["resolver"].get(key)
        baseline = shared._parsed(baseline_record)
        exact = shared._parsed(exact_record)
        visual = shared._parsed(visual_record)
        core = shared._parsed(core_record)
        resolver = shared._parsed(resolver_record)
        baseline_answer = str(baseline.get("answer", ""))
        core_answer = str(core.get("answer", ""))
        resolver_answer = str(resolver.get("answer", ""))
        options = [{"id": option_id, "text": text} for option_id, text in task.options]
        baseline_quality = shared._score(task, baseline_answer) if shared._ok(baseline_record) else 0.0
        resolver_transport = resolver.get("transport", {}) if isinstance(resolver, Mapping) else {}
        resolver_operational = bool(
            triggered
            and shared._ok(resolver_record)
            and isinstance(resolver_transport, Mapping)
            and resolver_transport.get("operational_complete")
        )
        resolver_view = resolver if resolver_operational else {
            "answer": "",
            "transport": {"operational_complete": False},
        }
        specialists = {"exact": exact, "visual": visual}
        v9_decision = decide_supported_blind_resolver_overwrite(
            baseline_answer=baseline_answer,
            resolver=resolver_view,
            specialists=specialists,
            metric=task.metric,
            options=options,
        )
        v9_two_decision = decide_supported_blind_resolver_overwrite(
            baseline_answer=baseline_answer,
            resolver=resolver_view,
            specialists=specialists,
            metric=task.metric,
            options=options,
            minimum_specialist_corroborators=2,
        )
        resolver_changed = bool(
            resolver_operational
            and canonical_answer(resolver_answer)
            and not answers_agree(
                baseline_answer, resolver_answer, metric=task.metric, options=options
            )
        )
        core_changed = bool(
            triggered
            and shared._ok(core_record)
            and canonical_answer(core_answer)
            and not answers_agree(
                baseline_answer, core_answer, metric=task.metric, options=options
            )
        )
        v8_decision = decide_answer_only_verified_overwrite(
            baseline_answer=baseline_answer,
            core_answer=core_answer,
            verifier=resolver_view,
            specialists=specialists,
            metric=task.metric,
            options=options,
        )
        specialist_pair = bool(
            triggered
            and valid_evidence_cards(exact)
            and valid_evidence_cards(visual)
            and answers_agree(
                exact.get("candidate_answer", ""),
                visual.get("candidate_answer", ""),
                metric=task.metric,
                options=options,
            )
            and not answers_agree(
                baseline_answer,
                exact.get("candidate_answer", ""),
                metric=task.metric,
                options=options,
            )
        )
        resolver_quality = (
            shared._score(task, resolver_answer) if resolver_changed else baseline_quality
        )
        exact_answer = str(exact.get("candidate_answer", exact.get("answer", "")))
        pair_quality = shared._score(task, exact_answer) if specialist_pair else baseline_quality
        v9_accept = bool(triggered and v9_decision.accept_overwrite)
        v9_two_accept = bool(triggered and v9_two_decision.accept_overwrite)
        v8_accept = bool(core_changed and v8_decision.accept_overwrite)
        core_quality = shared._score(task, core_answer) if shared._ok(core_record) else 0.0
        v9_quality = shared._score(task, resolver_answer) if v9_accept else baseline_quality
        v9_two_quality = shared._score(task, resolver_answer) if v9_two_accept else baseline_quality
        v8_quality = core_quality if v8_accept else baseline_quality

        baseline_usage = shared._usage_total([baseline_record])
        specialist_usage = (
            shared._usage_total([baseline_record, exact_record, visual_record])
            if triggered
            else baseline_usage
        )
        v9_usage = (
            shared._usage_total([baseline_record, exact_record, visual_record, resolver_record])
            if triggered
            else baseline_usage
        )
        direct_usage = (
            shared._usage_total([baseline_record, resolver_record])
            if triggered
            else baseline_usage
        )
        v8_usage = (
            shared._usage_total(
                [baseline_record, exact_record, visual_record, core_record]
                + ([resolver_record] if core_changed else [])
            )
            if triggered
            else baseline_usage
        )
        rows.append(
            {
                "source": source,
                "source_id": task.source_id,
                "task_id": task.task_id,
                "repairability_triggered": triggered,
                "resolver_called": triggered,
                "resolver_operational": resolver_operational,
                "resolver_changed": resolver_changed,
                "core_changed": core_changed,
                "accept_overwrite": v9_accept,
                "accept_two_specialists": v9_two_accept,
                "accept_v8_ablation": v8_accept,
                "decision_reason": v9_decision.reason,
                "corroborators": list(v9_decision.corroborating_specialists),
                "transport_mode": str(resolver_transport.get("mode", "invalid")),
                "sanitized_abstention": bool(resolver_transport.get("sanitized_abstention")),
                "baseline_quality": baseline_quality,
                "direct_resolver_quality": resolver_quality,
                "specialist_pair_quality": pair_quality,
                "v8_quality": v8_quality,
                "v9_two_quality": v9_two_quality,
                "v9_quality": v9_quality,
                "direct_resolver_delta": resolver_quality - baseline_quality,
                "specialist_pair_delta": pair_quality - baseline_quality,
                "v8_delta": v8_quality - baseline_quality,
                "v9_two_delta": v9_two_quality - baseline_quality,
                "v9_delta": v9_quality - baseline_quality,
                "baseline_usage": baseline_usage,
                "direct_resolver_usage": direct_usage,
                "specialist_pair_usage": specialist_usage,
                "v8_usage": v8_usage,
                "v9_two_usage": v9_usage,
                "v9_usage": v9_usage,
                "failed_stages": [
                    name
                    for name, record in (
                        ("baseline", baseline_record),
                        ("exact", exact_record if triggered else {"status": "ok"}),
                        ("visual", visual_record if triggered else {"status": "ok"}),
                        ("core", core_record if triggered else {"status": "ok"}),
                        ("resolver", resolver_record if triggered else {"status": "ok"}),
                    )
                    if not shared._ok(record)
                ],
            }
        )
    return rows


def summarize_source(
    rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    policies = {
        "baseline": shared._summarize_policy(rows, "baseline", bootstrap_samples=0, seed=seed),
        "direct_resolver": shared._summarize_policy(rows, "direct_resolver", bootstrap_samples=0, seed=seed),
        "specialist_pair": shared._summarize_policy(rows, "specialist_pair", bootstrap_samples=0, seed=seed),
        "v8": shared._summarize_policy(rows, "v8", bootstrap_samples=0, seed=seed),
        "v9_two": shared._summarize_policy(rows, "v9_two", bootstrap_samples=0, seed=seed),
        "v9": shared._summarize_policy(rows, "v9", bootstrap_samples=bootstrap_samples, seed=seed),
    }
    called = [row for row in rows if row["resolver_called"]]
    return {
        **policies,
        "tasks": len(rows),
        "repairability_triggered": sum(row["repairability_triggered"] for row in rows),
        "resolver_changed": sum(row["resolver_changed"] for row in rows),
        "accepted_overwrites": sum(row["accept_overwrite"] for row in rows),
        "accepted_two_specialists": sum(row["accept_two_specialists"] for row in rows),
        "accepted_v8_ablation": sum(row["accept_v8_ablation"] for row in rows),
        "resolver_operational_completion": (
            fmean(row["resolver_operational"] for row in called) if called else 0.0
        ),
        "transport_modes": dict(Counter(row["transport_mode"] for row in called)),
        "sanitized_abstentions": sum(row["sanitized_abstention"] for row in called),
        "decision_reasons": dict(Counter(row["decision_reason"] for row in rows)),
        "failure_counts": dict(Counter(stage for row in rows for stage in row["failed_stages"])),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "source", "source_id", "task_id", "repairability_triggered",
        "resolver_called", "resolver_operational", "resolver_changed",
        "core_changed", "accept_overwrite", "accept_two_specialists",
        "accept_v8_ablation", "decision_reason", "baseline_quality",
        "direct_resolver_quality", "specialist_pair_quality", "v8_quality",
        "v9_two_quality", "v9_quality", "direct_resolver_delta",
        "specialist_pair_delta", "v8_delta", "v9_two_delta", "v9_delta",
        "failed_stages",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = {field: row[field] for field in fields}
            payload["failed_stages"] = "|".join(row["failed_stages"])
            writer.writerow(payload)


def _load_visible_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        rows.append(
            {
                "source_id": str(raw.get("benchmark") or raw.get("source_id") or ""),
                "task_id": str(raw.get("task_id", "")),
                "image_path": str(Path(str(raw.get("image_path", ""))).resolve()),
            }
        )
    return rows


def _verify_sha256sums(path: Path, root: Path) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, raw_path = line.partition("  ")
        if not separator or len(digest) != 64:
            return False
        candidate = Path(raw_path).resolve()
        if root.resolve() not in candidate.parents or not candidate.is_file():
            return False
        if _sha256(candidate) != digest:
            return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chartqa-tasks", required=True)
    parser.add_argument("--chartqa-root", required=True)
    parser.add_argument("--ocrbench-tasks", required=True)
    parser.add_argument("--ocrbench-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--tasks-per-source", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260790)
    parser.add_argument("--rows-output", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report, rows = analyze_confirmation(
        sources={
            "chartqa": (Path(args.chartqa_tasks).resolve(), Path(args.chartqa_root).resolve()),
            "ocrbench": (Path(args.ocrbench_tasks).resolve(), Path(args.ocrbench_root).resolve()),
        },
        expected_commit=args.source_commit,
        tasks_per_source=args.tasks_per_source,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    if args.rows_output:
        _write_rows(Path(args.rows_output).resolve(), rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
