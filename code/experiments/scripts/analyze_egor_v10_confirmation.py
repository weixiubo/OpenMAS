#!/usr/bin/env python3
"""Score sealed EGOR-v10 confirmation after both sources pass preflight."""

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
from openagora_experiments.egor.verification import answers_agree, canonical_answer
from openagora_experiments.egor.verification_v9 import (
    decide_supported_blind_resolver_overwrite,
)


FILES = {
    "baseline": "baseline_qwen3.jsonl",
    "resolver": "resolver_v10_qwen35.jsonl",
    "exact": "specialist_exact_qwen25.jsonl",
    "visual": "specialist_visual_internvl3.jsonl",
    "repairability": "repairability.jsonl",
    "resolver_change": "resolver_change.jsonl",
}
EXPECTED_TASKS = {"chartqa": 230, "ocrbench": 570}
PRIMARY_POLICY = "v10"


def analyze_confirmation(
    *,
    sources: dict[str, tuple[Path, Path]],
    expected_commit: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preflight = preflight_confirmation(sources=sources, expected_commit=expected_commit)
    if not preflight["passed"]:
        raise ValueError("v10 integrity preflight failed before scoring")
    rows_by_source = {
        name: analyze_source(
            source=name, tasks=load_tasks(tasks_path), root=root
        )
        for name, (tasks_path, root) in sources.items()
    }
    rows = [row for name in ("chartqa", "ocrbench") for row in rows_by_source[name]]
    if len(rows) != sum(EXPECTED_TASKS.values()):
        raise ValueError("unexpected v10 confirmation denominator")
    summaries = {
        name: summarize_source(
            source_rows,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index + 1,
        )
        for index, (name, source_rows) in enumerate(rows_by_source.items())
    }
    pooled = summarize_source(rows, bootstrap_samples=bootstrap_samples, seed=seed)
    source_deltas = {
        name: summary[PRIMARY_POLICY]["quality_delta"]
        for name, summary in summaries.items()
    }
    source_harm = {
        name: summary[PRIMARY_POLICY]["strict_harm_rate"]
        for name, summary in summaries.items()
    }
    native_completion = {
        name: summary["resolver_native_transport_completion"]
        for name, summary in summaries.items()
    }
    accepted = {
        name: summary["accepted_overwrites"] for name, summary in summaries.items()
    }
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
            pooled[PRIMARY_POLICY]["strict_harm_rate"] <= 0.005
            and max(source_harm.values()) <= 0.01,
            {"pooled": pooled[PRIMARY_POLICY]["strict_harm_rate"], "sources": source_harm},
            "pooled <= 0.005 and every source <= 0.01",
        ),
        shared._criterion(
            "resolver_native_transport_completion",
            min(native_completion.values()) >= 0.95,
            native_completion,
            "every source >= 0.95 native one-line or explicit-abstention transport",
        ),
        shared._criterion(
            "accepted_overwrite_breadth",
            sum(accepted.values()) >= 8 and min(accepted.values()) >= 2,
            {"total": sum(accepted.values()), "sources": accepted},
            "at least eight total and at least two on each source",
        ),
        shared._criterion(
            "integrity_manifests",
            bool(preflight["passed"]),
            preflight["checks"],
            "all task, source, model, prompt, seed, code, stage, and checksum checks pass",
        ),
    ]
    direct_harms = {
        (row["source"], row["task_id"])
        for row in rows
        if row["direct_resolver_delta"] < -0.05
    }
    v10_harms = {
        (row["source"], row["task_id"])
        for row in rows
        if row["v10_delta"] < -0.05
    }
    report = {
        "schema_version": 1,
        "claim_boundary": "fresh_independent_sealed_v10_confirmation",
        "method": "egor_v10_two_support_blind_resolver",
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
            "v10_harms": len(v10_harms),
            "direct_harms_blocked_by_v10": len(direct_harms - v10_harms),
            "descriptive_not_a_pass_gate": True,
        },
        "criteria": criteria,
        "passed": all(item["passed"] for item in criteria),
    }
    return report, rows


def preflight_confirmation(
    *, sources: dict[str, tuple[Path, Path]], expected_commit: str
) -> dict[str, Any]:
    source_reports: dict[str, Any] = {}
    visible: dict[str, list[dict[str, str]]] = {}
    for name, (tasks_path, root) in sources.items():
        audit_path = root / "audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        task_sha = _sha256(tasks_path)
        checks = {
            "completion_marker": (root / "SOURCE_GENERATION_COMPLETE").is_file(),
            "checksums": _verify_sha256sums(
                root / "SOURCE_GENERATION_SHA256SUMS", root
            ),
            "source_commit": _read(root / "SOURCE_COMMIT") == expected_commit,
            "task_hash_marker": _read(root / "TASKS_SHA256") == task_sha,
            "audit_mode": audit.get("mode") == "confirmation",
            "audit_passed": audit.get("passed") is True,
            "audit_integrity": audit.get("integrity_passed") is True,
            "audit_outcome_blind": audit.get("outcome_fields_read") is False
            and audit.get("scores_computed") is False,
            "audit_commit": audit.get("source_commit") == expected_commit,
            "resolver_contract": audit.get("resolver_contract", {}).get("file")
            == FILES["resolver"],
            "specialist_contract": audit.get("specialist_contract", {}).get("ids")
            == "resolver_changed"
            and audit.get("specialist_contract", {}).get(
                "resolver_candidate_visible_to_specialists"
            )
            is False,
        }
        source_reports[name] = {
            "passed": all(checks.values()),
            "checks": checks,
            "tasks_sha256": task_sha,
            "audit_sha256": _sha256(audit_path),
            "generation_sha256sums_sha256": _sha256(
                root / "SOURCE_GENERATION_SHA256SUMS"
            ),
        }
        visible[name] = _load_visible_rows(tasks_path)
    all_keys = [
        (row["source_id"], row["task_id"])
        for values in visible.values()
        for row in values
    ]
    all_images = [row["image_path"] for values in visible.values() for row in values]
    cross_checks = {
        "two_sources": set(sources) == {"chartqa", "ocrbench"},
        "expected_tasks": {
            name: len(visible.get(name, [])) == expected
            for name, expected in EXPECTED_TASKS.items()
        },
        "unique_source_task_keys": len(set(all_keys)) == len(all_keys),
        "unique_images": len(set(all_images)) == len(all_images),
        "disjoint_task_ids": not (
            {row["task_id"] for row in visible["chartqa"]}
            & {row["task_id"] for row in visible["ocrbench"]}
        ),
    }
    checks = {
        "sources": all(report["passed"] for report in source_reports.values()),
        "cross_source": all(
            value if isinstance(value, bool) else all(value.values())
            for value in cross_checks.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "outcome_blind_before_scoring": True,
        "checks": checks,
        "cross_source_checks": cross_checks,
        "sources": source_reports,
    }


def analyze_source(
    *, source: str, tasks: list[PilotTask], root: Path
) -> list[dict[str, Any]]:
    """Reconstruct source rows and the frozen two-support decision.

    The explicit two-corroborator call below implements ``sec:authorization``
    and ``eq:overwrite-authorization`` for EGOR-v10.
    """
    records = {
        name: load_latest_stage_records(root / filename)
        for name, filename in FILES.items()
        if name not in {"repairability", "resolver_change"}
    }
    repairability = shared._load_decisions(
        root / FILES["repairability"], "repairability_triggered"
    )
    change_plan = shared._load_decisions(
        root / FILES["resolver_change"], "resolver_changed"
    )
    rows: list[dict[str, Any]] = []
    for task in tasks:
        key = (task.source_id, task.task_id)
        triggered = bool(repairability[key]["repairability_triggered"])
        planned_change = bool(change_plan[key]["resolver_changed"])
        baseline_record = records["baseline"].get(key)
        resolver_record = records["resolver"].get(key)
        exact_record = records["exact"].get(key)
        visual_record = records["visual"].get(key)
        baseline = shared._parsed(baseline_record)
        resolver = shared._parsed(resolver_record)
        exact = shared._parsed(exact_record)
        visual = shared._parsed(visual_record)
        baseline_answer = str(baseline.get("answer", ""))
        resolver_answer = str(resolver.get("answer", ""))
        options = [{"id": option_id, "text": text} for option_id, text in task.options]
        baseline_quality = (
            shared._score(task, baseline_answer) if shared._ok(baseline_record) else 0.0
        )
        transport = resolver.get("transport", {}) if isinstance(resolver, Mapping) else {}
        resolver_operational = bool(
            triggered
            and shared._ok(resolver_record)
            and isinstance(transport, Mapping)
            and transport.get("operational_complete")
        )
        resolver_native_transport = bool(
            resolver_operational
            and transport.get("mode")
            in {"plain_single_line", "explicit_abstention"}
        )
        resolver_view = resolver if resolver_operational else {
            "answer": "",
            "transport": {"operational_complete": False},
        }
        recomputed_change = bool(
            resolver_operational
            and transport.get("decisive")
            and not transport.get("sanitized_abstention")
            and canonical_answer(resolver_answer)
            and not answers_agree(
                baseline_answer, resolver_answer, metric=task.metric, options=options
            )
        )
        if recomputed_change != planned_change:
            raise ValueError(f"resolver-change drift: {key}")
        specialists = {"exact": exact, "visual": visual}
        one_decision = decide_supported_blind_resolver_overwrite(
            baseline_answer=baseline_answer,
            resolver=resolver_view,
            specialists=specialists,
            metric=task.metric,
            options=options,
            minimum_specialist_corroborators=1,
        )
        v10_decision = decide_supported_blind_resolver_overwrite(
            baseline_answer=baseline_answer,
            resolver=resolver_view,
            specialists=specialists,
            metric=task.metric,
            options=options,
            minimum_specialist_corroborators=2,
        )
        resolver_quality = (
            shared._score(task, resolver_answer) if planned_change else baseline_quality
        )
        one_accept = bool(triggered and one_decision.accept_overwrite)
        v10_accept = bool(triggered and v10_decision.accept_overwrite)
        one_quality = (
            shared._score(task, resolver_answer) if one_accept else baseline_quality
        )
        v10_quality = (
            shared._score(task, resolver_answer) if v10_accept else baseline_quality
        )
        baseline_usage = shared._usage_total([baseline_record])
        direct_usage = (
            shared._usage_total([baseline_record, resolver_record])
            if triggered
            else baseline_usage
        )
        support_usage = (
            shared._usage_total(
                [baseline_record, resolver_record, exact_record, visual_record]
            )
            if planned_change
            else direct_usage
        )
        rows.append(
            {
                "source": source,
                "source_id": task.source_id,
                "task_id": task.task_id,
                "repairability_triggered": triggered,
                "resolver_called": triggered,
                "resolver_operational": resolver_operational,
                "resolver_native_transport": resolver_native_transport,
                "resolver_changed": planned_change,
                "accept_one_support": one_accept,
                "accept_overwrite": v10_accept,
                "decision_reason": v10_decision.reason,
                "corroborators": list(v10_decision.corroborating_specialists),
                "transport_mode": str(transport.get("mode", "invalid")),
                "sanitized_abstention": bool(transport.get("sanitized_abstention")),
                "baseline_quality": baseline_quality,
                "direct_resolver_quality": resolver_quality,
                "one_support_quality": one_quality,
                "v10_quality": v10_quality,
                "direct_resolver_delta": resolver_quality - baseline_quality,
                "one_support_delta": one_quality - baseline_quality,
                "v10_delta": v10_quality - baseline_quality,
                "baseline_usage": baseline_usage,
                "direct_resolver_usage": direct_usage,
                "one_support_usage": support_usage,
                "v10_usage": support_usage,
                "failed_stages": [
                    name
                    for name, record in (
                        ("baseline", baseline_record),
                        ("resolver", resolver_record if triggered else {"status": "ok"}),
                        ("exact", exact_record if planned_change else {"status": "ok"}),
                        ("visual", visual_record if planned_change else {"status": "ok"}),
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
        "baseline": shared._summarize_policy(
            rows, "baseline", bootstrap_samples=0, seed=seed
        ),
        "direct_resolver": shared._summarize_policy(
            rows, "direct_resolver", bootstrap_samples=0, seed=seed
        ),
        "one_support": shared._summarize_policy(
            rows, "one_support", bootstrap_samples=0, seed=seed
        ),
        "v10": shared._summarize_policy(
            rows, "v10", bootstrap_samples=bootstrap_samples, seed=seed
        ),
    }
    called = [row for row in rows if row["resolver_called"]]
    return {
        **policies,
        "tasks": len(rows),
        "repairability_triggered": sum(row["repairability_triggered"] for row in rows),
        "resolver_changed": sum(row["resolver_changed"] for row in rows),
        "accepted_one_support": sum(row["accept_one_support"] for row in rows),
        "accepted_overwrites": sum(row["accept_overwrite"] for row in rows),
        "resolver_operational_completion": (
            fmean(row["resolver_operational"] for row in called) if called else 0.0
        ),
        "resolver_native_transport_completion": (
            fmean(row["resolver_native_transport"] for row in called)
            if called else 0.0
        ),
        "transport_modes": dict(Counter(row["transport_mode"] for row in called)),
        "sanitized_abstentions": sum(row["sanitized_abstention"] for row in called),
        "native_explicit_abstentions": sum(
            row["transport_mode"] == "explicit_abstention" for row in called
        ),
        "transport_sanitized_abstentions": sum(
            row["transport_mode"] == "sanitized_transport_abstention"
            for row in called
        ),
        "decision_reasons": dict(Counter(row["decision_reason"] for row in rows)),
        "failure_counts": dict(
            Counter(stage for row in rows for stage in row["failed_stages"])
        ),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "source",
        "source_id",
        "task_id",
        "repairability_triggered",
        "resolver_called",
        "resolver_operational",
        "resolver_native_transport",
        "resolver_changed",
        "accept_one_support",
        "accept_overwrite",
        "decision_reason",
        "baseline_quality",
        "direct_resolver_quality",
        "one_support_quality",
        "v10_quality",
        "direct_resolver_delta",
        "one_support_delta",
        "v10_delta",
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
        if line.strip():
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
        candidate = Path(raw_path).resolve()
        if (
            not separator
            or len(digest) != 64
            or root.resolve() not in candidate.parents
            or not candidate.is_file()
            or _sha256(candidate) != digest
        ):
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


def main() -> int:
    args = _parse_args()
    report, rows = analyze_confirmation(
        sources={
            "chartqa": (
                Path(args.chartqa_tasks).resolve(),
                Path(args.chartqa_root).resolve(),
            ),
            "ocrbench": (
                Path(args.ocrbench_tasks).resolve(),
                Path(args.ocrbench_root).resolve(),
            ),
        },
        expected_commit=args.source_commit,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    if args.rows_output:
        _write_rows(Path(args.rows_output).resolve(), rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chartqa-tasks", required=True)
    parser.add_argument("--chartqa-root", required=True)
    parser.add_argument("--ocrbench-tasks", required=True)
    parser.add_argument("--ocrbench-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260793)
    parser.add_argument("--rows-output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
