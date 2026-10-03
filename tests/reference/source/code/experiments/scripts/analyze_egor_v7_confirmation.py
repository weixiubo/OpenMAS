#!/usr/bin/env python3
"""Open and score both sealed v7 confirmation sources after integrity checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from openagora_experiments.cago.scoring import score_prediction
from openagora_experiments.egor.protocol import PilotTask
from openagora_experiments.egor.runner import load_latest_stage_records, load_tasks
from openagora_experiments.egor.verification import (
    answers_agree,
    canonical_answer,
    decide_answer_only_verified_overwrite,
    decide_verified_overwrite,
    valid_evidence_cards,
)


SOURCE_FILES = {
    "baseline": "baseline_qwen3.jsonl",
    "exact": "specialist_exact_qwen25.jsonl",
    "visual": "specialist_visual_internvl3.jsonl",
    "core": "core_pair.jsonl",
    "verifier": "verifier_v7_qwen35.jsonl",
    "repairability": "repairability.jsonl",
    "overwrite": "overwrite_plans.jsonl",
}


def main() -> int:
    args = _parse_args()
    sources = {
        "chartqa": (Path(args.chartqa_tasks).resolve(), Path(args.chartqa_root).resolve()),
        "infographic": (
            Path(args.infographic_tasks).resolve(),
            Path(args.infographic_root).resolve(),
        ),
    }
    preflight = preflight_confirmation(
        sources=sources,
        expected_commit=args.source_commit,
    )
    if not preflight["passed"]:
        raise SystemExit("confirmation integrity preflight failed before scoring")

    rows_by_source = {
        name: analyze_source(
            source=name,
            tasks=load_tasks(tasks_path),
            root=root,
        )
        for name, (tasks_path, root) in sources.items()
    }
    rows = [row for name in ("chartqa", "infographic") for row in rows_by_source[name]]
    if len(rows) != 480:
        raise ValueError(f"confirmation denominator is {len(rows)}, expected 480")
    summaries = {
        name: summarize_source(
            source_rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + index + 1,
        )
        for index, (name, source_rows) in enumerate(rows_by_source.items())
    }
    pooled = summarize_source(
        rows,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    completion = {
        name: summary["verifier_operational_completion"]
        for name, summary in summaries.items()
    }
    source_deltas = {
        name: summary["v7"]["quality_delta"] for name, summary in summaries.items()
    }
    source_harm = {
        name: summary["v7"]["strict_harm_rate"]
        for name, summary in summaries.items()
    }
    accepted_by_source = {
        name: summary["accepted_overwrites"] for name, summary in summaries.items()
    }
    criteria = [
        _criterion(
            "pooled_positive_with_strict_ci",
            pooled["v7"]["quality_delta"] > 0.0
            and pooled["v7"]["ci95_low"] > 0.0,
            {
                "delta": pooled["v7"]["quality_delta"],
                "ci95_low": pooled["v7"]["ci95_low"],
            },
            "pooled delta > 0 and paired-bootstrap 95% lower bound > 0",
        ),
        _criterion(
            "nonnegative_each_source",
            min(source_deltas.values()) >= 0.0,
            source_deltas,
            "each source point delta >= 0",
        ),
        _criterion(
            "bounded_strict_harm",
            pooled["v7"]["strict_harm_rate"] <= 0.01
            and max(source_harm.values()) <= 0.02,
            {"pooled": pooled["v7"]["strict_harm_rate"], "sources": source_harm},
            "pooled <= 0.01 and every source <= 0.02",
        ),
        _criterion(
            "verifier_operational_completion",
            min(completion.values()) >= 0.95,
            completion,
            "every source >= 0.95 on verifier-called rows",
        ),
        _criterion(
            "accepted_overwrite_breadth",
            sum(accepted_by_source.values()) >= 4
            and min(accepted_by_source.values()) >= 1,
            {"total": sum(accepted_by_source.values()), "sources": accepted_by_source},
            "at least four total and at least one on each source",
        ),
        _criterion(
            "integrity_manifests",
            bool(preflight["passed"]),
            preflight["checks"],
            "all task, source, seed, model, prompt, code, and stage checks pass",
        ),
    ]
    report = {
        "schema_version": 1,
        "claim_boundary": "independent_sealed_confirmation",
        "method": "egor_v7_verified_overwrite",
        "source_commit": args.source_commit,
        "task_denominator": len(rows),
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "unit": "paired task with equal task weight",
        },
        "preflight": preflight,
        "pooled": pooled,
        "sources": summaries,
        "criteria": criteria,
        "passed": all(item["passed"] for item in criteria),
    }
    if args.rows_output:
        _write_rows(Path(args.rows_output).resolve(), rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def preflight_confirmation(
    *,
    sources: dict[str, tuple[Path, Path]],
    expected_commit: str,
    expected_tasks_per_source: int = 240,
) -> dict[str, Any]:
    source_reports: dict[str, Any] = {}
    visible: dict[str, list[dict[str, str]]] = {}
    for name, (tasks_path, root) in sources.items():
        audit_path = root / "audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        checksum_ok = _verify_sha256sums(root / "SOURCE_GENERATION_SHA256SUMS", root)
        source_commit = (root / "SOURCE_COMMIT").read_text(encoding="utf-8").strip()
        tasks_marker = (root / "TASKS_SHA256").read_text(encoding="utf-8").strip()
        task_sha = _sha256(tasks_path)
        marker = root / "SOURCE_GENERATION_COMPLETE"
        checks = {
            "completion_marker": marker.is_file(),
            "checksums": checksum_ok,
            "source_commit": source_commit == expected_commit,
            "task_hash_marker": tasks_marker == task_sha,
            "audit_mode": audit.get("mode") == "confirmation",
            "audit_passed": audit.get("passed") is True,
            "audit_integrity": audit.get("integrity_passed") is True,
            "audit_outcome_blind": audit.get("outcome_fields_read") is False
            and audit.get("scores_computed") is False,
            "audit_commit": audit.get("source_commit") == expected_commit,
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

    all_task_keys = [
        (row["source_id"], row["task_id"])
        for values in visible.values()
        for row in values
    ]
    all_images = [row["image_path"] for values in visible.values() for row in values]
    cross_checks = {
        "two_sources": set(sources) == {"chartqa", "infographic"},
        "expected_tasks_each": all(
            len(values) == expected_tasks_per_source for values in visible.values()
        ),
        "unique_source_task_keys": len(set(all_task_keys)) == len(all_task_keys),
        "unique_images": len(set(all_images)) == len(all_images),
        "disjoint_task_ids": not (
            {row["task_id"] for row in visible["chartqa"]}
            & {row["task_id"] for row in visible["infographic"]}
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


def analyze_source(
    *,
    source: str,
    tasks: list[PilotTask],
    root: Path,
    verifier_filename: str = "verifier_v7_qwen35.jsonl",
    answer_only: bool = False,
    policy_key: str = "v7",
) -> list[dict[str, Any]]:
    source_files = {**SOURCE_FILES, "verifier": verifier_filename}
    records = {
        name: load_latest_stage_records(root / filename)
        for name, filename in source_files.items()
        if name in {"baseline", "exact", "visual", "core", "verifier"}
    }
    repairability = _load_decisions(
        root / SOURCE_FILES["repairability"], "repairability_triggered"
    )
    overwrite = _load_decisions(
        root / SOURCE_FILES["overwrite"], "verifier_triggered"
    )
    rows: list[dict[str, Any]] = []
    for task in tasks:
        key = (task.source_id, task.task_id)
        triggered = bool(repairability[key]["repairability_triggered"])
        verifier_called = bool(overwrite[key]["verifier_triggered"])
        baseline_record = records["baseline"].get(key)
        exact_record = records["exact"].get(key)
        visual_record = records["visual"].get(key)
        core_record = records["core"].get(key)
        verifier_record = records["verifier"].get(key)
        baseline = _parsed(baseline_record)
        exact = _parsed(exact_record)
        visual = _parsed(visual_record)
        core = _parsed(core_record)
        verifier = _parsed(verifier_record)
        baseline_answer = str(baseline.get("answer", ""))
        core_answer = str(core.get("answer", ""))
        options = [{"id": key, "text": text} for key, text in task.options]
        baseline_quality = _score(task, baseline_answer) if _ok(baseline_record) else 0.0
        core_quality = _score(task, core_answer) if _ok(core_record) else 0.0
        core_changed = bool(
            triggered
            and _ok(core_record)
            and canonical_answer(core_answer)
            and not answers_agree(
                baseline_answer, core_answer, metric=task.metric, options=options
            )
        )
        if core_changed != verifier_called:
            raise ValueError(f"frozen overwrite plan drift for {task.source_id}::{task.task_id}")
        transport = verifier.get("transport", {}) if isinstance(verifier, dict) else {}
        operational = bool(
            verifier_called
            and _ok(verifier_record)
            and isinstance(transport, Mapping)
            and transport.get("operational_complete")
        )
        if operational:
            verifier_view = {
                "candidate_answer": verifier.get("answer", ""),
                "evidence": verifier.get("evidence", []),
                "transport": transport,
            }
        else:
            verifier_view = {
                "candidate_answer": "",
                "evidence": [],
                "transport": {"operational_complete": False},
            }
        decision_function = (
            decide_answer_only_verified_overwrite
            if answer_only
            else decide_verified_overwrite
        )
        decision = decision_function(
            baseline_answer=baseline_answer,
            core_answer=core_answer,
            verifier=verifier_view,
            specialists={"exact": exact, "visual": visual},
            metric=task.metric,
            options=options,
        )
        accept = bool(verifier_called and operational and decision.accept_overwrite)
        strict_pair = bool(
            core_changed
            and valid_evidence_cards(exact)
            and valid_evidence_cards(visual)
            and answers_agree(
                exact.get("candidate_answer", ""),
                visual.get("candidate_answer", ""),
                metric=task.metric,
                options=options,
            )
            and answers_agree(
                exact.get("candidate_answer", ""),
                core_answer,
                metric=task.metric,
                options=options,
            )
        )
        core_policy_quality = core_quality if triggered else baseline_quality
        strict_quality = core_quality if strict_pair else baseline_quality
        method_quality = core_quality if accept else baseline_quality
        baseline_usage = _usage_total([baseline_record])
        core_usage = (
            _usage_total([baseline_record, exact_record, visual_record, core_record])
            if triggered
            else baseline_usage
        )
        method_usage = (
            _usage_add(core_usage, _usage_total([verifier_record]))
            if verifier_called
            else core_usage
        )
        rows.append(
            {
                "source": source,
                "source_id": task.source_id,
                "task_id": task.task_id,
                "repairability_triggered": triggered,
                "core_changed": core_changed,
                "verifier_called": verifier_called,
                "verifier_operational": operational,
                "transport_mode": str(transport.get("mode", "invalid")),
                "sanitized_abstention": bool(transport.get("sanitized_abstention")),
                "accept_overwrite": accept,
                "decision_reason": decision.reason,
                "corroborators": list(decision.corroborating_specialists),
                "baseline_quality": baseline_quality,
                "core_quality": core_policy_quality,
                "strict_pair_quality": strict_quality,
                f"{policy_key}_quality": method_quality,
                "core_delta": core_policy_quality - baseline_quality,
                "strict_pair_delta": strict_quality - baseline_quality,
                f"{policy_key}_delta": method_quality - baseline_quality,
                "baseline_usage": baseline_usage,
                "core_usage": core_usage,
                "strict_pair_usage": core_usage,
                f"{policy_key}_usage": method_usage,
                "failed_stages": [
                    name
                    for name, record in (
                        ("baseline", baseline_record),
                        ("exact", exact_record if triggered else {"status": "ok"}),
                        ("visual", visual_record if triggered else {"status": "ok"}),
                        ("core", core_record if triggered else {"status": "ok"}),
                        ("verifier", verifier_record if verifier_called else {"status": "ok"}),
                    )
                    if not _ok(record)
                ],
            }
        )
    return rows


def summarize_source(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
    policy_key: str = "v7",
) -> dict[str, Any]:
    controls = {
        "baseline": _summarize_policy(rows, "baseline", bootstrap_samples=0, seed=seed),
        "core": _summarize_policy(rows, "core", bootstrap_samples=0, seed=seed),
        "strict_pair": _summarize_policy(rows, "strict_pair", bootstrap_samples=0, seed=seed),
        policy_key: _summarize_policy(
            rows, policy_key, bootstrap_samples=bootstrap_samples, seed=seed
        ),
    }
    called = [row for row in rows if row["verifier_called"]]
    return {
        **controls,
        "tasks": len(rows),
        "repairability_triggered": sum(row["repairability_triggered"] for row in rows),
        "core_changed": sum(row["core_changed"] for row in rows),
        "accepted_overwrites": sum(row["accept_overwrite"] for row in rows),
        "rejected_core_changes": sum(
            row["core_changed"] and not row["accept_overwrite"] for row in rows
        ),
        "verifier_operational_completion": (
            fmean(row["verifier_operational"] for row in called) if called else 0.0
        ),
        "transport_modes": dict(Counter(row["transport_mode"] for row in called)),
        "sanitized_abstentions": sum(row["sanitized_abstention"] for row in called),
        "failure_counts": dict(
            Counter(stage for row in rows for stage in row["failed_stages"])
        ),
        "decision_reasons": dict(Counter(row["decision_reason"] for row in rows)),
    }


def _summarize_policy(
    rows: list[dict[str, Any]],
    policy: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    quality_key = f"{policy}_quality" if policy != "baseline" else "baseline_quality"
    delta_key = f"{policy}_delta" if policy != "baseline" else ""
    usage_key = f"{policy}_usage"
    qualities = [float(row[quality_key]) for row in rows]
    deltas = [float(row[delta_key]) if delta_key else 0.0 for row in rows]
    low, high = (
        _bootstrap_interval(deltas, samples=bootstrap_samples, seed=seed)
        if bootstrap_samples
        else (None, None)
    )
    usage = [row[usage_key] for row in rows]
    wins = sum(value > 0 for value in deltas)
    losses = sum(value < 0 for value in deltas)
    return {
        "quality": fmean(qualities),
        "quality_delta": fmean(deltas),
        "ci95_low": low,
        "ci95_high": high,
        "wins": wins,
        "losses": losses,
        "ties": len(deltas) - wins - losses,
        "strict_harm_rate": fmean(value < -0.05 for value in deltas),
        "useful_repair_rate": fmean(value > 0.05 for value in deltas),
        "exact_sign_p_two_sided": _exact_sign_p(wins, losses),
        "mean_calls": fmean(float(item["calls"]) for item in usage),
        "mean_total_tokens": fmean(float(item["total_tokens"]) for item in usage),
        "mean_latency_seconds": fmean(float(item["latency_seconds"]) for item in usage),
    }


def _score(task: PilotTask, answer: str) -> float:
    return score_prediction(
        metric=task.metric,
        predicted=answer,
        answer=task.answer,
        reference_answers=task.reference_answers,
        options=[{"id": key, "text": text} for key, text in task.options],
    )


def _parsed(record: Mapping[str, Any] | None) -> dict[str, Any]:
    if not _ok(record):
        return {}
    value = record.get("parsed", {}) if record else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _ok(record: Mapping[str, Any] | None) -> bool:
    return bool(record and str(record.get("status", "")) == "ok")


def _usage_total(records: list[Mapping[str, Any] | None]) -> dict[str, float]:
    output = {"calls": 0.0, "total_tokens": 0.0, "latency_seconds": 0.0}
    for record in records:
        usage = record.get("usage", {}) if record else {}
        if not isinstance(usage, Mapping):
            continue
        output["calls"] += _float(usage.get("attempts", 0.0))
        output["total_tokens"] += _float(usage.get("total_tokens", 0.0))
        output["latency_seconds"] += _float(usage.get("latency_seconds", 0.0))
    return output


def _usage_add(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: float(left[key]) + float(right[key]) for key in left}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_decisions(path: Path, field: str) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row.get("source_id", "")), str(row.get("task_id", "")))
        if not all(key) or key in output or not isinstance(row.get(field), bool):
            raise ValueError(f"invalid {field} plan row: {key}")
        output[key] = row
    return output


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


def _bootstrap_interval(
    values: list[float], *, samples: int, seed: int
) -> tuple[float, float]:
    generator = random.Random(seed)
    count = len(values)
    means = sorted(
        fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def _exact_sign_p(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**total))


def _criterion(name: str, passed: bool, actual: Any, required: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "actual": actual, "required": required}


def _write_rows(
    path: Path, rows: list[dict[str, Any]], *, policy_key: str = "v7"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source", "source_id", "task_id", "repairability_triggered",
        "core_changed", "verifier_called", "verifier_operational",
        "transport_mode", "sanitized_abstention", "accept_overwrite",
        "decision_reason", "baseline_quality", "core_quality",
        "strict_pair_quality", f"{policy_key}_quality", "core_delta",
        "strict_pair_delta", f"{policy_key}_delta", "failed_stages",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = {field: row[field] for field in fields}
            payload["failed_stages"] = "|".join(row["failed_stages"])
            writer.writerow(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chartqa-tasks", required=True)
    parser.add_argument("--chartqa-root", required=True)
    parser.add_argument("--infographic-tasks", required=True)
    parser.add_argument("--infographic-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--rows-output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
