"""qa_mini suite and ablation runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from openagora.config.loader import load_run_config
from openagora.config.models import RunConfig
from openagora.core.engine import OpenAGORAEngine
from openagora.plugins.registry import build_default_registry
from openagora.storage.sqlite_store import SQLiteStore
from openagora.types import TaskSpec
from openagora.utils.artifacts import build_file_hashes, write_hash_manifest


def run_qa_mini_ablation(
    config_path: str,
    output_csv: str,
    summary_csv: str | None = None,
    failure_csv: str | None = None,
    *,
    sample_limit: int | None = None,
    seed: int | None = None,
    variant_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run full/cached/ablation variants and export standardized outputs."""
    base_cfg = load_run_config(config_path)
    cfg = _override_cfg(base_cfg, sample_limit=sample_limit, seed=seed)
    tasks = _load_tasks(cfg.eval.dataset_path, limit=cfg.eval.sample_limit)
    rows: list[dict[str, Any]] = []
    variants = {
        "OpenAGORA_Full": {"toggles": {}, "always_on_rt": False},
        "OpenAGORA_AlwaysOnRT": {"toggles": {}, "always_on_rt": True},
        "OpenAGORA_Cache": {"toggles": {"enable_cache": True}, "always_on_rt": False},
        "OpenAGORA_NoNeg": {"toggles": {"enable_negotiation": False}, "always_on_rt": False},
        "OpenAGORA_NoOT": {"toggles": {}, "always_on_rt": False, "ot_mode": "fixed_ss"},
        "OpenAGORA_NoRT": {"toggles": {"enable_runtime_reconfig": False}, "always_on_rt": False},
        "OpenAGORA_NoCache": {"toggles": {"enable_cache": False}, "always_on_rt": False},
    }
    active_variants = (
        {name: spec for name, spec in variants.items() if name in set(variant_filter or [])}
        if variant_filter
        else variants
    )
    for variant_name, spec in active_variants.items():
        variant_cfg = _variant_config(
            cfg,
            spec["toggles"],
            always_on_rt=bool(spec.get("always_on_rt", False)),
            ot_mode=str(spec.get("ot_mode", "default")),
        )
        engine = _build_engine(variant_cfg)
        for task in tasks:
            result = engine.run(task, variant_cfg)
            rows.append(
                {
                    "variant": variant_name,
                    "run_id": result.run_id,
                    "task_id": result.task_id,
                    "quality_score": result.metrics["quality_score"],
                    "token_consumption": result.metrics["token_consumption"],
                    "latency_seconds": result.metrics["latency_seconds"],
                    "cost_usd": result.metrics["cost_usd"],
                    "perf_cost_ratio": result.metrics["perf_cost_ratio"],
                    "ot_alignment": result.ot_alignment.score,
                    "cache_used": int(result.cache_used),
                    "reconfig_count": result.reconfig_count,
                    "predicted_answer": str(result.result.get("answer", "")),
                    "expected_answer": str(task.metadata.get("reference_answer", "")),
                    "final_score": result.metrics["quality_score"],
                    "is_correct": int(bool(result.result.get("is_correct", False))),
                }
            )
    _write_csv(output_csv, rows)
    if summary_csv:
        _write_csv(summary_csv, _build_summary(rows))
    if failure_csv:
        _write_csv(failure_csv, [row for row in rows if int(row["is_correct"]) == 0])

    _write_run_artifacts(
        cfg=cfg,
        rows_path=Path(output_csv),
        summary_path=Path(summary_csv) if summary_csv else None,
        failure_path=Path(failure_csv) if failure_csv else None,
    )
    return rows


def _build_engine(cfg: RunConfig) -> OpenAGORAEngine:
    store = SQLiteStore(cfg.paths.db_path)
    registry = build_default_registry()
    registry.discover(cfg.plugins.discovery_modules)
    return OpenAGORAEngine(
        provider=registry.create_provider(cfg),
        task_adapter=registry.create_task_adapter(cfg),
        tools=registry.create_tool_adapter(cfg),
        store=store,
    )


def _variant_config(
    cfg: RunConfig,
    toggles: dict[str, bool],
    *,
    always_on_rt: bool = False,
    ot_mode: str = "default",
) -> RunConfig:
    merged = cfg.model_copy(deep=True)
    current = merged.toggles.model_dump(mode="json")
    current.update(toggles)
    merged.toggles = merged.toggles.model_validate(current)
    object.__setattr__(merged, "_always_on_rt", always_on_rt)
    object.__setattr__(merged, "_ot_mode", ot_mode)
    return merged


def _override_cfg(
    cfg: RunConfig,
    *,
    sample_limit: int | None,
    seed: int | None,
) -> RunConfig:
    next_cfg = cfg.model_copy(deep=True)
    eval_payload = next_cfg.eval.model_dump(mode="json")
    if sample_limit is not None:
        eval_payload["sample_limit"] = sample_limit
    next_cfg.eval = next_cfg.eval.model_validate(eval_payload)
    if seed is not None:
        provider_payload = next_cfg.provider.model_dump(mode="json")
        provider_payload["seed"] = seed
        next_cfg.provider = next_cfg.provider.model_validate(provider_payload)
    return next_cfg


def _load_tasks(path: str, *, limit: int | None = None) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip().lstrip("\ufeff")
            if not text:
                continue
            tasks.append(TaskSpec.model_validate(json.loads(text)))
            if limit is not None and len(tasks) >= limit:
                break
    return tasks


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    bucket: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        variant = str(row["variant"])
        bucket.setdefault(variant, []).append(row)
    summary: list[dict[str, Any]] = []
    numeric_cols = ["quality_score", "token_consumption", "latency_seconds", "cost_usd", "perf_cost_ratio"]
    for variant, grouped in bucket.items():
        line: dict[str, Any] = {"suite": "qa_mini", "variant": variant, "count": len(grouped)}
        for col in numeric_cols:
            line[f"{col}_mean"] = sum(float(item[col]) for item in grouped) / len(grouped)
        line["accuracy"] = sum(int(item["is_correct"]) for item in grouped) / len(grouped)
        summary.append(line)
    return summary


def _write_run_artifacts(
    *,
    cfg: RunConfig,
    rows_path: Path,
    summary_path: Path | None,
    failure_path: Path | None,
) -> None:
    payload = {
        "suite": "qa_mini",
        "dataset": cfg.eval.dataset_path,
        "seed": cfg.provider.seed,
    }
    manifest_out = Path(cfg.eval.output_manifest_json)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    files = [rows_path, manifest_out]
    if summary_path is not None:
        files.append(summary_path)
    if failure_path is not None:
        files.append(failure_path)

    hashes = build_file_hashes(files)
    write_hash_manifest(Path(cfg.eval.output_hashes_json), hashes)
