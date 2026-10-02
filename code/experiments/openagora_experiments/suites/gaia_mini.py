"""gaia_mini suite and evidence-chain runner."""

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
from openagora.utils.artifacts import build_file_hashes, write_hash_manifest

from openagora_experiments.adapters.gaia_mini import load_gaia_manifest, load_gaia_mini_tasks
from openagora_experiments.evaluation.judge import build_hybrid_judge
from openagora_experiments.evaluation.protocol import evaluate_rule
from openagora_experiments.evaluation.stats import build_failure_buckets, build_pairwise_stats, build_summary


def run_gaia_mini_ablation(
    config_path: str,
    output_csv: str,
    summary_csv: str,
    pairwise_csv: str,
    failure_csv: str,
    judge_audit_path: str,
    *,
    sample_limit: int | None = None,
    judge_mode: str | None = None,
    seed: int | None = None,
    variant_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run GAIA-40 variants and export evidence-chain artifacts."""
    base_cfg = load_run_config(config_path)
    cfg = _override_cfg(base_cfg, sample_limit=sample_limit, judge_mode=judge_mode, seed=seed)

    manifest = load_gaia_manifest(cfg.eval.gaia_manifest_path)
    tasks = load_gaia_mini_tasks(
        cfg.eval.dataset_path,
        manifest_path=cfg.eval.gaia_manifest_path,
        limit=cfg.eval.sample_limit,
    )

    rows: list[dict[str, Any]] = []
    judge_audits: list[dict[str, Any]] = []
    variants = _variant_specs(base_cfg_worker_count=cfg.runtime.worker_count)
    active_variants = (
        {name: spec for name, spec in variants.items() if name in set(variant_filter or [])}
        if variant_filter
        else variants
    )
    for variant_name, spec in active_variants.items():
        variant_cfg = _variant_config(
            cfg,
            toggles=spec["toggles"],
            worker_count=spec["worker_count"],
            always_on_rt=bool(spec.get("always_on_rt", False)),
            ot_mode=str(spec.get("ot_mode", "default")),
        )
        engine = _build_engine(variant_cfg)
        judge = build_hybrid_judge(variant_cfg, seed=variant_cfg.eval.stats_seed)
        for task in tasks:
            result = engine.run(task, variant_cfg)
            predicted = str(result.result.get("answer", "")).strip()
            expected = str(task.metadata.get("reference_answer", "")).strip()
            aliases = _as_str_list(task.metadata.get("answer_aliases", []))

            rule_eval = evaluate_rule(
                predicted=predicted,
                expected=expected,
                answer_aliases=aliases,
                borderline_low=variant_cfg.eval.borderline_low,
                borderline_high=variant_cfg.eval.borderline_high,
            )
            final_eval, audit = judge.evaluate(
                task_id=task.task_id,
                variant=variant_name,
                predicted=predicted,
                expected=expected,
                rule_eval=rule_eval,
            )
            if audit is not None:
                judge_audits.append(audit)

            rows.append(
                {
                    "suite": "gaia_mini",
                    "manifest_version": str(manifest.get("version", "unknown")),
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
                    "predicted_answer": predicted,
                    "expected_answer": expected,
                    "rule_score": rule_eval.rule_score,
                    "rule_verdict": rule_eval.rule_verdict,
                    "rule_conflict": int(rule_eval.conflict),
                    "rule_exact": int(rule_eval.exact_match),
                    "rule_normalized": int(rule_eval.normalized_match),
                    "rule_numeric": int(rule_eval.numeric_match),
                    "rule_alias": int(rule_eval.alias_match),
                    "rule_overlap": rule_eval.overlap,
                    "final_score": final_eval.final_score,
                    "final_verdict": final_eval.final_verdict,
                    "judge_source": final_eval.source,
                    "is_correct": int(final_eval.is_correct),
                    "error_type": final_eval.error_type,
                    "judge_mode": variant_cfg.eval.judge_mode,
                }
            )

    _write_ablation_outputs(
        cfg, rows, judge_audits, manifest, output_csv, summary_csv, pairwise_csv, failure_csv, judge_audit_path
    )
    return rows


def _variant_specs(*, base_cfg_worker_count: int) -> dict[str, dict[str, Any]]:
    disabled = {
        "enable_negotiation": False,
        "enable_runtime_reconfig": False,
        "enable_cache": False,
    }
    return {
        "OpenAGORA_Full": {"worker_count": base_cfg_worker_count, "toggles": {}},
        "OpenAGORA_AlwaysOnRT": {
            "worker_count": base_cfg_worker_count,
            "toggles": {},
            "always_on_rt": True,
        },
        "NoOT": {
            "worker_count": base_cfg_worker_count,
            "toggles": {},
            "ot_mode": "fixed_ss",
        },
        "SingleAgentTool": {"worker_count": 1, "toggles": disabled},
        "PlannerExecutor2": {"worker_count": 2, "toggles": disabled},
        "Static4AgentNoAdapt": {"worker_count": 4, "toggles": disabled},
        "NoNeg": {"worker_count": base_cfg_worker_count, "toggles": {"enable_negotiation": False}},
        "NoRT": {"worker_count": base_cfg_worker_count, "toggles": {"enable_runtime_reconfig": False}},
        "NoCache": {"worker_count": base_cfg_worker_count, "toggles": {"enable_cache": False}},
    }


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
    *,
    toggles: dict[str, bool],
    worker_count: int,
    always_on_rt: bool = False,
    ot_mode: str = "default",
) -> RunConfig:
    merged = cfg.model_copy(deep=True)
    toggle_payload = merged.toggles.model_dump(mode="json")
    toggle_payload.update(toggles)
    merged.toggles = merged.toggles.model_validate(toggle_payload)

    runtime_payload = merged.runtime.model_dump(mode="json")
    runtime_payload["worker_count"] = worker_count
    merged.runtime = merged.runtime.model_validate(runtime_payload)
    object.__setattr__(merged, "_always_on_rt", always_on_rt)
    object.__setattr__(merged, "_ot_mode", ot_mode)
    return merged


def _override_cfg(
    cfg: RunConfig,
    *,
    sample_limit: int | None,
    judge_mode: str | None,
    seed: int | None,
) -> RunConfig:
    next_cfg = cfg.model_copy(deep=True)
    eval_payload = next_cfg.eval.model_dump(mode="json")
    if sample_limit is not None:
        eval_payload["sample_limit"] = sample_limit
    if judge_mode is not None:
        eval_payload["judge_mode"] = judge_mode
    if seed is not None:
        eval_payload["stats_seed"] = seed
    next_cfg.eval = next_cfg.eval.model_validate(eval_payload)

    if seed is not None:
        provider_payload = next_cfg.provider.model_dump(mode="json")
        provider_payload["seed"] = seed
        next_cfg.provider = next_cfg.provider.model_validate(provider_payload)
    return next_cfg


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _write_run_artifacts(
    *,
    cfg: RunConfig,
    manifest: dict[str, Any],
    rows_path: Path,
    summary_path: Path,
    pairwise_path: Path,
    failure_path: Path,
    judge_audit_path: Path,
) -> None:
    payload = {
        "suite": "gaia_mini",
        "version": str(manifest.get("version", "unknown")),
        "dataset": cfg.eval.dataset_path,
        "sample_limit": cfg.eval.sample_limit,
        "judge_mode": cfg.eval.judge_mode,
        "seed": cfg.eval.stats_seed,
        "bootstrap_n": cfg.eval.stats_bootstrap_n,
    }
    manifest_out = Path(cfg.eval.output_manifest_json)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    files = [rows_path, summary_path, pairwise_path, failure_path, manifest_out]
    if cfg.eval.judge_mode == "hybrid":
        files.append(judge_audit_path)
    hashes = build_file_hashes(files)
    write_hash_manifest(Path(cfg.eval.output_hashes_json), hashes)


def _write_ablation_outputs(
    cfg: RunConfig,
    rows: list[dict[str, Any]],
    judge_audits: list[dict[str, Any]],
    manifest: dict[str, Any],
    output_csv: str,
    summary_csv: str,
    pairwise_csv: str,
    failure_csv: str,
    judge_audit_path: str,
) -> None:
    _write_csv(output_csv, rows)
    _write_csv(summary_csv, build_summary(rows))
    _write_csv(
        pairwise_csv,
        build_pairwise_stats(
            rows,
            reference_variant="OpenAGORA_Full",
            bootstrap_n=cfg.eval.stats_bootstrap_n,
            seed=cfg.eval.stats_seed,
        ),
    )
    _write_csv(failure_csv, build_failure_buckets(rows))
    if cfg.eval.judge_mode == "hybrid":
        _write_jsonl(judge_audit_path, judge_audits)

    _write_run_artifacts(
        cfg=cfg,
        manifest=manifest,
        rows_path=Path(output_csv),
        summary_path=Path(summary_csv),
        pairwise_path=Path(pairwise_csv),
        failure_path=Path(failure_csv),
        judge_audit_path=Path(judge_audit_path),
    )
    return None
