"""Typer CLI for OpenAGORA."""

from __future__ import annotations

import csv
import json
import os
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal

import typer

from openagora.config.loader import dump_config, load_run_config
from openagora.config.models import RunConfig
from openagora.core.engine import OpenAGORAEngine
from openagora.core.errors import ConfigurationError, normalize_error
from openagora.core.interfaces import ModelProvider
from openagora.plugins.registry import build_default_registry
from openagora.storage.sqlite_store import SQLiteStore
from openagora.templates.runtime import build_template_runtime
from openagora.types import TaskSpec
from openagora.utils.artifacts import (
    build_file_hashes,
    export_run_bundle,
    fingerprint_run_result,
    read_run_result,
    write_hash_manifest,
)
from openagora.utils.audit import assert_event_integrity
from openagora.utils.hash_utils import sha256_json

app = typer.Typer(no_args_is_help=True, add_completion=False)
template_app = typer.Typer(no_args_is_help=True, add_completion=False)
manifest_app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(template_app, name="template")
app.add_typer(manifest_app, name="manifest")


class ExitCode(IntEnum):
    OK = 0
    CONFIG_ERROR = 2
    ENV_ERROR = 3
    RUNTIME_ERROR = 4
    VALIDATION_ERROR = 5


def _build_engine(cfg: RunConfig) -> OpenAGORAEngine:
    store = SQLiteStore(cfg.paths.db_path)
    registry = build_default_registry()
    registry.discover(cfg.plugins.discovery_modules)
    provider = registry.create_provider(cfg)
    return OpenAGORAEngine(
        provider=provider,
        task_adapter=registry.create_task_adapter(cfg),
        tools=registry.create_tool_adapter(cfg),
        store=store,
    )


@app.command("run")
def run_command(
    config: str = typer.Option(..., "--config", help="Path to YAML config"),
    provider_mode: Literal["mock", "live", "local"] | None = typer.Option(
        None,
        "--provider-mode",
        help="Override provider mode",
    ),
) -> None:
    """Run one task defined in config."""
    try:
        cfg = _load_with_override(config, provider_mode=provider_mode)
        engine = _build_engine(cfg)
        result = engine.run(cfg.task, cfg)
        typer.echo(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))
    except Exception as exc:
        _exit_with_error(exc)


@app.command("replay")
def replay_command(
    run_id: str = typer.Option(..., "--run-id", help="Run ID"),
    config: str = typer.Option(..., "--config", help="Path to YAML config"),
) -> None:
    """Replay one completed run from persisted artifact."""
    try:
        cfg = load_run_config(config)
        engine = _build_engine(cfg)
        result = engine.replay(run_id)
        typer.echo(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))
    except Exception as exc:
        _exit_with_error(exc)


@app.command("eval")
def eval_command(
    suite: str = typer.Option(..., "--suite", help="Evaluation suite"),
    config: str = typer.Option(..., "--config", help="Path to YAML config"),
    provider_mode: Literal["mock", "live", "local"] | None = typer.Option(
        None,
        "--provider-mode",
        help="Override provider mode",
    ),
) -> None:
    """Run evaluation suite and export rows/summary/failure_cases CSVs."""
    try:
        if suite not in {"qa_mini", "gaia_mini"}:
            raise ConfigurationError("Only --suite qa_mini|gaia_mini is implemented", code="invalid_suite")
        cfg = _load_with_override(config, provider_mode=provider_mode)
        engine = _build_engine(cfg)
        dataset = _load_jsonl(cfg.eval.dataset_path)
        rows = _eval_rows(engine, cfg, dataset, suite=suite)
        _write_rows_csv(cfg.eval.output_csv, rows)
        summary = _summarize_rows(rows)
        _write_rows_csv(cfg.eval.output_summary_csv, [summary])
        failure_rows = [row for row in rows if not row["is_correct"]]
        _write_rows_csv(cfg.eval.output_failure_csv, failure_rows)
        _write_eval_artifacts(cfg=cfg, suite=suite, row_count=len(rows))
        typer.echo(f"Wrote rows: {cfg.eval.output_csv}")
        typer.echo(f"Wrote summary: {cfg.eval.output_summary_csv}")
        typer.echo(f"Wrote failures: {cfg.eval.output_failure_csv}")
    except Exception as exc:
        _exit_with_error(exc)


@app.command("doctor")
def doctor_command(
    config: str = typer.Option(..., "--config", help="Path to YAML config"),
    provider_mode: Literal["mock", "live", "local"] | None = typer.Option(None, "--provider-mode"),
) -> None:
    """Validate environment, plugins, templates and storage schema."""
    try:
        cfg = _load_with_override(config, provider_mode=provider_mode)
        registry = build_default_registry()
        registry.discover(cfg.plugins.discovery_modules)
        provider: ModelProvider = registry.create_provider(cfg)
        template_runtime = build_template_runtime(cfg)
        personas = template_runtime.assign_personas(cfg=cfg, worker_count=cfg.runtime.worker_count)
        store = SQLiteStore(cfg.paths.db_path)

        checks: dict[str, Any] = {
            "config_path": cfg.config_path,
            "provider_mode": cfg.provider.provider_mode,
            "provider_name": cfg.provider.name,
            "provider_model": cfg.provider.model,
            "provider_impl": provider.__class__.__name__,
            "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
            "task_adapter": cfg.plugins.task_adapter,
            "tool_adapter": cfg.plugins.tool_adapter,
            "plugin_modules": cfg.plugins.discovery_modules,
            "template_id": template_runtime.template_id,
            "template_version": template_runtime.version,
            "persona_count": len(personas),
            "db_path": cfg.paths.db_path,
            "db_schema_version": store.schema_version(),
            "event_schema_version": cfg.audit.event_schema_version,
            "status": "ok",
        }

        if cfg.provider.provider_mode == "live" and cfg.provider.name.lower() == "openai" and not checks[
            "openai_api_key_present"
        ]:
            checks["status"] = "degraded"
            checks["warning"] = "OPENAI_API_KEY missing in live mode"

        typer.echo(json.dumps(checks, ensure_ascii=False))
    except Exception as exc:
        _exit_with_error(exc)


@manifest_app.command("verify")
def manifest_verify_command(
    run_id: str = typer.Option(..., "--run-id", help="Run ID"),
    config: str = typer.Option(..., "--config", help="Path to YAML config"),
    compare_run_id: str | None = typer.Option(None, "--compare-run-id", help="Optional second run_id"),
) -> None:
    """Verify manifest/events/result consistency and reproducibility fingerprint."""
    try:
        cfg = load_run_config(config)
        store = SQLiteStore(cfg.paths.db_path)
        run = _require_run(store, run_id)

        manifest_path = Path(str(run["manifest_path"]))
        events_path = Path(str(run["events_path"]))
        result_path = Path(str(run["result_path"]))

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_config_hash = sha256_json(dump_config(cfg))
        config_match = manifest.get("config_hash") == current_config_hash

        integrity = assert_event_integrity(
            events_path=events_path,
            schema_version=str(manifest.get("event_schema_version", "v2")),
        )
        run_result = read_run_result(result_path)
        fingerprint = fingerprint_run_result(run_result)

        payload: dict[str, Any] = {
            "run_id": run_id,
            "manifest_path": str(manifest_path),
            "events_path": str(events_path),
            "result_path": str(result_path),
            "config_hash_match": config_match,
            "fingerprint": fingerprint,
            "event_count": integrity["event_count"],
            "redacted_hits": integrity["redacted_hits"],
            "prompt_truncations": integrity["prompt_truncations"],
            "status": "ok" if config_match else "mismatch",
        }

        if compare_run_id:
            other_run = _require_run(store, compare_run_id)
            other_result = read_run_result(Path(str(other_run["result_path"])))
            other_fp = fingerprint_run_result(other_result)
            payload["compare_run_id"] = compare_run_id
            payload["compare_fingerprint"] = other_fp
            payload["fingerprint_match"] = fingerprint == other_fp

        typer.echo(json.dumps(payload, ensure_ascii=False))
        if not config_match:
            raise typer.Exit(code=int(ExitCode.VALIDATION_ERROR))
    except typer.Exit:
        raise
    except Exception as exc:
        _exit_with_error(exc)


@app.command("export")
def export_command(
    run_id: str = typer.Option(..., "--run-id", help="Run ID"),
    config: str = typer.Option(..., "--config", help="Path to YAML config"),
    output: str | None = typer.Option(None, "--output", help="Output zip path"),
) -> None:
    """Export a run bundle with manifest/events/result/hashes."""
    try:
        cfg = load_run_config(config)
        store = SQLiteStore(cfg.paths.db_path)
        run = _require_run(store, run_id)

        if output is None:
            output_path = Path(cfg.paths.artifacts_dir).resolve() / "exports" / f"{run_id}.zip"
        else:
            output_path = Path(output).resolve()

        exported = export_run_bundle(
            run_id=run_id,
            manifest_path=Path(str(run["manifest_path"])),
            events_path=Path(str(run["events_path"])),
            result_path=Path(str(run["result_path"])),
            output_path=output_path,
        )
        typer.echo(str(exported))
    except Exception as exc:
        _exit_with_error(exc)


@template_app.command("validate")
def template_validate(config: str = typer.Option(..., "--config", help="Path to YAML config")) -> None:
    """Validate template registry and required schema."""
    try:
        cfg = load_run_config(config)
        runtime = build_template_runtime(cfg)
        personas = runtime.assign_personas(cfg=cfg, worker_count=cfg.runtime.worker_count)
        typer.echo(
            f"Template OK: id={runtime.template_id} version={runtime.version} personas={len(personas)}"
        )
    except Exception as exc:
        _exit_with_error(exc)


def _load_with_override(config: str, *, provider_mode: Literal["mock", "live", "local"] | None) -> RunConfig:
    cfg = load_run_config(config)
    if provider_mode is None:
        return cfg
    provider_payload = cfg.provider.model_dump(mode="json")
    provider_payload["provider_mode"] = provider_mode
    return cfg.model_copy(update={"provider": cfg.provider.model_validate(provider_payload)})


def _eval_rows(
    engine: OpenAGORAEngine,
    cfg: RunConfig,
    dataset: list[dict[str, Any]],
    *,
    suite: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in dataset:
        task = TaskSpec.model_validate(row)
        result = engine.run(task, cfg)
        expected = str(task.metadata.get("reference_answer", "")).strip()
        answer = str(result.result.get("answer", "")).strip()
        is_correct = _normalized_match(answer, expected)
        rows.append(
            {
                "suite": suite,
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
                "expected_answer": expected,
                "predicted_answer": answer,
                "is_correct": is_correct,
            }
        )
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    count = float(len(rows))
    numeric_cols = ["quality_score", "token_consumption", "latency_seconds", "cost_usd", "perf_cost_ratio"]
    summary: dict[str, Any] = {"count": int(count), "suite": rows[0]["suite"]}
    for col in numeric_cols:
        summary[f"{col}_mean"] = sum(float(row[col]) for row in rows) / count
    summary["accuracy"] = sum(1.0 if bool(row["is_correct"]) else 0.0 for row in rows) / count
    summary["cache_hit_rate"] = sum(float(row["cache_used"]) for row in rows) / count
    return summary


def _write_rows_csv(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_eval_artifacts(*, cfg: RunConfig, suite: str, row_count: int) -> None:
    manifest_path = Path(cfg.eval.output_manifest_json)
    hashes_path = Path(cfg.eval.output_hashes_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "suite": suite,
        "dataset_path": cfg.eval.dataset_path,
        "sample_limit": cfg.eval.sample_limit,
        "judge_mode": cfg.eval.judge_mode,
        "rows": row_count,
        "seed": cfg.provider.seed,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    files = [
        Path(cfg.eval.output_csv),
        Path(cfg.eval.output_summary_csv),
        Path(cfg.eval.output_failure_csv),
        manifest_path,
    ]
    hashes = build_file_hashes(files)
    write_hash_manifest(hashes_path, hashes)


def _normalized_match(predicted: str, expected: str) -> bool:
    if not expected:
        return False
    p = " ".join(predicted.strip().lower().split())
    e = " ".join(expected.strip().lower().split())
    return e in p or p == e


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip().lstrip("\ufeff")
            if not text:
                continue
            out.append(json.loads(text))
    return out


def _require_run(store: SQLiteStore, run_id: str) -> dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise ConfigurationError(f"run_id not found: {run_id}", code="run_id_not_found")
    if not run.get("result_path"):
        raise ConfigurationError(f"run_id has no result_path: {run_id}", code="result_missing")
    return run


def _exit_with_error(exc: Exception) -> None:
    info = normalize_error(exc)
    typer.echo(
        json.dumps(
            {
                "status": "error",
                "error_type": info.error_type,
                "category": info.category,
                "code": info.code,
                "retryable": info.retryable,
                "message": info.message,
            },
            ensure_ascii=False,
        ),
        err=True,
    )
    if info.category == "configuration":
        raise typer.Exit(code=int(ExitCode.CONFIG_ERROR))
    if info.category == "external_dependency":
        raise typer.Exit(code=int(ExitCode.ENV_ERROR))
    raise typer.Exit(code=int(ExitCode.RUNTIME_ERROR))


if __name__ == "__main__":
    app()
