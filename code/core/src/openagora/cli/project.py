"""Portable offline commands; optional experiments are imported only when selected."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

import typer

from openagora.adapters.qa_tools import LocalToolAdapter, QAToolTaskAdapter
from openagora.config.models import RunConfig
from openagora.core.engine import OpenAGORAEngine
from openagora.providers.mock import MockModelProvider
from openagora.storage.sqlite_store import SQLiteStore


def offline_demo(directory: Path) -> dict[str, Any]:
    cfg = RunConfig.model_validate(
        {
            "task": {
                "task_id": "offline-qa",
                "prompt": "Use tools to answer: what is 2 + 3?",
                "metadata": {
                    "reference_answer": "5",
                    "primary_type": "multi_step_qa",
                    "uncertainty": "high",
                    "complexity": "high",
                    "interdependence": "reciprocal",
                },
            },
            "paths": {"db_path": str(directory / "demo.db"), "artifacts_dir": str(directory / "artifacts")},
        }
    )
    engine = OpenAGORAEngine(
        provider=MockModelProvider(),
        task_adapter=QAToolTaskAdapter(),
        tools=LocalToolAdapter(),
        store=SQLiteStore(cfg.paths.db_path),
    )
    first = engine.run(cfg.task, cfg)
    second = engine.run(cfg.task, cfg)
    return {
        "example": "synthetic mock task; not a model benchmark",
        "result": first.result,
        "metrics": first.metrics,
        "second_run_cache_hit": second.cache_used,
    }


def register_project_commands(app: typer.Typer) -> None:
    @app.command("demo")
    def demo(output: Annotated[Path | None, typer.Option("--output", help="Keep demo artifacts here")] = None) -> None:
        """Execute a tiny, deterministic mock task; no models or network needed."""
        if output is None:
            with TemporaryDirectory(prefix="openagora-demo-") as directory:
                result = offline_demo(Path(directory))
        else:
            result = offline_demo(output.expanduser().resolve())
        typer.echo(json.dumps(result, ensure_ascii=False))

    @app.command("reproduce")
    def reproduce(evidence: Annotated[Path | None, typer.Option("--evidence")] = None) -> None:
        """Verify the sealed EGOR-v10 rows and recompute paired outcome counts."""
        try:
            from openagora_experiments.reproduction import verify_evidence
        except ImportError as exc:
            typer.echo("Install openagora-experiments (uv sync in the OpenMAS workspace).", err=True)
            raise typer.Exit(code=3) from exc
        try:
            typer.echo(json.dumps(verify_evidence(evidence), ensure_ascii=False))
        except (OSError, ValueError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=5) from exc
