"""One-command local demo for OpenAGORA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openagora.config.loader import load_run_config
from openagora.core.engine import OpenAGORAEngine
from openagora.plugins.registry import build_default_registry
from openagora.storage.sqlite_store import SQLiteStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAGORA local demo")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    registry = build_default_registry()
    registry.discover(cfg.plugins.discovery_modules)

    engine = OpenAGORAEngine(
        provider=registry.create_provider(cfg),
        task_adapter=registry.create_task_adapter(cfg),
        tools=registry.create_tool_adapter(cfg),
        store=SQLiteStore(cfg.paths.db_path),
    )
    result = engine.run(cfg.task, cfg)
    print(json.dumps({"run_id": result.run_id, "task_id": result.task_id, "quality": result.metrics["quality_score"]}))


if __name__ == "__main__":
    main()
