"""Local-only OpenAI live smoke runner (excluded from CI)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openagora.adapters.qa_tools import LocalToolAdapter, QAToolTaskAdapter
from openagora.config.loader import load_run_config
from openagora.core.engine import OpenAGORAEngine
from openagora.providers.openai_provider import OpenAIModelProvider
from openagora.storage.sqlite_store import SQLiteStore
from openagora.types import TaskSpec


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAGORA local OpenAI live smoke.")
    parser.add_argument("--config", default="configs/default.yaml", help="Core config path")
    parser.add_argument("--sample-limit", type=int, default=3, help="How many tasks to run")
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for live smoke")

    cfg = load_run_config(args.config)
    provider_payload = cfg.provider.model_dump(mode="json")
    provider_payload["name"] = "openai"
    provider_payload["provider_mode"] = "live"
    cfg = cfg.model_copy(update={"provider": cfg.provider.model_validate(provider_payload)})
    tasks = _load_tasks(cfg.eval.dataset_path, limit=args.sample_limit)

    engine = OpenAGORAEngine(
        provider=OpenAIModelProvider(
            cfg.provider.model,
            timeout_s=cfg.provider.timeout_s,
            max_retries=cfg.provider.max_retries,
            retry_backoff_base=cfg.provider.retry_backoff_base,
            max_concurrency=cfg.provider.max_concurrency,
            enable_idempotency_key=cfg.provider.enable_idempotency_key,
        ),
        task_adapter=QAToolTaskAdapter(),
        tools=LocalToolAdapter(),
        store=SQLiteStore(cfg.paths.db_path),
    )
    total = len(tasks)
    success = 0
    retries: list[int] = []
    for task in tasks:
        try:
            result = engine.run(task, cfg)
            success += 1
            retries.extend(_collect_retry_counts(result.events_path))
        except Exception as exc:
            print(f"[FAILED] {task.task_id}: {exc}")

    success_rate = (success / total) if total else 0.0
    avg_retry = (sum(retries) / len(retries)) if retries else 0.0
    print(f"success_rate={success_rate:.4f} ({success}/{total})")
    print(f"avg_retry_count={avg_retry:.4f}")


def _collect_retry_counts(events_path: str) -> list[int]:
    counts: list[int] = []
    path = Path(events_path)
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload.get("event_type") != "provider_call_audit":
            continue
        event_payload = payload.get("payload", {})
        if not isinstance(event_payload, dict):
            continue
        retry_count = event_payload.get("retry_count")
        if isinstance(retry_count, int):
            counts.append(retry_count)
    return counts


def _load_tasks(path: str, *, limit: int) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip().lstrip("\ufeff")
            if not text:
                continue
            tasks.append(TaskSpec.model_validate(json.loads(text)))
            if len(tasks) >= limit:
                break
    return tasks


if __name__ == "__main__":
    main()
