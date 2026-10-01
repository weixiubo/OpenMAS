from __future__ import annotations

import json
from pathlib import Path

from openagora.adapters.qa_tools import LocalToolAdapter, QAToolTaskAdapter
from openagora.config.models import RunConfig
from openagora.core.engine import OpenAGORAEngine
from openagora.providers.mock import MockModelProvider
from openagora.storage.sqlite_store import SQLiteStore


def test_full_cycle_cache_reuse_manifest_events_and_hashes(tmp_path) -> None:
    cfg = RunConfig.model_validate(
        {
            "task": {
                "task_id": "qa-int-1",
                "prompt": "Use tools to answer: what is 2 + 3?",
                "metadata": {
                    "reference_answer": "5",
                    "primary_type": "multi_step_qa",
                    "uncertainty": "high",
                    "complexity": "high",
                    "interdependence": "reciprocal",
                },
            },
            "paths": {
                "db_path": str(tmp_path / "openagora.db"),
                "artifacts_dir": str(tmp_path / "artifacts"),
            },
            "provider": {"provider_mode": "mock"},
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
    replayed = engine.replay(first.run_id)

    assert first.task_id == replayed.task_id
    assert first.metrics["quality_score"] >= 0.0
    assert second.cache_used

    manifest_payload = json.loads(Path(first.manifest_path).read_text(encoding="utf-8"))
    assert manifest_payload["template_id"] == cfg.templates.template_id
    assert manifest_payload["template_version"] == cfg.templates.version
    assert manifest_payload["event_schema_version"] == "v2"

    event_lines = Path(first.events_path).read_text(encoding="utf-8").strip().splitlines()
    assert event_lines
    first_event = json.loads(event_lines[0])
    assert first_event["schema_version"] == "v2"

    hashes_path = Path(first.manifest_path).parent / "hashes.json"
    assert hashes_path.exists()
    hashes_payload = json.loads(hashes_path.read_text(encoding="utf-8"))
    assert "hashes" in hashes_payload
    assert "events.jsonl" in hashes_payload["hashes"]
