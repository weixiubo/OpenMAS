"""Structured fixed-seed behavior derived from release 812f3e2; message text is omitted."""

import json
import re
from pathlib import Path

from openagora.adapters.qa_tools import LocalToolAdapter, QAToolTaskAdapter
from openagora.config.models import RunConfig
from openagora.core.engine import OpenAGORAEngine
from openagora.providers.mock import MockModelProvider
from openagora.storage.sqlite_store import SQLiteStore

REFERENCE = Path(__file__).parent / "fixtures" / "engine_behavior_812f3e2.json"


def _semantic(value):
    if isinstance(value, dict):
        return {
            k: _semantic(v)
            for k, v in value.items()
            if k
            not in {
                "created_at",
                "events_path",
                "manifest_path",
                "rationale",
                "run_id",
                "system_prompt_preview",
                "timestamp",
                "ts",
                "user_prompt_preview",
            }
        }
    if isinstance(value, list):
        return [_semantic(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<uuid>", value)
    return value


def capture(root):
    payloads = []
    for negotiation in (True, False):
        base = root / str(negotiation)
        cfg = RunConfig.model_validate(
            {
                "task": {
                    "task_id": "reference-5",
                    "prompt": "Use tools to answer: what is 2 + 3?",
                    "metadata": {
                        "reference_answer": "5",
                        "primary_type": "multi_step_qa",
                        "uncertainty": "high",
                        "complexity": "high",
                        "interdependence": "reciprocal",
                    },
                },
                "paths": {"db_path": str(base / "store.db"), "artifacts_dir": str(base / "runs")},
                "provider": {"seed": 42, "provider_mode": "mock"},
                "toggles": {"enable_negotiation": negotiation},
            }
        )
        engine = OpenAGORAEngine(
            provider=MockModelProvider(),
            task_adapter=QAToolTaskAdapter(),
            tools=LocalToolAdapter(),
            store=SQLiteStore(cfg.paths.db_path),
        )
        for _ in range(2):
            result = engine.run(cfg.task, cfg)
            events = [json.loads(line) for line in Path(result.events_path).read_text().splitlines()]
            payloads.append({"result": _semantic(result.model_dump(mode="json")), "events": _semantic(events)})
    return payloads


def test_fixed_seed_cold_and_warm_cache(tmp_path):
    # Compare structured behavior without timestamps or recorded message text.
    assert capture(tmp_path) == _semantic(json.loads(REFERENCE.read_text()))
