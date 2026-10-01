from __future__ import annotations

import pytest

from openagora.config.models import RunConfig
from openagora.templates.runtime import build_template_runtime


def test_template_render_missing_variable_fails() -> None:
    cfg = RunConfig.model_validate({"task": {"task_id": "t1", "prompt": "p"}})
    runtime = build_template_runtime(cfg)
    with pytest.raises(ValueError):
        runtime.render("leader_proposal", {"task_id": "t1"})


def test_seeded_persona_assignment_reproducible() -> None:
    cfg = RunConfig.model_validate(
        {
            "task": {"task_id": "t1", "prompt": "p"},
            "provider": {"seed": 7},
            "templates": {
                "persona_strategy": "seeded_random",
                "persona_pool": ["a", "b", "c"],
            },
            "runtime": {"worker_count": 5},
        }
    )
    runtime = build_template_runtime(cfg)
    first = runtime.assign_personas(cfg=cfg, worker_count=cfg.runtime.worker_count)
    second = runtime.assign_personas(cfg=cfg, worker_count=cfg.runtime.worker_count)
    assert first == second
