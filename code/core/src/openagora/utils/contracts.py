"""Public API contract checks."""

from __future__ import annotations

import inspect

from openagora.core.engine import OpenAGORAEngine
from openagora.types import RunResult

REQUIRED_RUN_RESULT_FIELDS = {
    "run_id",
    "task_id",
    "final_architecture",
    "result",
    "ot_alignment",
    "metrics",
    "cache_used",
    "negotiation_rounds",
    "reconfig_count",
    "override_used",
    "deadlock_breaker",
    "manifest_path",
    "events_path",
}


def check_public_contract() -> list[str]:
    """Return contract violations for stable public APIs."""
    violations: list[str] = []

    run_sig = inspect.signature(OpenAGORAEngine.run)
    expected_run = ["self", "task", "run_config"]
    if list(run_sig.parameters.keys()) != expected_run:
        violations.append(f"OpenAGORAEngine.run signature changed: {run_sig}")

    replay_sig = inspect.signature(OpenAGORAEngine.replay)
    expected_replay = ["self", "run_id"]
    if list(replay_sig.parameters.keys()) != expected_replay:
        violations.append(f"OpenAGORAEngine.replay signature changed: {replay_sig}")

    fields = set(RunResult.model_fields.keys())
    missing = REQUIRED_RUN_RESULT_FIELDS - fields
    if missing:
        violations.append(f"RunResult missing required fields: {sorted(missing)}")

    return violations
