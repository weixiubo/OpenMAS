from __future__ import annotations

import pytest

from openagora.config.models import RunConfig


def test_invalid_threshold_order() -> None:
    with pytest.raises(ValueError):
        RunConfig.model_validate(
            {
                "task": {"task_id": "x", "prompt": "x"},
                "thresholds": {
                    "cache_write_perf_threshold": 0.3,
                    "cache_invalidate_threshold": 0.6,
                },
            }
        )


def test_invalid_eval_borderline_order() -> None:
    with pytest.raises(ValueError):
        RunConfig.model_validate(
            {
                "task": {"task_id": "x", "prompt": "x"},
                "eval": {
                    "borderline_low": 0.8,
                    "borderline_high": 0.8,
                },
            }
        )
