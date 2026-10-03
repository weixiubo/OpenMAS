from __future__ import annotations

from openagora.adapters.qa_tools import LocalToolAdapter, QAToolTaskAdapter
from openagora.config.models import RunConfig
from openagora.mechanisms.ot_alignment import OTAlignmentScorer
from openagora.mechanisms.runtime import RuntimeExecutor
from openagora.types import RunEvent, TaskSignature


class _Observer:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def on_event(self, event: RunEvent) -> None:
        self.events.append(event)


def test_runtime_reconfig_respects_limits() -> None:
    observer = _Observer()
    scorer = OTAlignmentScorer()
    executor = RuntimeExecutor(QAToolTaskAdapter(), LocalToolAdapter(), scorer, observer)
    cfg = RunConfig.model_validate(
        {
            "task": {
                "task_id": "qa-1",
                "prompt": "what is 2 + 3?",
                "metadata": {"reference_answer": "5"},
            }
        }
    )
    signature = TaskSignature(
        primary_type="multi_step_qa",
        uncertainty="high",
        complexity="high",
        interdependence="reciprocal",
    )
    bad_arch = scorer.prototype_architecture("SS", worker_count=cfg.runtime.worker_count)
    out = executor.execute(task=cfg.task, signature=signature, architecture=bad_arch, cfg=cfg)
    assert out.reconfig_count <= cfg.thresholds.max_reconfigs
    assert len(out.step_results) == cfg.runtime.max_steps


def test_runtime_always_on_variant_consumes_reconfig_budget() -> None:
    observer = _Observer()
    scorer = OTAlignmentScorer()
    executor = RuntimeExecutor(QAToolTaskAdapter(), LocalToolAdapter(), scorer, observer)
    cfg = RunConfig.model_validate(
        {
            "task": {
                "task_id": "qa-1",
                "prompt": "what is 2 + 3?",
                "metadata": {"reference_answer": "5"},
            }
        }
    )
    signature = TaskSignature(
        primary_type="multi_step_qa",
        uncertainty="low",
        complexity="low",
        interdependence="pooled",
    )
    strong_arch = scorer.prototype_architecture("Adh", worker_count=cfg.runtime.worker_count)
    object.__setattr__(cfg, "_always_on_rt", True)
    out = executor.execute(task=cfg.task, signature=signature, architecture=strong_arch, cfg=cfg)
    assert out.reconfig_count == cfg.thresholds.max_reconfigs
    assert any(
        event.event_type == "runtime_reconfig" and str(event.payload.get("trigger")) == "always_on_rt"
        for event in observer.events
    )
