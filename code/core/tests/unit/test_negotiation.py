from __future__ import annotations

from openagora.config.models import RunConfig
from openagora.mechanisms.negotiation import NegotiationLoop
from openagora.mechanisms.ot_alignment import OTAlignmentScorer
from openagora.storage.sqlite_store import SQLiteStore
from openagora.types import RunEvent, TaskSignature, WorkerFeedback


class _Observer:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def on_event(self, event: RunEvent) -> None:
        self.events.append(event)


def test_consensus_and_override_logic(tmp_path) -> None:
    store = SQLiteStore(str(tmp_path / "test.db"))
    observer = _Observer()
    loop = NegotiationLoop(store, OTAlignmentScorer(), observer)
    feedback = [
        WorkerFeedback(worker_id=1, overall_assessment="agree"),
        WorkerFeedback(worker_id=2, overall_assessment="conditional_agree"),
    ]
    assert loop.consensus(feedback)

    prev = [
        WorkerFeedback(worker_id=1, overall_assessment="disagree", suggestion="x"),
        WorkerFeedback(worker_id=2, overall_assessment="conditional_agree", suggestion=""),
    ]
    curr = [
        WorkerFeedback(worker_id=1, overall_assessment="disagree", suggestion="x"),
        WorkerFeedback(worker_id=2, overall_assessment="conditional_agree", suggestion=""),
    ]
    assert loop.should_override(
        current_feedback=curr,
        previous_feedback=prev,
        round_idx=2,
        max_rounds=5,
        stable_threshold=0.9,
    )


def test_deadlock_breaker(tmp_path) -> None:
    class _DeadlockLoop(NegotiationLoop):
        def _collect_feedback(self, proposal, signature):  # type: ignore[override]
            return [
                WorkerFeedback(worker_id=1, overall_assessment="disagree"),
                WorkerFeedback(worker_id=2, overall_assessment="disagree"),
            ]

        def should_override(self, **kwargs):  # type: ignore[override]
            return False

    store = SQLiteStore(str(tmp_path / "test_deadlock.db"))
    observer = _Observer()
    loop = _DeadlockLoop(store, OTAlignmentScorer(), observer)
    cfg = RunConfig.model_validate(
        {
            "task": {"task_id": "t1", "prompt": "p"},
            "paths": {"db_path": str(tmp_path / "x.db"), "artifacts_dir": str(tmp_path / "a")},
        }
    )
    out = loop.negotiate(
        task=cfg.task,
        signature=TaskSignature(
            primary_type="multi_step_qa",
            uncertainty="high",
            complexity="high",
            interdependence="reciprocal",
        ),
        cfg=cfg,
    )
    assert out.deadlock_breaker
