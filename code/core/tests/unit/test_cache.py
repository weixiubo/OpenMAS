from __future__ import annotations

from openagora.config.models import RunConfig
from openagora.mechanisms.caching import CacheLifecycleManager
from openagora.mechanisms.ot_alignment import OTAlignmentScorer
from openagora.storage.sqlite_store import SQLiteStore
from openagora.types import RunEvent, TaskSignature


class _Observer:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def on_event(self, event: RunEvent) -> None:
        self.events.append(event)


def test_invalidation_and_rollback(tmp_path) -> None:
    store = SQLiteStore(str(tmp_path / "cache.db"))
    observer = _Observer()
    manager = CacheLifecycleManager(store, observer)
    scorer = OTAlignmentScorer()
    signature = TaskSignature(
        primary_type="multi_step_qa",
        uncertainty="medium",
        complexity="high",
        interdependence="pooled",
    )
    arch = scorer.prototype_architecture("PB", worker_count=4)
    store.upsert(signature=signature, architecture=arch, performance=0.9, ot_alignment=0.9, source="test")
    store.upsert(signature=signature, architecture=arch, performance=0.6, ot_alignment=0.9, source="test")
    store.rollback_if_needed(signature.primary_type)
    current = store.get_current(signature.primary_type)
    assert current is not None
    assert current.avg_perf == 0.9

    cfg = RunConfig.model_validate({"task": {"task_id": "x", "prompt": "x"}})
    manager.post_reuse_validation(signature=signature, reused_cache=True, performance=0.2, cfg=cfg)
    current2 = store.get_current(signature.primary_type)
    assert current2 is not None
    assert current2.needs_revalidation

