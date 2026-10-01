"""Cache lifecycle manager."""

from __future__ import annotations

from openagora.config.models import RunConfig
from openagora.core.interfaces import CacheStore, Observer
from openagora.types import ArchitectureSpec, RunEvent, TaskSignature


class CacheLifecycleManager:
    """Govern cache write, invalidation and rollback."""

    def __init__(self, store: CacheStore, observer: Observer) -> None:
        self._store = store
        self._observer = observer

    def write_if_eligible(
        self,
        *,
        signature: TaskSignature,
        architecture: ArchitectureSpec,
        performance: float,
        ot_alignment: float,
        cfg: RunConfig,
        source: str,
    ) -> bool:
        """Write cache entry when thresholds are satisfied."""
        if not cfg.toggles.enable_cache:
            return False
        if (
            performance < cfg.thresholds.cache_write_perf_threshold
            or ot_alignment < cfg.thresholds.cache_write_align_threshold
        ):
            self._observer.on_event(
                RunEvent(
                    event_type="cache_skip_write",
                    payload={
                        "task_type": signature.primary_type,
                        "performance": performance,
                        "ot_alignment": ot_alignment,
                    },
                )
            )
            return False
        entry = self._store.upsert(
            signature=signature,
            architecture=architecture,
            performance=performance,
            ot_alignment=ot_alignment,
            source=source,
        )
        self._observer.on_event(
            RunEvent(
                event_type="cache_write",
                payload={
                    "task_type": signature.primary_type,
                    "version_id": entry.version_id,
                    "performance": performance,
                    "ot_alignment": ot_alignment,
                },
            )
        )
        return True

    def post_reuse_validation(
        self,
        *,
        signature: TaskSignature,
        reused_cache: bool,
        performance: float,
        cfg: RunConfig,
    ) -> None:
        """Mark invalid and rollback if cache reuse underperforms."""
        if not (cfg.toggles.enable_cache and reused_cache):
            return
        if performance < cfg.thresholds.cache_invalidate_threshold:
            self._store.mark_needs_revalidation(signature.primary_type)
            self._observer.on_event(
                RunEvent(
                    event_type="cache_invalidate",
                    payload={
                        "task_type": signature.primary_type,
                        "performance": performance,
                    },
                )
            )
        self._store.rollback_if_needed(signature.primary_type)

