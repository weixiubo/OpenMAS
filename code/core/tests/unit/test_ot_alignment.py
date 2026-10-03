from __future__ import annotations

from openagora.mechanisms.ot_alignment import OTAlignmentScorer
from openagora.types import TaskSignature


def test_ot_alignment_score_range() -> None:
    scorer = OTAlignmentScorer()
    signature = TaskSignature(
        primary_type="multi_step_qa",
        uncertainty="high",
        complexity="high",
        interdependence="reciprocal",
    )
    arch = scorer.prototype_architecture("Adh", worker_count=4)
    out = scorer.compute(signature, arch)
    assert 0.0 <= out.score <= 1.0
    assert out.predicted_config == "Adh"
