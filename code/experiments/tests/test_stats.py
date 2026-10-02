from __future__ import annotations

from openagora_experiments.evaluation.stats import build_pairwise_stats


def test_pairwise_stats_stable_seed() -> None:
    rows = []
    for idx in range(6):
        task_id = f"t{idx}"
        rows.append(
            {
                "variant": "OpenAGORA_Full",
                "task_id": task_id,
                "is_correct": 1 if idx % 2 == 0 else 0,
                "final_score": 0.9 if idx % 2 == 0 else 0.2,
            }
        )
        rows.append(
            {
                "variant": "NoCache",
                "task_id": task_id,
                "is_correct": 0 if idx % 2 == 0 else 0,
                "final_score": 0.3,
            }
        )

    first = build_pairwise_stats(rows, reference_variant="OpenAGORA_Full", bootstrap_n=500, seed=2026)
    second = build_pairwise_stats(rows, reference_variant="OpenAGORA_Full", bootstrap_n=500, seed=2026)
    assert first == second
    assert first
