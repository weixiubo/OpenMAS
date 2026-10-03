"""Statistical analysis for GAIA mini pairwise comparisons."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate rows by variant into mean/variance style summary."""
    if not rows:
        return []
    bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket[str(row["variant"])].append(row)

    numeric_cols = ["quality_score", "token_consumption", "latency_seconds", "cost_usd", "perf_cost_ratio", "final_score"]
    out: list[dict[str, Any]] = []
    for variant, grouped in bucket.items():
        line: dict[str, Any] = {
            "variant": variant,
            "count": len(grouped),
            "accuracy": _mean(float(item["is_correct"]) for item in grouped),
            "cache_hit_rate": _mean(float(item.get("cache_used", 0.0)) for item in grouped),
        }
        for col in numeric_cols:
            values = [float(item[col]) for item in grouped]
            line[f"{col}_mean"] = _mean(values)
            line[f"{col}_var"] = _variance(values)
        out.append(line)
    return sorted(out, key=lambda row: str(row["variant"]))


def build_pairwise_stats(
    rows: list[dict[str, Any]],
    *,
    reference_variant: str,
    bootstrap_n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Compute pairwise stats against reference variant."""
    by_variant_task: dict[tuple[str, str], dict[str, Any]] = {
        (str(item["variant"]), str(item["task_id"])): item for item in rows
    }
    variants = sorted({str(item["variant"]) for item in rows if str(item["variant"]) != reference_variant})
    ref_tasks = sorted({str(item["task_id"]) for item in rows if str(item["variant"]) == reference_variant})

    out: list[dict[str, Any]] = []
    for variant in variants:
        paired = []
        for task_id in ref_tasks:
            ref = by_variant_task.get((reference_variant, task_id))
            other = by_variant_task.get((variant, task_id))
            if ref is None or other is None:
                continue
            paired.append((ref, other))
        if not paired:
            continue

        ref_correct = [int(pair[0]["is_correct"]) for pair in paired]
        var_correct = [int(pair[1]["is_correct"]) for pair in paired]
        b = sum(1 for r, v in zip(ref_correct, var_correct) if r == 1 and v == 0)
        c = sum(1 for r, v in zip(ref_correct, var_correct) if r == 0 and v == 1)
        acc_effect = _mean(ref_correct) - _mean(var_correct)
        acc_p = _mcnemar_exact_p_value(b, c)

        out.append(
            {
                "reference_variant": reference_variant,
                "variant": variant,
                "metric": "accuracy",
                "test": "mcnemar_exact",
                "n": len(paired),
                "effect": acc_effect,
                "p_value": acc_p,
                "ci_low": "",
                "ci_high": "",
                "discordant_b": b,
                "discordant_c": c,
            }
        )

        diffs = [float(pair[0]["final_score"]) - float(pair[1]["final_score"]) for pair in paired]
        eval_effect = _mean(diffs)
        ci_low, ci_high, eval_p = _paired_bootstrap(
            diffs,
            n_bootstrap=bootstrap_n,
            seed=seed,
        )
        out.append(
            {
                "reference_variant": reference_variant,
                "variant": variant,
                "metric": "eval_score",
                "test": "paired_bootstrap",
                "n": len(diffs),
                "effect": eval_effect,
                "p_value": eval_p,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "discordant_b": "",
                "discordant_c": "",
            }
        )

    return out


def build_failure_buckets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate incorrect samples by variant and error type."""
    bucket: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        if int(row["is_correct"]) == 1:
            continue
        key = (str(row["variant"]), str(row.get("error_type", "unknown")))
        bucket[key].append(str(row["task_id"]))

    out: list[dict[str, Any]] = []
    for (variant, error_type), task_ids in sorted(bucket.items()):
        out.append(
            {
                "variant": variant,
                "error_type": error_type,
                "count": len(task_ids),
                "sample_task_ids": "|".join(task_ids[:8]),
            }
        )
    return out


def _paired_bootstrap(diffs: list[float], *, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    if not diffs:
        return 0.0, 0.0, 1.0
    rng = random.Random(seed)
    n = len(diffs)
    sampled_means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        sampled_means.append(_mean(sample))
    sampled_means.sort()
    lo_idx = int(0.025 * (n_bootstrap - 1))
    hi_idx = int(0.975 * (n_bootstrap - 1))
    ci_low = sampled_means[lo_idx]
    ci_high = sampled_means[hi_idx]

    non_positive = sum(1 for value in sampled_means if value <= 0.0) / n_bootstrap
    non_negative = sum(1 for value in sampled_means if value >= 0.0) / n_bootstrap
    p_value = min(1.0, 2.0 * min(non_positive, non_negative))
    return ci_low, ci_high, p_value


def _mcnemar_exact_p_value(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cumulative = 0.0
    for i in range(k + 1):
        cumulative += _binomial_prob(n, i)
    return min(1.0, 2.0 * cumulative)


def _binomial_prob(n: int, k: int) -> float:
    return math.comb(n, k) * (0.5 ** n)


def _mean(values) -> float:  # type: ignore[no-untyped-def]
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def _variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _mean(values)
    return sum((value - m) ** 2 for value in values) / len(values)
