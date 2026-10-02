"""Paired, task-level uncertainty estimates for EGOR policies."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PairedContrast:
    tasks: int
    mean_delta: float
    ci95_low: float
    ci95_high: float
    wins: int
    losses: int
    ties: int
    sign_p_one_sided: float
    sign_p_two_sided: float
    bootstrap_samples: int
    seed: int


def paired_bootstrap_contrast(
    left: Sequence[float],
    right: Sequence[float],
    *,
    bootstrap_samples: int = 100_000,
    seed: int = 20_261_010,
    tie_tolerance: float = 1e-12,
) -> PairedContrast:
    """Estimate a paired mean difference and exact conditional sign test.

    Bootstrap resampling is over task pairs, never over individual system outputs.
    The one-sided sign test uses the predeclared alternative ``left > right``.
    """

    if len(left) != len(right) or not left:
        raise ValueError("left and right must contain the same positive number of tasks")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    deltas = [float(a) - float(b) for a, b in zip(left, right, strict=True)]
    if not all(math.isfinite(value) for value in deltas):
        raise ValueError("paired values must be finite")

    wins = sum(value > tie_tolerance for value in deltas)
    losses = sum(value < -tie_tolerance for value in deltas)
    ties = len(deltas) - wins - losses
    one_sided, two_sided = _exact_sign_p_values(wins=wins, losses=losses)

    rng = random.Random(seed)
    count = len(deltas)
    bootstrap_means: list[float] = []
    for _ in range(bootstrap_samples):
        total = sum(deltas[rng.randrange(count)] for _ in range(count))
        bootstrap_means.append(total / count)
    bootstrap_means.sort()

    return PairedContrast(
        tasks=count,
        mean_delta=sum(deltas) / count,
        ci95_low=_quantile(bootstrap_means, 0.025),
        ci95_high=_quantile(bootstrap_means, 0.975),
        wins=wins,
        losses=losses,
        ties=ties,
        sign_p_one_sided=one_sided,
        sign_p_two_sided=two_sided,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def _exact_sign_p_values(*, wins: int, losses: int) -> tuple[float, float]:
    non_ties = wins + losses
    if non_ties == 0:
        return 1.0, 1.0
    denominator = 2**non_ties
    one_sided = sum(math.comb(non_ties, successes) for successes in range(wins, non_ties + 1)) / denominator
    smaller_side = min(wins, losses)
    lower_tail = sum(math.comb(non_ties, successes) for successes in range(smaller_side + 1)) / denominator
    return one_sided, min(1.0, 2.0 * lower_tail)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)
