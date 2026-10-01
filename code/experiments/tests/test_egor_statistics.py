from __future__ import annotations

import pytest

from openagora_experiments.egor.statistics import paired_bootstrap_contrast


def test_paired_bootstrap_reports_exact_sign_test_and_deterministic_interval() -> None:
    result = paired_bootstrap_contrast(
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        bootstrap_samples=2_000,
        seed=7,
    )
    repeated = paired_bootstrap_contrast(
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        bootstrap_samples=2_000,
        seed=7,
    )

    assert result == repeated
    assert result.mean_delta == 0.5
    assert (result.wins, result.losses, result.ties) == (2, 0, 2)
    assert result.sign_p_one_sided == 0.25
    assert result.sign_p_two_sided == 0.5
    assert result.ci95_low <= result.mean_delta <= result.ci95_high


def test_paired_bootstrap_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="same positive number"):
        paired_bootstrap_contrast([1.0], [], bootstrap_samples=10)
