from __future__ import annotations

from math import isinf

import pytest

from openagora.mechanisms.conservative_adaptation import (
    CalibrationError,
    CalibrationExample,
    ConservativeOrganizationSelector,
    OrganizationOption,
    SimultaneousLowerBoundCalibrator,
)


def _example(
    predicted_a: float,
    observed_a: float,
    predicted_b: float,
    observed_b: float,
) -> CalibrationExample:
    return CalibrationExample(
        predicted_advantages={"a": predicted_a, "b": predicted_b},
        observed_advantages={"a": observed_a, "b": observed_b},
    )


def test_fit_uses_worst_candidate_residual_per_task() -> None:
    calibrator = SimultaneousLowerBoundCalibrator.fit(
        [
            _example(0.4, 0.3, 0.7, 0.2),
            _example(0.2, 0.1, 0.1, 0.0),
            _example(0.5, 0.3, 0.4, 0.3),
        ],
        alpha=0.5,
    )

    assert calibrator.correction == pytest.approx(0.2)
    assert calibrator.lower_bounds({"a": 0.6, "b": 0.25}) == pytest.approx({"a": 0.4, "b": 0.05})


def test_fit_returns_infinite_correction_when_coverage_exceeds_sample_resolution() -> None:
    calibrator = SimultaneousLowerBoundCalibrator.fit(
        [_example(0.2, 0.1, 0.1, 0.0)],
        alpha=0.1,
    )

    assert isinf(calibrator.correction)
    bounds = calibrator.lower_bounds({"a": 0.7, "b": 0.4})
    assert all(value == float("-inf") for value in bounds.values())


def test_selector_keeps_baseline_without_positive_lower_bound() -> None:
    calibrator = SimultaneousLowerBoundCalibrator.fit(
        [
            _example(0.2, 0.1, 0.2, 0.1),
            _example(0.3, 0.2, 0.3, 0.2),
            _example(0.4, 0.3, 0.4, 0.3),
        ],
        alpha=0.5,
    )
    selector = ConservativeOrganizationSelector()

    decision = selector.select(
        baseline_id="strong_static",
        options=[
            OrganizationOption("a", estimated_cost=0.2),
            OrganizationOption("b", estimated_cost=0.1),
        ],
        predicted_advantages={"a": 0.1, "b": 0.05},
        calibrator=calibrator,
    )

    assert decision.selected_id == "strong_static"
    assert decision.switched is False
    assert decision.reason == "no_admissible_candidate_with_positive_calibrated_advantage"


def test_selector_chooses_largest_positive_lower_bound() -> None:
    calibrator = SimultaneousLowerBoundCalibrator.fit(
        [
            _example(0.2, 0.2, 0.2, 0.2),
            _example(0.3, 0.3, 0.3, 0.3),
            _example(0.4, 0.4, 0.4, 0.4),
        ],
        alpha=0.5,
    )

    decision = ConservativeOrganizationSelector().select(
        baseline_id="strong_static",
        options=[
            OrganizationOption("a", estimated_cost=0.1),
            OrganizationOption("b", estimated_cost=0.4),
        ],
        predicted_advantages={"a": 0.2, "b": 0.5},
        calibrator=calibrator,
    )

    assert decision.selected_id == "b"
    assert decision.switched is True
    assert decision.selected_lower_bound == pytest.approx(0.5)


def test_selector_uses_edit_distance_then_cost_for_equal_bounds() -> None:
    calibrator = SimultaneousLowerBoundCalibrator.fit(
        [
            _example(0.2, 0.2, 0.2, 0.2),
            _example(0.3, 0.3, 0.3, 0.3),
            _example(0.4, 0.4, 0.4, 0.4),
        ],
        alpha=0.5,
    )
    selector = ConservativeOrganizationSelector(max_edit_distance=2)

    decision = selector.select(
        baseline_id="strong_static",
        options=[
            OrganizationOption("a", estimated_cost=0.1, edit_distance=3),
            OrganizationOption("b", estimated_cost=0.2, edit_distance=1),
        ],
        predicted_advantages={"a": 0.4, "b": 0.4},
        calibrator=calibrator,
    )

    assert decision.selected_id == "b"
    assert decision.switched is True
    assert decision.assessments[0].admissible is False


def test_calibration_rejects_mismatched_candidate_sets() -> None:
    with pytest.raises(CalibrationError, match="candidate ids"):
        SimultaneousLowerBoundCalibrator.fit(
            [
                CalibrationExample(
                    predicted_advantages={"a": 0.1},
                    observed_advantages={"b": 0.1},
                )
            ],
            alpha=0.1,
        )
