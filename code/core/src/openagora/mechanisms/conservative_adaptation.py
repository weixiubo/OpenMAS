"""Conservative organization selection with simultaneous conformal bounds.

This module is intentionally independent of any particular predictor. Training a
predictor belongs to the experiment layer; the runtime only consumes predicted
counterfactual advantages and a calibration object fitted on held-out examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf, isfinite
from typing import Mapping, Sequence


class CalibrationError(ValueError):
    """Raised when calibration inputs do not define one comparable candidate set."""


@dataclass(frozen=True)
class CalibrationExample:
    """Predicted and observed utility advantages for one held-out task."""

    predicted_advantages: Mapping[str, float]
    observed_advantages: Mapping[str, float]


@dataclass(frozen=True)
class SimultaneousLowerBoundCalibrator:
    """One-sided split-conformal bound shared by a finite candidate set.

    For every calibration task, the nonconformity score is the largest error
    ``predicted_advantage - observed_advantage`` over all candidates. Using the
    quantile of this maximum error gives a simultaneous lower bound: with the
    requested marginal coverage, all candidate advantages for a new exchangeable
    task are above their reported lower bounds at once. This permits adaptive
    candidate selection without invalidating the bound.
    """

    alpha: float
    correction: float
    calibration_size: int
    candidate_ids: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        examples: Sequence[CalibrationExample],
        *,
        alpha: float,
    ) -> SimultaneousLowerBoundCalibrator:
        """Fit a simultaneous one-sided lower-bound calibrator."""

        if not 0.0 < alpha < 1.0:
            raise CalibrationError("alpha must be strictly between 0 and 1")
        if not examples:
            raise CalibrationError("at least one calibration example is required")

        first_ids = cls._validate_example(examples[0])
        residuals: list[float] = []
        for example in examples:
            candidate_ids = cls._validate_example(example)
            if candidate_ids != first_ids:
                raise CalibrationError("all calibration examples must use the same candidate ids")
            residuals.append(
                max(
                    float(example.predicted_advantages[candidate_id])
                    - float(example.observed_advantages[candidate_id])
                    for candidate_id in candidate_ids
                )
            )

        residuals.sort()
        rank = ceil((len(residuals) + 1) * (1.0 - alpha))
        correction = inf if rank > len(residuals) else residuals[rank - 1]
        return cls(
            alpha=alpha,
            correction=correction,
            calibration_size=len(residuals),
            candidate_ids=first_ids,
        )

    @staticmethod
    def _validate_example(example: CalibrationExample) -> tuple[str, ...]:
        predicted_ids = tuple(sorted(example.predicted_advantages))
        observed_ids = tuple(sorted(example.observed_advantages))
        if not predicted_ids:
            raise CalibrationError("candidate advantages cannot be empty")
        if predicted_ids != observed_ids:
            raise CalibrationError("predicted and observed candidate ids must match")

        values = [
            *example.predicted_advantages.values(),
            *example.observed_advantages.values(),
        ]
        if any(not isfinite(float(value)) for value in values):
            raise CalibrationError("calibration advantages must be finite")
        return predicted_ids

    def lower_bounds(self, predicted_advantages: Mapping[str, float]) -> dict[str, float]:
        """Return simultaneous lower bounds for one new task."""

        candidate_ids = tuple(sorted(predicted_advantages))
        if candidate_ids != self.candidate_ids:
            raise CalibrationError("prediction candidate ids do not match the calibrated candidate set")
        if any(not isfinite(float(value)) for value in predicted_advantages.values()):
            raise CalibrationError("predicted advantages must be finite")
        return {
            candidate_id: float(predicted_advantages[candidate_id]) - self.correction
            for candidate_id in self.candidate_ids
        }


@dataclass(frozen=True)
class OrganizationOption:
    """One executable organization considered against the baseline organization."""

    candidate_id: str
    estimated_cost: float
    edit_distance: int = 0
    admissible: bool = True

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id cannot be empty")
        if not isfinite(self.estimated_cost) or self.estimated_cost < 0.0:
            raise ValueError("estimated_cost must be finite and non-negative")
        if self.edit_distance < 0:
            raise ValueError("edit_distance must be non-negative")


@dataclass(frozen=True)
class CandidateAssessment:
    """Auditable assessment of one organization candidate."""

    candidate_id: str
    predicted_advantage: float
    lower_bound: float
    estimated_cost: float
    edit_distance: int
    admissible: bool


@dataclass(frozen=True)
class OrganizationSelectionDecision:
    """Result of a conservative organization selection."""

    baseline_id: str
    selected_id: str
    switched: bool
    selected_lower_bound: float
    assessments: tuple[CandidateAssessment, ...]
    reason: str


class ConservativeOrganizationSelector:
    """Switch organization only when a calibrated advantage lower bound is positive."""

    def __init__(self, *, min_advantage: float = 0.0, max_edit_distance: int | None = None) -> None:
        if not isfinite(min_advantage):
            raise ValueError("min_advantage must be finite")
        if max_edit_distance is not None and max_edit_distance < 0:
            raise ValueError("max_edit_distance must be non-negative")
        self._min_advantage = min_advantage
        self._max_edit_distance = max_edit_distance

    def select(
        self,
        *,
        baseline_id: str,
        options: Sequence[OrganizationOption],
        predicted_advantages: Mapping[str, float],
        calibrator: SimultaneousLowerBoundCalibrator,
    ) -> OrganizationSelectionDecision:
        """Select the safest positive-advantage organization, or keep the baseline."""

        if not baseline_id.strip():
            raise ValueError("baseline_id cannot be empty")
        if not options:
            raise ValueError("at least one organization option is required")

        option_ids = tuple(sorted(option.candidate_id for option in options))
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("organization option ids must be unique")
        if option_ids != calibrator.candidate_ids:
            raise CalibrationError("organization options do not match the calibrated candidate set")

        lower_bounds = calibrator.lower_bounds(predicted_advantages)
        assessments = tuple(
            CandidateAssessment(
                candidate_id=option.candidate_id,
                predicted_advantage=float(predicted_advantages[option.candidate_id]),
                lower_bound=lower_bounds[option.candidate_id],
                estimated_cost=option.estimated_cost,
                edit_distance=option.edit_distance,
                admissible=option.admissible
                and (
                    self._max_edit_distance is None
                    or option.edit_distance <= self._max_edit_distance
                ),
            )
            for option in sorted(options, key=lambda item: item.candidate_id)
        )
        eligible = [
            assessment
            for assessment in assessments
            if assessment.admissible and assessment.lower_bound > self._min_advantage
        ]
        if not eligible:
            return OrganizationSelectionDecision(
                baseline_id=baseline_id,
                selected_id=baseline_id,
                switched=False,
                selected_lower_bound=0.0,
                assessments=assessments,
                reason="no_admissible_candidate_with_positive_calibrated_advantage",
            )

        selected = min(
            eligible,
            key=lambda item: (
                -item.lower_bound,
                item.edit_distance,
                item.estimated_cost,
                item.candidate_id,
            ),
        )
        return OrganizationSelectionDecision(
            baseline_id=baseline_id,
            selected_id=selected.candidate_id,
            switched=True,
            selected_lower_bound=selected.lower_bound,
            assessments=assessments,
            reason="positive_simultaneous_lower_bound",
        )
