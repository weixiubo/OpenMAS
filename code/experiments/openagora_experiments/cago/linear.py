"""Dependency-free ridge models for the first CAGO kill pilot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt

from openagora_experiments.cago.data import MatrixValidationError


@dataclass(frozen=True)
class RidgeModel:
    """Small standardized ridge regressor solved by pivoted elimination."""

    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    weights: tuple[float, ...]
    l2: float

    @classmethod
    def fit(
        cls,
        features: Sequence[Sequence[float]],
        targets: Sequence[float],
        *,
        l2: float = 1.0,
    ) -> RidgeModel:
        if not features or len(features) != len(targets):
            raise MatrixValidationError("features and targets must be non-empty and aligned")
        if l2 <= 0.0:
            raise MatrixValidationError("l2 must be positive")
        width = len(features[0])
        if width == 0 or any(len(row) != width for row in features):
            raise MatrixValidationError("feature rows must have one fixed positive width")

        means, scales, design = _standardized_design(features, width)
        gram, rhs = _normal_equations(design, targets, width, l2)
        weights = _solve_linear_system(gram, rhs)
        return cls(
            feature_means=means,
            feature_scales=tuple(scales),
            weights=tuple(weights),
            l2=l2,
        )

    def predict(self, features: Sequence[float]) -> float:
        if len(features) != len(self.feature_means):
            raise MatrixValidationError("prediction feature width does not match fitted model")
        standardized = [
            (float(value) - self.feature_means[index]) / self.feature_scales[index]
            for index, value in enumerate(features)
        ]
        return self.weights[0] + sum(
            weight * value for weight, value in zip(self.weights[1:], standardized, strict=True)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_means": list(self.feature_means),
            "feature_scales": list(self.feature_scales),
            "weights": list(self.weights),
            "l2": self.l2,
        }


@dataclass(frozen=True)
class AdvantagePredictor:
    """One paired-advantage model per alternative organization."""

    baseline_id: str
    models: Mapping[str, RidgeModel]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.models))

    def predict(self, features: Sequence[float]) -> dict[str, float]:
        return {candidate_id: self.models[candidate_id].predict(features) for candidate_id in self.candidate_ids}


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]
    for pivot_column in range(size):
        pivot_row = max(range(pivot_column, size), key=lambda row: abs(augmented[row][pivot_column]))
        if abs(augmented[pivot_row][pivot_column]) < 1e-12:
            raise MatrixValidationError("ridge normal equations are singular")
        augmented[pivot_column], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_column]
        pivot = augmented[pivot_column][pivot_column]
        augmented[pivot_column] = [value / pivot for value in augmented[pivot_column]]
        for row in range(size):
            if row == pivot_column:
                continue
            factor = augmented[row][pivot_column]
            if factor == 0.0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[pivot_column], strict=True)
            ]
    return [augmented[row][-1] for row in range(size)]


def _standardized_design(
    features: Sequence[Sequence[float]], width: int
) -> tuple[tuple[float, ...], list[float], list[list[float]]]:
    means = tuple(sum(float(row[column]) for row in features) / len(features) for column in range(width))
    scales: list[float] = []
    for column, mean in enumerate(means):
        variance = sum((float(row[column]) - mean) ** 2 for row in features) / len(features)
        scales.append(sqrt(variance) if variance > 1e-12 else 1.0)
    design = [
        [1.0, *((float(value) - means[index]) / scales[index] for index, value in enumerate(row))] for row in features
    ]
    return means, scales, design


def _normal_equations(
    design: Sequence[Sequence[float]], targets: Sequence[float], width: int, l2: float
) -> tuple[list[list[float]], list[float]]:
    dimension = width + 1
    gram = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    rhs = [0.0 for _ in range(dimension)]
    for row, target in zip(design, targets, strict=True):
        for left in range(dimension):
            rhs[left] += row[left] * float(target)
            for right in range(dimension):
                gram[left][right] += row[left] * row[right]
    for index in range(1, dimension):
        gram[index][index] += l2
    return gram, rhs
