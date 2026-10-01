"""Counterfactual data and evaluation utilities for CAGO experiments."""

from openagora_experiments.cago.data import (
    CounterfactualMatrix,
    MatrixAudit,
    OutcomeRecord,
    TaskKey,
    TaskRecord,
    load_counterfactual_matrix,
)
from openagora_experiments.cago.scoring import score_prediction

__all__ = [
    "CounterfactualMatrix",
    "MatrixAudit",
    "OutcomeRecord",
    "TaskKey",
    "TaskRecord",
    "load_counterfactual_matrix",
    "score_prediction",
]
