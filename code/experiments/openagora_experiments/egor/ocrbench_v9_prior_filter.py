"""Visible-input-only prior-question exclusion for fresh OCRBench-v2 data."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence, Set
from typing import Any

from openagora_experiments.egor.ocrbench_splits import OCRBenchCandidate


def filter_prior_questions(
    candidates: Sequence[OCRBenchCandidate],
    *,
    excluded_question_hashes: Set[str],
) -> tuple[list[OCRBenchCandidate], dict[str, Any]]:
    """Remove prior normalized questions without inspecting answer fields.

    Official OCRBench-v2 rows may legitimately reuse a question template for
    different images. The confirmation unit is image-unique, so this function
    excludes historical question overlap but preserves within-split templates.
    """

    selected: list[OCRBenchCandidate] = []
    excluded_prior = 0
    for row in candidates:
        question = " ".join(row.question.split()).strip()
        question_hash = hashlib.sha256(question.encode()).hexdigest()
        if question_hash in excluded_question_hashes:
            excluded_prior += 1
            continue
        selected.append(row)
    return selected, {
        "source_rows_before_question_exclusion": len(candidates),
        "rows_excluded_by_prior_question": excluded_prior,
        "selection_input_rows_after_question_exclusion": len(selected),
    }
