from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from openagora_experiments.cago.data import (
    CounterfactualMatrix,
    MatrixAudit,
    MatrixValidationError,
    OutcomeRecord,
    TaskKey,
    TaskRecord,
    load_counterfactual_matrix,
)
from openagora_experiments.cago.evaluation import UtilitySpec, run_cago_evaluation
from openagora_experiments.cago.features import visible_feature_vector
from openagora_experiments.cago.linear import RidgeModel
from openagora_experiments.cago.scoring import score_prediction
from openagora_experiments.cago.splits import source_stratified_split


def test_anls_uses_best_official_reference() -> None:
    score = score_prediction(
        metric="anls",
        predicted="University of California, San Diego",
        answer="university of california",
        reference_answers=[
            "university of california",
            "University of California, San Diego",
        ],
    )

    assert score == 1.0


def test_choice_scorer_does_not_treat_an_article_as_option_a() -> None:
    options = [
        {"id": "A", "text": "red square"},
        {"id": "B", "text": "blue circle"},
    ]

    assert (
        score_prediction(
            metric="choice_accuracy",
            predicted="A picture of a blue circle",
            answer="A",
            reference_answers=["A"],
            options=options,
        )
        == 0.0
    )
    assert (
        score_prediction(
            metric="choice_accuracy",
            predicted="The answer is B.",
            answer="B",
            reference_answers=["B"],
            options=options,
        )
        == 1.0
    )


def test_chartqa_relaxed_accuracy_uses_five_percent_numeric_tolerance() -> None:
    assert (
        score_prediction(
            metric="relaxed_accuracy",
            predicted="The answer is 104",
            answer="100",
        )
        == 1.0
    )
    assert (
        score_prediction(
            metric="relaxed_accuracy",
            predicted="106",
            answer="100",
        )
        == 0.0
    )
    assert (
        score_prediction(
            metric="relaxed_accuracy",
            predicted="50%",
            answer="0.5",
        )
        == 1.0
    )


def test_ocrbench_v2_vqa_matches_short_containment_and_long_anls() -> None:
    assert (
        score_prediction(
            metric="ocrbench_v2_vqa",
            predicted="The answer is Steve Madden.",
            answer="Steve Madden",
        )
        == 1.0
    )
    assert score_prediction(
        metric="ocrbench_v2_vqa",
        predicted="one two three four six",
        answer="one two three four five",
    ) == pytest.approx(20.0 / 23.0)
    assert (
        score_prediction(
            metric="relaxed_accuracy",
            predicted="United States",
            answer="united states",
        )
        == 1.0
    )


def test_visible_features_are_answer_blind() -> None:
    task = _task(source="source", task_id="1", question="What number is shown: 42?")
    changed_secret_fields = replace(
        task,
        answer="a completely different answer",
        reference_answers=("hidden reference",),
        metadata={"score": 1.0, "expected_answer": "leak"},
    )

    assert visible_feature_vector(task) == visible_feature_vector(changed_secret_fields)


def test_matrix_loader_rescores_every_organization_with_one_scorer(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "question": "What is the value?",
                "answer": "0.28",
                "reference_answers": ["0.28"],
                "metric": "anls",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_rows(
        first, [{"task_id": "t1", "predicted_answer": "0.24", "quality_score": "0.75"}]
    )
    _write_rows(
        second, [{"task_id": "t1", "predicted_answer": "0.24", "quality_score": "0.0"}]
    )

    matrix = load_counterfactual_matrix(
        task_paths={"source": tasks_path},
        outcome_paths={
            "first": {"source": first},
            "second": {"source": second},
        },
    )

    key = TaskKey("source", "t1")
    assert matrix.outcome(key, "first").quality == pytest.approx(0.75)
    assert matrix.outcome(key, "second").quality == pytest.approx(0.75)
    assert matrix.audit.reported_score_disagreements == 1


def test_matrix_loader_rejects_a_missing_counterfactual_cell(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "question": "q",
                    "answer": "a",
                    "metric": "accuracy",
                }
            )
            for task_id in ("t1", "t2")
        )
        + "\n",
        encoding="utf-8",
    )
    rows_path = tmp_path / "rows.csv"
    _write_rows(
        rows_path, [{"task_id": "t1", "predicted_answer": "a", "quality_score": "1"}]
    )

    with pytest.raises(MatrixValidationError, match="completeness audit"):
        load_counterfactual_matrix(
            task_paths={"source": tasks_path},
            outcome_paths={"candidate": {"source": rows_path}},
        )


def test_source_stratified_split_is_deterministic_and_disjoint() -> None:
    keys = [TaskKey(source, str(index)) for source in ("a", "b") for index in range(20)]

    first = source_stratified_split(keys, salt="frozen")
    second = source_stratified_split(reversed(keys), salt="frozen")

    assert first == second
    assert len(first.train) == 20
    assert len(first.calibration) == 10
    assert len(first.test) == 10
    assert set(first.train).isdisjoint(first.calibration)
    assert set(first.train).isdisjoint(first.test)


def test_dependency_free_ridge_fits_a_simple_relation() -> None:
    model = RidgeModel.fit(
        [[0.0], [1.0], [2.0], [3.0]],
        [1.0, 3.0, 5.0, 7.0],
        l2=1e-6,
    )

    assert model.predict([4.0]) == pytest.approx(9.0, abs=1e-4)


def test_cago_evaluation_produces_auditable_decisions() -> None:
    tasks: dict[TaskKey, TaskRecord] = {}
    outcomes: dict[tuple[TaskKey, str], OutcomeRecord] = {}
    for index in range(40):
        key = TaskKey("source", str(index))
        question = "What is shown?" + (" detailed evidence" * index)
        tasks[key] = _task(source="source", task_id=str(index), question=question)
        baseline_quality = 0.5
        candidate_quality = min(
            1.0, max(0.0, baseline_quality + (len(question) - 200) / 1000.0)
        )
        outcomes[(key, "baseline")] = _outcome(key, "baseline", baseline_quality)
        outcomes[(key, "candidate")] = _outcome(key, "candidate", candidate_quality)
    audit = MatrixAudit(
        task_count=40,
        organization_count=2,
        outcome_count=80,
        expected_outcome_count=80,
        duplicate_rows=0,
        missing_cells=(),
        extra_cells=(),
        reported_score_disagreements=0,
        max_reported_score_difference=0.0,
        source_task_counts={"source": 40},
        organization_row_counts={"baseline": 40, "candidate": 40},
        input_sha256={},
    )
    matrix = CounterfactualMatrix(
        tasks=tasks,
        outcomes=outcomes,
        organization_ids=("baseline", "candidate"),
        audit=audit,
    )
    split = source_stratified_split(matrix.task_keys, salt="evaluation-test")

    result = run_cago_evaluation(
        matrix=matrix,
        split=split,
        baseline_id="baseline",
        utility=UtilitySpec(),
        alpha=0.2,
        ridge_l2=0.01,
    )

    assert result.calibration_size == len(split.calibration)
    assert len(result.cago_decisions) == len(split.test)
    assert {summary.policy for summary in result.summaries} == {
        "baseline",
        "best_static",
        "best_static_by_source",
        "uncalibrated_advantage",
        "cago",
        "oracle",
    }
    assert all(
        tuple(decision.predicted_advantages) == ("candidate",)
        for decision in result.cago_decisions
    )


def _task(*, source: str, task_id: str, question: str) -> TaskRecord:
    return TaskRecord(
        key=TaskKey(source, task_id),
        question=question,
        answer="secret",
        reference_answers=("secret",),
        metric="accuracy",
        question_type="qa",
        options=(),
        metadata={},
    )


def _outcome(key: TaskKey, organization_id: str, quality: float) -> OutcomeRecord:
    return OutcomeRecord(
        key=key,
        organization_id=organization_id,
        predicted_answer="",
        quality=quality,
        reported_quality=quality,
        tokens=100.0,
        latency_seconds=1.0,
        cost_usd=0.01,
        source_path="synthetic",
    )


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["task_id", "predicted_answer", "quality_score"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
