from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from openmas_test_history import historical_root

SCRIPT = (historical_root() / "code/experiments") / "scripts" / "build_egor_v9_confirmation_data.py"
SPEC = importlib.util.spec_from_file_location("build_egor_v9_data_test", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def candidate(index: int, question_type: str, *, answer: str = "secret"):
    return module.ChartQACandidate(
        row_index=index,
        image_name=f"image-{index}.png",
        image_sha256=f"{index + 1:064x}",
        image_bytes=b"image",
        question=f"visible question {index}",
        answer=answer,
        question_type=question_type,
    )


def empty_exclusions():
    return module.Exclusions(
        source_ids=frozenset(),
        image_hashes=frozenset(),
        image_basenames=frozenset(),
        question_hashes=frozenset(),
    )


def test_chartqa_selection_is_balanced_unique_and_answer_blind() -> None:
    rows = [
        *(candidate(index, "human") for index in range(6)),
        *(candidate(index + 10, "augmented") for index in range(6)),
    ]
    selected, audit = module.select_chartqa_rows(
        rows,
        exclusions=empty_exclusions(),
        per_type=2,
        salt="frozen",
    )
    changed_answers = [
        module.ChartQACandidate(
            row.row_index,
            row.image_name,
            row.image_sha256,
            row.image_bytes,
            row.question,
            f"changed-{row.row_index}",
            row.question_type,
        )
        for row in rows
    ]
    selected_changed, _ = module.select_chartqa_rows(
        changed_answers,
        exclusions=empty_exclusions(),
        per_type=2,
        salt="frozen",
    )
    assert [row.row_index for row in selected] == [row.row_index for row in selected_changed]
    assert audit["selected_rows_by_type"] == {"human": 2, "augmented": 2}
    assert len({row.image_sha256 for row in selected}) == 4


def test_chartqa_selection_accepts_frozen_unequal_quotas() -> None:
    rows = [
        *(candidate(index, "human") for index in range(6)),
        *(candidate(index + 10, "augmented") for index in range(8)),
    ]

    selected, audit = module.select_chartqa_rows(
        rows,
        exclusions=empty_exclusions(),
        per_type={"human": 2, "augmented": 4},
        salt="frozen-unequal",
    )

    assert audit["selected_rows_by_type"] == {"human": 2, "augmented": 4}
    assert len(selected) == 6
    assert len({row.image_sha256 for row in selected}) == 6


def test_partition_is_stratified_and_disjoint() -> None:
    rows = [
        candidate(0, "human"),
        candidate(1, "human"),
        candidate(10, "augmented"),
        candidate(11, "augmented"),
    ]
    format_rows, confirmation_rows = module.partition_by_stratum(
        rows,
        stratum=lambda row: row.question_type,
        strata=module.CHARTQA_TYPES,
        format_per_stratum=1,
        expected_per_stratum=2,
        salt="frozen",
    )
    assert len(format_rows) == len(confirmation_rows) == 2
    assert {row.question_type for row in format_rows} == set(module.CHARTQA_TYPES)
    assert {row.task_id for row in format_rows}.isdisjoint({row.task_id for row in confirmation_rows})


def test_partition_accepts_frozen_unequal_stratum_quotas() -> None:
    rows = [
        *(candidate(index, "alpha") for index in range(3)),
        *(candidate(index + 10, "beta") for index in range(4)),
    ]

    format_rows, confirmation_rows = module.partition_by_stratum(
        rows,
        stratum=lambda row: row.question_type,
        strata=("alpha", "beta"),
        format_per_stratum={"alpha": 1, "beta": 2},
        expected_per_stratum={"alpha": 3, "beta": 4},
        salt="frozen-unequal",
    )

    assert len(format_rows) == 3
    assert len(confirmation_rows) == 4
    assert {row.task_id for row in format_rows}.isdisjoint({row.task_id for row in confirmation_rows})


def test_amended_ocrbench_quota_contract_is_exact() -> None:
    assert tuple(module.OCRBENCH_V9_FORMAT_QUOTA) == module.OCRBENCH_V9_TYPES
    assert tuple(module.OCRBENCH_V9_CONFIRMATION_QUOTA) == module.OCRBENCH_V9_TYPES
    assert tuple(module.OCRBENCH_V9_TOTAL_QUOTA) == module.OCRBENCH_V9_TYPES
    assert sum(module.OCRBENCH_V9_FORMAT_QUOTA.values()) == 40
    assert sum(module.OCRBENCH_V9_CONFIRMATION_QUOTA.values()) == 200
    assert sum(module.OCRBENCH_V9_TOTAL_QUOTA.values()) == 240
    assert set(module.OCRBENCH_V9_TOTAL_QUOTA.values()) == {34, 35}
    assert "document classification en" not in module.OCRBENCH_V9_TYPES
    assert "text recognition en" not in module.OCRBENCH_V9_TYPES
    assert "math QA en" in module.OCRBENCH_V9_TYPES
    for task_type in module.OCRBENCH_V9_TYPES:
        assert module.OCRBENCH_V9_TOTAL_QUOTA[task_type] == (
            module.OCRBENCH_V9_FORMAT_QUOTA[task_type] + module.OCRBENCH_V9_CONFIRMATION_QUOTA[task_type]
        )


def test_exclusion_loader_reads_only_visible_identity_fields(tmp_path: Path) -> None:
    path = tmp_path / "prior.jsonl"
    row = {
        "task_id": "ocrbench-v2-00123",
        "question": "  A visible   question? ",
        "answer": "must-not-affect-selection",
        "image_path": "/prior/image.png",
        "metadata": {"source_image_sha256": "a" * 64, "source_id": 123},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    exclusions = module.collect_exclusions([path])
    assert exclusions.source_ids == frozenset({123})
    assert exclusions.image_hashes == frozenset({"a" * 64})
    assert exclusions.image_basenames == frozenset({"image.png"})
    assert len(exclusions.question_hashes) == 1
