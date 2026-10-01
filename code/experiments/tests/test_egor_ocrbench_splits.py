from __future__ import annotations

import json
from pathlib import Path

from openagora_experiments.egor.ocrbench_splits import (
    OCRBenchCandidate,
    load_exclusions,
    normalize_transfer_row,
    select_transfer_rows,
    select_transfer_rows_by_type,
)


TYPES = ("APP agent en", "text recognition en")


def test_transfer_selection_is_answer_blind_balanced_and_image_unique() -> None:
    candidates = [_candidate(index) for index in range(20)]
    changed = [
        OCRBenchCandidate(
            source_id=row.source_id,
            question=row.question,
            answers=(f"secret-{row.source_id}",),
            dataset_name=row.dataset_name,
            task_type=row.task_type,
            image_sha256=row.image_sha256,
        )
        for row in candidates
    ]

    first, audit = select_transfer_rows(
        candidates,
        excluded_source_ids={0},
        excluded_image_hashes={"image-1"},
        task_types=TYPES,
        per_type=3,
        salt="fixed-salt",
    )
    second, _ = select_transfer_rows(
        changed,
        excluded_source_ids={0},
        excluded_image_hashes={"image-1"},
        task_types=TYPES,
        per_type=3,
        salt="fixed-salt",
    )

    assert [row.source_id for row in first] == [row.source_id for row in second]
    assert len(first) == 6
    assert len({row.image_sha256 for row in first}) == 6
    assert audit["selected_rows_by_type"] == {
        "APP agent en": 3,
        "text recognition en": 3,
    }
    assert not ({0, 1} & {row.source_id for row in first})


def test_transfer_selection_supports_frozen_unequal_quotas() -> None:
    candidates = [_candidate(index) for index in range(20)]

    selected, audit = select_transfer_rows_by_type(
        candidates,
        excluded_source_ids=set(),
        excluded_image_hashes=set(),
        quota_by_type={"APP agent en": 2, "text recognition en": 4},
        salt="fixed-unequal-quota",
    )

    assert len(selected) == 6
    assert len({row.image_sha256 for row in selected}) == 6
    assert audit["requested_rows_by_type"] == {
        "APP agent en": 2,
        "text recognition en": 4,
    }
    assert audit["selected_rows_by_type"] == {
        "APP agent en": 2,
        "text recognition en": 4,
    }


def test_exclusions_read_raw_json_and_normalized_jsonl(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "old.png").write_bytes(b"old-image")
    raw = tmp_path / "raw.json"
    raw.write_text(
        json.dumps(
            [
                {
                    "id": 17,
                    "question": "hidden",
                    "answers": ["hidden"],
                    "image": "images/old.png",
                }
            ]
        ),
        encoding="utf-8",
    )
    normalized = tmp_path / "old.jsonl"
    normalized.write_text(
        json.dumps(
            {
                "task_id": "ocrbench-v2-00023",
                "benchmark": "OCRBench-v2",
                "image_path": str(images / "old.png"),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    source_ids, image_hashes = load_exclusions([raw, normalized])

    assert source_ids == {17, 23}
    assert len(image_hashes) == 1


def test_normalized_transfer_row_uses_official_metric(tmp_path: Path) -> None:
    candidate = OCRBenchCandidate(
        source_id=9,
        question="read this",
        answers=("alpha", "Alpha"),
        dataset_name="fixture",
        task_type="text recognition en",
        image_sha256="a" * 64,
    )
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")

    row = normalize_transfer_row(candidate, image_path=image)

    assert row["task_id"] == "ocrbench-v2-00009"
    assert row["metric"] == "ocrbench_v2_vqa"
    assert row["metadata"]["official_capability"] == "text_recognition"
    assert row["metadata"]["egor_zero_adaptation_transfer"] is True


def test_math_qa_uses_the_official_vqa_metric_group(tmp_path: Path) -> None:
    candidate = OCRBenchCandidate(
        source_id=10,
        question="What is the total?",
        answers=("42",),
        dataset_name="fixture",
        task_type="math QA en",
        image_sha256="b" * 64,
    )
    image = tmp_path / "math.png"
    image.write_bytes(b"fixture")

    row = normalize_transfer_row(candidate, image_path=image)

    assert row["metric"] == "ocrbench_v2_vqa"
    assert row["metadata"]["official_capability"] == "knowledge_reasoning"


def _candidate(index: int) -> OCRBenchCandidate:
    return OCRBenchCandidate(
        source_id=index,
        question=f"question {index}",
        answers=(f"answer {index}",),
        dataset_name="fixture",
        task_type=TYPES[index % 2],
        image_sha256=f"image-{index}",
    )
