from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from openagora_experiments.egor.chartqa_splits import (
    load_exclusions,
    materialize_chartqa_rows,
    select_chartqa_rows,
)


def test_chartqa_selection_is_answer_blind_and_image_disjoint(tmp_path: Path) -> None:
    rows = [_row(index) for index in range(12)]
    changed_answers = deepcopy(rows)
    for row in changed_answers:
        row["label"] = f"secret-{row['label']}"

    first_dev, first_test, audit = select_chartqa_rows(
        rows,
        excluded_task_ids={"chartqa-hf-test-00001"},
        excluded_images={"chart-2.png"},
        development_size=2,
        heldout_size=2,
        salt="fixed-salt",
    )
    second_dev, second_test, _ = select_chartqa_rows(
        changed_answers,
        excluded_task_ids={"chartqa-hf-test-00001"},
        excluded_images={"chart-2.png"},
        development_size=2,
        heldout_size=2,
        salt="fixed-salt",
    )

    assert [item.task_id for item in first_dev] == [item.task_id for item in second_dev]
    assert [item.task_id for item in first_test] == [
        item.task_id for item in second_test
    ]
    assert {item.image_name for item in first_dev}.isdisjoint(
        item.image_name for item in first_test
    )
    assert audit["image_overlap_between_splits"] == 0

    normalized = materialize_chartqa_rows(first_dev, images_root=tmp_path / "images")
    assert all(row["metric"] == "relaxed_accuracy" for row in normalized)
    assert all(Path(row["image_path"]).is_file() for row in normalized)


def test_chartqa_exclusions_read_source_image_or_materialized_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "old-task",
                "image_path": "/tmp/fallback.png",
                "metadata": {"source_image": "source.png"},
            }
        )
        + "\n"
        + json.dumps({"task_id": "other", "image_path": "/tmp/other.png"})
        + "\n",
        encoding="utf-8",
    )

    task_ids, images = load_exclusions([path])

    assert task_ids == {"old-task", "other"}
    assert images == {"source.png", "other.png"}


def _row(index: int) -> dict[str, object]:
    image_index = index // 2
    return {
        "imgname": f"chart-{image_index}.png",
        "query": f"question {index}",
        "label": str(index),
        "type": "human",
        "image": f"encoded-image-{image_index}".encode(),
    }
