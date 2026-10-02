from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_builder_is_deterministic_excludes_old_images_and_separates_splits(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    source_rows = []
    for index in range(8):
        image_name = f"image-{index}.png"
        (images / image_name).write_bytes(b"fixture")
        source_rows.append(
            {
                "questionId": 100 + index,
                "question": f"question {index}",
                "answers": [f"answer {index}"],
                "image_local_name": image_name,
                "answer_type": ["single span"],
                "evidence": ["text"],
                "operation/reasoning": [],
                "data_split": "val",
            }
        )
    source = tmp_path / "val.json"
    source.write_text(json.dumps({"data": source_rows}), encoding="utf-8")
    excluded = tmp_path / "old.jsonl"
    excluded.write_text(
        json.dumps(
            {
                "task_id": "100",
                "image_path": str((images / "image-0.png").resolve()),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_egor_infographic_splits.py"

    first = _run_builder(
        script=script,
        source=source,
        images=images,
        excluded=excluded,
        output_root=tmp_path / "first",
    )
    second = _run_builder(
        script=script,
        source=source,
        images=images,
        excluded=excluded,
        output_root=tmp_path / "second",
    )

    first_calibration = _rows(first[0])
    first_heldout = _rows(first[1])
    assert [row["task_id"] for row in first_calibration] == [
        row["task_id"] for row in _rows(second[0])
    ]
    assert [row["task_id"] for row in first_heldout] == [row["task_id"] for row in _rows(second[1])]
    assert len(first_calibration) == 2
    assert len(first_heldout) == 3
    calibration_images = {Path(row["image_path"]).name for row in first_calibration}
    heldout_images = {Path(row["image_path"]).name for row in first_heldout}
    assert not calibration_images & heldout_images
    assert "image-0.png" not in calibration_images | heldout_images
    assert {row["answer"] for row in first_calibration + first_heldout}

    manifest = json.loads(first[2].read_text(encoding="utf-8"))
    assert manifest["image_overlap_between_splits"] == 0
    assert manifest["rows_excluded_by_task_id"] == 1
    assert "never inspect answers or prior outcomes" in manifest["selection_policy"]


def _run_builder(
    *,
    script: Path,
    source: Path,
    images: Path,
    excluded: Path,
    output_root: Path,
) -> tuple[Path, Path, Path]:
    calibration = output_root / "calibration.jsonl"
    heldout = output_root / "heldout.jsonl"
    manifest = output_root / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--source",
            str(source),
            "--images-root",
            str(images),
            "--exclude-jsonl",
            str(excluded),
            "--calibration-out",
            str(calibration),
            "--heldout-out",
            str(heldout),
            "--manifest",
            str(manifest),
            "--calibration-size",
            "2",
            "--heldout-size",
            "3",
            "--salt",
            "fixed-test-salt",
        ],
        check=True,
    )
    return calibration, heldout, manifest


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
