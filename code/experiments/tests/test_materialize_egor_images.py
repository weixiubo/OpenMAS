from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from openmas_test_history import historical_root
from PIL import Image


def test_materializer_caps_pixels_without_using_answer_fields(tmp_path: Path) -> None:
    source_image = tmp_path / "source.png"
    Image.new("RGB", (120, 80), color=(20, 40, 60)).save(source_image)
    tasks = tmp_path / "tasks.jsonl"
    rows = [
        {
            "task_id": "task-a",
            "benchmark": "fixture",
            "question": "question a",
            "answer": "private-a",
            "reference_answers": ["private-a"],
            "image_path": str(source_image),
        },
        {
            "task_id": "task-b",
            "benchmark": "fixture",
            "question": "different question",
            "answer": "private-b",
            "reference_answers": ["private-b"],
            "image_path": str(source_image),
        },
    ]
    tasks.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output_tasks = tmp_path / "output" / "tasks.jsonl"
    output_images = tmp_path / "output" / "images"
    manifest = tmp_path / "output" / "manifest.json"
    script = (historical_root() / "code/experiments") / "scripts" / "materialize_egor_images.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--tasks",
            str(tasks),
            "--out-tasks",
            str(output_tasks),
            "--out-images",
            str(output_images),
            "--max-pixels",
            "2500",
            "--manifest",
            str(manifest),
        ],
        check=True,
    )

    output_rows = [json.loads(line) for line in output_tasks.read_text(encoding="utf-8").splitlines()]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    with Image.open(output_rows[0]["image_path"]) as first, Image.open(output_rows[1]["image_path"]) as second:
        assert first.width * first.height <= 2500
        assert first.size == second.size
    assert output_rows[0]["answer"] == "private-a"
    assert output_rows[1]["answer"] == "private-b"
    assert payload["resized_tasks"] == 2
    assert payload["images"][0]["output_sha256"] == payload["images"][1]["output_sha256"]
    assert "answers" in payload["selection_policy"]
