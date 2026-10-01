from __future__ import annotations

import json

from openagora.utils.hash_utils import sha256_file
from openagora_experiments.adapters.gaia_mini import load_gaia_mini_tasks, normalize_text


def test_gaia_manifest_and_loader_success(tmp_path) -> None:
    dataset = tmp_path / "gaia.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "gaia-mini-001",
                "question": "What is 2 + 2?",
                "answer": "4",
                "answer_aliases": ["four"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "version": "gaia40-v1",
        "task_ids": ["gaia-mini-001"],
        "expected_rows": 1,
        "dataset_sha256": sha256_file(str(dataset)),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    tasks = load_gaia_mini_tasks(str(dataset), manifest_path=str(manifest_path), limit=1)
    assert len(tasks) == 1
    assert tasks[0].task_id == "gaia-mini-001"
    assert normalize_text(" Hello,  World! ") == "hello world"
