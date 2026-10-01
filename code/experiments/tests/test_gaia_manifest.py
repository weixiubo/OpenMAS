from __future__ import annotations

import json

import pytest

from openagora_experiments.adapters.gaia_mini import load_gaia_mini_tasks


def test_gaia_manifest_dataset_hash_validation(tmp_path) -> None:
    dataset = tmp_path / "gaia.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "gaia-mini-001",
                "question": "q",
                "answer": "a",
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
        "dataset_sha256": "deadbeef",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError):
        load_gaia_mini_tasks(str(dataset), manifest_path=str(manifest_path), limit=1)

