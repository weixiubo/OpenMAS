from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from uuid import uuid4

import yaml

from openagora.config.loader import load_run_config
from openagora_experiments.suites.gaia_mini import run_gaia_mini_ablation


def test_gaia_pipeline_outputs_and_reproducibility() -> None:
    base = Path(__file__).resolve().parents[1]
    config_path = str(base / "configs" / "gaia_mini.yaml")
    run_root = base / ".tmp_test_runs" / str(uuid4())
    run_root.mkdir(parents=True, exist_ok=True)

    rows1 = run_root / "rows1.csv"
    summary1 = run_root / "summary1.csv"
    pairwise1 = run_root / "pairwise1.csv"
    failure1 = run_root / "failure1.csv"
    audit1 = run_root / "audit1.jsonl"

    rows2 = run_root / "rows2.csv"
    summary2 = run_root / "summary2.csv"
    pairwise2 = run_root / "pairwise2.csv"
    failure2 = run_root / "failure2.csv"
    audit2 = run_root / "audit2.jsonl"

    cfg1 = load_run_config(config_path)
    cfg1.paths.db_path = str(run_root / "openagora1.db")
    cfg1.paths.artifacts_dir = str(run_root / "artifacts1")
    cfg1.eval.output_manifest_json = str(run_root / "manifest1.json")
    cfg1.eval.output_hashes_json = str(run_root / "hashes1.json")

    cfg2 = load_run_config(config_path)
    cfg2.paths.db_path = str(run_root / "openagora2.db")
    cfg2.paths.artifacts_dir = str(run_root / "artifacts2")
    cfg2.eval.output_manifest_json = str(run_root / "manifest2.json")
    cfg2.eval.output_hashes_json = str(run_root / "hashes2.json")

    cfg1_path = run_root / "cfg1.yaml"
    cfg2_path = run_root / "cfg2.yaml"
    _write_temp_cfg(cfg1_path, cfg1)
    _write_temp_cfg(cfg2_path, cfg2)

    run_gaia_mini_ablation(
        str(cfg1_path),
        str(rows1),
        str(summary1),
        str(pairwise1),
        str(failure1),
        str(audit1),
        sample_limit=6,
        judge_mode="rule",
        seed=2026,
    )
    run_gaia_mini_ablation(
        str(cfg2_path),
        str(rows2),
        str(summary2),
        str(pairwise2),
        str(failure2),
        str(audit2),
        sample_limit=6,
        judge_mode="rule",
        seed=2026,
    )

    assert rows1.exists() and rows1.read_text(encoding="utf-8").strip()
    assert summary1.exists() and summary1.read_text(encoding="utf-8").strip()
    assert pairwise1.exists() and pairwise1.read_text(encoding="utf-8").strip()
    assert failure1.exists()

    for path in [cfg1.eval.output_manifest_json, cfg1.eval.output_hashes_json, cfg2.eval.output_manifest_json, cfg2.eval.output_hashes_json]:
        assert Path(path).exists()

    manifest_payload = json.loads(Path(cfg1.eval.output_manifest_json).read_text(encoding="utf-8"))
    assert manifest_payload["suite"] == "gaia_mini"

    first_rows = _read_rows_without_run_id(rows1)
    second_rows = _read_rows_without_run_id(rows2)
    assert first_rows == second_rows
    assert "OpenAGORA_AlwaysOnRT" in {row["variant"] for row in first_rows}
    assert "NoOT" in {row["variant"] for row in first_rows}

    shutil.rmtree(run_root, ignore_errors=True)


def _read_rows_without_run_id(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = []
        for row in reader:
            row = dict(row)
            row.pop("run_id", None)
            rows.append(row)
    return rows


def _write_temp_cfg(path: Path, cfg) -> None:  # type: ignore[no-untyped-def]
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
