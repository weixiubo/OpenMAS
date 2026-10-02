from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from openmas_test_history import historical_root


def test_selector_statistics_binds_inputs_and_computes_paired_contrasts(
    tmp_path: Path,
) -> None:
    selector = tmp_path / "selector.csv"
    fixed = tmp_path / "fixed.csv"
    standalone = tmp_path / "standalone.csv"
    _write(
        selector,
        [
            ("source-a", "t1", "exact", 1.0, 1.0, 0),
            ("source-b", "t2", "baseline", 0.0, 0.0, 0),
        ],
    )
    _write(
        fixed,
        [
            ("source-a", "t1", "visual", 0.5, 2.0, 0),
            ("source-b", "t2", "visual", -0.5, 2.0, 1),
        ],
    )
    _write(
        standalone,
        [
            ("source-a", "t1", "exact", 0.75, 1.0, 0),
            ("source-b", "t2", "baseline", 0.0, 0.0, 0),
        ],
    )
    output = tmp_path / "statistics.json"
    script = (historical_root() / "code/experiments") / "scripts" / "analyze_egor_selector_statistics.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--selector-rows",
            str(selector),
            "--best-fixed-rows",
            str(fixed),
            "--standalone-rows",
            str(standalone),
            "--bootstrap-samples",
            "1000",
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["tasks"] == 2
    assert payload["selector_counts"]["selections"] == {
        "baseline": 1,
        "exact": 1,
    }
    assert payload["contrasts"]["selector_minus_baseline_quality"]["mean_delta"] == 0.5
    assert payload["contrasts"]["selector_minus_best_fixed_quality"]["mean_delta"] == 0.5
    assert payload["contrasts"]["calls_saved_vs_best_fixed"]["mean_delta"] == 1.5
    assert payload["inputs"]["selector"]["sha256"]


def _write(path: Path, rows: list[tuple[str, str, str, float, float, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_id",
                "task_id",
                "selected_id",
                "quality_delta",
                "incremental_calls",
                "strict_harm",
            ]
        )
        writer.writerows(rows)
