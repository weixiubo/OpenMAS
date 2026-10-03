from __future__ import annotations

import csv

from openagora_experiments.analysis.aggregate import summarize_rows


def test_summarize_rows(tmp_path) -> None:
    input_csv = tmp_path / "rows.csv"
    output_csv = tmp_path / "summary.csv"

    rows = [
        {
            "variant": "A",
            "quality_score": "0.8",
            "token_consumption": "100",
            "latency_seconds": "2",
            "cost_usd": "0.2",
            "perf_cost_ratio": "4",
        },
        {
            "variant": "A",
            "quality_score": "0.6",
            "token_consumption": "120",
            "latency_seconds": "3",
            "cost_usd": "0.3",
            "perf_cost_ratio": "2",
        },
    ]
    with input_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summarize_rows(str(input_csv), str(output_csv))

    with output_csv.open("r", encoding="utf-8") as fp:
        out_rows = list(csv.DictReader(fp))
    assert len(out_rows) == 1
    assert out_rows[0]["variant"] == "A"
    assert out_rows[0]["quality_score_mean"] == "0.700000"
