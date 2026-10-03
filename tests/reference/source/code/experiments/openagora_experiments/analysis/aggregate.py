"""CSV aggregation helpers for experiment outputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def summarize_rows(input_csv: str, output_csv: str) -> None:
    """Aggregate row-level csv by variant."""
    rows: list[dict[str, str]] = []
    with Path(input_csv).open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows.extend(reader)
    bucket: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    numeric_cols = ["quality_score", "token_consumption", "latency_seconds", "cost_usd", "perf_cost_ratio"]
    for row in rows:
        name = row["variant"]
        for col in numeric_cols:
            bucket[name][col].append(float(row[col]))
    out_rows: list[dict[str, str]] = []
    for variant, stats in bucket.items():
        out = {"variant": variant}
        for col in numeric_cols:
            values = stats[col]
            out[f"{col}_mean"] = f"{sum(values)/len(values):.6f}"
        out_rows.append(out)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

