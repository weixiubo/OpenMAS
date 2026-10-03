#!/usr/bin/env python3
"""Combine heterogeneous EGOR policy CSVs into one statistics input."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


REQUIRED = (
    "source_id",
    "task_id",
    "policy_id",
    "triggered",
    "baseline_quality",
    "policy_quality",
    "quality_delta",
    "calls",
    "total_tokens",
    "latency_seconds",
    "strict_harm",
    "useful_repair",
    "provenance_rejected",
    "failed_stages",
)


def main() -> int:
    args = _parse_args()
    inputs = [Path(value).resolve() for value in args.input]
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    tasks_by_policy: dict[str, set[tuple[str, str]]] = {}
    for path in inputs:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or any(
                field not in reader.fieldnames for field in REQUIRED
            ):
                raise ValueError(f"policy CSV lacks required fields: {path}")
            for raw in reader:
                key = (raw["policy_id"], raw["source_id"], raw["task_id"])
                if key in seen:
                    raise ValueError(f"duplicate policy task: {key}")
                seen.add(key)
                tasks_by_policy.setdefault(key[0], set()).add((key[1], key[2]))
                rows.append({field: str(raw.get(field, "")) for field in REQUIRED})
    if not rows:
        raise ValueError("no policy rows found")
    expected = next(iter(tasks_by_policy.values()))
    mismatched = {
        policy: len(keys)
        for policy, keys in tasks_by_policy.items()
        if keys != expected
    }
    if mismatched:
        raise ValueError(f"policy task sets differ: {mismatched}")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED)
        writer.writeheader()
        writer.writerows(
            sorted(
                rows,
                key=lambda row: (row["policy_id"], row["source_id"], row["task_id"]),
            )
        )
    manifest = {
        "schema_version": 1,
        "source_commit": os.environ.get("EGOR_SOURCE_COMMIT", "").strip(),
        "policies": sorted(tasks_by_policy),
        "tasks_per_policy": len(expected),
        "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in inputs],
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"combined {len(tasks_by_policy)} policies over {len(expected)} tasks")
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
