"""Assert evidence-chain outputs exist and have required columns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Assert experiment output artifacts")
    parser.add_argument("--rows", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--pairwise", required=True)
    parser.add_argument("--failure", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--hashes", required=True)
    parser.add_argument("--judge-audit", default=None)
    args = parser.parse_args()

    _assert_csv(
        args.rows,
        required={"variant", "task_id", "rule_score", "final_score", "is_correct", "error_type"},
    )
    _assert_csv(
        args.summary,
        required={"variant", "count", "accuracy", "final_score_mean"},
    )
    _assert_csv(
        args.pairwise,
        required={"variant", "metric", "test", "effect", "p_value"},
    )
    _assert_csv(
        args.failure,
        required={"variant", "error_type", "count"},
    )

    _assert_manifest(args.manifest)
    _assert_hashes(args.hashes)

    if args.judge_audit:
        _assert_jsonl(args.judge_audit)

    print("ASSERT_OK")


def _assert_csv(path: str, *, required: set[str]) -> None:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing csv: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing columns {sorted(missing)}: {csv_path}")
        rows = list(reader)
        if not rows:
            raise ValueError(f"CSV has no rows: {csv_path}")


def _assert_manifest(path: str) -> None:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be object: {manifest_path}")
    required = {"suite", "sample_limit", "judge_mode", "seed"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(f"Manifest missing keys {sorted(missing)}: {manifest_path}")


def _assert_hashes(path: str) -> None:
    hashes_path = Path(path)
    if not hashes_path.exists():
        raise FileNotFoundError(f"Missing hashes: {hashes_path}")
    payload = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes = payload.get("hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"Hashes must contain non-empty map: {hashes_path}")


def _assert_jsonl(path: str) -> None:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing jsonl: {file_path}")
    lines = [line for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"JSONL has no entries: {file_path}")
    for line in lines[:5]:
        json.loads(line)


if __name__ == "__main__":
    main()
