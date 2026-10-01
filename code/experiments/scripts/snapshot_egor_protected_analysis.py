#!/usr/bin/env python3
"""Snapshot or verify immutable EGOR diagnostic and failed-analysis artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openagora_experiments.egor.analysis_recovery import (
    snapshot_paths,
    verify_snapshot,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--path", action="append", required=True)
    snapshot.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--snapshot", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "snapshot":
        report = snapshot_paths([Path(value) for value in args.path])
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"snapshotted {len(report['entries'])} protected files")
        return 0
    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict):
        raise ValueError("protected-artifact snapshot must be a JSON object")
    verify_snapshot(snapshot)
    print(f"verified {len(snapshot['entries'])} protected files unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
