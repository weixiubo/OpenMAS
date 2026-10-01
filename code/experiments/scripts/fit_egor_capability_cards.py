#!/usr/bin/env python3
"""Fit frozen cross-type EGOR capability cards from singleton matrix rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from openagora.mechanisms.evidence_gap_repair import EvidenceType
from openagora_experiments.egor.planning import (
    capability_cards_payload,
    fit_capability_cards,
)


def main() -> int:
    args = _parse_args()
    matrix_path = Path(args.matrix_rows).resolve()
    action_map_path = Path(args.action_map).resolve()
    with matrix_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    action_payload = json.loads(action_map_path.read_text(encoding="utf-8"))
    if not isinstance(action_payload, dict):
        raise ValueError("action map must be a JSON object")
    action_candidates: dict[str, str] = {}
    admissible_gap_types: dict[str, list[EvidenceType]] = {}
    for action_id, raw_spec in action_payload.items():
        if not isinstance(raw_spec, dict):
            raise ValueError("each action map value must be an object")
        action_candidates[str(action_id)] = str(raw_spec.get("candidate_id", ""))
        raw_types = raw_spec.get("gap_types", [])
        if not isinstance(raw_types, list):
            raise ValueError("gap_types must be a list")
        admissible_gap_types[str(action_id)] = [
            EvidenceType(str(value)) for value in raw_types
        ]
    cards, estimates = fit_capability_cards(
        rows,
        action_candidates=action_candidates,
        admissible_gap_types=admissible_gap_types,
        minimum_gap=args.minimum_gap,
        minimum_eligible=args.minimum_eligible,
        useful_margin=args.useful_margin,
        correct_threshold=args.correct_threshold,
        noninferiority_tolerance=args.noninferiority_tolerance,
    )
    payload = capability_cards_payload(
        cards,
        estimates,
        metadata={
            "matrix_rows": str(matrix_path),
            "matrix_rows_sha256": _sha256(matrix_path),
            "action_map": str(action_map_path),
            "action_map_sha256": _sha256(action_map_path),
            "minimum_gap": args.minimum_gap,
            "minimum_eligible": args.minimum_eligible,
            "useful_margin": args.useful_margin,
            "correct_threshold": args.correct_threshold,
            "noninferiority_tolerance": args.noninferiority_tolerance,
            "supply_estimator": "Beta(1,1) posterior mean; zero when eligible count is below minimum",
            "provenance_policy": "unsupported changed answers revert to baseline before closure scoring",
        },
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(cards)} capability cards to {output_path}")
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-rows", required=True)
    parser.add_argument("--action-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-gap", type=float, default=0.05)
    parser.add_argument("--minimum-eligible", type=int, default=5)
    parser.add_argument("--useful-margin", type=float, default=0.05)
    parser.add_argument("--correct-threshold", type=float, default=0.95)
    parser.add_argument("--noninferiority-tolerance", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
