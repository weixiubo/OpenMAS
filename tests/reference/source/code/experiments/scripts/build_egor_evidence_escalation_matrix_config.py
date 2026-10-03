#!/usr/bin/env python3
"""Build a sparse-matrix config for staged evidence escalation and controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    args = _parse_args()
    payload = build_config(args)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote evidence-escalation matrix config to {output_path}")
    return 0


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    exact = _specialist(
        "qwen25_ocr_span",
        "exact_span",
        args.exact_specialist,
    )
    visual = _specialist(
        "internvl3_visual_relation",
        "visual_relation",
        args.visual_specialist,
    )
    numerical = _specialist(
        "qwen25_numerical_derivation",
        "numerical_derivation",
        args.numerical_specialist,
    )
    verifier = _specialist(
        "qwen3_independent_verifier",
        "independent_verification",
        args.verifier_specialist,
    )
    candidates = [
        _candidate("core_pair", args.core_synthesis, [exact, visual]),
        _candidate(
            "full_team",
            args.full_synthesis,
            [exact, visual, numerical, verifier],
        ),
    ]
    if args.baseline_resynthesis:
        candidates.append(
            _candidate(
                "baseline_resynthesis",
                args.baseline_resynthesis,
                [],
            )
        )
    if args.visual_synthesis:
        candidates.append(
            _candidate("visual_control", args.visual_synthesis, [visual])
        )
    return {
        "schema_version": 1,
        "baseline": str(Path(args.baseline).resolve()),
        "candidates": candidates,
        "policies": [],
    }


def _candidate(
    candidate_id: str,
    synthesis: str,
    specialists: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "specialists": specialists,
        "synthesis": str(Path(synthesis).resolve()),
    }


def _specialist(
    action_id: str,
    evidence_type: str,
    path: str,
) -> dict[str, str]:
    return {
        "action_id": action_id,
        "evidence_type": evidence_type,
        "path": str(Path(path).resolve()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--exact-specialist", required=True)
    parser.add_argument("--visual-specialist", required=True)
    parser.add_argument("--numerical-specialist", required=True)
    parser.add_argument("--verifier-specialist", required=True)
    parser.add_argument("--core-synthesis", required=True)
    parser.add_argument("--full-synthesis", required=True)
    parser.add_argument("--baseline-resynthesis", default="")
    parser.add_argument("--visual-synthesis", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
