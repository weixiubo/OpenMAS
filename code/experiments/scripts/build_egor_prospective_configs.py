#!/usr/bin/env python3
"""Build deterministic planned-policy and full-matrix configs for one run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    args = _parse_args()
    root = Path(args.output_root).resolve()
    execution_path = Path(args.execution_output).resolve()
    matrix_path = Path(args.matrix_output).resolve()
    execution = build_execution_config(root)
    matrix = build_matrix_config(root)
    for path, payload in ((execution_path, execution), (matrix_path, matrix)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"wrote prospective configs under {root}")
    return 0


def build_execution_config(root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "actions": {
            "exact": str(root / "specialist_exact_qwen25.jsonl"),
            "numerical": str(root / "specialist_numerical_qwen25.jsonl"),
            "verifier": str(root / "specialist_verify_qwen3.jsonl"),
            "visual": str(root / "specialist_visual_internvl3.jsonl"),
        },
        "combinations": {
            "exact": str(root / "exact_repair.jsonl"),
            "exact__visual": str(root / "exact_visual_repair.jsonl"),
            "numerical": str(root / "numerical_repair.jsonl"),
            "verifier": str(root / "verifier_repair.jsonl"),
            "visual": str(root / "visual_repair.jsonl"),
        },
    }


def build_matrix_config(root: Path) -> dict[str, Any]:
    candidates = [
        _candidate(root, "baseline_resynthesis", "baseline_resynthesis.jsonl", []),
        _candidate(
            root,
            "exact_repair",
            "exact_repair.jsonl",
            [("qwen25_ocr_span", "exact_span", "specialist_exact_qwen25.jsonl")],
        ),
        _candidate(
            root,
            "visual_repair",
            "visual_repair.jsonl",
            [
                (
                    "internvl3_visual_relation",
                    "visual_relation",
                    "specialist_visual_internvl3.jsonl",
                )
            ],
        ),
        _candidate(
            root,
            "numerical_repair",
            "numerical_repair.jsonl",
            [
                (
                    "qwen25_numerical_derivation",
                    "numerical_derivation",
                    "specialist_numerical_qwen25.jsonl",
                )
            ],
        ),
        _candidate(
            root,
            "verifier_repair",
            "verifier_repair.jsonl",
            [
                (
                    "qwen3_independent_verifier",
                    "independent_verification",
                    "specialist_verify_qwen3.jsonl",
                )
            ],
        ),
        _candidate(
            root,
            "exact_visual_repair",
            "exact_visual_repair.jsonl",
            [
                ("qwen25_ocr_span", "exact_span", "specialist_exact_qwen25.jsonl"),
                (
                    "internvl3_visual_relation",
                    "visual_relation",
                    "specialist_visual_internvl3.jsonl",
                ),
            ],
        ),
        _candidate(
            root,
            "full_team",
            "full_team.jsonl",
            [
                ("qwen25_ocr_span", "exact_span", "specialist_exact_qwen25.jsonl"),
                (
                    "qwen25_numerical_derivation",
                    "numerical_derivation",
                    "specialist_numerical_qwen25.jsonl",
                ),
                (
                    "qwen3_independent_verifier",
                    "independent_verification",
                    "specialist_verify_qwen3.jsonl",
                ),
                (
                    "internvl3_visual_relation",
                    "visual_relation",
                    "specialist_visual_internvl3.jsonl",
                ),
            ],
        ),
    ]
    return {
        "schema_version": 1,
        "baseline": str(root / "baseline_qwen3.jsonl"),
        "candidates": candidates,
        "policies": [
            _policy(
                "baseline_resynthesis_all",
                "baseline_resynthesis",
                None,
                False,
                trigger_mode="all",
            ),
            _policy("fixed_exact_gap", "exact_repair", 0.05, True),
            # Visual has the highest all-development fixed utility under the
            # frozen 0.01 call price (0.04387394781144781).
            _policy("best_fixed_visual_gap", "visual_repair", 0.05, True),
            _policy("fixed_numerical_gap", "numerical_repair", 0.05, True),
            _policy("fixed_verifier_gap", "verifier_repair", 0.05, True),
            _policy("fixed_exact_visual_gap", "exact_visual_repair", 0.05, True),
            _policy(
                "full_team_all",
                "full_team",
                None,
                True,
                trigger_mode="all",
            ),
        ],
    }


def _candidate(
    root: Path,
    candidate_id: str,
    synthesis: str,
    specialists: list[tuple[str, str, str]],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "specialists": [
            {
                "action_id": action_id,
                "evidence_type": evidence_type,
                "path": str(root / path),
            }
            for action_id, evidence_type, path in specialists
        ],
        "synthesis": str(root / synthesis),
    }


def _policy(
    policy_id: str,
    candidate_id: str,
    threshold: float | None,
    require_evidence: bool,
    *,
    trigger_mode: str = "gap",
) -> dict[str, Any]:
    policy = {
        "policy_id": policy_id,
        "candidate_id": candidate_id,
        "trigger_mode": trigger_mode,
        "require_evidence_for_change": require_evidence,
    }
    if threshold is not None:
        policy["threshold"] = threshold
    return policy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--execution-output", required=True)
    parser.add_argument("--matrix-output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
