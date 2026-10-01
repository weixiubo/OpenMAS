"""Run OpenAGORA qa_mini ablations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORE_SRC = ROOT / "code" / "core" / "src"
EXPERIMENTS_ROOT = ROOT / "code" / "experiments"
for _path in (str(CORE_SRC), str(EXPERIMENTS_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from openagora.config.loader import load_run_config
from openagora_experiments.suites.qa_mini import run_qa_mini_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run qa_mini ablation suite.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "qa_mini.yaml"),
        help="Experiment config yaml path",
    )
    parser.add_argument(
        "--rows",
        default=str(Path(__file__).resolve().parents[1] / "results" / "qa_rows.csv"),
        help="Row-level output csv",
    )
    parser.add_argument(
        "--summary",
        default=str(Path(__file__).resolve().parents[1] / "results" / "qa_summary.csv"),
        help="Summary output csv",
    )
    parser.add_argument(
        "--failure",
        default=str(Path(__file__).resolve().parents[1] / "results" / "qa_failure_cases.csv"),
        help="Failure cases csv",
    )
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--seed", type=int, default=None, help="Seed override")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    run_qa_mini_ablation(
        args.config,
        args.rows,
        args.summary,
        args.failure,
        sample_limit=args.sample_limit,
        seed=args.seed,
    )
    print(f"Rows: {args.rows}")
    print(f"Summary: {args.summary}")
    print(f"Failure cases: {args.failure}")
    print(f"Manifest: {cfg.eval.output_manifest_json}")
    print(f"Hashes: {cfg.eval.output_hashes_json}")


if __name__ == "__main__":
    main()
