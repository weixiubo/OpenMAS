"""Run OpenAGORA GAIA mini evidence-chain suite."""

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
from openagora_experiments.suites.gaia_mini import run_gaia_mini_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GAIA mini evidence-chain suite.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "gaia_mini.yaml"),
        help="Experiment config yaml path",
    )
    parser.add_argument("--rows", default=None, help="Rows output csv")
    parser.add_argument("--summary", default=None, help="Summary output csv")
    parser.add_argument("--pairwise", default=None, help="Pairwise stats csv")
    parser.add_argument("--failure", default=None, help="Failure buckets csv")
    parser.add_argument("--judge-audit", default=None, help="Judge audit jsonl output")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--judge-mode", choices=["rule", "hybrid"], default=None, help="Judge mode override")
    parser.add_argument("--seed", type=int, default=None, help="Seed override")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    rows_path = args.rows or cfg.eval.output_csv
    summary_path = args.summary or cfg.eval.output_summary_csv
    pairwise_path = args.pairwise or cfg.eval.output_pairwise_csv
    failure_path = args.failure or cfg.eval.output_failure_csv
    judge_audit_path = args.judge_audit or cfg.eval.output_judge_audit_jsonl

    run_gaia_mini_ablation(
        args.config,
        rows_path,
        summary_path,
        pairwise_path,
        failure_path,
        judge_audit_path,
        sample_limit=args.sample_limit,
        judge_mode=args.judge_mode,
        seed=args.seed,
    )
    print(f"Rows: {rows_path}")
    print(f"Summary: {summary_path}")
    print(f"Pairwise stats: {pairwise_path}")
    print(f"Failure cases: {failure_path}")
    print(f"Manifest: {cfg.eval.output_manifest_json}")
    print(f"Hashes: {cfg.eval.output_hashes_json}")
    if (args.judge_mode or cfg.eval.judge_mode) == "hybrid":
        print(f"Judge audit: {judge_audit_path}")


if __name__ == "__main__":
    main()
