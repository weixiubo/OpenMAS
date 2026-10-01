"""Run a unified mechanism evidence pass for the paper.

This is an internal paper-side helper. It refreshes the revision result files
expected by `paper/scripts/build_revision_results.py` under one controlled
variant set and provider configuration.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CORE_SRC = ROOT / "code" / "core" / "src"
EXPERIMENTS_ROOT = ROOT / "code" / "experiments"
for _path in (str(CORE_SRC), str(EXPERIMENTS_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from openagora.config.loader import load_run_config
from openagora_experiments.suites.gaia_mini import run_gaia_mini_ablation
from openagora_experiments.suites.qa_mini import run_qa_mini_ablation

CONFIGS = EXPERIMENTS_ROOT / "configs"
DEFAULT_REVISION_DIR = EXPERIMENTS_ROOT / "results" / "revision"
PAPER_ROOT = ROOT / "paper"
BUILD_RESULTS = PAPER_ROOT / "scripts" / "build_revision_results.py"
BUILD_ASSETS = PAPER_ROOT / "scripts" / "build_paper_assets.py"

DEFAULT_VARIANTS = ["Full", "AlwaysOnRT", "NoNeg", "NoOT", "NoRT", "NoCache", "ADAS"]
GAIA_VARIANT_MAP = {
    "Full": "OpenAGORA_Full",
    "AlwaysOnRT": "OpenAGORA_AlwaysOnRT",
    "NoNeg": "NoNeg",
    "NoOT": "NoOT",
    "NoRT": "NoRT",
    "NoCache": "NoCache",
    "ADAS": "SingleAgentTool",
    "AFlow": "PlannerExecutor2",
}
QA_VARIANT_MAP = {
    "Full": "OpenAGORA_Full",
    "AlwaysOnRT": "OpenAGORA_AlwaysOnRT",
    "NoNeg": "OpenAGORA_NoNeg",
    "NoOT": "OpenAGORA_NoOT",
    "NoRT": "OpenAGORA_NoRT",
    "NoCache": "OpenAGORA_NoCache",
    "ADAS": "SingleAgentTool",
}


def _write_cfg(
    *,
    source_path: Path,
    temp_dir: Path,
    provider_mode: str,
    provider_name: str,
    provider_model: str,
    local_backend: str,
    local_model_path: str,
    local_model_profile: str,
    local_allow_download: bool,
    max_gpu_mem_gb: float,
    gpu_devices: str,
    sample_limit: int,
    judge_mode: str,
    db_stem: str,
    manifest_stem: str,
) -> Path:
    cfg = load_run_config(str(source_path))
    provider_payload = cfg.provider.model_dump(mode="json")
    provider_payload["provider_mode"] = provider_mode
    provider_payload["name"] = provider_name
    provider_payload["model"] = provider_model
    provider_payload["local_backend"] = local_backend
    provider_payload["local_model_path"] = local_model_path
    provider_payload["local_model_profile"] = local_model_profile
    provider_payload["local_allow_download"] = local_allow_download
    provider_payload["max_gpu_mem_gb"] = max_gpu_mem_gb
    provider_payload["gpu_devices"] = gpu_devices
    cfg.provider = cfg.provider.model_validate(provider_payload)

    eval_payload = cfg.eval.model_dump(mode="json")
    eval_payload["sample_limit"] = sample_limit
    eval_payload["judge_mode"] = judge_mode
    eval_payload["output_manifest_json"] = str(temp_dir / f"{manifest_stem}_manifest.json")
    eval_payload["output_hashes_json"] = str(temp_dir / f"{manifest_stem}_hashes.json")
    cfg.eval = cfg.eval.model_validate(eval_payload)

    path_payload = cfg.paths.model_dump(mode="json")
    path_payload["db_path"] = str(temp_dir / f"{db_stem}.db")
    path_payload["artifacts_dir"] = str(temp_dir / f"{db_stem}_artifacts")
    cfg.paths = cfg.paths.model_validate(path_payload)

    out_path = temp_dir / f"{source_path.stem}_{db_stem}.yaml"
    out_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    return out_path


def _refresh_paper(py_exe: str, revision_dir: Path) -> None:
    subprocess.run(
        [py_exe, str(BUILD_RESULTS), "--mode", "real", "--revision-dir", str(revision_dir)],
        cwd=str(PAPER_ROOT),
        check=True,
    )
    subprocess.run([py_exe, str(BUILD_ASSETS)], cwd=str(PAPER_ROOT), check=True)


def _clean_revision_dir(revision_dir: Path) -> None:
    revision_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.csv", "*.json", "*.jsonl"):
        for path in revision_dir.glob(pattern):
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh mechanism evidence under one unified harness.")
    parser.add_argument("--gaia-config", default=str(CONFIGS / "gaia_mini.yaml"))
    parser.add_argument("--qa-config", default=str(CONFIGS / "qa_mini.yaml"))
    parser.add_argument("--revision-dir", default=str(DEFAULT_REVISION_DIR))
    parser.add_argument("--provider-mode", choices=["mock", "live", "local"], default="mock")
    parser.add_argument("--provider-name", default="")
    parser.add_argument("--provider-model", default="")
    parser.add_argument("--local-backend", choices=["transformers-local", "vllm-local"], default="transformers-local")
    parser.add_argument("--local-model-path", default="")
    parser.add_argument("--local-model-profile", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--local-allow-download", action="store_true")
    parser.add_argument("--max-gpu-mem-gb", type=float, default=0.0)
    parser.add_argument("--gpu-devices", default="")
    parser.add_argument("--gaia-seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--qa-seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--sample-limit", type=int, default=40)
    parser.add_argument("--qa-sample-limit", type=int, default=5)
    parser.add_argument("--judge-mode", choices=["rule", "hybrid"], default="rule")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--refresh-paper", action="store_true")
    parser.add_argument("--clean-revision-dir", action="store_true")
    args = parser.parse_args()

    py_exe = sys.executable
    revision_dir = Path(args.revision_dir)
    revision_dir.mkdir(parents=True, exist_ok=True)
    provider_name = args.provider_name or ("local" if args.provider_mode == "local" else "mock")
    provider_model = args.provider_model or (args.local_model_path or "mock-leader-v1")

    temp_root = Path(tempfile.mkdtemp(prefix="openagora_mechanism_"))
    try:
        if args.clean_revision_dir:
            _clean_revision_dir(revision_dir)

        gaia_cfg = _write_cfg(
            source_path=Path(args.gaia_config),
            temp_dir=temp_root,
            provider_mode=args.provider_mode,
            provider_name=provider_name,
            provider_model=provider_model,
            local_backend=args.local_backend,
            local_model_path=args.local_model_path,
            local_model_profile=args.local_model_profile,
            local_allow_download=args.local_allow_download,
            max_gpu_mem_gb=args.max_gpu_mem_gb,
            gpu_devices=args.gpu_devices,
            sample_limit=args.sample_limit,
            judge_mode=args.judge_mode,
            db_stem="gaia_mechanism",
            manifest_stem="gaia_mechanism",
        )
        qa_cfg = _write_cfg(
            source_path=Path(args.qa_config),
            temp_dir=temp_root,
            provider_mode=args.provider_mode,
            provider_name=provider_name,
            provider_model=provider_model,
            local_backend=args.local_backend,
            local_model_path=args.local_model_path,
            local_model_profile=args.local_model_profile,
            local_allow_download=args.local_allow_download,
            max_gpu_mem_gb=args.max_gpu_mem_gb,
            gpu_devices=args.gpu_devices,
            sample_limit=args.qa_sample_limit,
            judge_mode="rule",
            db_stem="qa_mechanism",
            manifest_stem="qa_mechanism",
        )

        gaia_variants = [GAIA_VARIANT_MAP[name] for name in args.variants if name in GAIA_VARIANT_MAP]
        qa_variants = [QA_VARIANT_MAP[name] for name in args.variants if name in QA_VARIANT_MAP]

        for seed in args.gaia_seeds:
            run_gaia_mini_ablation(
                str(gaia_cfg),
                str(revision_dir / f"gaia_rows_seed{seed}.csv"),
                str(revision_dir / f"gaia_summary_seed{seed}.csv"),
                str(revision_dir / f"gaia_pairwise_seed{seed}.csv"),
                str(revision_dir / f"gaia_failure_seed{seed}.csv"),
                str(revision_dir / f"gaia_judge_audit_seed{seed}.jsonl"),
                sample_limit=args.sample_limit,
                judge_mode=args.judge_mode,
                seed=seed,
                variant_filter=gaia_variants,
            )

        for seed in args.qa_seeds:
            run_qa_mini_ablation(
                str(qa_cfg),
                str(revision_dir / f"qa_rows_seed{seed}.csv"),
                str(revision_dir / f"qa_summary_seed{seed}.csv"),
                str(revision_dir / f"qa_failure_seed{seed}.csv"),
                sample_limit=args.qa_sample_limit,
                seed=seed,
                variant_filter=qa_variants,
            )

        if args.refresh_paper:
            _refresh_paper(py_exe, revision_dir)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
