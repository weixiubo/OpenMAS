"""Export a filtered public-repository view from the private mother repo."""

from __future__ import annotations

import argparse
import fnmatch
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO_ROOT / ".tmp" / "public_repo_stage" / "OpenMAS"
PUBLIC_INCLUDE_PATHS: tuple[str, ...] = (
    ".github/workflows/openagora-ci.yml",
    ".github/workflows/openagora-release.yml",
    ".gitignore",
    ".gitattributes",
    "scripts/export_public.py",
    "scripts/export_public_repo.py",
    "PUBLIC_MANIFEST.json",
    "运行.py",
    "README.md",
    "LICENSE",
    "THIRD_PARTY.md",
    "OPEN_SOURCE_STATUS.md",
    "DEVELOPMENT.md",
    "examples",
    "reproduce",
    "docs/ARCHITECTURE.md",
    "docs/MIGRATION.md",
    "docs/model_sources.json",
    "SOURCE_PROVENANCE.json",
    "requirements-dev.lock",
    "code/core",
    "code/experiments",
)
PUBLIC_EXCLUDE_PATTERNS: tuple[str, ...] = (
    ".git/**",
    "paper/**",
    "resource/**",
    ".github/workflows/openagora-paper-private.yml",
    "code/core/.venv/**",
    "code/core/.coverage",
    "code/core/.pytest_cache/**",
    "code/core/.mypy_cache/**",
    "code/core/.ruff_cache/**",
    "code/core/.tmp*",
    "code/core/configs/.openagora/**",
    "code/core/zz_tmp_dir/**",
    "code/core/**/__pycache__/**",
    "code/core/src/openagora.egg-info/**",
    "code/experiments/.coverage",
    "code/experiments/.pytest_cache/**",
    "code/experiments/.ruff_cache/**",
    "code/experiments/.tmp*",
    "code/experiments/pytest-cache-files-*/**",
    "code/experiments/openagora_experiments.egg-info/**",
    "code/experiments/results/**",
    "code/experiments/scripts/build_server_baseline_report.py",
    "code/experiments/scripts/build_zero_cost_ceiling_report.py",
    "code/experiments/scripts/remote_local_repro.py",
    "code/experiments/scripts/run_strong_accept_suite.py",
    "code/experiments/tests/test_acmmm_compliance.py",
    "code/experiments/tests/test_analyze_pdf_quality.py",
    "code/experiments/tests/test_build_revision_thresholds.py",
    "code/experiments/tests/test_claim_consistency_camera_ready.py",
    "code/experiments/tests/test_evidence_pressure_summary.py",
    "code/experiments/tests/test_export_public_repo.py",
    "code/experiments/tests/test_free_uplift_loop.py",
    "code/experiments/tests/test_l1_l2_projection.py",
    "code/experiments/tests/test_pdf_hygiene.py",
    "code/experiments/tests/test_pdf_review_simulation.py",
    "code/experiments/tests/test_remote_local_repro.py",
    "code/experiments/tests/test_reviewer_evidence_summary.py",
    "code/experiments/tests/test_server_baseline_report.py",
    "code/experiments/tests/test_strong_accept_suite.py",
    "code/experiments/tests/test_submission_cycle_pipeline.py",
    "code/experiments/tests/test_validate_claim_chain_strict_core.py",
    "code/experiments/tests/test_zero_cost_ceiling_report.py",
    "code/experiments/**/__pycache__/**",
)


def _normalize_rel(path: str | Path) -> str:
    rel = Path(path).as_posix()
    if rel == ".":
        return ""
    if rel.startswith("./"):
        return rel[2:]
    return rel


def _is_under_included_root(rel_path: str) -> bool:
    return any(rel_path == root or rel_path.startswith(root + "/") for root in PUBLIC_INCLUDE_PATHS)


def is_public_path(rel_path: str | Path) -> bool:
    rel = _normalize_rel(rel_path)
    if not rel or not _is_under_included_root(rel):
        return False
    return not any(fnmatch.fnmatch(rel, pattern) for pattern in PUBLIC_EXCLUDE_PATTERNS)


def iter_public_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for root in PUBLIC_INCLUDE_PATHS:
        src = repo_root / root
        if not src.exists():
            continue
        if src.is_file():
            if is_public_path(root):
                files.append(src)
            continue
        for path in src.rglob("*"):
            if path.is_file():
                rel = path.relative_to(repo_root)
                if is_public_path(rel):
                    files.append(path)
    return sorted(files)


def export_public_tree(repo_root: Path, target_root: Path, *, clean: bool = False) -> int:
    target_root = target_root.resolve()
    if target_root == repo_root.resolve() or target_root in repo_root.resolve().parents:
        raise ValueError("Export target cannot replace the source repository")
    if clean and target_root.exists():
        if repo_root.resolve() / ".tmp" not in target_root.parents:
            raise ValueError("--clean is limited to the repository .tmp staging area")
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in iter_public_files(repo_root):
        rel = src.relative_to(repo_root)
        dst = target_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the public OpenMAS repository view.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--print-files", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    target_root = Path(args.target).resolve()
    copied = export_public_tree(repo_root, target_root, clean=args.clean)

    print(f"PUBLIC_EXPORT_OK target={target_root} files={copied}")
    if args.print_files:
        for path in iter_public_files(repo_root):
            print(path.relative_to(repo_root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
