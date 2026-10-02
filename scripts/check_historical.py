"""Verify and test the immutable historical workflow snapshot in isolation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def extract_verified(root: Path, destination: Path) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location("reference_support", root / "tests/reference_support.py")
    assert spec and spec.loader
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    return support.copy_reference(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix="openmas-history-") as directory:
        target = Path(directory)
        metadata = extract_verified(root, target)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            str(target / path) for path in ("code/core/src", "code/experiments", "code/experiments/scripts")
        )
        command = [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib", *metadata["test_files"]]
        if args.junit:
            command.append(f"--junitxml={args.junit.resolve()}")
        return subprocess.run(command, cwd=target, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
