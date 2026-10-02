"""Verify and test the immutable historical workflow snapshot in isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory


def extract_verified(root: Path, destination: Path) -> dict:
    metadata = json.loads((root / "reproduce/frozen/SOURCE.json").read_text())
    archive = root / "reproduce/frozen" / metadata["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Historical source archive checksum mismatch")
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination.resolve() not in target.parents or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe historical archive member: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read archive member: {member.name}")
            target.write_bytes(stream.read())
    for record in metadata["files"]:
        if hashlib.sha256((destination / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Historical source mismatch: {record['path']}")
    return metadata


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
