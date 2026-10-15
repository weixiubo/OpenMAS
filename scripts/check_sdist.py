"""Rebuild both library sdists and test them in a reconstructed workspace."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from check_release import check_install

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = {"openagora-": "code/core", "openagora_experiments-": "code/experiments"}


def command(args, *, cwd, env):
    result = subprocess.run(list(map(str, args)), cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def wheel_contents(path):
    with zipfile.ZipFile(path) as archive:
        return {n: archive.read(n) for n in archive.namelist() if not n.endswith(".dist-info/RECORD")}


def unpack_member(source, prefix, relative, workspace):
    public = json.loads((ROOT / "PUBLIC_MANIFEST.json").read_text())
    required = {
        row["path"].removeprefix(relative + "/")
        for row in public["files"]
        if row["path"].startswith(relative + "/") and not Path(row["path"]).name.startswith(".")
    }
    with tarfile.open(source) as archive:
        members = {m.name.split("/", 1)[-1]: m for m in archive.getmembers() if m.isfile()}
        assert not required - members.keys(), f"Missing {prefix} members: {sorted(required - members.keys())}"
        for name in sorted(required):
            stream = archive.extractfile(members[name])
            assert stream is not None and stream.read() == (ROOT / relative / name).read_bytes(), name
        staging = workspace / (prefix + "unpack")
        archive.extractall(staging, filter="data")
    target = workspace / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    next(staging.iterdir()).rename(target)
    staging.rmdir()


def main():
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}}
    env.update(UV_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    with TemporaryDirectory(prefix="openmas-sdist-") as directory:
        workspace = Path(directory)
        for prefix, relative in MEMBERS.items():
            sources = list((ROOT / "dist").glob(prefix + "*.tar.gz"))
            assert len(sources) == 1, f"Build one current {prefix} sdist first"
            unpack_member(sources[0], prefix, relative, workspace)
        # Workspace metadata and cross-package reference drivers are released in
        # the public source export, separately from the two library sdists.
        for relative in (
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            "scripts/check_historical.py",
            "scripts/handoff.py",
        ):
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        shutil.copytree(ROOT / "tests", workspace / "tests", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        command([uv, "sync", "--locked", "--offline", "--python", sys.executable], cwd=workspace, env=env)
        tests = command([workspace / ".venv/bin/python", "-m", "pytest", "-q"], cwd=workspace, env=env)
        command([uv, "build", "--all-packages", "--wheel", "--offline"], cwd=workspace, env=env)
        wheels = sorted((workspace / "dist").glob("*.whl"))
        for wheel in wheels:
            assert wheel_contents(wheel) == wheel_contents(ROOT / "dist" / wheel.name), wheel.name
        installed = check_install(wheels, workspace / "install-check")
    print(
        json.dumps(
            {"sdists": 2, "workspace_tests": tests.strip(), "wheel_payload_matches": True, "installed": installed},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
