"""Inspect distributions and run installed wheel commands outside the source tree."""

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

ROOT = Path(__file__).resolve().parents[1]


def inspect_distributions() -> list[Path]:
    wheels = sorted((ROOT / "dist").glob("*.whl"))
    sources = sorted((ROOT / "dist").glob("*.tar.gz"))
    if len(wheels) != 2 or len(sources) != 2:
        raise ValueError("Run uv build --all-packages into a clean dist directory first")
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            if not any(name.endswith("/licenses/LICENSE") for name in names):
                raise ValueError(f"License missing from {wheel.name}")
            metadata = next(name for name in names if name.endswith(".dist-info/METADATA"))
            if "License-Expression: Apache-2.0" not in archive.read(metadata).decode():
                raise ValueError(f"License metadata missing from {wheel.name}")
            required = (
                "openagora/resource_catalog.json"
                if wheel.name.startswith("openagora-")
                else "openagora_experiments/evidence/egor-v10/task_rows.csv"
            )
            if required not in names:
                raise ValueError(f"Required resource {required} missing from {wheel.name}")
    for source in sources:
        with tarfile.open(source) as archive:
            if not any(name.endswith("/LICENSE") for name in archive.getnames()):
                raise ValueError(f"License missing from {source.name}")
    return wheels


def command(args, *, cwd, env=None) -> str:
    completed = subprocess.run(list(map(str, args)), cwd=cwd, env=env, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    return completed.stdout


def check_install(wheels: list[Path], temporary: Path) -> dict:
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    environment = temporary / "environment"
    install_env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT", "UV_PROJECT_ENVIRONMENT"}
    }
    install_env["UV_PROJECT_ENVIRONMENT"] = str(environment)
    # Seed runtime dependencies directly from the lock, with no index lookup.
    command(
        [uv, "sync", "--locked", "--offline", "--no-dev", "--no-install-workspace", "--python", sys.executable],
        cwd=ROOT,
        env=install_env,
    )
    python = environment / "bin/python"
    command(
        [uv, "pip", "install", "--offline", "--no-deps", "--python", python, *wheels],
        cwd=temporary,
    )
    command([uv, "pip", "check", "--python", python], cwd=temporary)
    directory = temporary / "unrelated-working-directory"
    directory.mkdir()
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH" and not key.startswith("OPENMAS_")}
    env.update(HOME=str(temporary / "home"), XDG_CONFIG_HOME=str(temporary / "home/.config"))
    result = {}
    for name in ("doctor", "demo", "reproduce"):
        output = command([environment / "bin/openagora", name], cwd=directory, env=env)
        result[name] = json.loads(output)
    assert result["doctor"]["status"] == "ok"
    assert result["demo"]["result"]["is_correct"] and result["demo"]["second_run_cache_hit"]
    assert result["reproduce"]["tasks"] == 800 and result["reproduce"]["evidence_verified"]
    assert not result["reproduce"]["scientific_gate_passed"]
    assert not list(directory.iterdir()), "Offline commands leaked output into caller's working directory"
    return {"doctor": "ok", "demo": "task and cache verified", "reproduce": "800 frozen rows verified"}


def main() -> None:
    wheels = inspect_distributions()
    with TemporaryDirectory(prefix="openmas-release-") as directory:
        commands = check_install(wheels, Path(directory))
    print(
        json.dumps({"status": "ok", "wheels": [path.name for path in wheels], "installed_commands": commands}, indent=2)
    )


if __name__ == "__main__":
    main()
