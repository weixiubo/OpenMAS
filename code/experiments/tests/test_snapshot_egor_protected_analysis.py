from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def test_snapshot_detects_content_metadata_and_file_set_changes(tmp_path: Path) -> None:
    module = _load_module()
    protected = tmp_path / "matrix_analysis"
    protected.mkdir()
    first = protected / "matrix_rows.csv"
    first.write_text("one\n", encoding="utf-8")
    log = tmp_path / "orchestrator.log"
    log.write_text("failed deterministically\n", encoding="utf-8")

    snapshot = module.snapshot_paths([protected, log])
    module.verify_snapshot(snapshot)

    first.write_text("two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        module.verify_snapshot(snapshot)

    first.write_text("one\n", encoding="utf-8")
    snapshot = module.snapshot_paths([protected, log])
    (protected / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        module.verify_snapshot(snapshot)


def test_snapshot_rejects_symlinked_protected_files(tmp_path: Path) -> None:
    module = _load_module()
    target = tmp_path / "target"
    target.write_text("protected\n", encoding="utf-8")
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "link").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        module.snapshot_paths([protected])


def _load_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "snapshot_egor_protected_analysis.py"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
