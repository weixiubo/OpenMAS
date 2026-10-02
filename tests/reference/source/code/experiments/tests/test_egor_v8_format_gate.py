from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_egor_v8_format_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_egor_v8_format_gate", SCRIPT
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_v8_format_gate_is_outcome_blind_and_rejects_missing_sources(
    tmp_path: Path,
) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "score_prediction" not in source
    assert "reference_answers" not in source
    assert '"scores_computed": False' in source
    report = module.build_gate(
        tmp_path,
        "commit",
        {"chartqa": "a" * 64, "infographic": "b" * 64},
    )
    assert report["passed"] is False
    assert report["v7_format_outputs_used"] is False
    assert report["sources"]["chartqa"]["checks"]["verifier_stage"] is False
