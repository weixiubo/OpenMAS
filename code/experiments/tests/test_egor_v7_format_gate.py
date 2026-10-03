from __future__ import annotations

import importlib.util
import sys

from openmas_test_history import historical_root

SCRIPT = (historical_root() / "code/experiments") / "scripts" / "build_egor_v7_format_gate.py"
SPEC = importlib.util.spec_from_file_location("build_egor_v7_format_gate", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_format_gate_contract_has_no_scoring_path() -> None:
    assert set(module.EXPECTED_TASK_HASHES) == {"chartqa", "infographic"}
    source = SCRIPT.read_text(encoding="utf-8")
    assert "score_prediction" not in source
    assert "reference_answers" not in source
    assert '"scores_computed": False' in source
