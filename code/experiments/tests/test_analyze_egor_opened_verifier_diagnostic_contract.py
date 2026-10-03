from openmas_test_history import historical_root

ROOT = historical_root()
SCRIPT = ROOT / "code/experiments/scripts/analyze_egor_opened_verifier_diagnostic_v1.py"


def test_opened_verifier_analysis_freezes_primary_and_controls() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"primary_policy": "independently_verified_overwrite"' in source
    assert '"core_capped_delta"' in source
    assert '"strict_pair_delta"' in source
    assert '"development_signal"' in source
    assert '"opened_ocrbench_signal"' in source
    assert '"harm_rejection"' in source
    assert '"verifier_completion_definition"' in source
    assert "valid_evidence_cards(verifier_parsed)" in source
    assert "canonical_answer(core_answer)" in source
    assert "reference_answers" not in source
