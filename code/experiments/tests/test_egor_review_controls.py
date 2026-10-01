from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_nested_report_keeps_posthoc_boundary_and_isolates_conditions() -> None:
    module = _load("analyze_egor_v8_review_controls.py")
    base = {
        "source": "chartqa",
        "source_id": "s",
        "repairability_triggered": True,
        "selector_template": "visual",
        "specialist_agreement": True,
        "structural_card_corroboration": True,
        "blind_verifier_agreement": True,
        "cited_card_corroboration": True,
        "corroborating_specialists": ["exact"],
        "used_evidence_ids": ["exact::e1"],
        "core_answer": "B",
        "verifier_answer": "B",
    }
    rows = [
        base
        | {
            "task_id": "win",
            "baseline_quality": 0.0,
            "core_changed": True,
            "policy_quality": {name: 1.0 for name in module.POLICIES},
        },
        base
        | {
            "task_id": "blocked-loss",
            "baseline_quality": 1.0,
            "core_changed": True,
            "blind_verifier_agreement": False,
            "cited_card_corroboration": False,
            "policy_quality": {
                "core": 0.0,
                "strict_pair": 0.0,
                "specialist_only": 0.0,
                "specialist_agreement_no_card": 0.0,
                "blind_verifier_no_card": 1.0,
                "egor_v8": 1.0,
                "egor_v8_cited_card": 1.0,
            },
        },
    ]
    report = module.build_report(
        primary={"source_commit": "a" * 40, "passed": True},
        rows=rows,
        bootstrap_samples=200,
        seed=7,
    )

    assert report["claim_boundary"] == (
        "posthoc_review_diagnostic_not_new_confirmation"
    )
    assert report["policies"]["specialist_only"]["losses"] == 1
    assert report["policies"]["egor_v8"]["losses"] == 0
    assert report["decision_funnel"]["egor_v8_accept"] == 1
    assert report["gate_action_audit"]["other_action_argmax"] == 2


def test_zero_event_interval_does_not_turn_no_observation_into_safety() -> None:
    module = _load("analyze_egor_v8_review_controls.py")
    assert module._zero_event_cp_upper(0, alpha=0.05) is None
    assert math.isclose(
        module._zero_event_cp_upper(8, alpha=0.05),
        0.3694166475528192,
    )


def test_direct_control_summary_reports_both_required_contrasts() -> None:
    module = _load("analyze_egor_qwen35_direct_control.py")
    rows = [
        {
            "baseline_quality": 0.0,
            "direct_quality": 1.0,
            "egor_quality": 1.0,
            "direct_calls": 1.0,
            "direct_tokens": 10.0,
            "direct_latency_seconds": 2.0,
        },
        {
            "baseline_quality": 1.0,
            "direct_quality": 0.0,
            "egor_quality": 1.0,
            "direct_calls": 1.0,
            "direct_tokens": 12.0,
            "direct_latency_seconds": 4.0,
        },
    ]
    summary = module.summarize(rows, bootstrap_samples=200, seed=11)

    assert summary["qwen35_direct_wins"] == 1
    assert summary["qwen35_direct_losses"] == 1
    assert summary["egor_minus_qwen35_direct"] == 0.5
    assert summary["mean_direct_calls"] == 1.0


def _load(filename: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
