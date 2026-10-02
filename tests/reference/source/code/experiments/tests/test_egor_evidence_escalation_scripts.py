from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def test_stage_validator_normalizes_benchmark_as_source_id(tmp_path: Path) -> None:
    module = _load_script("validate_egor_stage_output.py")
    tasks_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "baseline.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {"benchmark": "ocr-transfer", "task_id": "t0"},
            {"benchmark": "ocr-transfer", "task_id": "t1"},
        ],
    )
    _write_jsonl(
        output_path,
        [
            {"source_id": "ocr-transfer", "task_id": "t0", "status": "error"},
            {"source_id": "ocr-transfer", "task_id": "t0", "status": "ok"},
            {"source_id": "ocr-transfer", "task_id": "t1", "status": "ok"},
        ],
    )

    summary = module.validate_stage_output(
        stage_output=output_path,
        tasks_path=tasks_path,
    )

    assert summary == {"expected": 2, "history": 3, "latest": 2}


def test_stage_validator_rejects_missing_source_identity(tmp_path: Path) -> None:
    module = _load_script("validate_egor_stage_output.py")
    tasks_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "baseline.jsonl"
    _write_jsonl(tasks_path, [{"task_id": "t0"}])
    _write_jsonl(output_path, [])

    with pytest.raises(ValueError, match="benchmark/source_id"):
        module.validate_stage_output(
            stage_output=output_path,
            tasks_path=tasks_path,
        )


def test_materializer_projects_nested_decisions_without_outcomes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script("materialize_egor_repairability_gate.py")
    input_path = tmp_path / "cross_validated_rows.csv"
    output_path = tmp_path / "repairability.jsonl"
    fields = [*module.CV_ALLOWED_INPUT_FIELDS, "quality_delta", "strict_harm"]
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "fixture",
                    "task_id": "t0",
                    "selected_id": "baseline",
                    "reason": "no_residual_gap",
                    "selection_protocol": "nested_loso",
                    "selector_target": "paired_advantage",
                    "fold_config_json": '{"threshold": 0.05}',
                    "quality_delta": "1.0",
                    "strict_harm": "0",
                },
                {
                    "source_id": "fixture",
                    "task_id": "t1",
                    "selected_id": "visual",
                    "reason": "positive_net_advantage",
                    "selection_protocol": "nested_loso",
                    "selector_target": "paired_advantage",
                    "fold_config_json": '{"threshold": 0.05}',
                    "quality_delta": "-1.0",
                    "strict_harm": "1",
                },
            ]
        )
    monkeypatch.setenv("EGOR_SOURCE_COMMIT", "fixture-commit")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module.__file__,
            "--cross-validated-rows",
            str(input_path),
            "--output",
            str(output_path),
            "--expected-tasks",
            "2",
        ],
    )

    assert module.main() == 0

    rows = _read_jsonl(output_path)
    assert [row["repairability_triggered"] for row in rows] == [False, True]
    assert all("selected_id" not in row for row in rows)
    assert all("quality_delta" not in row for row in rows)
    assert all("strict_harm" not in row for row in rows)
    manifest = json.loads(
        output_path.with_suffix(".jsonl.manifest.json").read_text()
    )
    assert manifest["outcome_fields_read"] is False
    assert "quality_delta" not in manifest["allowed_input_fields_read"]
    assert manifest["trigger_counts"] == {"abstain": 1, "trigger": 1}


def test_materializer_projects_frozen_selector_plans_and_task_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script("materialize_egor_repairability_gate.py")
    input_path = tmp_path / "selector_plans.jsonl"
    output_path = tmp_path / "repairability.jsonl"
    task_ids_path = tmp_path / "triggered.txt"
    _write_jsonl(
        input_path,
        [
            {
                "source_id": "fixture",
                "task_id": "t0",
                "selected_template_id": "baseline",
                "reason": "no_residual_gap",
                "predicted_advantages": {"visual": 999.0},
            },
            {
                "source_id": "fixture",
                "task_id": "t1",
                "selected_template_id": "verifier",
                "reason": "positive_net_advantage",
                "predicted_advantages": {"visual": -999.0},
            },
        ],
    )
    monkeypatch.setenv("EGOR_SOURCE_COMMIT", "fixture-commit")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module.__file__,
            "--selector-plans",
            str(input_path),
            "--output",
            str(output_path),
            "--task-ids-output",
            str(task_ids_path),
            "--expected-tasks",
            "2",
        ],
    )

    assert module.main() == 0

    assert [
        row["repairability_triggered"] for row in _read_jsonl(output_path)
    ] == [False, True]
    assert task_ids_path.read_text() == "t1\n"
    manifest = json.loads(
        output_path.with_suffix(".jsonl.manifest.json").read_text()
    )
    assert manifest["input_kind"] == "frozen_selector_plans"
    assert "frozen development-trained selector" in manifest["policy"]
    assert manifest["outcome_fields_read"] is False
    assert "predicted_advantages" not in manifest["allowed_input_fields_read"]


def test_sparse_matrix_config_has_staged_candidates_and_controls(
    tmp_path: Path,
) -> None:
    module = _load_script("build_egor_evidence_escalation_matrix_config.py")
    args = argparse.Namespace(
        baseline=str(tmp_path / "baseline.jsonl"),
        exact_specialist=str(tmp_path / "exact.jsonl"),
        visual_specialist=str(tmp_path / "visual.jsonl"),
        numerical_specialist=str(tmp_path / "numerical.jsonl"),
        verifier_specialist=str(tmp_path / "verifier.jsonl"),
        core_synthesis=str(tmp_path / "core.jsonl"),
        full_synthesis=str(tmp_path / "full.jsonl"),
        baseline_resynthesis=str(tmp_path / "resynthesis.jsonl"),
        visual_synthesis=str(tmp_path / "visual_synthesis.jsonl"),
    )

    config = module.build_config(args)

    by_id = {row["candidate_id"]: row for row in config["candidates"]}
    assert list(by_id) == [
        "core_pair",
        "full_team",
        "baseline_resynthesis",
        "visual_control",
    ]
    assert len(by_id["core_pair"]["specialists"]) == 2
    assert len(by_id["full_team"]["specialists"]) == 4
    assert by_id["baseline_resynthesis"]["specialists"] == []
    assert config["policies"] == []


def test_planner_freezes_only_outcome_blind_stage_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script("plan_egor_evidence_escalation.py")
    plans_path = tmp_path / "repairability.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    exact_path = tmp_path / "exact.jsonl"
    visual_path = tmp_path / "visual.jsonl"
    core_path = tmp_path / "core.jsonl"
    output_dir = tmp_path / "plan"

    _write_jsonl(
        plans_path,
        [
            _repairability("t0", "baseline"),
            _repairability("t1", "exact"),
            _repairability("t2", "visual"),
        ],
    )
    _write_jsonl(
        baseline_path,
        [_stage(task, _baseline()) for task in ("t0", "t1", "t2")],
    )
    _write_jsonl(
        exact_path,
        [
            _stage("t1", _specialist("exact", "yes")),
            _stage("t2", _specialist("exact", "yes")),
        ],
    )
    _write_jsonl(
        visual_path,
        [
            _stage("t1", _specialist("visual", "yes")),
            _stage("t2", _specialist("visual", "yes")),
        ],
    )
    _write_jsonl(
        core_path,
        [
            _stage("t1", _synthesis("yes", used=["exact::s1"])),
            _stage(
                "t2",
                _synthesis(
                    "yes",
                    used=["visual::s1"],
                    remaining=["independent_verification"],
                ),
            ),
        ],
    )
    monkeypatch.setenv("EGOR_SOURCE_COMMIT", "fixture-commit")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module.__file__,
            "--repairability-plans",
            str(plans_path),
            "--baseline",
            str(baseline_path),
            "--exact-specialist",
            str(exact_path),
            "--visual-specialist",
            str(visual_path),
            "--core-synthesis",
            str(core_path),
            "--output-dir",
            str(output_dir),
            "--expected-tasks",
            "3",
        ],
    )

    assert module.main() == 0

    frozen = {
        row["task_id"]: row for row in _read_jsonl(output_dir / "plans.jsonl")
    }
    assert frozen["t0"]["tier"] == "baseline"
    assert frozen["t1"]["tier"] == "core_pair"
    assert frozen["t2"]["tier"] == "full_team"
    assert frozen["t2"]["reason"] == "core_gap_remains"
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["source_commit"] == "fixture-commit"
    assert manifest["outcome_fields_read"] is False
    assert manifest["tier_counts"] == {
        "baseline": 1,
        "core_pair": 1,
        "full_team": 1,
    }
    assert "task scores" in manifest["forbidden_inputs"]
    assert "full-team synthesis records" in manifest["forbidden_inputs"]
    assert (output_dir / "task_ids" / "core_pair.txt").read_text() == "t1\nt2\n"
    assert (output_dir / "task_ids" / "full_team.txt").read_text() == "t2\n"
    core_control = {
        row["task_id"]: row
        for row in _read_jsonl(output_dir / "core_pair_control_plans.jsonl")
    }
    assert core_control["t0"]["tier"] == "baseline"
    assert core_control["t1"]["tier"] == "core_pair"
    assert core_control["t2"]["tier"] == "core_pair"
    assert all("legacy_gate_witness_id" not in row for row in frozen.values())


def test_analyzer_charges_sequential_core_once_and_enforces_certificate() -> None:
    module = _load_script("analyze_egor_evidence_escalation.py")
    plans = {
        ("fixture", "t0"): {"tier": "baseline", "reason": "abstain"},
        ("fixture", "t1"): {"tier": "core_pair", "reason": "stop"},
        ("fixture", "t2"): {"tier": "full_team", "reason": "escalate"},
        ("fixture", "t3"): {"tier": "full_team", "reason": "escalate"},
    }
    matrix: dict[tuple[str, str, str], dict[str, str]] = {}
    for task_id in ("t0", "t1", "t2", "t3"):
        matrix[("fixture", task_id, "core")] = _matrix_row(
            task_id,
            candidate_quality=0.7,
            calls=4,
            supported=True,
        )
        if task_id in {"t2", "t3"}:
            matrix[("fixture", task_id, "full")] = _matrix_row(
                task_id,
                candidate_quality=0.9 if task_id != "t3" else 1.0,
                calls=6,
                supported=task_id != "t3",
            )
    core_records = {
        ("fixture", task_id): _stage(
            task_id,
            _synthesis("yes", used=["exact::s1"]),
            usage={"attempts": 1, "total_tokens": 5, "latency_seconds": 0.5},
        )
        for task_id in ("t2", "t3")
    }

    rows, audit = module.analyze_escalation_policy(
        plans=plans,
        matrix=matrix,
        core_synthesis_records=core_records,
        policy_id="fixture_policy",
        core_candidate_id="core",
        full_candidate_id="full",
    )
    by_task = {row.task_id: row for row in rows}
    audit_by_task = {row["task_id"]: row for row in audit}

    assert by_task["t0"].candidate_id == "baseline"
    assert by_task["t0"].policy_quality == 0.2
    assert by_task["t0"].calls == 1.0
    assert by_task["t1"].policy_quality == 0.7
    assert by_task["t1"].calls == 4.0
    assert by_task["t2"].policy_quality == 0.9
    assert by_task["t2"].calls == 7.0
    assert by_task["t2"].total_tokens == 65.0
    assert by_task["t2"].latency_seconds == 6.5
    assert audit_by_task["t2"]["extra_core_synthesis_calls"] == 1.0
    assert by_task["t3"].policy_quality == 0.2
    assert by_task["t3"].provenance_rejected is True
    assert by_task["t3"].calls == 7.0

    raw_rows, _ = module.analyze_escalation_policy(
        plans=plans,
        matrix=matrix,
        core_synthesis_records=core_records,
        policy_id="raw_control",
        core_candidate_id="core",
        full_candidate_id="full",
        require_evidence_for_change=False,
    )
    raw_by_task = {row.task_id: row for row in raw_rows}
    assert raw_by_task["t3"].policy_quality == 1.0
    assert raw_by_task["t3"].provenance_rejected is False


def _load_script(filename: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repairability(task_id: str, template: str) -> dict[str, Any]:
    return {
        "source_id": "fixture",
        "task_id": task_id,
        "repairability_triggered": template != "baseline",
    }


def _stage(
    task_id: str,
    parsed: dict[str, Any],
    *,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source_id": "fixture",
        "task_id": task_id,
        "status": "ok",
        "parsed": parsed,
        "usage": usage or {},
    }


def _baseline() -> dict[str, Any]:
    return {
        "answer": "no",
        "evidence": [_evidence("baseline", "baseline::b1")],
    }


def _specialist(producer: str, answer: str) -> dict[str, Any]:
    return {
        "candidate_answer": answer,
        "evidence": [_evidence(producer, f"{producer}::s1")],
    }


def _evidence(producer: str, evidence_id: str) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "producer_id": producer,
        "type": "exact_span",
        "claim": "fixture claim",
        "support": "fixture support",
        "provenance": "fixture region",
        "confidence": 1.0,
    }


def _synthesis(
    answer: str,
    *,
    used: list[str],
    remaining: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "answer": answer,
        "changed_baseline": answer != "no",
        "used_evidence_ids": used,
        "remaining_gap_types": remaining or [],
    }


def _matrix_row(
    task_id: str,
    *,
    candidate_quality: float,
    calls: float,
    supported: bool,
) -> dict[str, str]:
    return {
        "source_id": "fixture",
        "task_id": task_id,
        "baseline_answer": "no",
        "baseline_quality": "0.2",
        "baseline_calls": "1",
        "baseline_total_tokens": "10",
        "baseline_latency_seconds": "1",
        "candidate_quality": str(candidate_quality),
        "actual_changed": "1",
        "evidence_supported_change": str(int(supported)),
        "calls": str(calls),
        "total_tokens": str(calls * 10),
        "latency_seconds": str(calls),
        "failed_stages": "",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]
