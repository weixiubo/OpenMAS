import json
from pathlib import Path

import pytest
from openagora.resources import resolve_paths
from openagora_experiments.egor.catalog import load_scheduler_catalog
from openagora_experiments.egor.job import execute_stage, resolve_stage, stage_problems


class Provider:
    model_name = "offline-contract-provider"

    def __init__(self, interrupt_at=None):
        self.calls = []
        self.interrupt_at = interrupt_at

    def complete_multimodal(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt("interrupted test run")
        return json.dumps({"answer": "5", "confidence": 0.9, "demands": [], "evidence": [], "open_gap_types": []})

    def pop_audit_events(self):
        return [{"prompt_tokens": 10, "completion_tokens": 4, "attempts": 1, "latency_ms": 0}]


def stage_config(tmp_path):
    image = tmp_path / "small.png"
    image.write_bytes(b"offline provider does not decode pixels")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": str(i),
                    "source_id": "fixture",
                    "question": "2+3?",
                    "image_path": "small.png",
                    "answer": "5",
                }
            )
            + "\n"
            for i in range(2)
        )
    )
    (tmp_path / "model").mkdir()
    return resolve_stage(
        {
            "stage": "baseline",
            "tasks": "tasks.jsonl",
            "model_path": "model",
            "model_name": "fixture",
            "model_revision": "test-fixture-v1",
            "run_id": "test",
            "seed": 41,
        },
        tmp_path / "config.yaml",
        resolve_paths({"runs": tmp_path / "runs"}, env={}, home=tmp_path),
    )


def test_interrupted_stage_resumes_original_seed_schedule(tmp_path):
    cfg = stage_config(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        execute_stage(cfg, provider=Provider(interrupt_at=2))
    output = Path(cfg.output)
    assert len(output.read_text().splitlines()) == 1
    assert json.loads(output.with_suffix(".manifest.json").read_text())["status"] == "interrupted"
    cfg.resume = True
    provider = Provider()
    report = execute_stage(cfg, provider=provider)
    assert report["attempted"] == report["successful"] == 1
    assert provider.calls[0]["seed"] == 42
    assert provider.calls[0]["artifacts"][0].path == str(tmp_path / "small.png")
    assert len(output.read_text().splitlines()) == 2
    assert execute_stage(cfg, provider=Provider())["attempted"] == 0


def test_resume_rejects_modified_inputs_and_parameters(tmp_path):
    cfg = stage_config(tmp_path)
    execute_stage(cfg, provider=Provider())
    with pytest.raises(FileExistsError, match="Output exists"):
        execute_stage(cfg, provider=Provider())
    cfg.resume = True
    cfg.seed += 1
    with pytest.raises(ValueError, match="mismatch"):
        execute_stage(cfg, provider=Provider())
    cfg.seed -= 1
    Path(cfg.tasks).write_text(Path(cfg.tasks).read_text() + "\n")
    with pytest.raises(ValueError, match="mismatch"):
        execute_stage(cfg, provider=Provider())


def test_stage_preflight_missing_image_and_checksum_create_no_output(tmp_path):
    cfg = stage_config(tmp_path)
    cfg.input_sha256[cfg.tasks] = "0" * 64
    assert any("checksum" in message for message in stage_problems(cfg, check_packages=False))
    with pytest.raises(ValueError, match="checksum"):
        execute_stage(cfg, provider=Provider())
    cfg.input_sha256 = {}
    (tmp_path / "small.png").unlink()
    with pytest.raises(ValueError, match="Task image missing"):
        execute_stage(cfg, provider=Provider())
    assert not Path(cfg.output).parent.exists()


def test_scheduler_catalog_validates_references(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {"actions": {"read": {"evidence_type": "exact_span"}}, "templates": {"one": {"action_ids": ["read"]}}}
        )
    )
    assert load_scheduler_catalog(path) == [
        {"template_id": "one", "action_ids": ["read"], "evidence_types": ["exact_span"]}
    ]
    path.write_text(json.dumps({"templates": {"broken": {"action_ids": ["missing"]}}}))
    with pytest.raises(ValueError, match="unknown actions"):
        load_scheduler_catalog(path)
