from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def test_two_safe_readings_are_required_and_recorded() -> None:
    module = _load_module()
    snapshots = iter([_snapshot(), _snapshot(free0=15500, free1=18100)])
    sleeps: list[float] = []

    report = module.run_resource_gate(
        allowed_pids={10, 20},
        minimum_free_mib={0: 15000, 1: 18000},
        max_utilization_percent=5,
        readings=2,
        interval_seconds=30,
        snapshot_loader=lambda: next(snapshots),
        sleeper=sleeps.append,
    )

    assert report["passed"] is True
    assert report["read_only"] is True
    assert report["outcome_blind"] is True
    assert report["completed_readings"] == 2
    assert sleeps == [30]
    assert all(item["passed"] for item in report["readings"])


def test_unknown_compute_pid_fails_closed_without_waiting() -> None:
    module = _load_module()
    snapshot = _snapshot()
    snapshot["gpus"][0]["compute_processes"].append(
        {
            "gpu_uuid": "GPU-0",
            "pid": 999,
            "used_memory_mib": 100,
            "process_name": "other-training",
        }
    )
    sleeps: list[float] = []

    report = module.run_resource_gate(
        allowed_pids={10, 20},
        minimum_free_mib={0: 15000, 1: 18000},
        max_utilization_percent=5,
        readings=2,
        interval_seconds=30,
        snapshot_loader=lambda: snapshot,
        sleeper=sleeps.append,
    )

    assert report["passed"] is False
    assert report["completed_readings"] == 1
    assert sleeps == []
    assert report["readings"][0]["violations"] == [
        {"code": "unknown_compute_pids", "gpu": 0, "pids": [999]}
    ]


@pytest.mark.parametrize(
    ("snapshot_kwargs", "code"),
    [
        ({"free0": 14999}, "insufficient_free_memory"),
        ({"util1": 6}, "excess_utilization"),
    ],
)
def test_memory_and_utilization_thresholds_fail_closed(
    snapshot_kwargs: dict[str, int], code: str
) -> None:
    module = _load_module()

    violations = module.evaluate_snapshot(
        _snapshot(**snapshot_kwargs),
        allowed_pids={10, 20},
        minimum_free_mib={0: 15000, 1: 18000},
        max_utilization_percent=5,
    )

    assert code in {item["code"] for item in violations}


def test_nvidia_csv_parser_maps_processes_by_uuid() -> None:
    module = _load_module()

    snapshot = module.parse_nvidia_snapshot(
        gpu_text="0, GPU-A, 16000, 2\n1, GPU-B, 19000, 0\n",
        process_text=(
            "GPU-A, 12, 900, /service/a\n"
            "GPU-B, 34, 5000, /service/b\n"
        ),
        observed_at="2026-07-15T00:00:00+00:00",
    )

    assert [gpu["index"] for gpu in snapshot["gpus"]] == [0, 1]
    assert snapshot["gpus"][0]["compute_processes"][0]["pid"] == 12
    assert snapshot["gpus"][1]["compute_processes"][0]["pid"] == 34
    assert snapshot["unmapped_compute_processes"] == []


@pytest.mark.parametrize(
    ("gpu_text", "process_text", "message"),
    [
        (
            "0, GPU-A, 16000, 2\n0, GPU-B, 19000, 0\n",
            "",
            "duplicate GPU index",
        ),
        (
            "0, GPU-A, 16000, 2\n",
            "GPU-A, 12, 900\n",
            "unexpected compute-process query row",
        ),
        (
            "0, GPU-A, 16000, 2\n",
            "malformed, 12, 900, /service/a\n",
            "unexpected compute-process query row",
        ),
    ],
)
def test_nvidia_csv_parser_rejects_ambiguous_rows(
    gpu_text: str, process_text: str, message: str
) -> None:
    module = _load_module()

    with pytest.raises(ValueError, match=message):
        module.parse_nvidia_snapshot(
            gpu_text=gpu_text,
            process_text=process_text,
            observed_at="2026-07-15T00:00:00+00:00",
        )


def test_injected_malformed_snapshot_fails_closed() -> None:
    module = _load_module()
    snapshot = _snapshot()
    snapshot["gpus"].append({"index": 0, "compute_processes": []})
    snapshot["gpus"][1]["compute_processes"].append({"pid": "unknown"})
    snapshot["unmapped_compute_processes"] = {"unexpected": "object"}

    violations = module.evaluate_snapshot(
        snapshot,
        allowed_pids={10, 20},
        minimum_free_mib={0: 15000, 1: 18000},
        max_utilization_percent=5,
    )

    assert {item["code"] for item in violations} >= {
        "duplicate_gpu_index",
        "invalid_compute_processes",
        "invalid_unmapped_process_snapshot",
    }


@pytest.mark.parametrize("raw", ["", "0", "-1", "10,bad"])
def test_allowed_pid_parser_rejects_ambiguous_values(raw: str) -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="positive integers"):
        module.parse_allowed_pids(raw)


def _snapshot(
    *, free0: int = 16000, free1: int = 19000, util0: int = 2, util1: int = 1
) -> dict[str, object]:
    return {
        "observed_at": "2026-07-15T00:00:00+00:00",
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-0",
                "memory_free_mib": free0,
                "utilization_percent": util0,
                "compute_processes": [
                    {
                        "gpu_uuid": "GPU-0",
                        "pid": 10,
                        "used_memory_mib": 900,
                        "process_name": "service-0",
                    }
                ],
            },
            {
                "index": 1,
                "uuid": "GPU-1",
                "memory_free_mib": free1,
                "utilization_percent": util1,
                "compute_processes": [
                    {
                        "gpu_uuid": "GPU-1",
                        "pid": 20,
                        "used_memory_mib": 5000,
                        "process_name": "service-1",
                    }
                ],
            },
        ],
        "unmapped_compute_processes": [],
    }


def _load_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "check_egor_sharded_resource_gate.py"
    )
    spec = importlib.util.spec_from_file_location("egor_sharded_resource_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
