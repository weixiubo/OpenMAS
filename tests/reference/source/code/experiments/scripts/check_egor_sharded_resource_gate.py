#!/usr/bin/env python3
"""Read-only two-snapshot GPU gate for EGOR sharded inference.

The gate never signals or modifies a process. It accepts only explicitly
preapproved compute PIDs and records the complete decision in a JSON artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


Snapshot = dict[str, Any]


def main() -> int:
    args = _parse_args()
    report = run_resource_gate(
        allowed_pids=parse_allowed_pids(args.allowed_pids),
        minimum_free_mib=_assignments(args.gpu_min_free),
        max_utilization_percent=args.max_utilization,
        readings=args.readings,
        interval_seconds=args.interval_seconds,
    )
    output = Path(args.output).resolve()
    _write_json_atomic(output, report)
    print(
        "EGOR sharded resource gate "
        f"{'passed' if report['passed'] else 'failed'}: {output}"
    )
    if args.require_pass and not report["passed"]:
        return 75
    return 0


def run_resource_gate(
    *,
    allowed_pids: set[int],
    minimum_free_mib: dict[int, int],
    max_utilization_percent: int,
    readings: int,
    interval_seconds: float,
    snapshot_loader: Callable[[], Snapshot] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if not allowed_pids:
        raise ValueError("at least one preapproved service PID is required")
    if not minimum_free_mib or any(
        index < 0 or required <= 0 for index, required in minimum_free_mib.items()
    ):
        raise ValueError("GPU free-memory requirements must be positive")
    if not 0 <= max_utilization_percent <= 100:
        raise ValueError("maximum utilization must lie in [0, 100]")
    if readings < 1:
        raise ValueError("at least one resource reading is required")
    if interval_seconds < 0:
        raise ValueError("resource-reading interval cannot be negative")

    loader = snapshot_loader or query_nvidia_snapshot
    audited_readings: list[dict[str, Any]] = []
    for reading in range(1, readings + 1):
        snapshot = loader()
        violations = evaluate_snapshot(
            snapshot,
            allowed_pids=allowed_pids,
            minimum_free_mib=minimum_free_mib,
            max_utilization_percent=max_utilization_percent,
        )
        audited_readings.append(
            {
                "reading": reading,
                "observed_at": snapshot.get("observed_at"),
                "gpus": snapshot.get("gpus", []),
                "unmapped_compute_processes": snapshot.get(
                    "unmapped_compute_processes", []
                ),
                "passed": not violations,
                "violations": violations,
            }
        )
        if violations:
            break
        if reading < readings:
            sleeper(interval_seconds)

    passed = len(audited_readings) == readings and all(
        bool(item["passed"]) for item in audited_readings
    )
    return {
        "schema_version": 1,
        "created_at": _now(),
        "outcome_blind": True,
        "read_only": True,
        "passed": passed,
        "requirements": {
            "allowed_compute_pids": sorted(allowed_pids),
            "minimum_free_mib": {
                str(index): required
                for index, required in sorted(minimum_free_mib.items())
            },
            "max_utilization_percent": max_utilization_percent,
            "readings": readings,
            "interval_seconds": interval_seconds,
            "unknown_compute_pids_allowed": False,
        },
        "completed_readings": len(audited_readings),
        "readings": audited_readings,
    }


def query_nvidia_snapshot() -> Snapshot:
    observed_at = _now()
    gpu_text = _run_nvidia_smi(
        "--query-gpu=index,uuid,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    process_text = _run_nvidia_smi(
        "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    )
    return parse_nvidia_snapshot(
        gpu_text=gpu_text,
        process_text=process_text,
        observed_at=observed_at,
    )


def parse_nvidia_snapshot(
    *, gpu_text: str, process_text: str, observed_at: str
) -> Snapshot:
    gpus: list[dict[str, Any]] = []
    by_uuid: dict[str, dict[str, Any]] = {}
    seen_indices: set[int] = set()
    for row in csv.reader(gpu_text.splitlines()):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != 4:
            raise ValueError(f"unexpected GPU query row: {row}")
        index = _integer(row[0], "GPU index")
        if index in seen_indices:
            raise ValueError(f"duplicate GPU index: {index}")
        uuid = row[1].strip()
        if not uuid or uuid in by_uuid:
            raise ValueError(f"invalid or duplicate GPU UUID: {uuid}")
        gpu = {
            "index": index,
            "uuid": uuid,
            "memory_free_mib": _integer(row[2], "free GPU memory"),
            "utilization_percent": _integer(row[3], "GPU utilization"),
            "compute_processes": [],
        }
        seen_indices.add(index)
        by_uuid[uuid] = gpu
        gpus.append(gpu)
    if not gpus:
        raise ValueError("nvidia-smi returned no GPUs")

    unmapped: list[dict[str, Any]] = []
    for row in csv.reader(process_text.splitlines()):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != 4 or not row[0].strip().startswith("GPU-"):
            raise ValueError(f"unexpected compute-process query row: {row}")
        process = {
            "gpu_uuid": row[0].strip(),
            "pid": _integer(row[1], "compute PID"),
            "used_memory_mib": _optional_integer(row[2]),
            "process_name": row[3].strip(),
        }
        gpu = by_uuid.get(process["gpu_uuid"])
        if gpu is None:
            unmapped.append(process)
        else:
            gpu["compute_processes"].append(process)
    for gpu in gpus:
        gpu["compute_processes"].sort(key=lambda item: int(item["pid"]))
    gpus.sort(key=lambda item: int(item["index"]))
    return {
        "observed_at": observed_at,
        "gpus": gpus,
        "unmapped_compute_processes": unmapped,
    }


def evaluate_snapshot(
    snapshot: Snapshot,
    *,
    allowed_pids: set[int],
    minimum_free_mib: dict[int, int],
    max_utilization_percent: int,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    raw_gpus = snapshot.get("gpus")
    if not isinstance(raw_gpus, list):
        return [{"code": "invalid_gpu_snapshot"}]
    by_index: dict[int, dict[str, Any]] = {}
    for position, gpu in enumerate(raw_gpus):
        if not isinstance(gpu, dict) or not _is_integer_like(gpu.get("index")):
            violations.append(
                {"code": "invalid_gpu_entry", "position": position}
            )
            continue
        index = int(gpu["index"])
        if index in by_index:
            violations.append({"code": "duplicate_gpu_index", "gpu": index})
            continue
        by_index[index] = gpu
    for index, minimum in sorted(minimum_free_mib.items()):
        gpu = by_index.get(index)
        if gpu is None:
            violations.append({"code": "missing_gpu", "gpu": index})
            continue
        free = _finite_number(gpu.get("memory_free_mib"))
        utilization = _finite_number(gpu.get("utilization_percent"))
        if free is None or free < minimum:
            violations.append(
                {
                    "code": "insufficient_free_memory",
                    "gpu": index,
                    "required_mib": minimum,
                    "observed_mib": gpu.get("memory_free_mib"),
                }
            )
        if utilization is None or utilization > max_utilization_percent:
            violations.append(
                {
                    "code": "excess_utilization",
                    "gpu": index,
                    "maximum_percent": max_utilization_percent,
                    "observed_percent": gpu.get("utilization_percent"),
                }
            )
        processes = gpu.get("compute_processes", [])
        if not isinstance(processes, list):
            violations.append({"code": "invalid_process_snapshot", "gpu": index})
            continue
        invalid_processes = [
            position
            for position, item in enumerate(processes)
            if not isinstance(item, dict) or not _is_integer_like(item.get("pid"))
        ]
        if invalid_processes:
            violations.append(
                {
                    "code": "invalid_compute_processes",
                    "gpu": index,
                    "positions": invalid_processes,
                }
            )
        unknown = sorted(
            {
                int(item["pid"])
                for item in processes
                if isinstance(item, dict)
                and _is_integer_like(item.get("pid"))
                and int(item["pid"]) not in allowed_pids
            }
        )
        if unknown:
            violations.append(
                {"code": "unknown_compute_pids", "gpu": index, "pids": unknown}
            )
    unmapped = snapshot.get("unmapped_compute_processes", [])
    if not isinstance(unmapped, list):
        violations.append({"code": "invalid_unmapped_process_snapshot"})
    elif unmapped:
        violations.append(
            {
                "code": "unmapped_compute_processes",
                "count": len(unmapped),
            }
        )
    return violations


def parse_allowed_pids(raw: str) -> set[int]:
    values = [item for item in re.split(r"[\s,]+", str(raw).strip()) if item]
    if not values or any(not item.isdigit() or int(item) <= 0 for item in values):
        raise ValueError("allowed PIDs must be positive integers")
    return {int(item) for item in values}


def _assignments(values: list[str]) -> dict[int, int]:
    parsed: dict[int, int] = {}
    for raw in values:
        left, separator, right = raw.partition("=")
        if not separator:
            raise ValueError(f"expected GPU=MIB, received: {raw}")
        index = _integer(left, "GPU requirement index")
        required = _integer(right, "GPU free-memory requirement")
        if index in parsed:
            raise ValueError(f"duplicate GPU requirement: {index}")
        parsed[index] = required
    return parsed


def _run_nvidia_smi(*args: str) -> str:
    completed = subprocess.run(
        ["nvidia-smi", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _integer(value: Any, label: str) -> int:
    text = str(value).strip()
    if re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError(f"{label} is not a non-negative integer: {value}")
    return int(text)


def _optional_integer(value: Any) -> int | None:
    text = str(value).strip()
    return int(text) if re.fullmatch(r"[0-9]+", text) else None


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _is_integer_like(value: Any) -> bool:
    return re.fullmatch(r"[0-9]+", str(value).strip()) is not None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowed-pids", required=True)
    parser.add_argument("--gpu-min-free", action="append", required=True)
    parser.add_argument("--max-utilization", type=int, default=5)
    parser.add_argument("--readings", type=int, default=2)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
