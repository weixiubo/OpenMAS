"""Verify frozen EGOR-v10 evidence and recompute paired outcome summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean


def verify(root: Path) -> dict:
    for filename in ("task_rows.csv", "confirmation_gate.json"):
        digest_file = root / (filename.rsplit(".", 1)[0] + ".sha256")
        expected = digest_file.read_text(encoding="utf-8-sig").split()[0]
        actual = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen evidence hash mismatch: {filename}")
    with (root / "task_rows.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    gate = json.loads((root / "confirmation_gate.json").read_text(encoding="utf-8-sig"))
    if len(rows) != gate["task_denominator"]:
        raise ValueError("Task denominator mismatch")
    expected = gate["pooled"]["v10"]
    deltas = [float(row["v10_quality"]) - float(row["baseline_quality"]) for row in rows]
    counts = {
        "wins": sum(x > 0 for x in deltas),
        "losses": sum(x < 0 for x in deltas),
        "ties": sum(x == 0 for x in deltas),
    }
    if any(counts[key] != expected[key] for key in counts):
        raise ValueError("Paired outcome counts differ from the frozen gate")
    if not math.isclose(fmean(deltas), expected["quality_delta"], abs_tol=1e-12):
        raise ValueError("Quality delta differs from the frozen gate")
    accepted = sum(row["accept_overwrite"].lower() == "true" for row in rows)
    if accepted != gate["pooled"]["accepted_overwrites"]:
        raise ValueError("Accepted overwrite count mismatch")
    if gate["passed"] != all(criterion["passed"] for criterion in gate["criteria"]):
        raise ValueError("Frozen gate status is inconsistent with its criteria")
    return {
        "tasks": len(rows),
        "sources": dict(Counter(row["source"] for row in rows)),
        **counts,
        "quality_delta": fmean(deltas),
        "accepted_overwrites": accepted,
        "scientific_gate_passed": gate["passed"],
        "evidence_verified": True,
        "bootstrap_interval": [expected["ci95_low"], expected["ci95_high"]],
        "bootstrap_interval_status": "hash-bound recorded interval; not recomputed by this arithmetic verifier",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path(__file__).resolve().parent / "egor-v10")
    args = parser.parse_args()
    print(json.dumps(verify(args.evidence), indent=2))
