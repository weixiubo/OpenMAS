#!/usr/bin/env python3
"""Replay sealed v5 outputs through the frozen v7 transport contract."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import analyze_egor_v6_opened_gate as v6
from openagora_experiments.egor.protocol import parse_verifier_output_v7


def main() -> None:
    v6.parse_verifier_output_v6 = parse_verifier_output_v7
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        v6.main()
    report = json.loads(buffer.getvalue())
    report.update(
        {
            "method": "egor_v7_verified_overwrite",
            "parent_v6_gate_sha256": (
                "755163da9bdb3d0295081f64e2ec04131672422a28e550cc2859e503fb968f61"
            ),
            "v7_transport_change": (
                "remove exactly one trailing backtick only when the remaining "
                "entire response is a single exact JSON object"
            ),
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
