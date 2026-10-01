from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def main() -> int:
    args = _parse_args()
    validate_chartqa_release(
        result_root=Path(args.chartqa_result_root).resolve(),
        generation_commit=args.generation_source_commit,
        analysis_commit=args.chartqa_analysis_commit,
        expected_tasks=args.expected_tasks,
    )
    if args.smoke_result_root:
        validate_smoke_release(
            result_root=Path(args.smoke_result_root).resolve(),
            generation_commit=args.generation_source_commit,
            control_commit=args.smoke_control_commit,
        )
    print("EGOR downstream release prerequisites passed")
    return 0


def validate_chartqa_release(
    *,
    result_root: Path,
    generation_commit: str,
    analysis_commit: str,
    expected_tasks: int,
) -> None:
    _validate_commit(generation_commit, "generation source commit")
    _validate_commit(analysis_commit, "ChartQA analysis commit")
    _require_file(result_root / "ANALYSIS_AMENDMENT_COMPLETE")
    _require_commit(result_root / "SOURCE_COMMIT", generation_commit)
    _require_commit(
        result_root / "ANALYSIS_AMENDMENT_SOURCE_COMMIT", analysis_commit
    )

    gate_path = result_root / "paper_gate_report.json"
    manifest_path = result_root / "analysis_amendment_manifest.json"
    gate = _load_object(gate_path)
    manifest = _load_object(manifest_path)
    if (
        gate.get("passed") is not True
        or gate.get("source_commit") != generation_commit
        or int(gate.get("tasks", -1)) != expected_tasks
    ):
        raise ValueError("ChartQA paper gate is not a passing bound result")
    if (
        manifest.get("integrity_passed") is not True
        or manifest.get("paper_gate_passed") is not True
        or manifest.get("generation_source_commit") != generation_commit
        or manifest.get("analysis_commit") != analysis_commit
        or int(manifest.get("expected_tasks", -1)) != expected_tasks
    ):
        raise ValueError("ChartQA analysis amendment manifest is incompatible")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("ChartQA amendment manifest has no bound artifacts")
    bound: set[Path] = set()
    for index, raw in enumerate(artifacts):
        if not isinstance(raw, dict):
            raise ValueError(f"invalid ChartQA artifact entry {index}")
        path = Path(str(raw.get("path", ""))).resolve()
        expected_sha = str(raw.get("sha256", ""))
        if not _is_within(path, result_root):
            raise ValueError(f"ChartQA artifact escapes result root: {path}")
        _require_sha(path, expected_sha)
        bound.add(path)
    if gate_path.resolve() not in bound:
        raise ValueError("ChartQA gate report is not hash-bound by the amendment")


def validate_smoke_release(
    *, result_root: Path, generation_commit: str, control_commit: str
) -> None:
    _validate_commit(generation_commit, "generation source commit")
    _validate_commit(control_commit, "smoke control commit")
    _require_file(result_root / "SMOKE_COMPLETE")
    _require_commit(result_root / "SOURCE_COMMIT", generation_commit)
    _require_commit(result_root / "SMOKE_CONTROL_COMMIT", control_commit)

    audit = _load_object(result_root / "smoke_audit.json")
    tasks = audit.get("tasks", {})
    stages = audit.get("stages", {})
    baseline = stages.get("baseline", {}) if isinstance(stages, dict) else {}
    stage_manifest = baseline.get("manifest", {})
    if (
        audit.get("passed") is not True
        or audit.get("outcome_blind") is not True
        or audit.get("require_manifest") is not True
        or int(tasks.get("count", -1)) != 2
        or baseline.get("complete") is not True
        or int(baseline.get("successful_tasks", -1)) != 2
        or stage_manifest.get("compatible") is not True
        or stage_manifest.get("git_commit") != generation_commit
    ):
        raise ValueError("Qwen3.5 smoke audit is not a passing two-task result")

    for raw_path, raw_sha in (
        (baseline.get("path"), baseline.get("sha256")),
        (stage_manifest.get("path"), stage_manifest.get("sha256")),
    ):
        path = Path(str(raw_path or "")).resolve()
        if not _is_within(path, result_root):
            raise ValueError(f"smoke artifact escapes result root: {path}")
        _require_sha(path, str(raw_sha or ""))


def _require_commit(path: Path, expected: str) -> None:
    _require_file(path)
    if path.read_text(encoding="utf-8").strip() != expected:
        raise ValueError(f"commit mismatch: {path}")


def _require_sha(path: Path, expected: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"invalid SHA-256 for {path}")
    _require_file(path)
    if _sha256(path) != expected:
        raise ValueError(f"artifact SHA-256 mismatch: {path}")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing required file: {path}")


def _load_object(path: Path) -> dict[str, Any]:
    _require_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_commit(value: str, label: str) -> None:
    if COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact 40-character Git commit")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chartqa-result-root", required=True)
    parser.add_argument("--generation-source-commit", required=True)
    parser.add_argument("--chartqa-analysis-commit", required=True)
    parser.add_argument("--expected-tasks", type=int, default=240)
    parser.add_argument("--smoke-result-root")
    parser.add_argument("--smoke-control-commit")
    args = parser.parse_args()
    if bool(args.smoke_result_root) != bool(args.smoke_control_commit):
        parser.error(
            "--smoke-result-root and --smoke-control-commit must be supplied together"
        )
    return args


if __name__ == "__main__":
    raise SystemExit(main())
