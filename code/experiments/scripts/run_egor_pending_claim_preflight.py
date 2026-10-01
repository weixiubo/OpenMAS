from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = (
    REPO_ROOT
    / "code"
    / "experiments"
    / "configs"
    / "egor_pending_claim_preflight_v1.json"
)
DEFAULT_CLAIM_REGISTRY = REPO_ROOT / "paper_aaai27" / "evidence" / "claim_registry.json"
TEST_PATH_RE = re.compile(
    r"(?:code/experiments/tests|paper_aaai27/tests)/test_[A-Za-z0-9_]+\.py"
)


class _CollectionCapture:
    def __init__(self) -> None:
        self.collected = 0

    def pytest_collection_finish(self, session: Any) -> None:
        self.collected = len(session.items)


def load_preflight_spec(*, spec_path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _load_object(spec_path)
    if payload.get("schema_version") != 1:
        raise ValueError("pending-claim preflight spec must use schema version 1")

    modules = payload.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError("pending-claim preflight spec has no test modules")
    if any(not isinstance(value, str) or not value for value in modules):
        raise ValueError("pending-claim preflight modules must be nonempty strings")
    if len(set(modules)) != len(modules):
        raise ValueError("pending-claim preflight modules must be unique")

    expected_count = payload.get("expected_module_count")
    if not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError("expected_module_count must be a positive integer")
    if len(modules) != expected_count:
        raise ValueError(
            "pending-claim module count mismatch: "
            f"expected {expected_count}, found {len(modules)}"
        )

    minimum_tests = payload.get("minimum_test_count")
    if not isinstance(minimum_tests, int) or minimum_tests <= 0:
        raise ValueError("minimum_test_count must be a positive integer")

    allowed_roots = (
        (repo_root / "code" / "experiments" / "tests").resolve(),
        (repo_root / "paper_aaai27" / "tests").resolve(),
    )
    for value in modules:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"preflight module must be repository-relative: {value}")
        path = (repo_root / relative).resolve()
        if not any(_is_within(path, root) for root in allowed_roots):
            raise ValueError(f"preflight module escapes the allowed test roots: {value}")
        if path.suffix != ".py" or not path.name.startswith("test_"):
            raise ValueError(f"preflight module is not a Python test module: {value}")
        if not path.is_file():
            raise ValueError(f"missing preflight test module: {value}")

    covered_claim_ids = payload.get("covered_claim_ids")
    if (
        not isinstance(covered_claim_ids, list)
        or not covered_claim_ids
        or any(not isinstance(value, str) or not value for value in covered_claim_ids)
        or len(set(covered_claim_ids)) != len(covered_claim_ids)
    ):
        raise ValueError("covered_claim_ids must be a nonempty unique string list")
    return payload


def validate_pending_claim_coverage(
    *, spec: dict[str, Any], claim_registry_path: Path
) -> set[str]:
    registry = _load_object(claim_registry_path)
    raw_claims = registry.get("claims")
    if not isinstance(raw_claims, list):
        raise ValueError("claim registry has no claims list")
    claims = {
        str(claim.get("claim_id")): claim
        for claim in raw_claims
        if isinstance(claim, dict) and claim.get("claim_id")
    }

    covered_ids = set(spec["covered_claim_ids"])
    unknown = covered_ids - set(claims)
    if unknown:
        raise ValueError(f"preflight references unknown claim IDs: {sorted(unknown)}")
    for claim_id in sorted(covered_ids):
        status = claims[claim_id].get("status")
        if status not in {"pending_experiment", "pending_human"}:
            raise ValueError(
                f"preflight claim {claim_id} is no longer pending: {status!r}"
            )

    explicit_tests: set[str] = set()
    for claim_id in sorted(covered_ids):
        claim = claims[claim_id]
        sources = [str(value) for value in claim.get("artifacts", [])]
        sources.append(str(claim.get("analysis_command", "")))
        for source in sources:
            explicit_tests.update(TEST_PATH_RE.findall(source))

    configured = set(spec["modules"])
    missing = explicit_tests - configured
    if missing:
        raise ValueError(
            "preflight omits tests named directly by pending claims: "
            f"{sorted(missing)}"
        )
    return explicit_tests


def run_preflight(
    *,
    repo_root: Path,
    spec_path: Path,
    claim_registry_path: Path,
    collect_only: bool,
) -> tuple[int, dict[str, Any]]:
    spec = load_preflight_spec(spec_path=spec_path, repo_root=repo_root)
    explicit_tests = validate_pending_claim_coverage(
        spec=spec, claim_registry_path=claim_registry_path
    )

    for path in (
        repo_root / "code" / "core" / "src",
        repo_root / "code" / "experiments",
    ):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    try:
        import pytest
    except ImportError as error:  # pragma: no cover - environment failure
        raise ValueError("pytest is required for the pending-claim preflight") from error

    capture = _CollectionCapture()
    arguments = ["-q"]
    if collect_only:
        arguments.append("--collect-only")
    arguments.extend(str(repo_root / value) for value in spec["modules"])
    exit_code = int(pytest.main(arguments, plugins=[capture]))

    minimum_tests = int(spec["minimum_test_count"])
    count_passed = capture.collected >= minimum_tests
    passed = exit_code == 0 and count_passed
    if exit_code == 0 and not count_passed:
        exit_code = 3

    report = {
        "schema_version": 1,
        "source_commit": _git_head(repo_root),
        "spec": str(spec_path.resolve().relative_to(repo_root.resolve())),
        "spec_sha256": _sha256(spec_path),
        "claim_registry_sha256": _sha256(claim_registry_path),
        "module_count": len(spec["modules"]),
        "collected_tests": capture.collected,
        "minimum_test_count": minimum_tests,
        "explicit_pending_claim_tests": sorted(explicit_tests),
        "collect_only": collect_only,
        "pytest_exit_code": int(exit_code),
        "passed": passed,
    }
    return exit_code, report


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value) else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--claim-registry", type=Path, default=DEFAULT_CLAIM_REGISTRY
    )
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    spec_path = args.spec.resolve()
    claim_registry_path = args.claim_registry.resolve()
    try:
        exit_code, report = run_preflight(
            repo_root=repo_root,
            spec_path=spec_path,
            claim_registry_path=claim_registry_path,
            collect_only=args.collect_only,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"EGOR pending-claim preflight failed: {error}")
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.report is not None:
        report_path = args.report
        if not report_path.is_absolute():
            report_path = repo_root / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered, encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
