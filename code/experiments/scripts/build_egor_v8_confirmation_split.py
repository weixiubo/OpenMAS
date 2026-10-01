#!/usr/bin/env python3
"""Deterministically split untouched v6 pools for v8 format and confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SALT = "egor-aaai27-v8-resplit-20260718"
SOURCE_SPECS = {
    "chartqa": {
        "benchmark": "ChartQA-EGOR-Fresh",
        "format_file": "format40.jsonl",
        "confirmation_file": "confirmation200.jsonl",
    },
    "infographic": {
        "benchmark": "InfographicVQA-EGOR-Fresh",
        "format_file": "format40.jsonl",
        "confirmation_file": "confirmation200.jsonl",
    },
}


@dataclass(frozen=True)
class VisibleRow:
    raw: str
    benchmark: str
    task_id: str
    image_basename: str
    question_sha256: str


def load_visible_rows(path: Path, *, expected_benchmark: str) -> list[VisibleRow]:
    rows: list[VisibleRow] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_images: set[str] = set()
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        payload = json.loads(raw)
        benchmark = str(
            payload.get("benchmark") or payload.get("source_id") or ""
        ).strip()
        task_id = str(payload.get("task_id", "")).strip()
        image_basename = Path(str(payload.get("image_path", ""))).name.strip()
        question = _normalize_question(
            str(payload.get("question", ""))
        )
        key = (benchmark, task_id)
        if benchmark != expected_benchmark:
            raise ValueError(
                f"unexpected benchmark at line {line_number}: {benchmark!r}"
            )
        if (
            not task_id
            or not image_basename
            or not question
            or key in seen_keys
        ):
            raise ValueError(f"invalid or duplicate visible key at line {line_number}")
        if image_basename in seen_images:
            raise ValueError(f"duplicate image at line {line_number}")
        seen_keys.add(key)
        seen_images.add(image_basename)
        rows.append(
            VisibleRow(
                raw=raw.strip(),
                benchmark=benchmark,
                task_id=task_id,
                image_basename=image_basename,
                question_sha256=hashlib.sha256(
                    question.encode("utf-8")
                ).hexdigest(),
            )
        )
    if not rows:
        raise ValueError(f"empty pool: {path}")
    return rows


def partition(
    rows: list[VisibleRow],
    *,
    source: str,
    salt: str,
    format_count: int,
    forced_format_question_sha256: set[str] | None = None,
) -> tuple[list[VisibleRow], list[VisibleRow]]:
    if format_count <= 0 or format_count >= len(rows):
        raise ValueError("format_count must leave nonempty confirmation data")

    def rank(row: VisibleRow) -> tuple[str, str]:
        visible_key = "\0".join(
            (salt, source, row.benchmark, row.task_id, row.image_basename)
        )
        return hashlib.sha256(visible_key.encode("utf-8")).hexdigest(), row.task_id

    forced_hashes = set(forced_format_question_sha256 or set())
    unknown = forced_hashes - {row.question_sha256 for row in rows}
    if unknown:
        raise ValueError(
            f"{source} prior-use audit contains unknown question hashes"
        )
    forced_rows = sorted(
        [row for row in rows if row.question_sha256 in forced_hashes],
        key=rank,
    )
    if len(forced_rows) > format_count:
        raise ValueError(f"{source} forced format rows exceed format_count")
    remaining_rows = sorted(
        [row for row in rows if row.question_sha256 not in forced_hashes],
        key=rank,
    )
    remaining_slots = format_count - len(forced_rows)
    format_rows = forced_rows + remaining_rows[:remaining_slots]
    confirmation_rows = remaining_rows[remaining_slots:]
    if any(row.question_sha256 in forced_hashes for row in confirmation_rows):
        raise AssertionError("prior-used question leaked into confirmation")
    return format_rows, confirmation_rows


def build_split(
    *,
    pools: dict[str, Path],
    output_root: Path,
    expected_pool_rows: int,
    format_count: int,
    salt: str,
    format_only_question_sha256: dict[str, set[str]] | None = None,
    prior_use_audit_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    format_constraints = {
        source: set(values)
        for source, values in (format_only_question_sha256 or {}).items()
    }
    unknown_sources = set(format_constraints) - set(SOURCE_SPECS)
    if unknown_sources:
        raise ValueError("prior-use audit contains unknown sources")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing nonempty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    sources: dict[str, Any] = {}
    checksum_paths: list[Path] = []
    for source, spec in SOURCE_SPECS.items():
        pool = pools[source].resolve()
        rows = load_visible_rows(
            pool, expected_benchmark=str(spec["benchmark"])
        )
        if len(rows) != expected_pool_rows:
            raise ValueError(
                f"{source} pool has {len(rows)} rows, expected {expected_pool_rows}"
            )
        format_rows, confirmation_rows = partition(
            rows,
            source=source,
            salt=salt,
            format_count=format_count,
            forced_format_question_sha256=format_constraints.get(
                source, set()
            ),
        )
        source_root = output_root / source
        source_root.mkdir(parents=True, exist_ok=True)
        format_path = source_root / str(spec["format_file"])
        confirmation_path = source_root / str(spec["confirmation_file"])
        _write_rows(format_path, format_rows)
        _write_rows(confirmation_path, confirmation_rows)
        checksum_paths.extend((format_path, confirmation_path))

        format_keys = {(row.benchmark, row.task_id) for row in format_rows}
        confirmation_keys = {
            (row.benchmark, row.task_id) for row in confirmation_rows
        }
        format_images = {row.image_basename for row in format_rows}
        confirmation_images = {row.image_basename for row in confirmation_rows}
        overlap = {
            "task_keys": len(format_keys & confirmation_keys),
            "image_basenames": len(format_images & confirmation_images),
        }
        if any(overlap.values()):
            raise ValueError(f"split overlap for {source}: {overlap}")
        sources[source] = {
            "benchmark": spec["benchmark"],
            "pool": {
                "path": str(pool),
                "rows": len(rows),
                "sha256": _sha256(pool),
            },
            "prior_use_quarantine": {
                "question_sha256": sorted(
                    format_constraints.get(source, set())
                ),
                "unique_question_hashes": len(
                    format_constraints.get(source, set())
                ),
                "format_rows": sum(
                    row.question_sha256 in format_constraints.get(source, set())
                    for row in format_rows
                ),
                "confirmation_rows": sum(
                    row.question_sha256 in format_constraints.get(source, set())
                    for row in confirmation_rows
                ),
            },
            "format": _split_receipt(format_path, format_rows),
            "confirmation": _split_receipt(
                confirmation_path, confirmation_rows
            ),
            "overlap": overlap,
        }

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_policy": (
            "Prior-used normalized-question SHA-256 values are forced into "
            "the unscored format split; remaining rows are SHA-256 ranked "
            "over salt, source, benchmark, task id, and image basename; "
            "labels, reference answers, and outcomes do not affect membership"
        ),
        "salt": salt,
        "prior_use_audit": prior_use_audit_receipt,
        "expected_pool_rows": expected_pool_rows,
        "format_count": format_count,
        "confirmation_count": expected_pool_rows - format_count,
        "selection_fields": [
            "benchmark/source_id",
            "task_id",
            "image basename",
            "normalized question SHA-256 (prior-use quarantine only)",
        ],
        "outcome_fields_used_for_selection": False,
        "raw_task_ids_emitted_to_logs": False,
        "questions_emitted_to_logs": False,
        "question_hashes_emitted_to_manifest": True,
        "reference_answers_emitted_to_logs": False,
        "scores_computed": False,
        "sources": sources,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_paths.append(manifest_path)
    checksum_file = output_root / "SPLIT_SHA256SUMS"
    checksum_file.write_text(
        "".join(f"{_sha256(path)}  {path}\n" for path in sorted(checksum_paths)),
        encoding="utf-8",
    )
    (output_root / "SPLIT_FROZEN").touch()
    return manifest


def _write_rows(path: Path, rows: list[VisibleRow]) -> None:
    path.write_text(
        "".join(f"{row.raw}\n" for row in rows),
        encoding="utf-8",
    )


def _split_receipt(path: Path, rows: list[VisibleRow]) -> dict[str, Any]:
    task_keys = sorted(f"{row.benchmark}\0{row.task_id}" for row in rows)
    images = sorted(row.image_basename for row in rows)
    question_hashes = sorted(row.question_sha256 for row in rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "sha256": _sha256(path),
        "task_key_set_sha256": _hash_lines(task_keys),
        "image_basename_set_sha256": _hash_lines(images),
        "question_sha256_set_sha256": _hash_lines(question_hashes),
        "unique_task_keys": len(set(task_keys)),
        "unique_image_basenames": len(set(images)),
        "unique_question_hashes": len(set(question_hashes)),
    }


def _normalize_question(value: str) -> str:
    return " ".join(value.split()).strip()


def _hash_lines(values: list[str]) -> str:
    rendered = "".join(f"{value}\n" for value in values)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prior_use_constraints(
    path: Path,
    *,
    pools: dict[str, Path],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    resolved = path.resolve()
    report = json.loads(resolved.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ValueError("prior-use audit did not pass")
    task_sets = report.get("task_sets")
    raw_constraints = report.get("format_only_question_sha256")
    if not isinstance(task_sets, dict) or not isinstance(
        raw_constraints, dict
    ):
        raise ValueError("prior-use audit lacks required receipts")

    constraints: dict[str, set[str]] = {}
    for source in SOURCE_SPECS:
        task_receipt = task_sets.get(source)
        if not isinstance(task_receipt, dict):
            raise ValueError(f"prior-use audit lacks {source} task receipt")
        if task_receipt.get("sha256") != _sha256(pools[source].resolve()):
            raise ValueError(f"prior-use audit {source} pool hash mismatch")
        values = raw_constraints.get(source, [])
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not _is_sha256(value)
            for value in values
        ):
            raise ValueError(
                f"prior-use audit has invalid {source} question hashes"
            )
        constraints[source] = set(values)

    receipt = {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "audit_kind": report.get("audit_kind"),
        "passed": True,
    }
    return constraints, receipt


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and set(value) <= set("0123456789abcdef")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chartqa-pool", required=True)
    parser.add_argument("--infographic-pool", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prior-use-audit", required=True)
    parser.add_argument("--expected-pool-rows", type=int, default=240)
    parser.add_argument("--format-count", type=int, default=40)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    pools = {
        "chartqa": Path(args.chartqa_pool),
        "infographic": Path(args.infographic_pool),
    }
    constraints, prior_use_receipt = load_prior_use_constraints(
        Path(args.prior_use_audit),
        pools=pools,
    )
    manifest = build_split(
        pools=pools,
        output_root=Path(args.output_root).resolve(),
        expected_pool_rows=args.expected_pool_rows,
        format_count=args.format_count,
        salt=args.salt,
        format_only_question_sha256=constraints,
        prior_use_audit_receipt=prior_use_receipt,
    )
    summary = {
        source: {
            split: manifest["sources"][source][split]["sha256"]
            for split in ("format", "confirmation")
        }
        for source in SOURCE_SPECS
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
