#!/usr/bin/env python3
"""Audit that sealed v8 confirmation pools have no prior structured use."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


STRUCTURED_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv"}
SOURCE_ALIASES = {
    "chartqa": ("chartqa",),
    "infographic": ("infographic", "infographicvqa"),
}


def load_targets(path: Path) -> dict[str, set[str]]:
    targets = {"task_ids": set(), "images": set(), "questions": set()}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        task_id = str(payload.get("task_id", "")).strip()
        image = Path(str(payload.get("image_path", ""))).name.strip()
        question = _normalize_question(str(payload.get("question", "")))
        if not task_id or not image or not question:
            raise ValueError(f"target row lacks visible identifiers: {path}")
        targets["task_ids"].add(task_id)
        targets["images"].add(image)
        targets["questions"].add(question)
    if not targets["task_ids"]:
        raise ValueError(f"empty target set: {path}")
    return targets


def audit_prior_use(
    *,
    pools: dict[str, Path],
    scan_roots: list[Path],
    excluded_paths: set[Path] | None = None,
) -> dict[str, Any]:
    targets = {
        source: load_targets(path.resolve()) for source, path in pools.items()
    }
    excluded = {path.resolve() for path in (excluded_paths or set())}
    excluded.update(path.resolve() for path in pools.values())
    files = sorted(
        {
            path.resolve()
            for root in scan_roots
            if root.exists()
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in STRUCTURED_SUFFIXES
            and path.resolve() not in excluded
        }
    )

    strong: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    question_only: list[dict[str, Any]] = []
    format_only_question_sha256: dict[str, set[str]] = {
        source: set() for source in targets
    }
    parse_fallbacks = 0
    for path in files:
        source_hint = _source_hint(str(path))
        matches: dict[str, dict[str, int]] = {
            source: {"task_ids": 0, "images": 0, "questions": 0}
            for source in targets
        }
        ambiguous_counts = {"task_ids": 0, "images": 0, "questions": 0}
        try:
            records = _load_records(path)
            for record in records:
                _inspect_value(
                    record,
                    inherited_source=source_hint,
                    targets=targets,
                    matches=matches,
                    ambiguous=ambiguous_counts,
                    format_only_questions=format_only_question_sha256,
                )
        except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError):
            parse_fallbacks += 1
            _fallback_scan(
                path,
                source_hint=source_hint,
                targets=targets,
                matches=matches,
                ambiguous=ambiguous_counts,
                format_only_questions=format_only_question_sha256,
            )

        source_matches = {
            source: counts
            for source, counts in matches.items()
            if any(counts.values())
        }
        blocking_source_matches = {
            source: counts
            for source, counts in source_matches.items()
            if counts["task_ids"] or counts["images"]
        }
        question_source_matches = {
            source: {"questions": counts["questions"]}
            for source, counts in source_matches.items()
            if counts["questions"]
        }
        path_receipt = {
            "path_sha256": hashlib.sha256(
                str(path).encode("utf-8")
            ).hexdigest()
        }
        if blocking_source_matches:
            strong.append(
                {
                    **path_receipt,
                    "matches": blocking_source_matches,
                }
            )
        if question_source_matches or ambiguous_counts["questions"]:
            question_matches = dict(question_source_matches)
            if ambiguous_counts["questions"]:
                question_matches["ambiguous"] = {
                    "questions": ambiguous_counts["questions"]
                }
            question_only.append(
                {
                    **path_receipt,
                    "matches": question_matches,
                }
            )
        if ambiguous_counts["task_ids"] or ambiguous_counts["images"]:
            ambiguous.append(
                {
                    **path_receipt,
                    "matches": ambiguous_counts,
                }
            )

    split_receipts = {
        source: {
            "path": str(path.resolve()),
            "sha256": _sha256(path.resolve()),
            "rows": len(targets[source]["task_ids"]),
            "task_id_set_sha256": _hash_set(targets[source]["task_ids"]),
            "image_basename_set_sha256": _hash_set(targets[source]["images"]),
            "question_set_sha256": _hash_set(targets[source]["questions"]),
        }
        for source, path in pools.items()
    }
    passed = not strong and not ambiguous
    return {
        "schema_version": 1,
        "audit_kind": "benchmark_aware_outcome_blind_v8_prior_use",
        "passed": passed,
        "privacy_contract": {
            "model_answers_emitted": False,
            "questions_emitted": False,
            "question_hashes_emitted": True,
            "raw_task_ids_emitted": False,
            "reference_answers_emitted": False,
            "scores_emitted": False,
        },
        "scan_roots": [str(path.resolve()) for path in scan_roots],
        "scanned_structured_files": len(files),
        "parse_fallbacks": parse_fallbacks,
        "task_sets": split_receipts,
        "strong_matching_artifacts": strong,
        "ambiguous_matching_artifacts": ambiguous,
        "question_matching_artifacts": question_only,
        "format_only_question_sha256": {
            source: sorted(values)
            for source, values in format_only_question_sha256.items()
        },
        "usage_summary": {
            source: {
                "appears_in_prior_result_or_analysis": any(
                    source in artifact["matches"]
                    for artifact in (*strong, *question_only)
                ),
                "strong_matching_artifacts": sum(
                    source in artifact["matches"] for artifact in strong
                ),
                "question_matching_artifacts": sum(
                    source in artifact["matches"]
                    for artifact in question_only
                ),
                "format_only_questions": len(
                    format_only_question_sha256[source]
                ),
            }
            for source in targets
        },
    }


def _load_records(path: Path) -> Iterable[Any]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if suffix == ".json":
        yield json.loads(path.read_text(encoding="utf-8"))
        return
    delimiter = "\t" if suffix == ".tsv" else ","
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter=delimiter)


def _inspect_value(
    value: Any,
    *,
    inherited_source: str | None,
    targets: dict[str, dict[str, set[str]]],
    matches: dict[str, dict[str, int]],
    ambiguous: dict[str, int],
    format_only_questions: dict[str, set[str]],
) -> None:
    if isinstance(value, Mapping):
        source = _source_from_mapping(value) or inherited_source
        for key, nested in value.items():
            kind = _field_kind(str(key))
            if kind:
                for scalar in _scalars(nested):
                    _record_scalar(
                        scalar,
                        kind=kind,
                        source=source,
                        targets=targets,
                        matches=matches,
                        ambiguous=ambiguous,
                        format_only_questions=format_only_questions,
                    )
            if isinstance(nested, (Mapping, list, tuple)):
                _inspect_value(
                    nested,
                    inherited_source=source,
                    targets=targets,
                    matches=matches,
                    ambiguous=ambiguous,
                    format_only_questions=format_only_questions,
                )
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _inspect_value(
                nested,
                inherited_source=inherited_source,
                targets=targets,
                matches=matches,
                ambiguous=ambiguous,
                format_only_questions=format_only_questions,
            )


def _record_scalar(
    value: Any,
    *,
    kind: str,
    source: str | None,
    targets: dict[str, dict[str, set[str]]],
    matches: dict[str, dict[str, int]],
    ambiguous: dict[str, int],
    format_only_questions: dict[str, set[str]],
) -> None:
    raw = str(value).strip()
    if not raw:
        return
    candidate = (
        Path(raw).name
        if kind == "images"
        else _normalize_question(raw)
        if kind == "questions"
        else raw
    )
    matched_sources = [
        name for name, values in targets.items() if candidate in values[kind]
    ]
    if not matched_sources:
        return
    if kind == "questions":
        question_hash = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        for matched_source in matched_sources:
            format_only_questions[matched_source].add(question_hash)
    if source in matched_sources:
        matches[source][kind] += 1
    else:
        ambiguous[kind] += 1


def _fallback_scan(
    path: Path,
    *,
    source_hint: str | None,
    targets: dict[str, dict[str, set[str]]],
    matches: dict[str, dict[str, int]],
    ambiguous: dict[str, int],
    format_only_questions: dict[str, set[str]],
) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for source, fields in targets.items():
        counts = {"task_ids": 0, "images": 0, "questions": 0}
        for kind, values in fields.items():
            for value in values:
                if value not in text:
                    continue
                counts[kind] += 1
                if kind == "questions":
                    question_hash = hashlib.sha256(
                        value.encode("utf-8")
                    ).hexdigest()
                    format_only_questions[source].add(question_hash)
        if source_hint == source:
            for kind, count in counts.items():
                matches[source][kind] += count
        else:
            for kind, count in counts.items():
                ambiguous[kind] += count


def _source_from_mapping(value: Mapping[str, Any]) -> str | None:
    for key in ("benchmark", "source_id", "source", "dataset"):
        if key in value:
            source = _source_hint(str(value[key]))
            if source:
                return source
    return None


def _source_hint(value: str) -> str | None:
    lowered = value.lower()
    for source, aliases in SOURCE_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return source
    return None


def _field_kind(key: str) -> str | None:
    lowered = key.lower()
    if "task" in lowered and "id" in lowered:
        return "task_ids"
    if "image" in lowered:
        return "images"
    if lowered in {"question", "query", "question_text"}:
        return "questions"
    return None


def _scalars(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if not isinstance(item, (Mapping, list, tuple, set)):
                yield item
    elif not isinstance(value, Mapping):
        yield value


def _normalize_question(value: str) -> str:
    return " ".join(value.split()).strip()


def _hash_set(values: set[str]) -> str:
    rendered = "".join(f"{value}\n" for value in sorted(values))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chartqa-pool", required=True)
    parser.add_argument("--infographic-pool", required=True)
    parser.add_argument("--scan-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = Path(args.output).resolve()
    report = audit_prior_use(
        pools={
            "chartqa": Path(args.chartqa_pool),
            "infographic": Path(args.infographic_pool),
        },
        scan_roots=[Path(value) for value in args.scan_root],
        excluded_paths={output},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "scanned_structured_files": report[
                    "scanned_structured_files"
                ],
                "strong_matching_artifacts": len(
                    report["strong_matching_artifacts"]
                ),
                "ambiguous_matching_artifacts": len(
                    report["ambiguous_matching_artifacts"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
