#!/usr/bin/env python3
"""Build answer-blind, image-disjoint InfographicVQA splits for EGOR."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> int:
    args = _parse_args()
    source_path = Path(args.source).resolve()
    images_root = Path(args.images_root).resolve()
    calibration_path = Path(args.calibration_out).resolve()
    heldout_path = Path(args.heldout_out).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else calibration_path.with_name("egor_infographic_splits.manifest.json")
    )
    excluded_task_ids, excluded_images = _load_exclusions(
        [Path(item).resolve() for item in args.exclude_jsonl]
    )
    rows = _source_rows(source_path)
    calibration_rows, heldout_rows, audit = build_splits(
        rows,
        images_root=images_root,
        excluded_task_ids=excluded_task_ids,
        excluded_images=excluded_images,
        calibration_size=args.calibration_size,
        heldout_size=args.heldout_size,
        max_questions_per_image=args.max_questions_per_image,
        salt=args.salt,
    )
    _write_jsonl(calibration_path, calibration_rows)
    _write_jsonl(heldout_path, heldout_rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "images_root": str(images_root),
        "exclusion_files": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in [Path(item).resolve() for item in args.exclude_jsonl]
        ],
        "salt": args.salt,
        "selection_policy": (
            "rank image groups and task ids by SHA-256(salt, visible identifiers); "
            "never inspect answers or prior outcomes; keep calibration and heldout image-disjoint"
        ),
        "max_questions_per_image": args.max_questions_per_image,
        "calibration": {
            "path": str(calibration_path),
            "rows": len(calibration_rows),
            "sha256": _sha256(calibration_path),
            "task_ids_sha256": _identifier_hash(calibration_rows),
        },
        "heldout": {
            "path": str(heldout_path),
            "rows": len(heldout_rows),
            "sha256": _sha256(heldout_path),
            "task_ids_sha256": _identifier_hash(heldout_rows),
        },
        **audit,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote calibration={len(calibration_rows)} heldout={len(heldout_rows)} "
        f"manifest={manifest_path}"
    )
    return 0


def build_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    images_root: Path,
    excluded_task_ids: set[str],
    excluded_images: set[str],
    calibration_size: int,
    heldout_size: int,
    max_questions_per_image: int,
    salt: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if calibration_size <= 0 or heldout_size <= 0:
        raise ValueError("calibration and heldout sizes must be positive")
    if max_questions_per_image <= 0:
        raise ValueError("max_questions_per_image must be positive")
    if not salt.strip():
        raise ValueError("a non-empty salt is required")
    by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_task_ids: set[str] = set()
    excluded_by_task = 0
    excluded_by_image = 0
    for row in rows:
        task_id = str(row.get("questionId", row.get("task_id", ""))).strip()
        image_name = Path(str(row.get("image_local_name", row.get("image", ""))).strip()).name
        question = str(row.get("question", "")).strip()
        if not task_id or not image_name or not question:
            raise ValueError("every source row needs questionId/task_id, question, and image_local_name/image")
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate source task id: {task_id}")
        seen_task_ids.add(task_id)
        if task_id in excluded_task_ids:
            excluded_by_task += 1
            continue
        if image_name in excluded_images:
            excluded_by_image += 1
            continue
        image_path = images_root / image_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        by_image[image_name].append(row)

    ranked_images = sorted(by_image, key=lambda name: (_rank(salt, "image", name), name))
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for image_name in ranked_images:
        ranked_rows = sorted(
            by_image[image_name],
            key=lambda row: (
                _rank(salt, "task", str(row.get("questionId", row.get("task_id", "")))),
                str(row.get("questionId", row.get("task_id", ""))),
            ),
        )
        candidates.extend((image_name, row) for row in ranked_rows[:max_questions_per_image])

    required = calibration_size + heldout_size
    if len(candidates) < required:
        raise ValueError(
            f"only {len(candidates)} eligible image-capped tasks for {required} requested rows"
        )
    calibration_pairs = candidates[:calibration_size]
    calibration_images = {image_name for image_name, _ in calibration_pairs}
    heldout_pairs: list[tuple[str, Mapping[str, Any]]] = []
    for image_name, row in candidates[calibration_size:]:
        if image_name in calibration_images:
            continue
        heldout_pairs.append((image_name, row))
        if len(heldout_pairs) == heldout_size:
            break
    if len(heldout_pairs) < heldout_size:
        raise ValueError("not enough tasks to keep calibration and heldout images disjoint")

    calibration_rows = [_normalize(row, images_root, image_name) for image_name, row in calibration_pairs]
    heldout_rows = [_normalize(row, images_root, image_name) for image_name, row in heldout_pairs]
    heldout_images = {image_name for image_name, _ in heldout_pairs}
    if calibration_images & heldout_images:
        raise AssertionError("calibration and heldout images overlap")
    return (
        calibration_rows,
        heldout_rows,
        {
            "source_rows": len(rows),
            "source_unique_images": len({
                Path(str(row.get("image_local_name", row.get("image", "")))).name for row in rows
            }),
            "excluded_task_ids_declared": len(excluded_task_ids),
            "excluded_images_declared": len(excluded_images),
            "rows_excluded_by_task_id": excluded_by_task,
            "rows_excluded_by_image": excluded_by_image,
            "eligible_images": len(by_image),
            "eligible_tasks_after_image_cap": len(candidates),
            "image_overlap_between_splits": 0,
        },
    )


def _normalize(row: Mapping[str, Any], images_root: Path, image_name: str) -> dict[str, Any]:
    task_id = str(row.get("questionId", row.get("task_id", ""))).strip()
    raw_answers = row.get("answers", row.get("reference_answers", []))
    answers = [str(item).strip() for item in raw_answers] if isinstance(raw_answers, list) else []
    answers = [item for item in answers if item]
    if not answers:
        answer = str(row.get("answer", "")).strip()
        answers = [answer] if answer else []
    if not answers:
        raise ValueError(f"selected task {task_id} has no reference answer")
    answer_types = _string_list(row.get("answer_type", []))
    evidence = _string_list(row.get("evidence", []))
    operations = _string_list(row.get("operation/reasoning", row.get("operation_reasoning", [])))
    return {
        "task_id": task_id,
        "benchmark": "InfographicVQA-EGOR-Fresh",
        "question": str(row.get("question", "")).strip(),
        "answer": answers[0],
        "reference_answers": answers,
        "metric": "anls",
        "question_type": "qa",
        "image_path": str((images_root / image_name).resolve()),
        "metadata": {
            "primary_type": "infographicvqa",
            "source_split": str(row.get("data_split", "val")),
            "source_image": image_name,
            "answer_type": answer_types,
            "evidence": evidence,
            "operation_reasoning": operations,
            "uncertainty": "medium",
            "complexity": "high",
            "interdependence": "sequential",
            "egor_fresh_split": True,
        },
    }


def _source_rows(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rows = payload.get("data", []) if isinstance(payload, Mapping) else payload
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("source must be a non-empty list or an object with non-empty data")
    if any(not isinstance(row, Mapping) for row in raw_rows):
        raise ValueError("every source row must be an object")
    return list(raw_rows)


def _load_exclusions(paths: Sequence[Path]) -> tuple[set[str], set[str]]:
    task_ids: set[str] = set()
    images: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = str(row.get("task_id", "")).strip()
                image_path = str(row.get("image_path", "")).strip()
                if task_id:
                    task_ids.add(task_id)
                if image_path:
                    images.add(Path(image_path).name)
    return task_ids, images


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rank(salt: str, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{salt}\0{namespace}\0{value}".encode()).hexdigest()


def _identifier_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(str(row.get("task_id", "")) for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string_list(value: Any) -> list[str]:
    return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    parser.add_argument("--calibration-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--calibration-size", type=int, default=80)
    parser.add_argument("--heldout-size", type=int, default=160)
    parser.add_argument("--max-questions-per-image", type=int, default=1)
    parser.add_argument("--salt", default="egor-aaai27-preregistered-v1")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
