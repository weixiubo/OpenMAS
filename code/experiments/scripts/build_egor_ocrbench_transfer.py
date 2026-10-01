#!/usr/bin/env python3
"""Build a sealed zero-adaptation OCRBench-v2 BasicVQA transfer split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openagora_experiments.egor.ocrbench_splits import (
    DEFAULT_BASIC_VQA_TYPES,
    OCRBenchCandidate,
    load_exclusions,
    normalize_transfer_row,
    select_transfer_rows,
    selected_identifier_hash,
    selected_image_hash,
    sha256,
    write_jsonl,
)


def main() -> int:
    args = _parse_args()
    arrow_paths = _arrow_paths(Path(args.arrow_dir).resolve())
    exclusion_paths = [Path(value).resolve() for value in args.exclude]
    excluded_source_ids, excluded_image_hashes = load_exclusions(exclusion_paths)
    candidates = _index_candidates(arrow_paths)
    task_types = tuple(args.task_type or DEFAULT_BASIC_VQA_TYPES)
    selected, audit = select_transfer_rows(
        candidates,
        excluded_source_ids=excluded_source_ids,
        excluded_image_hashes=excluded_image_hashes,
        task_types=task_types,
        per_type=args.per_type,
        salt=args.salt,
    )
    counts = Counter(row.task_type for row in candidates)
    print(
        "source_type_counts="
        + json.dumps(dict(sorted(counts.items())), ensure_ascii=False, sort_keys=True)
    )
    print("selection_audit=" + json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if args.audit_only:
        return 0

    output_path = Path(args.output).resolve()
    images_dir = Path(args.images_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    image_paths = _materialize_images(selected, images_dir=images_dir)
    normalized = [
        normalize_transfer_row(row, image_path=image_paths[row.source_id])
        for row in selected
    ]
    write_jsonl(output_path, normalized)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    revision = arrow_paths[0].parent.name if arrow_paths else ""
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_dataset": "lmms-lab/OCRBench-v2",
        "official_dataset_revision": revision,
        "source_arrow_shards": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in arrow_paths
        ],
        "exclusion_files": [
            {"path": str(path), "sha256": sha256(path)} for path in exclusion_paths
        ],
        "salt": args.salt,
        "selection_policy": (
            "balance the preregistered official BasicVQA task types; rank only "
            "source id, question, dataset name, task type, and image digest with "
            "SHA-256; exclude every prior OCRBench-v2 source id and every prior "
            "image digest; never inspect answers or model outcomes"
        ),
        "policy_use": (
            "zero-adaptation transfer only: no OCRBench-v2 outcome may train, "
            "select, calibrate, or threshold EGOR-v2"
        ),
        "official_metric": (
            "OCRBench-v2 BasicVQA: case-insensitive reference containment for "
            "answers shorter than five words, otherwise thresholded ANLS"
        ),
        "task_types": list(task_types),
        "per_type": args.per_type,
        "output": {
            "path": str(output_path),
            "rows": len(normalized),
            "sha256": sha256(output_path),
            "task_ids_sha256": selected_identifier_hash(selected),
            "selected_images_sha256": selected_image_hash(selected),
            "images_dir": str(images_dir),
        },
        **audit,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote transfer={len(normalized)} output={output_path} "
        f"manifest={manifest_path}"
    )
    return 0


def _arrow_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.arrow"))
    if not paths:
        raise FileNotFoundError(f"no Arrow shards in {directory}")
    return paths


def _index_candidates(paths: list[Path]) -> list[OCRBenchCandidate]:
    try:
        from datasets import Dataset, Image  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - extraction environment only
        raise RuntimeError("the extraction interpreter must provide datasets") from exc
    candidates: list[OCRBenchCandidate] = []
    for path in paths:
        dataset = Dataset.from_file(str(path)).cast_column("image", Image(decode=False))
        for row_index in range(len(dataset)):
            row = dataset[row_index]
            image_bytes = _image_bytes(row.get("image"), base_dir=path.parent)
            answers = row.get("answers", [])
            references = (
                tuple(str(value).strip() for value in answers if str(value).strip())
                if isinstance(answers, list)
                else ()
            )
            source_id = int(row["id"])
            question = str(row.get("question", "")).strip()
            dataset_name = str(row.get("dataset_name", "")).strip()
            task_type = str(row.get("type", "")).strip()
            if not question or not references or not dataset_name or not task_type:
                raise ValueError(f"invalid OCRBench-v2 row id={source_id}")
            candidates.append(
                OCRBenchCandidate(
                    source_id=source_id,
                    question=question,
                    answers=references,
                    dataset_name=dataset_name,
                    task_type=task_type,
                    image_sha256=hashlib.sha256(image_bytes).hexdigest(),
                    shard_path=str(path),
                    row_index=row_index,
                )
            )
    return candidates


def _materialize_images(
    rows: list[OCRBenchCandidate], *, images_dir: Path
) -> dict[int, Path]:
    try:
        from datasets import Dataset, Image  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - extraction environment only
        raise RuntimeError("the extraction interpreter must provide datasets") from exc
    images_dir.mkdir(parents=True, exist_ok=True)
    by_shard: dict[str, list[OCRBenchCandidate]] = defaultdict(list)
    for row in rows:
        by_shard[row.shard_path].append(row)
    outputs: dict[int, Path] = {}
    for shard, selected_rows in by_shard.items():
        path = Path(shard)
        dataset = Dataset.from_file(str(path)).cast_column("image", Image(decode=False))
        for selected in selected_rows:
            source = dataset[selected.row_index]
            payload = _image_bytes(source.get("image"), base_dir=path.parent)
            if hashlib.sha256(payload).hexdigest() != selected.image_sha256:
                raise AssertionError(
                    f"image changed for source id {selected.source_id}"
                )
            suffix = _image_suffix(payload)
            output = images_dir / f"{selected.task_id}{suffix}"
            output.write_bytes(payload)
            outputs[selected.source_id] = output
    return outputs


def _image_bytes(value: Any, *, base_dir: Path) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict):
        payload = value.get("bytes")
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        raw_path = value.get("path")
        if isinstance(raw_path, str) and raw_path:
            path = Path(raw_path)
            if not path.is_absolute():
                path = base_dir / path
            return path.read_bytes()
    raise ValueError("OCRBench-v2 image is neither encoded bytes nor a readable path")


def _image_suffix(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if payload.startswith(b"BM"):
        return ".bmp"
    if payload[:4] in {b"II*\x00", b"MM\x00*"}:
        return ".tiff"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return ".webp"
    return ".img"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow-dir", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--output", default="")
    parser.add_argument("--images-dir", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--task-type", action="append", default=[])
    parser.add_argument("--per-type", type=int, default=30)
    parser.add_argument("--salt", default="egor-aaai27-ocrbench-transfer-v1")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not args.audit_only and not (args.output and args.images_dir and args.manifest):
        parser.error(
            "--output, --images-dir, and --manifest are required unless --audit-only"
        )
    return args


if __name__ == "__main__":
    raise SystemExit(main())
