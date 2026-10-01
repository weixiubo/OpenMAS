#!/usr/bin/env python3
"""Build fresh ChartQA development/held-out splits from a cached Arrow file."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openagora_experiments.egor.chartqa_splits import (
    identifier_hash,
    load_exclusions,
    materialize_chartqa_rows,
    select_chartqa_rows,
    selected_image_hash,
    sha256,
    write_jsonl,
)


def main() -> int:
    args = _parse_args()
    arrow_path = Path(args.arrow).resolve()
    images_root = Path(args.images_root).resolve()
    development_path = Path(args.development_out).resolve()
    heldout_path = Path(args.heldout_out).resolve()
    manifest_path = Path(args.manifest).resolve()
    exclusion_paths = [Path(value).resolve() for value in args.exclude_jsonl]
    excluded_task_ids, excluded_images = load_exclusions(exclusion_paths)
    rows = _load_arrow_rows(arrow_path)
    development_selection, heldout_selection, audit = select_chartqa_rows(
        rows,
        excluded_task_ids=excluded_task_ids,
        excluded_images=excluded_images,
        development_size=args.development_size,
        heldout_size=args.heldout_size,
        salt=args.salt,
    )
    development_rows = materialize_chartqa_rows(
        development_selection,
        images_root=images_root,
    )
    heldout_rows = materialize_chartqa_rows(
        heldout_selection,
        images_root=images_root,
    )
    write_jsonl(development_path, development_rows)
    write_jsonl(heldout_path, heldout_rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_arrow": str(arrow_path),
        "source_arrow_sha256": sha256(arrow_path),
        "images_root": str(images_root),
        "exclusion_files": [
            {"path": str(path), "sha256": sha256(path)} for path in exclusion_paths
        ],
        "salt": args.salt,
        "selection_policy": (
            "rank image names and question identifiers with SHA-256; select at most "
            "one visible question per image; never inspect labels or prior outcomes"
        ),
        "metric": "ChartQA relaxed accuracy with 5% relative numeric tolerance",
        "development": _split_manifest(development_path, development_rows),
        "heldout": _split_manifest(heldout_path, heldout_rows),
        **audit,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote development={len(development_rows)} heldout={len(heldout_rows)} "
        f"manifest={manifest_path}"
    )
    return 0


def _load_arrow_rows(path: Path) -> list[dict[str, Any]]:
    try:
        from datasets import Dataset  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on extraction environment
        raise RuntimeError(
            "the extraction interpreter must provide Hugging Face datasets"
        ) from exc
    dataset = Dataset.from_file(str(path))
    return [dict(dataset[index]) for index in range(len(dataset))]


def _split_manifest(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path),
        "rows": len(rows),
        "sha256": sha256(path),
        "task_ids_sha256": identifier_hash(rows),
        "selected_images_sha256": selected_image_hash(rows),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    parser.add_argument("--development-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--development-size", type=int, default=120)
    parser.add_argument("--heldout-size", type=int, default=240)
    parser.add_argument("--salt", default="egor-aaai27-chartqa-v1")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
