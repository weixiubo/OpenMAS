#!/usr/bin/env python3
"""Build answer-blind fresh ChartQA and OCRBench-v2 v9 confirmation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openagora_experiments.egor.ocrbench_splits import (
    OCRBenchCandidate,
    normalize_transfer_row,
    select_transfer_rows_by_type,
    sha256,
    write_jsonl,
)
from openagora_experiments.egor.ocrbench_v9_prior_filter import (
    filter_prior_questions,
)


DEFAULT_SALT = "egor-aaai27-v9-fresh-confirmation-20260718"
CHARTQA_TYPES = ("human", "augmented")
CHARTQA_BENCHMARK = "ChartQA-EGOR-V9-Fresh"
CHARTQA_TASK_PREFIX = "chartqa-v9-test"
OCRBENCH_BENCHMARK = "OCRBench-v2-English-VQA-EGOR-V9-Fresh"
OCRBENCH_OFFICIAL_EVAL_REPOSITORY = (
    "https://github.com/Yuliang-Liu/MultimodalOCR"
)
OCRBENCH_OFFICIAL_EVAL_COMMIT = "9ec46950c87e1b683cd3af9e6f6d45dea088ecee"
OCRBENCH_V9_TYPES = (
    "APP agent en",
    "ASCII art classification en",
    "cognition VQA en",
    "diagram QA en",
    "math QA en",
    "reasoning VQA en",
    "science QA en",
)
OCRBENCH_V9_FORMAT_QUOTA = {
    "APP agent en": 6,
    "ASCII art classification en": 5,
    "cognition VQA en": 6,
    "diagram QA en": 6,
    "math QA en": 6,
    "reasoning VQA en": 5,
    "science QA en": 6,
}
OCRBENCH_V9_CONFIRMATION_QUOTA = {
    "APP agent en": 29,
    "ASCII art classification en": 29,
    "cognition VQA en": 28,
    "diagram QA en": 28,
    "math QA en": 29,
    "reasoning VQA en": 29,
    "science QA en": 28,
}
OCRBENCH_V9_TOTAL_QUOTA = {
    task_type: OCRBENCH_V9_FORMAT_QUOTA[task_type]
    + OCRBENCH_V9_CONFIRMATION_QUOTA[task_type]
    for task_type in OCRBENCH_V9_TYPES
}


@dataclass(frozen=True)
class Exclusions:
    source_ids: frozenset[int]
    image_hashes: frozenset[str]
    image_basenames: frozenset[str]
    question_hashes: frozenset[str]


@dataclass(frozen=True)
class ChartQACandidate:
    row_index: int
    image_name: str
    image_sha256: str
    image_bytes: bytes
    question: str
    answer: str
    question_type: str

    @property
    def task_id(self) -> str:
        return f"{CHARTQA_TASK_PREFIX}-{self.row_index + 1:05d}"


def collect_exclusions(paths: Sequence[Path]) -> Exclusions:
    """Collect only visible identifiers, questions, and image identities."""

    source_ids: set[int] = set()
    image_hashes: set[str] = set()
    image_basenames: set[str] = set()
    question_hashes: set[str] = set()
    for path in paths:
        for row in _json_rows(path):
            metadata = row.get("metadata", {})
            metadata = metadata if isinstance(metadata, Mapping) else {}
            benchmark = str(row.get("benchmark", "")).lower()
            primary_type = str(metadata.get("primary_type", "")).lower()
            is_ocrbench = "ocrbench" in benchmark or "ocrbench" in primary_type
            task_id = str(row.get("task_id", "")).strip().lower()
            question = _normalize_question(
                str(row.get("question", row.get("query", "")))
            )
            if question:
                question_hashes.add(_text_sha256(question))
            for value in (
                row.get("imgname"),
                row.get("image_path"),
                row.get("image") if isinstance(row.get("image"), str) else None,
                metadata.get("source_image"),
                metadata.get("source_image_name"),
            ):
                if isinstance(value, str) and value.strip():
                    image_basenames.add(Path(value).name)
                    image_path = Path(value)
                    if not image_path.is_absolute():
                        image_path = path.parent / image_path
                    if image_path.is_file():
                        image_hashes.add(sha256(image_path))
            recorded_hash = str(
                metadata.get("source_image_sha256", "")
            ).strip().lower()
            if _is_sha256(recorded_hash):
                image_hashes.add(recorded_hash)
            for value in (row.get("id"), metadata.get("source_id")):
                parsed = _integer(value)
                if parsed is not None:
                    source_ids.add(parsed)
            if task_id.startswith("ocrbench-v2-"):
                parsed = _integer(task_id.removeprefix("ocrbench-v2-"))
                if parsed is not None:
                    source_ids.add(parsed)
            elif is_ocrbench and (parsed := _integer(task_id)) is not None:
                source_ids.add(parsed)
    return Exclusions(
        source_ids=frozenset(source_ids),
        image_hashes=frozenset(image_hashes),
        image_basenames=frozenset(image_basenames),
        question_hashes=frozenset(question_hashes),
    )


def select_chartqa_rows(
    candidates: Sequence[ChartQACandidate],
    *,
    exclusions: Exclusions,
    per_type: int | Mapping[str, int],
    salt: str,
) -> tuple[list[ChartQACandidate], dict[str, Any]]:
    """Select type-balanced rows without using candidate answers."""

    groups: dict[str, list[ChartQACandidate]] = {
        value: [] for value in CHARTQA_TYPES
    }
    excluded_image = 0
    excluded_question = 0
    for row in candidates:
        if row.question_type not in groups:
            continue
        if (
            row.image_name in exclusions.image_basenames
            or row.image_sha256 in exclusions.image_hashes
        ):
            excluded_image += 1
            continue
        if _text_sha256(_normalize_question(row.question)) in exclusions.question_hashes:
            excluded_question += 1
            continue
        groups[row.question_type].append(row)

    for question_type, rows in groups.items():
        rows.sort(
            key=lambda row: (
                _rank(
                    salt,
                    question_type,
                    row.image_sha256,
                    row.image_name,
                    _normalize_question(row.question),
                    str(row.row_index),
                ),
                row.row_index,
            )
        )

    selected: list[ChartQACandidate] = []
    used_images: set[str] = set()
    cursors = {value: 0 for value in CHARTQA_TYPES}
    selected_by_type = {value: 0 for value in CHARTQA_TYPES}
    quotas = {
        value: _stratum_quota(per_type, value) for value in CHARTQA_TYPES
    }
    while any(selected_by_type[key] < quotas[key] for key in CHARTQA_TYPES):
        for question_type in CHARTQA_TYPES:
            quota = quotas[question_type]
            if selected_by_type[question_type] >= quota:
                continue
            rows = groups[question_type]
            while True:
                if cursors[question_type] >= len(rows):
                    raise ValueError(
                        f"ChartQA type {question_type!r} lacks {quota} "
                        "eligible image-unique rows"
                    )
                row = rows[cursors[question_type]]
                cursors[question_type] += 1
                if row.image_sha256 in used_images:
                    continue
                selected.append(row)
                used_images.add(row.image_sha256)
                selected_by_type[question_type] += 1
                break
    return selected, {
        "source_rows": len(candidates),
        "eligible_rows_by_type": {
            key: len(value) for key, value in groups.items()
        },
        "selected_rows_by_type": selected_by_type,
        "selected_unique_images": len(used_images),
        "rows_excluded_by_image": excluded_image,
        "rows_excluded_by_question": excluded_question,
    }


def partition_by_stratum(
    rows: Sequence[Any],
    *,
    stratum: Any,
    strata: Sequence[str],
    format_per_stratum: int | Mapping[str, int],
    expected_per_stratum: int | Mapping[str, int],
    salt: str,
) -> tuple[list[Any], list[Any]]:
    format_rows: list[Any] = []
    confirmation_rows: list[Any] = []
    for value in strata:
        group = [row for row in rows if stratum(row) == value]
        expected = _stratum_quota(expected_per_stratum, value)
        format_count = _stratum_quota(format_per_stratum, value)
        if format_count < 0 or format_count > expected:
            raise ValueError(
                f"invalid format quota {format_count} for stratum {value!r}"
            )
        if len(group) != expected:
            raise ValueError(
                f"stratum {value!r} has {len(group)} selected rows, "
                f"expected {expected}"
            )
        ranked = sorted(
            group,
            key=lambda row: (
                _rank(salt, "format-partition", value, _stable_row_id(row)),
                _stable_row_id(row),
            ),
        )
        format_rows.extend(ranked[:format_count])
        confirmation_rows.extend(ranked[format_count:])
    return format_rows, confirmation_rows


def build_data(
    *,
    chartqa_arrow: Path,
    ocrbench_arrow_dir: Path,
    exclusion_paths: Sequence[Path],
    output_root: Path,
    salt: str,
    chartqa_per_type: int = 120,
    chartqa_format_per_type: int = 20,
    chartqa_format_quota: Mapping[str, int] | None = None,
    chartqa_confirmation_quota: Mapping[str, int] | None = None,
    ocrbench_format_quota: Mapping[str, int] | None = None,
    ocrbench_confirmation_quota: Mapping[str, int] | None = None,
    chartqa_benchmark: str = CHARTQA_BENCHMARK,
    ocrbench_benchmark: str = OCRBENCH_BENCHMARK,
    metadata_flag: str = "egor_v9_fresh_confirmation",
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing nonempty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if (chartqa_format_quota is None) != (chartqa_confirmation_quota is None):
        raise ValueError("ChartQA format and confirmation quotas must be paired")
    chart_format_quota = dict(
        {task_type: chartqa_format_per_type for task_type in CHARTQA_TYPES}
        if chartqa_format_quota is None
        else chartqa_format_quota
    )
    chart_confirmation_quota = dict(
        {
            task_type: chartqa_per_type - chartqa_format_per_type
            for task_type in CHARTQA_TYPES
        }
        if chartqa_confirmation_quota is None
        else chartqa_confirmation_quota
    )
    if set(chart_format_quota) != set(CHARTQA_TYPES) or set(
        chart_confirmation_quota
    ) != set(CHARTQA_TYPES):
        raise ValueError("ChartQA quotas must cover the official strata")
    chart_total_quota = {
        task_type: chart_format_quota[task_type]
        + chart_confirmation_quota[task_type]
        for task_type in CHARTQA_TYPES
    }
    ocr_format_quota = dict(
        OCRBENCH_V9_FORMAT_QUOTA
        if ocrbench_format_quota is None
        else ocrbench_format_quota
    )
    ocr_confirmation_quota = dict(
        OCRBENCH_V9_CONFIRMATION_QUOTA
        if ocrbench_confirmation_quota is None
        else ocrbench_confirmation_quota
    )
    if set(ocr_format_quota) != set(ocr_confirmation_quota):
        raise ValueError("OCRBench format and confirmation strata differ")
    ocr_types = tuple(ocr_format_quota)
    ocr_total_quota = {
        task_type: ocr_format_quota[task_type]
        + ocr_confirmation_quota[task_type]
        for task_type in ocr_types
    }
    if not metadata_flag or any(value < 0 for value in ocr_total_quota.values()):
        raise ValueError("invalid reusable split parameters")
    chartqa_format_count = sum(chart_format_quota.values())
    chartqa_confirmation_count = sum(chart_confirmation_quota.values())
    ocr_format_count = sum(ocr_format_quota.values())
    ocr_confirmation_count = sum(ocr_confirmation_quota.values())
    resolved_exclusions = sorted({path.resolve() for path in exclusion_paths})
    for path in resolved_exclusions:
        if not path.is_file():
            raise FileNotFoundError(path)
    exclusions = collect_exclusions(resolved_exclusions)

    chartqa_candidates = _index_chartqa(chartqa_arrow)
    chartqa_selected, chartqa_audit = select_chartqa_rows(
        chartqa_candidates,
        exclusions=exclusions,
        per_type=chart_total_quota,
        salt=f"{salt}:chartqa",
    )
    chartqa_format, chartqa_confirmation = partition_by_stratum(
        chartqa_selected,
        stratum=lambda row: row.question_type,
        strata=CHARTQA_TYPES,
        format_per_stratum=chart_format_quota,
        expected_per_stratum=chart_total_quota,
        salt=f"{salt}:chartqa",
    )
    chartqa_images = _materialize_chartqa(
        chartqa_selected, output_root / "chartqa" / "images"
    )
    chartqa_normalized = {
        row.task_id: _normalize_chartqa(
            row,
            chartqa_images[row.row_index],
            benchmark=chartqa_benchmark,
            metadata_flag=metadata_flag,
        )
        for row in chartqa_selected
    }

    ocrbench_arrows = sorted(ocrbench_arrow_dir.glob("*.arrow"))
    if not ocrbench_arrows:
        raise FileNotFoundError(f"no Arrow shards in {ocrbench_arrow_dir}")
    ocrbench_candidates = _index_ocrbench(ocrbench_arrows)
    ocrbench_selection_candidates, question_audit = (
        filter_prior_questions(
            ocrbench_candidates,
            excluded_question_hashes=exclusions.question_hashes,
        )
    )
    ocrbench_selected, ocrbench_audit = select_transfer_rows_by_type(
        ocrbench_selection_candidates,
        excluded_source_ids=set(exclusions.source_ids),
        excluded_image_hashes=set(exclusions.image_hashes),
        quota_by_type=ocr_total_quota,
        salt=f"{salt}:ocrbench",
    )
    ocrbench_audit = {
        **ocrbench_audit,
        **question_audit,
        "source_rows": len(ocrbench_candidates),
    }
    ocrbench_format, ocrbench_confirmation = partition_by_stratum(
        ocrbench_selected,
        stratum=lambda row: row.task_type,
        strata=ocr_types,
        format_per_stratum=ocr_format_quota,
        expected_per_stratum=ocr_total_quota,
        salt=f"{salt}:ocrbench",
    )
    ocrbench_images = _materialize_ocrbench(
        ocrbench_selected, output_root / "ocrbench" / "images"
    )
    ocrbench_normalized: dict[str, dict[str, Any]] = {}
    for row in ocrbench_selected:
        normalized = normalize_transfer_row(
            row, image_path=ocrbench_images[row.source_id]
        )
        normalized["benchmark"] = ocrbench_benchmark
        normalized["mime_type"] = _mime(ocrbench_images[row.source_id])
        normalized["metadata"][metadata_flag] = True
        normalized["metadata"][
            "primary_type"
        ] = "ocrbench_v2_english_vqa"
        ocrbench_normalized[row.task_id] = normalized

    task_paths = {
        "chartqa": _write_source_tasks(
            output_root / "chartqa",
            chartqa_format,
            chartqa_confirmation,
            chartqa_normalized,
            expected_format=chartqa_format_count,
            expected_confirmation=chartqa_confirmation_count,
        ),
        "ocrbench": _write_source_tasks(
            output_root / "ocrbench",
            ocrbench_format,
            ocrbench_confirmation,
            ocrbench_normalized,
            expected_format=ocr_format_count,
            expected_confirmation=ocr_confirmation_count,
        ),
    }
    _assert_disjoint(task_paths)
    image_paths = sorted(
        list(chartqa_images.values()) + list(ocrbench_images.values())
    )
    image_sums = output_root / "IMAGE_SHA256SUMS"
    image_sums.write_text(
        "".join(f"{sha256(path)}  {path}\n" for path in image_paths),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "outcome_blind_task_and_image_selection",
        "salt": salt,
        "selection_policy": (
            "exclude every recorded prior source id, normalized question, "
            "image basename, and image digest; rank only source identifiers, "
            "question text, official stratum, and image identity; labels, "
            "reference answers, model outputs, and scores never affect selection"
        ),
        "outcome_fields_used_for_selection": False,
        "exclusions": {
            "files": [
                {"path": str(path), "sha256": sha256(path)}
                for path in resolved_exclusions
            ],
            "source_ids": len(exclusions.source_ids),
            "image_hashes": len(exclusions.image_hashes),
            "image_basenames": len(exclusions.image_basenames),
            "question_hashes": len(exclusions.question_hashes),
        },
        "sources": {
            "chartqa": {
                "official_dataset": "ahmed-masry/ChartQA",
                "source_arrow": _file_receipt(chartqa_arrow),
                "format": _task_receipt(task_paths["chartqa"]["format"]),
                "confirmation": _task_receipt(
                    task_paths["chartqa"]["confirmation"]
                ),
                "strata": list(CHARTQA_TYPES),
                **chartqa_audit,
            },
            "ocrbench": {
                "official_dataset": "lmms-lab/OCRBench-v2",
                "source_arrows": [
                    _file_receipt(path) for path in ocrbench_arrows
                ],
                "format": _task_receipt(task_paths["ocrbench"]["format"]),
                "confirmation": _task_receipt(
                    task_paths["ocrbench"]["confirmation"]
                ),
                "strata": list(ocr_types),
                "stratum_quotas": {
                    task_type: {
                        "format": ocr_format_quota[task_type],
                        "confirmation": ocr_confirmation_quota[task_type],
                    }
                    for task_type in ocr_types
                },
                "official_evaluator": {
                    "repository": OCRBENCH_OFFICIAL_EVAL_REPOSITORY,
                    "commit": OCRBENCH_OFFICIAL_EVAL_COMMIT,
                    "metric_function": "vqa_evaluation",
                    "selection_group": "English types dispatched to vqa_evaluation",
                },
                **ocrbench_audit,
            },
        },
        "image_files": len(image_paths),
        "image_sha256sums": _file_receipt(image_sums),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_inputs = sorted(
        [
            manifest_path,
            image_sums,
            *(path for paths in task_paths.values() for path in paths.values()),
        ]
    )
    (output_root / "SPLIT_SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path}\n" for path in checksum_inputs),
        encoding="utf-8",
    )
    (output_root / "SPLIT_FROZEN").touch()
    return manifest


def _index_chartqa(path: Path) -> list[ChartQACandidate]:
    Dataset, Image = _datasets()
    dataset = Dataset.from_file(str(path)).cast_column("image", Image(decode=False))
    output: list[ChartQACandidate] = []
    for index in range(len(dataset)):
        row = dataset[index]
        payload = _image_bytes(row.get("image"), base_dir=path.parent)
        question = str(row.get("query", "")).strip()
        answer = str(row.get("label", "")).strip()
        question_type = str(row.get("type", "")).strip()
        image_name = Path(str(row.get("imgname", ""))).name
        if not all((question, answer, question_type, image_name, payload)):
            raise ValueError(f"invalid ChartQA row {index}")
        output.append(
            ChartQACandidate(
                row_index=index,
                image_name=image_name,
                image_sha256=hashlib.sha256(payload).hexdigest(),
                image_bytes=payload,
                question=question,
                answer=answer,
                question_type=question_type,
            )
        )
    return output


def _index_ocrbench(paths: Sequence[Path]) -> list[OCRBenchCandidate]:
    Dataset, Image = _datasets()
    output: list[OCRBenchCandidate] = []
    for path in paths:
        dataset = Dataset.from_file(str(path)).cast_column("image", Image(decode=False))
        for row_index in range(len(dataset)):
            row = dataset[row_index]
            payload = _image_bytes(row.get("image"), base_dir=path.parent)
            raw_answers = row.get("answers", [])
            answers = (
                tuple(str(value).strip() for value in raw_answers if str(value).strip())
                if isinstance(raw_answers, list)
                else ()
            )
            candidate = OCRBenchCandidate(
                source_id=int(row["id"]),
                question=str(row.get("question", "")).strip(),
                answers=answers,
                dataset_name=str(row.get("dataset_name", "")).strip(),
                task_type=str(row.get("type", "")).strip(),
                image_sha256=hashlib.sha256(payload).hexdigest(),
                shard_path=str(path),
                row_index=row_index,
            )
            if not all(
                (candidate.question, candidate.answers, candidate.dataset_name, candidate.task_type)
            ):
                raise ValueError(f"invalid OCRBench row {candidate.source_id}")
            output.append(candidate)
    return output


def _materialize_chartqa(
    rows: Sequence[ChartQACandidate], directory: Path
) -> dict[int, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[int, Path] = {}
    for row in rows:
        suffix = Path(row.image_name).suffix.lower() or _image_suffix(row.image_bytes)
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(row.image_name).stem)
        path = directory / f"{row.image_sha256[:16]}_{safe_stem}{suffix}"
        path.write_bytes(row.image_bytes)
        outputs[row.row_index] = path
    return outputs


def _materialize_ocrbench(
    rows: Sequence[OCRBenchCandidate], directory: Path
) -> dict[int, Path]:
    Dataset, Image = _datasets()
    directory.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[OCRBenchCandidate]] = defaultdict(list)
    for row in rows:
        grouped[row.shard_path].append(row)
    outputs: dict[int, Path] = {}
    for shard, values in grouped.items():
        source = Path(shard)
        dataset = Dataset.from_file(str(source)).cast_column("image", Image(decode=False))
        for row in values:
            payload = _image_bytes(dataset[row.row_index].get("image"), base_dir=source.parent)
            if hashlib.sha256(payload).hexdigest() != row.image_sha256:
                raise AssertionError(f"OCRBench image changed for {row.source_id}")
            path = directory / f"{row.task_id}{_image_suffix(payload)}"
            path.write_bytes(payload)
            outputs[row.source_id] = path
    return outputs


def _normalize_chartqa(
    row: ChartQACandidate,
    image_path: Path,
    *,
    benchmark: str = CHARTQA_BENCHMARK,
    metadata_flag: str = "egor_v9_fresh_confirmation",
) -> dict[str, Any]:
    return {
        "task_id": row.task_id,
        "benchmark": benchmark,
        "question": row.question,
        "answer": row.answer,
        "reference_answers": [row.answer],
        "metric": "relaxed_accuracy",
        "question_type": "qa",
        "mime_type": _mime(image_path),
        "image_path": str(image_path.resolve()),
        "metadata": {
            "primary_type": "chartqa",
            "source_split": "test",
            "source_index": row.row_index,
            "source_image": row.image_name,
            "source_image_sha256": row.image_sha256,
            "source_subset": row.question_type,
            metadata_flag: True,
            "uncertainty": "medium",
            "complexity": "high",
            "interdependence": "reciprocal",
        },
    }


def _write_source_tasks(
    root: Path,
    format_rows: Sequence[Any],
    confirmation_rows: Sequence[Any],
    normalized: Mapping[str, Mapping[str, Any]],
    *,
    expected_format: int = 40,
    expected_confirmation: int = 200,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "format": root / f"format{expected_format}.jsonl",
        "confirmation": root / f"confirmation{expected_confirmation}.jsonl",
    }
    write_jsonl(paths["format"], [normalized[_stable_row_id(row)] for row in format_rows])
    write_jsonl(
        paths["confirmation"],
        [normalized[_stable_row_id(row)] for row in confirmation_rows],
    )
    if sum(1 for _ in paths["format"].open(encoding="utf-8")) != expected_format:
        raise AssertionError("format denominator mismatch")
    if (
        sum(1 for _ in paths["confirmation"].open(encoding="utf-8"))
        != expected_confirmation
    ):
        raise AssertionError("confirmation denominator mismatch")
    return paths


def _assert_disjoint(task_paths: Mapping[str, Mapping[str, Path]]) -> None:
    seen_tasks: set[tuple[str, str]] = set()
    seen_images: set[str] = set()
    for paths in task_paths.values():
        for path in paths.values():
            for row in _json_rows(path):
                key = (str(row.get("benchmark", "")), str(row.get("task_id", "")))
                image = str(row.get("metadata", {}).get("source_image_sha256", ""))
                if not all(key) or key in seen_tasks or not image or image in seen_images:
                    raise ValueError("fresh format/confirmation data are not disjoint")
                seen_tasks.add(key)
                seen_images.add(image)


def _task_receipt(path: Path) -> dict[str, Any]:
    rows = _json_rows(path)
    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "sha256": sha256(path),
        "task_ids_sha256": _hash_lines(sorted(str(row["task_id"]) for row in rows)),
        "image_hashes_sha256": _hash_lines(
            sorted(str(row["metadata"]["source_image_sha256"]) for row in rows)
        ),
    }


def _file_receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def _json_rows(path: Path) -> list[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("data", "rows", "examples", "items"):
            values = payload.get(key)
            if isinstance(values, list):
                return [row for row in values if isinstance(row, Mapping)]
    return []


def _datasets() -> tuple[Any, Any]:
    try:
        from datasets import Dataset, Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("data extraction requires the datasets package") from exc
    return Dataset, Image


def _image_bytes(value: Any, *, base_dir: Path) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, Mapping):
        payload = value.get("bytes")
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        raw_path = value.get("path")
        if isinstance(raw_path, str) and raw_path:
            path = Path(raw_path)
            if not path.is_absolute():
                path = base_dir / path
            return path.read_bytes()
    raise ValueError("image is neither encoded bytes nor a readable path")


def _image_suffix(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return ".webp"
    return ".img"


def _mime(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")


def _stable_row_id(row: Any) -> str:
    return str(row.task_id)


def _stratum_quota(value: int | Mapping[str, int], stratum: str) -> int:
    if isinstance(value, Mapping):
        if stratum not in value:
            raise ValueError(f"missing quota for stratum {stratum!r}")
        quota = int(value[stratum])
    else:
        quota = int(value)
    if quota < 0:
        raise ValueError(f"negative quota for stratum {stratum!r}")
    return quota


def _normalize_question(value: str) -> str:
    return " ".join(value.split()).strip()


def _rank(salt: str, *visible_values: str) -> str:
    return hashlib.sha256("\0".join((salt, *visible_values)).encode()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _hash_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode()).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and set(value) <= set("0123456789abcdef")


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chartqa-arrow", required=True)
    parser.add_argument("--ocrbench-arrow-dir", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = build_data(
        chartqa_arrow=Path(args.chartqa_arrow).resolve(),
        ocrbench_arrow_dir=Path(args.ocrbench_arrow_dir).resolve(),
        exclusion_paths=[Path(value).resolve() for value in args.exclude],
        output_root=Path(args.output_root).resolve(),
        salt=args.salt,
    )
    print(
        json.dumps(
            {
                source: {
                    split: manifest["sources"][source][split]["sha256"]
                    for split in ("format", "confirmation")
                }
                for source in ("chartqa", "ocrbench")
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
