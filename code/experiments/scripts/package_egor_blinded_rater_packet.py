#!/usr/bin/env python3
"""Package a deterministic, rater-only EGOR evidence-audit release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


COMMIT_RE = re.compile(r"[0-9a-f]{40}")
BLIND_ID_RE = re.compile(r"[0-9a-f]{64}")
WINDOWS_PATH_RE = re.compile(r"(?i)(?:^|[\s\"'(])[a-z]:[\\/]")
HOME_PATH_RE = re.compile(r"(?:^|[\s\"'(])/(?:home|users|mnt)/", re.IGNORECASE)
LABEL_FIELDS = (
    "support_observable",
    "card_claim_supported",
    "change_supported",
    "change_material",
    "failure_mode",
    "notes",
)
RATING_COLUMNS = ("release_id", "rater_id", "blind_item_id", *LABEL_FIELDS)
RELEASE_SCHEMA_VERSION = 2
RELEASE_DOMAIN = "openmas-egor-blinded-rater-release-v2"
ITEM_KEYS = {
    "schema_version",
    "blind_item_id",
    "visible_question",
    "image_file",
    "image_sha256",
    "baseline_answer",
    "candidate_answer",
    "cited_evidence_ids",
    "cited_cards",
}
CARD_KEYS = {
    "evidence_id",
    "type",
    "claim",
    "support",
    "provenance",
    "producer_id",
    "confidence",
}
FORBIDDEN_KEYS = {
    "answer_key",
    "automatic_outcome",
    "baseline_quality",
    "control_id",
    "gold",
    "gold_answer",
    "model",
    "model_id",
    "policy_id",
    "policy_quality",
    "quality_delta",
    "reference",
    "references",
    "selector_score",
    "source_id",
    "strict_harm",
    "task_id",
    "useful_repair",
}
FIXED_ZIP_TIME = (2026, 7, 13, 0, 0, 0)


def main() -> int:
    args = _parse_args()
    report = package_rater_packet(
        audit_dir=Path(args.audit_dir).resolve(),
        rater_id=args.rater_id,
        output=Path(args.output).resolve(),
        manifest_output=Path(args.manifest_output).resolve(),
        expected_dataset_id=args.expected_dataset_id,
        expected_policy_id=args.expected_policy_id,
        expected_tasks=args.expected_tasks,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def package_rater_packet(
    *,
    audit_dir: Path,
    rater_id: str,
    output: Path,
    manifest_output: Path,
    expected_dataset_id: str,
    expected_policy_id: str,
    expected_tasks: int,
) -> dict[str, Any]:
    if rater_id not in {"a", "b"}:
        raise ValueError("rater_id must be a or b")
    if not expected_dataset_id.strip() or not expected_policy_id.strip():
        raise ValueError("expected dataset and policy identifiers are required")
    if expected_tasks <= 0:
        raise ValueError("expected_tasks must be positive")
    if output == manifest_output:
        raise ValueError("archive and sidecar manifest outputs must differ")

    population_manifest_path = audit_dir / "population_manifest.json"
    population_manifest = _load_object(population_manifest_path)
    generation_commit, analysis_commit = _validate_population_manifest(
        population_manifest,
        expected_dataset_id=expected_dataset_id,
        expected_policy_id=expected_policy_id,
        expected_tasks=expected_tasks,
    )
    blinded_path = audit_dir / "blinded_items.jsonl"
    template_path = audit_dir / f"annotation_template_rater_{rater_id}.csv"
    _require_bound_output(population_manifest, blinded_path)
    _require_bound_output(population_manifest, template_path)

    items = _load_and_validate_items(blinded_path, audit_dir=audit_dir)
    item_ids = [str(item["blind_item_id"]) for item in items]
    _validate_blank_template(template_path, expected_ids=item_ids)
    population_manifest_sha256 = _sha256(population_manifest_path)
    release_id = _release_id(
        rater_id=rater_id,
        dataset_id=expected_dataset_id,
        policy_id=expected_policy_id,
        generation_commit=generation_commit,
        analysis_commit=analysis_commit,
        task_population=expected_tasks,
        audit_item_count=len(items),
        population_manifest_sha256=population_manifest_sha256,
    )

    files: dict[str, bytes] = {
        "RATER_GUIDE.md": _render_guide(
            rater_id=rater_id,
            release_id=release_id,
            item_count=len(items),
        ).encode("utf-8"),
        "items.html": _render_html(items).encode("utf-8"),
        "blinded_items.jsonl": blinded_path.read_bytes(),
        "ratings.csv": _render_rating_template(
            rater_id=rater_id,
            release_id=release_id,
            item_ids=item_ids,
        ),
    }
    for item in items:
        relative = str(item["image_file"])
        image_path = (audit_dir / relative).resolve()
        _require_bound_output(population_manifest, image_path)
        files[relative] = image_path.read_bytes()

    archive_a = _build_zip(files)
    archive_b = _build_zip(files)
    if archive_a != archive_b:
        raise RuntimeError("rater packet is not byte deterministic")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(archive_a)
    temporary.replace(output)

    report = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "audience": f"independent blinded rater {rater_id.upper()}",
        "rater_id": rater_id,
        "release_id": release_id,
        "rating_columns": list(RATING_COLUMNS),
        "dataset_id": expected_dataset_id,
        "policy_id": expected_policy_id,
        "generation_source_commit": generation_commit,
        "packet_analysis_commit": analysis_commit,
        "task_population": expected_tasks,
        "audit_item_count": len(items),
        "population_manifest_sha256": population_manifest_sha256,
        "archive_sha256": hashlib.sha256(archive_a).hexdigest(),
        "archive_bytes": len(archive_a),
        "deterministic": True,
        "rater_archive_excludes": [
            "gold answers and references",
            "automatic outcomes and quality deltas",
            "source and task identifiers",
            "policy, control, and model identities",
            "population index and selected author traces",
            "the other rater's template and adjudication template",
        ],
        "files": [
            {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
            for name, payload in sorted(files.items())
        ],
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest_output.with_suffix(manifest_output.suffix + ".tmp")
    manifest_tmp.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_tmp.replace(manifest_output)
    return report


def _validate_population_manifest(
    manifest: dict[str, Any],
    *,
    expected_dataset_id: str,
    expected_policy_id: str,
    expected_tasks: int,
) -> tuple[str, str]:
    if (
        int(manifest.get("schema_version", -1)) != 1
        or manifest.get("dataset_id") != expected_dataset_id
        or manifest.get("policy_id") != expected_policy_id
        or int(manifest.get("task_count", -1)) != expected_tasks
    ):
        raise ValueError("population manifest provenance or task count mismatch")
    generation_commit = str(manifest.get("source_commit", ""))
    analysis_commit = str(manifest.get("analysis_commit", ""))
    if COMMIT_RE.fullmatch(generation_commit) is None:
        raise ValueError("population manifest has invalid generation commit")
    if COMMIT_RE.fullmatch(analysis_commit) is None:
        raise ValueError("population manifest has invalid analysis commit")
    guarantee = manifest.get("blinding_guarantee")
    if not isinstance(guarantee, dict) or not all(
        guarantee.get(key) is True
        for key in (
            "population_index_is_not_rater_facing",
            "certificate_rejections_are_not_rater_facing",
        )
    ):
        raise ValueError("population manifest lacks the frozen blinding guarantee")
    return generation_commit, analysis_commit


def _load_and_validate_items(path: Path, *, audit_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict) or set(raw) != ITEM_KEYS:
                raise ValueError(f"blinded item schema mismatch on line {line_number}")
            _reject_forbidden_keys(raw)
            blind_id = str(raw.get("blind_item_id", ""))
            if BLIND_ID_RE.fullmatch(blind_id) is None:
                raise ValueError(f"invalid blind item ID on line {line_number}")
            if int(raw.get("schema_version", -1)) != 1:
                raise ValueError(f"invalid blinded schema on line {line_number}")
            relative = str(raw.get("image_file", ""))
            posix_image = PurePosixPath(relative)
            suffix = posix_image.suffix.lower()
            if (
                posix_image.is_absolute()
                or ".." in posix_image.parts
                or posix_image.parent != PurePosixPath("images")
                or posix_image.stem != blind_id
                or suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            ):
                raise ValueError(f"unsafe or mismatched image path: {relative}")
            image_path = (audit_dir / relative).resolve()
            try:
                image_path.relative_to(audit_dir.resolve())
            except ValueError as exc:
                raise ValueError(f"image escapes audit directory: {relative}") from exc
            if not image_path.is_file() or _sha256(image_path) != raw.get(
                "image_sha256"
            ):
                raise ValueError(f"blinded image hash mismatch: {relative}")
            if not isinstance(raw.get("cited_evidence_ids"), list) or not isinstance(
                raw.get("cited_cards"), list
            ):
                raise ValueError("blinded evidence fields must be lists")
            cited_ids = raw["cited_evidence_ids"]
            cards = raw["cited_cards"]
            if (
                not cited_ids
                or not cards
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in cited_ids
                )
                or any(
                    not isinstance(card, dict) or set(card) != CARD_KEYS
                    for card in cards
                )
                or [str(card["evidence_id"]) for card in cards] != cited_ids
            ):
                raise ValueError("blinded cited-card population is malformed")
            if any(
                not isinstance(raw.get(key), str) or not str(raw.get(key)).strip()
                for key in ("visible_question", "baseline_answer", "candidate_answer")
            ):
                raise ValueError(
                    "blinded question and answers must be nonempty strings"
                )
            _reject_absolute_paths(raw)
            items.append(raw)
    ids = [str(item["blind_item_id"]) for item in items]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("blinded items must have unique hash-sorted IDs")
    return items


def _validate_blank_template(path: Path, *, expected_ids: Sequence[str]) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("blind_item_id", *LABEL_FIELDS):
            raise ValueError("rater template header mismatch")
        rows = list(reader)
    ids = [str(row.get("blind_item_id", "")) for row in rows]
    if ids != list(expected_ids):
        raise ValueError("rater template item IDs do not match blinded items")
    if any(str(row.get(field, "")).strip() for row in rows for field in LABEL_FIELDS):
        raise ValueError("refusing to package a prefilled rater template")


def _render_rating_template(
    *, rater_id: str, release_id: str, item_ids: Sequence[str]
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=RATING_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    for blind_id in item_ids:
        writer.writerow(
            {
                "release_id": release_id,
                "rater_id": rater_id,
                "blind_item_id": blind_id,
            }
        )
    return buffer.getvalue().encode("utf-8")


def _release_id(
    *,
    rater_id: str,
    dataset_id: str,
    policy_id: str,
    generation_commit: str,
    analysis_commit: str,
    task_population: int,
    audit_item_count: int,
    population_manifest_sha256: str,
) -> str:
    payload = {
        "domain": RELEASE_DOMAIN,
        "rater_id": rater_id,
        "dataset_id": dataset_id,
        "policy_id": policy_id,
        "generation_source_commit": generation_commit,
        "packet_analysis_commit": analysis_commit,
        "task_population": task_population,
        "audit_item_count": audit_item_count,
        "population_manifest_sha256": population_manifest_sha256,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _require_bound_output(manifest: dict[str, Any], path: Path) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("population manifest has no outputs")
    actual = _sha256(path) if path.is_file() else ""
    if not any(
        isinstance(entry, dict)
        and Path(str(entry.get("path", ""))).resolve() == path.resolve()
        and entry.get("sha256") == actual
        for entry in outputs
    ):
        raise ValueError(f"file is absent from bound audit outputs: {path}")


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).strip().lower() in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden rater-facing field: {key}")
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)


def _reject_absolute_paths(value: Any) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_absolute_paths(child)
    elif isinstance(value, list):
        for child in value:
            _reject_absolute_paths(child)
    elif isinstance(value, str) and (
        WINDOWS_PATH_RE.search(value) or HOME_PATH_RE.search(value)
    ):
        raise ValueError("absolute local path leaked into rater-facing content")


def _render_guide(*, rater_id: str, release_id: str, item_count: int) -> str:
    return f"""# Blinded Evidence-Alignment Rating Guide

Role: independent rater {rater_id.upper()}
Release ID: {release_id}
Items: {item_count}

Open `items.html` locally and enter one complete row per item in `ratings.csv`.
Do not change `release_id`, `rater_id`, `blind_item_id`, row order, or column
names. Work independently:
do not discuss labels, inspect project result directories, search for reference
answers, or consult the other rater before both files are returned.

Judge only the supplied image, visible question, two answers, and cited cards:

- `support_observable`: `yes`, `no`, or `unclear` -- can the cited raw support
  be checked in the supplied image or shown trace?
- `card_claim_supported`: `yes`, `no`, or `unclear` -- does that support justify
  the card claim without relying on the candidate answer?
- `change_supported`: `yes`, `no`, or `unclear` -- does at least one cited card
  materially justify changing the baseline answer to the candidate answer?
- `change_material`: `yes` or `no` -- is the change semantic rather than only
  punctuation, casing, spacing, or formatting?
- `failure_mode`: `none`, `false evidence`, `irrelevant evidence`,
  `unsupported inference`, `provenance too vague`, or `other`.
- `notes`: concise rationale; required whenever any label is `no`, `unclear`,
  or `other`.

Use `unclear` when the supplied packet is insufficient. Do not infer missing
evidence from model reputation or likely benchmark answers. Return only the
completed `ratings.csv` to the coordinator.
"""


def _render_html(items: Sequence[Mapping[str, Any]]) -> str:
    sections = []
    for index, item in enumerate(items, start=1):
        cards = json.dumps(item["cited_cards"], ensure_ascii=False, indent=2)
        cited_ids = ", ".join(str(value) for value in item["cited_evidence_ids"])
        sections.append(
            f"""<section>
<h2>Item {index} of {len(items)}</h2>
<p><code>{html.escape(str(item["blind_item_id"]))}</code></p>
<img src="{html.escape(str(item["image_file"]), quote=True)}" alt="Task image">
<h3>Visible question</h3><pre>{html.escape(str(item["visible_question"]))}</pre>
<h3>Baseline answer</h3><pre>{html.escape(str(item["baseline_answer"]))}</pre>
<h3>Candidate answer</h3><pre>{html.escape(str(item["candidate_answer"]))}</pre>
<h3>Cited evidence IDs</h3><pre>{html.escape(cited_ids)}</pre>
<h3>Cited cards</h3><pre>{html.escape(cards)}</pre>
</section>"""
        )
    body = "\n".join(sections) or "<p>No accepted changed items exist.</p>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Blinded Evidence Audit</title>
<style>body{{font:16px/1.45 sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem}}
section{{border-top:3px solid #333;margin:3rem 0;padding-top:1rem}}
img{{max-width:100%;max-height:720px;border:1px solid #aaa}}
pre{{white-space:pre-wrap;background:#f5f5f5;padding:.8rem;overflow-wrap:anywhere}}
code{{overflow-wrap:anywhere}}</style></head><body>
<h1>Blinded Evidence-Alignment Items</h1>
<p>Use the separate rating guide and record labels in <code>ratings.csv</code>.</p>
{body}
</body></html>
"""


def _build_zip(files: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, payload in sorted(files.items()):
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive path: {name}")
            info = zipfile.ZipInfo(str(relative), date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload, compresslevel=9)
    return buffer.getvalue()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--rater-id", choices=("a", "b"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--expected-dataset-id", default="chartqa_prospective_v1")
    parser.add_argument("--expected-policy-id", default="frozen_egor_selector")
    parser.add_argument("--expected-tasks", type=int, default=240)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
