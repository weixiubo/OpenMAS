#!/usr/bin/env python3
"""Analyze two blinded evidence-alignment ratings plus adjudication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any


MINIMUM_SUPPORT_RATE = 0.90
MAXIMUM_CLEARLY_UNSUPPORTED_RATE = 0.05
MINIMUM_RAW_AGREEMENT = 0.80
KAPPA_WARNING_BELOW = 0.60
RELEASE_SCHEMA_VERSION = 2
RELEASE_DOMAIN = "openmas-egor-blinded-rater-release-v2"
HASH_RE = re.compile(r"[0-9a-f]{64}")
LABEL_DOMAINS = {
    "support_observable": {"yes", "no", "unclear"},
    "card_claim_supported": {"yes", "no", "unclear"},
    "change_supported": {"yes", "no", "unclear"},
    "change_material": {"yes", "no"},
    "failure_mode": {
        "none",
        "false evidence",
        "irrelevant evidence",
        "unsupported inference",
        "provenance too vague",
        "other",
    },
}
LABEL_FIELDS = (*LABEL_DOMAINS, "notes")
RATING_COLUMNS = ("release_id", "rater_id", "blind_item_id", *LABEL_FIELDS)
ADJUDICATION_COLUMNS = ("blind_item_id", *LABEL_FIELDS)
REQUIRED_RATER_EXCLUSIONS = {
    "gold answers and references",
    "automatic outcomes and quality deltas",
    "source and task identifiers",
    "policy, control, and model identities",
    "population index and selected author traces",
    "the other rater's template and adjudication template",
}


def main() -> int:
    args = _parse_args()
    rating_analysis_commit = _exact_commit(
        args.rating_analysis_commit, name="rating analysis"
    )
    manifest_path = Path(args.population_manifest).resolve()
    index_path = Path(args.population_index).resolve()
    rater_a_path = Path(args.rater_a).resolve()
    rater_b_path = Path(args.rater_b).resolve()
    rater_a_manifest_path = Path(args.rater_a_release_manifest).resolve()
    rater_b_manifest_path = Path(args.rater_b_release_manifest).resolve()
    rater_a_archive_path = Path(args.rater_a_archive).resolve()
    rater_b_archive_path = Path(args.rater_b_archive).resolve()
    adjudication_path = Path(args.adjudication).resolve()
    output_path = Path(args.output).resolve()
    manifest = _load_object(manifest_path)
    index_rows = _load_index(index_path)
    accepted_ids = sorted(
        blind_id
        for blind_id, row in index_rows.items()
        if row["population"] == "accepted_changed"
    )
    rejected_ids = sorted(
        blind_id
        for blind_id, row in index_rows.items()
        if row["population"] == "certificate_rejected"
    )
    _validate_population_manifest(
        manifest,
        index_path=index_path,
        accepted_count=len(accepted_ids),
        rejected_count=len(rejected_ids),
        expected_dataset_id=args.expected_dataset_id,
        expected_policy_id=args.expected_policy_id,
    )
    source_commit = _exact_commit(manifest["source_commit"], name="generation source")
    packet_analysis_commit = _exact_commit(
        manifest["analysis_commit"], name="packet analysis"
    )
    task_population = int(manifest["task_count"])
    population_manifest_sha256 = _sha256(manifest_path)
    rater_releases = {
        "a": _validate_rater_release(
            manifest_path=rater_a_manifest_path,
            archive_path=rater_a_archive_path,
            expected_rater_id="a",
            expected_dataset_id=args.expected_dataset_id,
            expected_policy_id=args.expected_policy_id,
            generation_commit=source_commit,
            analysis_commit=packet_analysis_commit,
            task_population=task_population,
            audit_item_count=len(accepted_ids),
            population_manifest_sha256=population_manifest_sha256,
            expected_ids=accepted_ids,
        ),
        "b": _validate_rater_release(
            manifest_path=rater_b_manifest_path,
            archive_path=rater_b_archive_path,
            expected_rater_id="b",
            expected_dataset_id=args.expected_dataset_id,
            expected_policy_id=args.expected_policy_id,
            generation_commit=source_commit,
            analysis_commit=packet_analysis_commit,
            task_population=task_population,
            audit_item_count=len(accepted_ids),
            population_manifest_sha256=population_manifest_sha256,
            expected_ids=accepted_ids,
        ),
    }
    if rater_releases["a"]["release_id"] == rater_releases["b"]["release_id"]:
        raise ValueError("rater A and B release IDs must differ")
    if rater_releases["a"]["archive_sha256"] == rater_releases["b"]["archive_sha256"]:
        raise ValueError("rater-specific archives unexpectedly have the same hash")
    rater_a = _load_labels(
        rater_a_path,
        expected_ids=accepted_ids,
        expected_release_id=rater_releases["a"]["release_id"],
        expected_rater_id="a",
    )
    rater_b = _load_labels(
        rater_b_path,
        expected_ids=accepted_ids,
        expected_release_id=rater_releases["b"]["release_id"],
        expected_rater_id="b",
    )
    adjudicated = _load_labels(adjudication_path, expected_ids=accepted_ids)
    adjudication_audit = _audit_adjudication(
        rater_a,
        rater_b,
        adjudicated,
        blind_ids=accepted_ids,
    )
    agreements = {
        field: _agreement_report(
            [rater_a[blind_id][field] for blind_id in accepted_ids],
            [rater_b[blind_id][field] for blind_id in accepted_ids],
        )
        for field in LABEL_DOMAINS
    }
    material_ids = [
        blind_id
        for blind_id in accepted_ids
        if adjudicated[blind_id]["change_material"] == "yes"
    ]
    support_counts = Counter(
        adjudicated[blind_id]["change_supported"] for blind_id in material_ids
    )
    material_count = len(material_ids)
    support_yes = support_counts["yes"]
    support_no = support_counts["no"]
    support_rate = support_yes / material_count if material_count else 0.0
    clearly_unsupported_rate = support_no / material_count if material_count else 0.0
    support_low, support_high = _wilson_interval(support_yes, material_count)
    raw_change_agreement = agreements["change_supported"]["raw_agreement"]
    strong_gate = (
        material_count > 0
        and support_rate >= MINIMUM_SUPPORT_RATE
        and clearly_unsupported_rate <= MAXIMUM_CLEARLY_UNSUPPORTED_RATE
        and raw_change_agreement >= MINIMUM_RAW_AGREEMENT
    )
    kappa = agreements["change_supported"]["cohens_kappa"]
    report = {
        "schema_version": 2,
        "source_commit": source_commit,
        "analysis_commit": packet_analysis_commit,
        "rating_analysis_commit": rating_analysis_commit,
        "dataset_id": str(manifest["dataset_id"]),
        "policy_id": str(manifest["policy_id"]),
        "population": {
            "accepted_changed": len(accepted_ids),
            "certificate_rejected": len(rejected_ids),
            "accepted_material_changes": material_count,
            "accepted_nonmaterial_changes": len(accepted_ids) - material_count,
        },
        "primary_endpoint": {
            "adjudicated_supported": support_yes,
            "adjudicated_clearly_unsupported": support_no,
            "adjudicated_unclear": support_counts["unclear"],
            "support_rate": support_rate,
            "support_wilson95_low": support_low,
            "support_wilson95_high": support_high,
            "clearly_unsupported_rate": clearly_unsupported_rate,
            "unclear_treated_as_unsupported": True,
        },
        "interrater": agreements,
        "adjudication_integrity": adjudication_audit,
        "rater_releases": {
            rater_id: {
                "release_id": release["release_id"],
                "release_manifest_sha256": release["manifest_sha256"],
                "archive_sha256": release["archive_sha256"],
                "returned_ratings_sha256": _sha256(
                    rater_a_path if rater_id == "a" else rater_b_path
                ),
            }
            for rater_id, release in sorted(rater_releases.items())
        },
        "adjudicated_label_counts_accepted_material": {
            field: dict(
                sorted(
                    Counter(
                        adjudicated[blind_id][field] for blind_id in material_ids
                    ).items()
                )
            )
            for field in LABEL_DOMAINS
        },
        "strong_certificate_gate": {
            "minimum_support_rate": MINIMUM_SUPPORT_RATE,
            "maximum_clearly_unsupported_rate": MAXIMUM_CLEARLY_UNSUPPORTED_RATE,
            "minimum_raw_change_supported_agreement": MINIMUM_RAW_AGREEMENT,
            "kappa_warning_below": KAPPA_WARNING_BELOW,
            "passed": strong_gate,
            "kappa_warning": kappa is not None and kappa < KAPPA_WARNING_BELOW,
            "kappa_undefined_constant_labels": kappa is None,
        },
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (
                manifest_path,
                index_path,
                rater_a_manifest_path,
                rater_b_manifest_path,
                rater_a_archive_path,
                rater_b_archive_path,
                rater_a_path,
                rater_b_path,
                adjudication_path,
            )
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"analyzed {len(accepted_ids)} accepted changes; "
        f"strong certificate gate passed={strong_gate}"
    )
    return 0


def _agreement_report(first: Sequence[str], second: Sequence[str]) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("rater vectors differ in length")
    if not first:
        return {"items": 0, "raw_agreement": 0.0, "cohens_kappa": None}
    observed = sum(
        left == right for left, right in zip(first, second, strict=True)
    ) / len(first)
    categories = set(first) | set(second)
    expected = sum(
        (first.count(category) / len(first)) * (second.count(category) / len(second))
        for category in categories
    )
    kappa = (
        None
        if math.isclose(expected, 1.0)
        else (observed - expected) / (1.0 - expected)
    )
    return {"items": len(first), "raw_agreement": observed, "cohens_kappa": kappa}


def _audit_adjudication(
    rater_a: dict[str, dict[str, str]],
    rater_b: dict[str, dict[str, str]],
    adjudicated: dict[str, dict[str, str]],
    *,
    blind_ids: Sequence[str],
) -> dict[str, Any]:
    disagreements: dict[str, int] = {}
    for field in LABEL_DOMAINS:
        disagreements[field] = 0
        for blind_id in blind_ids:
            left = rater_a[blind_id][field]
            right = rater_b[blind_id][field]
            final = adjudicated[blind_id][field]
            if left == right:
                if final != left:
                    raise ValueError(
                        "adjudication changed a consensus label: "
                        f"{blind_id}:{field}:{left}->{final}"
                    )
            else:
                disagreements[field] += 1
    return {
        "consensus_labels_preserved": True,
        "disagreements_by_field": disagreements,
        "total_disagreements": sum(disagreements.values()),
    }


def _wilson_interval(successes: int, count: int) -> tuple[float, float]:
    if count <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def _load_index(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            blind_id = str(row.get("blind_item_id", "")).strip()
            population = str(row.get("population", "")).strip()
            if not blind_id or blind_id in rows:
                raise ValueError(f"invalid or duplicate blind_item_id: {blind_id!r}")
            if population not in {"accepted_changed", "certificate_rejected"}:
                raise ValueError(f"invalid population for {blind_id}: {population!r}")
            rows[blind_id] = row
    return rows


def _validate_population_manifest(
    manifest: dict[str, Any],
    *,
    index_path: Path,
    accepted_count: int,
    rejected_count: int,
    expected_dataset_id: str,
    expected_policy_id: str,
) -> None:
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("population manifest schema is invalid")
    if manifest.get("dataset_id") != expected_dataset_id:
        raise ValueError("population manifest dataset ID mismatch")
    if manifest.get("policy_id") != expected_policy_id:
        raise ValueError("population manifest policy ID mismatch")
    _exact_commit(manifest.get("source_commit"), name="generation source")
    _exact_commit(manifest.get("analysis_commit"), name="packet analysis")
    population_counts = _integer_mapping(
        manifest.get("population_counts"), name="population counts"
    )
    exclusion_counts = _integer_mapping(
        manifest.get("exclusion_counts"), name="exclusion counts"
    )
    if population_counts.get("accepted_changed", 0) != accepted_count:
        raise ValueError("accepted-change population count mismatch")
    if population_counts.get("certificate_rejected", 0) != rejected_count:
        raise ValueError("certificate-rejection population count mismatch")
    if int(manifest.get("audit_item_count", -1)) != accepted_count:
        raise ValueError("audit item count mismatch")
    partition_count = sum(population_counts.values()) + sum(exclusion_counts.values())
    if int(manifest.get("task_count", -1)) != partition_count:
        raise ValueError("population manifest does not partition every task")
    index_hash = _sha256(index_path)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not any(
        isinstance(entry, dict)
        and Path(str(entry.get("path", ""))).name == index_path.name
        and str(entry.get("sha256", "")) == index_hash
        for entry in outputs
    ):
        raise ValueError("population manifest does not bind the supplied index")


def _validate_rater_release(
    *,
    manifest_path: Path,
    archive_path: Path,
    expected_rater_id: str,
    expected_dataset_id: str,
    expected_policy_id: str,
    generation_commit: str,
    analysis_commit: str,
    task_population: int,
    audit_item_count: int,
    population_manifest_sha256: str,
    expected_ids: Sequence[str],
) -> dict[str, str]:
    release = _load_object(manifest_path)
    expected_release_id = _release_id(
        rater_id=expected_rater_id,
        dataset_id=expected_dataset_id,
        policy_id=expected_policy_id,
        generation_commit=generation_commit,
        analysis_commit=analysis_commit,
        task_population=task_population,
        audit_item_count=audit_item_count,
        population_manifest_sha256=population_manifest_sha256,
    )
    if (
        int(release.get("schema_version", -1)) != RELEASE_SCHEMA_VERSION
        or release.get("audience")
        != f"independent blinded rater {expected_rater_id.upper()}"
        or release.get("rater_id") != expected_rater_id
        or release.get("release_id") != expected_release_id
        or release.get("rating_columns") != list(RATING_COLUMNS)
        or release.get("dataset_id") != expected_dataset_id
        or release.get("policy_id") != expected_policy_id
        or release.get("generation_source_commit") != generation_commit
        or release.get("packet_analysis_commit") != analysis_commit
        or int(release.get("task_population", -1)) != task_population
        or int(release.get("audit_item_count", -1)) != audit_item_count
        or release.get("population_manifest_sha256") != population_manifest_sha256
        or release.get("deterministic") is not True
    ):
        raise ValueError(
            f"rater {expected_rater_id.upper()} release manifest is incompatible"
        )
    exclusions = release.get("rater_archive_excludes")
    if not isinstance(exclusions, list) or not REQUIRED_RATER_EXCLUSIONS.issubset(
        {str(value) for value in exclusions}
    ):
        raise ValueError(
            f"rater {expected_rater_id.upper()} release is not least privilege"
        )

    archive_sha256 = _sha256(archive_path)
    if (
        HASH_RE.fullmatch(str(release.get("archive_sha256", ""))) is None
        or release.get("archive_sha256") != archive_sha256
        or int(release.get("archive_bytes", -1)) != archive_path.stat().st_size
        or archive_path.stat().st_size <= 0
    ):
        raise ValueError(
            f"rater {expected_rater_id.upper()} archive hash or size mismatch"
        )
    expected_files = _release_file_entries(release)
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if (
            len(names) != len(set(names))
            or set(names) != set(expected_files)
            or any(info.is_dir() for info in infos)
        ):
            raise ValueError(
                f"rater {expected_rater_id.upper()} archive inventory mismatch"
            )
        _validate_release_archive_inventory(names, expected_ids=expected_ids)
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe path in rater archive: {name}")
            payload = archive.read(name)
            entry = expected_files[name]
            if (
                hashlib.sha256(payload).hexdigest() != entry["sha256"]
                or len(payload) != entry["bytes"]
            ):
                raise ValueError(
                    f"rater {expected_rater_id.upper()} archive member mismatch: {name}"
                )
        if "ratings.csv" not in names:
            raise ValueError("rater archive lacks ratings.csv")
        _validate_blank_release_template(
            archive.read("ratings.csv"),
            expected_ids=expected_ids,
            expected_release_id=expected_release_id,
            expected_rater_id=expected_rater_id,
        )
    return {
        "release_id": expected_release_id,
        "manifest_sha256": _sha256(manifest_path),
        "archive_sha256": archive_sha256,
    }


def _validate_release_archive_inventory(
    names: Sequence[str], *, expected_ids: Sequence[str]
) -> None:
    core = {
        "RATER_GUIDE.md",
        "items.html",
        "blinded_items.jsonl",
        "ratings.csv",
    }
    actual = set(names)
    image_names = actual - core
    if not core.issubset(actual) or len(image_names) != len(expected_ids):
        raise ValueError("rater archive lacks the exact core/image inventory")
    image_ids: set[str] = set()
    for name in image_names:
        path = PurePosixPath(name)
        if path.parent != PurePosixPath("images") or path.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
        }:
            raise ValueError(f"unexpected rater archive member: {name}")
        image_ids.add(path.stem)
    if image_ids != set(expected_ids):
        raise ValueError("rater archive images do not match the blinded population")


def _release_file_entries(release: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_files = release.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("rater release file inventory is missing")
    entries: dict[str, dict[str, Any]] = {}
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ValueError("rater release file inventory is malformed")
        name = str(raw.get("path", ""))
        digest = str(raw.get("sha256", ""))
        size = int(raw.get("bytes", -1))
        relative = PurePosixPath(name)
        if (
            not name
            or name in entries
            or relative.is_absolute()
            or ".." in relative.parts
            or HASH_RE.fullmatch(digest) is None
            or size < 0
        ):
            raise ValueError("rater release file inventory is malformed")
        entries[name] = {"sha256": digest, "bytes": size}
    return entries


def _validate_blank_release_template(
    payload: bytes,
    *,
    expected_ids: Sequence[str],
    expected_release_id: str,
    expected_rater_id: str,
) -> None:
    with io.StringIO(payload.decode("utf-8"), newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RATING_COLUMNS:
            raise ValueError("released rating template header mismatch")
        rows = list(reader)
    if [str(row.get("blind_item_id", "")) for row in rows] != list(expected_ids):
        raise ValueError("released rating template item order mismatch")
    for row in rows:
        if (
            row.get("release_id") != expected_release_id
            or row.get("rater_id") != expected_rater_id
            or any(str(row.get(field, "")).strip() for field in LABEL_FIELDS)
        ):
            raise ValueError("released rating template metadata or blanks mismatch")


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


def _integer_mapping(raw: Any, *, name: str) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise ValueError(f"invalid {name}")
    result = {str(key): int(value) for key, value in raw.items()}
    if any(value < 0 for value in result.values()):
        raise ValueError(f"negative {name}")
    return result


def _exact_commit(raw: Any, *, name: str) -> str:
    value = str(raw or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{name} commit must be an exact 40-hex commit")
    return value


def _load_labels(
    path: Path,
    *,
    expected_ids: Sequence[str],
    expected_release_id: str | None = None,
    expected_rater_id: str | None = None,
) -> dict[str, dict[str, str]]:
    if (expected_release_id is None) != (expected_rater_id is None):
        raise ValueError("release ID and rater ID must be supplied together")
    rows: dict[str, dict[str, str]] = {}
    ordered_ids: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = (
            ADJUDICATION_COLUMNS if expected_release_id is None else RATING_COLUMNS
        )
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError(f"annotation header mismatch in {path}")
        for row in reader:
            blind_id = str(row.get("blind_item_id", "")).strip()
            if not blind_id or blind_id in rows:
                raise ValueError(f"invalid or duplicate annotation ID in {path}")
            if expected_release_id is not None and (
                str(row.get("release_id", "")).strip() != expected_release_id
                or str(row.get("rater_id", "")).strip() != expected_rater_id
            ):
                raise ValueError(f"rating release identity mismatch in {path}")
            parsed: dict[str, str] = {}
            for field, allowed in LABEL_DOMAINS.items():
                value = str(row.get(field, "")).strip().lower()
                if value not in allowed:
                    raise ValueError(
                        f"invalid {field}={value!r} for {blind_id} in {path}"
                    )
                parsed[field] = value
            parsed["notes"] = str(row.get("notes", "")).strip()
            if not parsed["notes"] and any(
                parsed[field] in {"no", "unclear", "other"} for field in LABEL_DOMAINS
            ):
                raise ValueError(f"required annotation note is missing for {blind_id}")
            rows[blind_id] = parsed
            ordered_ids.append(blind_id)
    if ordered_ids != list(expected_ids):
        expected_set = set(expected_ids)
        raise ValueError(
            f"annotation IDs or order mismatch in {path}: "
            f"missing={sorted(expected_set - set(rows))}, "
            f"extra={sorted(set(rows) - expected_set)}"
        )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-manifest", required=True)
    parser.add_argument("--population-index", required=True)
    parser.add_argument("--rater-a", required=True)
    parser.add_argument("--rater-b", required=True)
    parser.add_argument("--rater-a-release-manifest", required=True)
    parser.add_argument("--rater-b-release-manifest", required=True)
    parser.add_argument("--rater-a-archive", required=True)
    parser.add_argument("--rater-b-archive", required=True)
    parser.add_argument("--adjudication", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rating-analysis-commit", required=True)
    parser.add_argument("--expected-dataset-id", default="chartqa_prospective_v1")
    parser.add_argument("--expected-policy-id", default="frozen_egor_selector")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
