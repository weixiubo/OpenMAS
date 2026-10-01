"""Deterministic source-stratified train/calibration/test assignment."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from openagora_experiments.cago.data import MatrixValidationError, TaskKey


@dataclass(frozen=True)
class SplitAssignment:
    """Disjoint task sets with a separate conformal calibration partition."""

    train: tuple[TaskKey, ...]
    calibration: tuple[TaskKey, ...]
    test: tuple[TaskKey, ...]

    def __post_init__(self) -> None:
        train = set(self.train)
        calibration = set(self.calibration)
        test = set(self.test)
        if not train or not calibration or not test:
            raise MatrixValidationError("train, calibration, and test splits must be non-empty")
        if train.intersection(calibration) or train.intersection(test) or calibration.intersection(test):
            raise MatrixValidationError("train, calibration, and test splits must be disjoint")

    def counts_by_source(self) -> Mapping[str, Mapping[str, int]]:
        counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "calibration": 0, "test": 0})
        for split_name, keys in (
            ("train", self.train),
            ("calibration", self.calibration),
            ("test", self.test),
        ):
            for key in keys:
                counts[key.source_id][split_name] += 1
        return {source: dict(values) for source, values in sorted(counts.items())}


def source_stratified_split(
    keys: Iterable[TaskKey],
    *,
    train_fraction: float = 0.5,
    calibration_fraction: float = 0.25,
    salt: str = "cago-v1",
) -> SplitAssignment:
    """Rank stable hashes within each source, then allocate exact disjoint counts."""

    if not 0.0 < train_fraction < 1.0:
        raise MatrixValidationError("train_fraction must be between zero and one")
    if not 0.0 < calibration_fraction < 1.0:
        raise MatrixValidationError("calibration_fraction must be between zero and one")
    if train_fraction + calibration_fraction >= 1.0:
        raise MatrixValidationError("train and calibration fractions must leave a test split")

    by_source: dict[str, list[TaskKey]] = defaultdict(list)
    for key in set(keys):
        by_source[key.source_id].append(key)
    if not by_source:
        raise MatrixValidationError("cannot split an empty task set")

    train: list[TaskKey] = []
    calibration: list[TaskKey] = []
    test: list[TaskKey] = []
    for source_id, source_keys in sorted(by_source.items()):
        ranked = sorted(source_keys, key=lambda key: (_stable_hash(key, salt), key.task_id))
        if len(ranked) < 3:
            raise MatrixValidationError(f"source {source_id} needs at least three tasks")
        train_count = max(1, int(len(ranked) * train_fraction))
        calibration_count = max(1, int(len(ranked) * calibration_fraction))
        if train_count + calibration_count >= len(ranked):
            calibration_count = max(1, len(ranked) - train_count - 1)
        train.extend(ranked[:train_count])
        calibration.extend(ranked[train_count : train_count + calibration_count])
        test.extend(ranked[train_count + calibration_count :])
    return SplitAssignment(
        train=tuple(sorted(train)),
        calibration=tuple(sorted(calibration)),
        test=tuple(sorted(test)),
    )


def _stable_hash(key: TaskKey, salt: str) -> str:
    return hashlib.sha256(f"{salt}\x00{key.stable_id}".encode()).hexdigest()
