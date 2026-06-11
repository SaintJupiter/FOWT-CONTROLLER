from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_REPLAY_ARRAY_KEYS = ("X", "y_uv_raw", "y_event")


@dataclass(frozen=True)
class DatasetValidationReport:
    dataset_dir: Path
    split: str
    ok: bool
    checked_paths: tuple[Path, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    array_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def raise_for_errors(self) -> None:
        if self.ok:
            return
        joined = "\n".join(f"- {item}" for item in self.errors)
        raise ValueError(f"dataset manifest validation failed for split={self.split}:\n{joined}")


@dataclass(frozen=True)
class DatasetManifest:
    """Typed access to a wind-preview dataset directory.

    The manifest owns the file-layout and shape contract that used to be
    re-learned by each caller. Callers still decide what they want to load, but
    path resolution, split checks, scaler checks, and cheap array shape
    validation live in one module.
    """

    dataset_dir: Path
    metadata: dict[str, Any]
    scaler: dict[str, Any]

    @classmethod
    def load(cls, dataset_dir: str | Path) -> "DatasetManifest":
        root = Path(dataset_dir)
        metadata_path = root / "metadata.json"
        scaler_path = root / "scaler_train.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing dataset metadata: {metadata_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"missing train scaler: {scaler_path}")
        return cls(
            dataset_dir=root,
            metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
            scaler=json.loads(scaler_path.read_text(encoding="utf-8")),
        )

    @property
    def feature_columns(self) -> list[str]:
        return list(self.metadata.get("feature_columns", []))

    @property
    def event_columns(self) -> list[str]:
        return list(self.metadata.get("event_columns", []))

    @property
    def history_steps(self) -> int:
        return int(self.metadata["history_steps"])

    @property
    def future_steps(self) -> int:
        return int(self.metadata["future_steps"])

    @property
    def input_resolution_minutes(self) -> int:
        return int(self.metadata.get("input_resolution_minutes", 10))

    @property
    def arrays_by_split(self) -> dict[str, dict[str, Any]]:
        return dict(self.metadata.get("arrays", {}))

    @property
    def sample_index_path(self) -> Path:
        return self.dataset_dir / "sample_index.csv.gz"

    def split_arrays(self, split: str) -> dict[str, Any]:
        split = str(split)
        try:
            return dict(self.arrays_by_split[split])
        except KeyError as exc:
            available = ", ".join(sorted(self.arrays_by_split)) or "<none>"
            raise KeyError(f"unknown dataset split {split!r}; available: {available}") from exc

    def array_path(self, split: str, key: str) -> Path:
        arrays = self.split_arrays(split)
        try:
            rel = arrays[key]
        except KeyError as exc:
            raise KeyError(f"split={split!r} has no array key {key!r}") from exc
        return self.dataset_dir / str(rel)

    def expected_shape(self, split: str, key: str) -> tuple[int, ...] | None:
        arrays = self.split_arrays(split)
        shape_key = f"shape_{key}"
        if shape_key not in arrays:
            return None
        return tuple(int(v) for v in arrays[shape_key])

    def default_fino1_canonical_path(self) -> Path:
        return self.dataset_dir.parent / "fino1_platform_10min" / "canonical_observations_10min.csv.gz"

    def validate(
        self,
        split: str,
        *,
        require_replay_arrays: bool = True,
        require_sample_index: bool = True,
        canonical_path: str | Path | None = None,
        check_array_shapes: bool = True,
    ) -> DatasetValidationReport:
        split = str(split)
        errors: list[str] = []
        warnings: list[str] = []
        checked_paths: list[Path] = []
        array_shapes: dict[str, tuple[int, ...]] = {}

        try:
            arrays = self.split_arrays(split)
        except KeyError as exc:
            errors.append(str(exc))
            arrays = {}

        if not self.feature_columns:
            errors.append("metadata.feature_columns is empty")
        if not self.event_columns:
            errors.append("metadata.event_columns is empty")

        feature_scaler = dict(self.scaler.get("feature_scaler", {}))
        missing_scaler = [name for name in self.feature_columns if name not in feature_scaler]
        if missing_scaler:
            errors.append(
                "feature_scaler is missing feature columns: "
                + ", ".join(missing_scaler[:12])
                + (" ..." if len(missing_scaler) > 12 else "")
            )

        target_scaler = dict(self.scaler.get("target_uv_scaler", {}))
        for name in ("wind_u_ms", "wind_v_ms"):
            if name not in target_scaler:
                errors.append(f"target_uv_scaler is missing {name}")

        if require_sample_index:
            checked_paths.append(self.sample_index_path)
            if not self.sample_index_path.exists():
                errors.append(f"missing sample index: {self.sample_index_path}")
            else:
                try:
                    with gzip.open(self.sample_index_path, "rt", encoding="utf-8", newline="") as f:
                        header = f.readline()
                    for col in ("split", "series_id", "history_end"):
                        if col not in header:
                            errors.append(f"sample_index.csv.gz header missing {col!r}")
                except OSError as exc:
                    errors.append(f"cannot read sample index {self.sample_index_path}: {exc}")

        if canonical_path is not None:
            canonical = Path(canonical_path)
            checked_paths.append(canonical)
            if not canonical.exists():
                errors.append(f"missing canonical replay observations: {canonical}")

        array_keys = REQUIRED_REPLAY_ARRAY_KEYS if require_replay_arrays else tuple(arrays)
        for key in array_keys:
            if key not in arrays:
                errors.append(f"split={split!r} metadata missing array key {key!r}")
                continue
            path = self.dataset_dir / str(arrays[key])
            checked_paths.append(path)
            if not path.exists():
                errors.append(f"missing array file for {split}.{key}: {path}")
                continue
            if not check_array_shapes:
                continue
            try:
                arr = np.load(path, mmap_mode="r")
                shape = tuple(int(v) for v in arr.shape)
                array_shapes[key] = shape
            except Exception as exc:
                errors.append(f"cannot read array shape for {split}.{key}: {exc}")
                continue
            expected = self.expected_shape(split, key)
            if expected is not None and shape != expected:
                errors.append(
                    f"array shape mismatch for {split}.{key}: metadata {expected}, file {shape}"
                )

        if self.history_steps <= 0:
            errors.append("metadata.history_steps must be positive")
        if self.future_steps <= 0:
            errors.append("metadata.future_steps must be positive")
        if self.input_resolution_minutes <= 0:
            errors.append("metadata.input_resolution_minutes must be positive")

        return DatasetValidationReport(
            dataset_dir=self.dataset_dir,
            split=split,
            ok=not errors,
            checked_paths=tuple(checked_paths),
            errors=tuple(errors),
            warnings=tuple(warnings),
            array_shapes=array_shapes,
        )

    def validate_replay_split(
        self,
        split: str,
        *,
        canonical_path: str | Path | None = None,
    ) -> DatasetValidationReport:
        report = self.validate(
            split,
            require_replay_arrays=True,
            require_sample_index=True,
            canonical_path=canonical_path,
            check_array_shapes=True,
        )
        report.raise_for_errors()
        return report
