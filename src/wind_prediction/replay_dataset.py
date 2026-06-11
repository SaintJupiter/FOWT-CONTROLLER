from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np

from .dataset_manifest import DatasetManifest


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


def _parse_timestamp(raw: str) -> datetime:
    return datetime.strptime(str(raw).strip(), TIMESTAMP_FMT)


@dataclass(frozen=True)
class ReplayTraceRow:
    timestamp: datetime
    wind_speed_ms: float
    wind_dir_deg: float
    air_temp_c: float | None
    rel_humidity_pct: float | None
    split: str


@dataclass(frozen=True)
class ReplaySample:
    split: str
    series_id: str
    history_start: datetime
    history_end: datetime
    future_start: datetime
    future_end: datetime
    x_window: np.ndarray
    y_uv_raw: np.ndarray
    y_event: np.ndarray
    wind_obs: dict
    meteo_obs: dict


class Fino1ReplayDataset:
    """Replay-oriented access to FINO1 dataset windows and 10-minute observations.

    This class deliberately prioritizes mechanism validation on real FINO1
    sequences over a generic online-feature-reconstruction workflow.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "test",
        canonical_path: str | Path | None = None,
    ) -> None:
        self.manifest = DatasetManifest.load(dataset_dir)
        self.dataset_dir = self.manifest.dataset_dir
        self.split = str(split)
        self.metadata = self.manifest.metadata
        self.scaler = self.manifest.scaler
        self.event_columns = self.manifest.event_columns
        self.feature_columns = self.manifest.feature_columns
        self.history_steps = self.manifest.history_steps
        self.future_steps = self.manifest.future_steps
        self.update_interval_minutes = self.manifest.input_resolution_minutes
        self.update_interval_s = int(self.update_interval_minutes * 60)

        if canonical_path is None:
            canonical_path = self.manifest.default_fino1_canonical_path()
        self.canonical_path = Path(canonical_path)
        self.validation_report = self.manifest.validate_replay_split(
            self.split,
            canonical_path=self.canonical_path,
        )

        self._trace_rows_cache: list[ReplayTraceRow] | None = None
        self._trace_by_timestamp: dict[datetime, ReplayTraceRow] = {}
        self._sample_rows = self._load_sample_rows()
        self._sample_by_history_end = {
            _parse_timestamp(row["history_end"]): idx
            for idx, row in enumerate(self._sample_rows)
        }
        self._x_all = np.load(self.manifest.array_path(self.split, "X"), mmap_mode="r")
        self._y_uv_raw_all = np.load(self.manifest.array_path(self.split, "y_uv_raw"), mmap_mode="r")
        self._y_event_all = np.load(self.manifest.array_path(self.split, "y_event"), mmap_mode="r")
        if len(self._sample_rows) != self._x_all.shape[0]:
            raise ValueError(
                f"sample_index rows for split={self.split} do not match X rows: "
                f"{len(self._sample_rows)} vs {self._x_all.shape[0]}"
            )
        self._samples_cache: list[ReplaySample] | None = None

    @staticmethod
    def _trace_row_from_csv(row: dict) -> ReplayTraceRow:
        air_temp = row.get("air_temp_c", "")
        rel_humidity = row.get("rel_humidity_pct", "")
        return ReplayTraceRow(
            timestamp=_parse_timestamp(row["timestamp"]),
            wind_speed_ms=float(row["wind_speed_ms"]),
            wind_dir_deg=float(row["wind_dir_deg"]),
            air_temp_c=float(air_temp) if air_temp not in ("", None) else None,
            rel_humidity_pct=float(rel_humidity) if rel_humidity not in ("", None) else None,
            split=str(row.get("split", "")).strip(),
        )

    def _load_trace_rows(self) -> list[ReplayTraceRow]:
        rows: list[ReplayTraceRow] = []
        with gzip.open(self.canonical_path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trace_row = self._trace_row_from_csv(row)
                if trace_row.split == self.split:
                    rows.append(trace_row)
        if not rows:
            raise ValueError(f"no canonical replay rows found in {self.canonical_path}")
        self._trace_by_timestamp.update({row.timestamp: row for row in rows})
        return rows

    def _trace_row_for_timestamp(self, timestamp: datetime) -> ReplayTraceRow | None:
        cached = self._trace_by_timestamp.get(timestamp)
        if cached is not None:
            return cached
        with gzip.open(self.canonical_path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("split", "")).strip() != self.split:
                    continue
                if _parse_timestamp(row["timestamp"]) != timestamp:
                    continue
                trace_row = self._trace_row_from_csv(row)
                self._trace_by_timestamp[timestamp] = trace_row
                return trace_row
        return None

    def _scan_trace_rows_from(
        self,
        start_timestamp: datetime,
        row_count: int | None = None,
    ) -> list[ReplayTraceRow]:
        rows: list[ReplayTraceRow] = []
        with gzip.open(self.canonical_path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("split", "")).strip() != self.split:
                    continue
                timestamp = _parse_timestamp(row["timestamp"])
                if timestamp < start_timestamp:
                    continue
                trace_row = self._trace_row_from_csv(row)
                rows.append(trace_row)
                self._trace_by_timestamp[timestamp] = trace_row
                if row_count is not None and len(rows) >= int(row_count):
                    break
        return rows

    def _load_sample_rows(self) -> list[dict]:
        sample_rows: list[dict] = []
        with gzip.open(self.dataset_dir / "sample_index.csv.gz", "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row["split"]).strip() == self.split:
                    sample_rows.append(row)
        return sample_rows

    def _inverse_feature_value(self, feature_name: str, value: float) -> float:
        scaler = self.scaler.get("feature_scaler", {}).get(feature_name, {})
        mean = float(scaler.get("mean", 0.0))
        std = float(scaler.get("std", 1.0))
        return float(value) * std + mean

    def _wind_obs_from_x_row(self, x_row: np.ndarray) -> dict:
        speed_idx = self.feature_columns.index("wind_speed_ms")
        sin_idx = self.feature_columns.index("wind_dir_sin")
        cos_idx = self.feature_columns.index("wind_dir_cos")
        speed = self._inverse_feature_value("wind_speed_ms", float(x_row[speed_idx]))
        sin_val = self._inverse_feature_value("wind_dir_sin", float(x_row[sin_idx]))
        cos_val = self._inverse_feature_value("wind_dir_cos", float(x_row[cos_idx]))
        direction = (np.rad2deg(np.arctan2(sin_val, cos_val)) + 360.0) % 360.0
        return {"ws": float(speed), "wd_deg": float(direction)}

    def _meteo_obs_from_x_row(self, x_row: np.ndarray) -> dict:
        out = {"air_temp_c": None, "rel_humidity_pct": None}
        for feature_name, out_name in (
            ("air_temp_c_filled", "air_temp_c"),
            ("rel_humidity_pct_filled", "rel_humidity_pct"),
        ):
            if feature_name in self.feature_columns:
                idx = self.feature_columns.index(feature_name)
                out[out_name] = self._inverse_feature_value(feature_name, float(x_row[idx]))
        return out

    def _trace_row_from_sample_index(self, idx: int) -> ReplayTraceRow:
        x_row = np.asarray(self._x_all[int(idx), -1], dtype=np.float32)
        wind_obs = self._wind_obs_from_x_row(x_row)
        meteo_obs = self._meteo_obs_from_x_row(x_row)
        return ReplayTraceRow(
            timestamp=self._sample_history_end(idx),
            wind_speed_ms=float(wind_obs["ws"]),
            wind_dir_deg=float(wind_obs["wd_deg"]),
            air_temp_c=meteo_obs["air_temp_c"],
            rel_humidity_pct=meteo_obs["rel_humidity_pct"],
            split=self.split,
        )

    def _materialize_sample(self, idx: int) -> ReplaySample:
        row = self._sample_rows[int(idx)]
        history_end = _parse_timestamp(row["history_end"])
        x_window = np.array(self._x_all[idx], dtype=np.float32, copy=True)
        x_current = x_window[-1]
        return ReplaySample(
            split=self.split,
            series_id=str(row["series_id"]),
            history_start=_parse_timestamp(row["history_start"]),
            history_end=history_end,
            future_start=_parse_timestamp(row["future_start"]),
            future_end=_parse_timestamp(row["future_end"]),
            x_window=x_window,
            y_uv_raw=np.array(self._y_uv_raw_all[idx], dtype=np.float32, copy=True),
            y_event=np.array(self._y_event_all[idx], dtype=np.float32, copy=True),
            wind_obs=self._wind_obs_from_x_row(x_current),
            meteo_obs=self._meteo_obs_from_x_row(x_current),
        )

    def _load_samples(self) -> list[ReplaySample]:
        return [self._materialize_sample(idx) for idx in range(len(self._sample_rows))]

    def _sample_index_for_history_end(self, timestamp: datetime) -> int | None:
        return self._sample_by_history_end.get(timestamp)

    def _event_values(self) -> np.ndarray:
        return np.asarray(self._y_event_all, dtype=np.float32)

    def _sample_history_end(self, idx: int) -> datetime:
        return _parse_timestamp(self._sample_rows[int(idx)]["history_end"])

    @property
    def samples(self) -> list[ReplaySample]:
        if self._samples_cache is None:
            self._samples_cache = self._load_samples()
        return self._samples_cache

    @property
    def trace_rows(self) -> list[ReplayTraceRow]:
        if self._trace_rows_cache is None:
            self._trace_rows_cache = self._load_trace_rows()
        return self._trace_rows_cache

    def first_sample_timestamp(self) -> datetime:
        if not self._sample_rows:
            raise ValueError(f"no replay samples found for split={self.split}")
        return self._sample_history_end(0)

    def first_positive_sample_timestamp(self, event_name: str = "ballast_attention_event") -> datetime:
        try:
            event_idx = self.event_columns.index(event_name)
        except ValueError as exc:
            raise KeyError(f"unsupported event_name={event_name}") from exc
        y_event = self._event_values()
        for idx in range(y_event.shape[0]):
            if float(y_event[idx, event_idx]) >= 0.5:
                return self._sample_history_end(idx)
        return self.first_sample_timestamp()

    def sample_for_history_end(self, timestamp: datetime) -> ReplaySample | None:
        idx = self._sample_index_for_history_end(timestamp)
        if idx is None:
            return None
        return self._materialize_sample(idx)

    def trace_rows_from(self, start_timestamp: datetime, row_count: int | None = None) -> list[ReplayTraceRow]:
        sample_idx = self._sample_index_for_history_end(start_timestamp)
        if sample_idx is not None and row_count is not None:
            end_idx = int(sample_idx) + int(row_count)
            if end_idx <= len(self._sample_rows):
                return [
                    self._trace_row_from_sample_index(idx)
                    for idx in range(int(sample_idx), end_idx)
                ]
        if self._trace_rows_cache is None:
            return self._scan_trace_rows_from(start_timestamp, row_count)
        rows = [row for row in self._trace_rows_cache if row.timestamp >= start_timestamp]
        if row_count is not None:
            rows = rows[: int(row_count)]
        return rows

    def build_wind_trace(
        self,
        start_timestamp: datetime,
        row_count: int,
        dt_s: float,
    ) -> dict:
        row_count = int(row_count)
        if row_count <= 0:
            raise ValueError("row_count must be positive")
        dt_s = float(dt_s)
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")

        rows = self.trace_rows_from(start_timestamp=start_timestamp, row_count=row_count)
        if len(rows) < row_count:
            raise ValueError(f"requested {row_count} replay rows, only found {len(rows)}")

        hold_steps = int(round(self.update_interval_s / dt_s))
        if abs(hold_steps * dt_s - self.update_interval_s) > 1e-6:
            raise ValueError(
                f"dt_s={dt_s} does not evenly divide {self.update_interval_s}s replay interval"
            )

        ws_vals = np.repeat(np.array([row.wind_speed_ms for row in rows], dtype=float), hold_steps)
        wd_vals = np.repeat(np.array([row.wind_dir_deg for row in rows], dtype=float), hold_steps)
        timestamps = np.repeat(
            np.array([row.timestamp.strftime(TIMESTAMP_FMT) for row in rows], dtype=object),
            hold_steps,
        )
        return {
            "ws": ws_vals,
            "wd": wd_vals,
            "timestamp": timestamps,
            "n_steps": int(len(ws_vals)),
            "dt": float(dt_s),
            "update_interval_s": float(self.update_interval_s),
            "mean_lpf_tau_s": 0.0,
            "seed": 0,
        }

    def simulation_timestamp(self, start_timestamp: datetime, current_time_s: float) -> datetime:
        bucket = int(np.floor(max(float(current_time_s), 0.0) / self.update_interval_s + 1e-9))
        return start_timestamp + timedelta(seconds=bucket * self.update_interval_s)

    def iter_sample_timestamps(self) -> Iterator[datetime]:
        for idx in range(len(self._sample_rows)):
            yield self._sample_history_end(idx)
