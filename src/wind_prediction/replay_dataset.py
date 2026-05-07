from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np


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
        self.dataset_dir = Path(dataset_dir)
        self.split = str(split)
        self.metadata = json.loads((self.dataset_dir / "metadata.json").read_text(encoding="utf-8"))
        self.scaler = json.loads((self.dataset_dir / "scaler_train.json").read_text(encoding="utf-8"))
        self.event_columns = list(self.metadata["event_columns"])
        self.feature_columns = list(self.metadata["feature_columns"])
        self.history_steps = int(self.metadata["history_steps"])
        self.future_steps = int(self.metadata["future_steps"])
        self.update_interval_minutes = int(self.metadata.get("input_resolution_minutes", 10))
        self.update_interval_s = int(self.update_interval_minutes * 60)

        if canonical_path is None:
            canonical_path = (
                self.dataset_dir.parent / "fino1_platform_10min" / "canonical_observations_10min.csv.gz"
            )
        self.canonical_path = Path(canonical_path)

        self._trace_rows_all = self._load_trace_rows()
        self._trace_rows = [row for row in self._trace_rows_all if row.split == self.split]
        self._trace_by_timestamp = {row.timestamp: row for row in self._trace_rows}
        self._samples = self._load_samples()
        self._sample_by_history_end = {sample.history_end: sample for sample in self._samples}

    def _load_trace_rows(self) -> list[ReplayTraceRow]:
        rows: list[ReplayTraceRow] = []
        with gzip.open(self.canonical_path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                air_temp = row.get("air_temp_c", "")
                rel_humidity = row.get("rel_humidity_pct", "")
                rows.append(
                    ReplayTraceRow(
                        timestamp=_parse_timestamp(row["timestamp"]),
                        wind_speed_ms=float(row["wind_speed_ms"]),
                        wind_dir_deg=float(row["wind_dir_deg"]),
                        air_temp_c=float(air_temp) if air_temp not in ("", None) else None,
                        rel_humidity_pct=float(rel_humidity) if rel_humidity not in ("", None) else None,
                        split=str(row.get("split", "")).strip(),
                    )
                )
        if not rows:
            raise ValueError(f"no canonical replay rows found in {self.canonical_path}")
        return rows

    def _load_samples(self) -> list[ReplaySample]:
        arrays_meta = self.metadata["arrays"][self.split]
        x_all = np.load(self.dataset_dir / arrays_meta["X"], mmap_mode="r")
        y_uv_raw_all = np.load(self.dataset_dir / arrays_meta["y_uv_raw"], mmap_mode="r")
        y_event_all = np.load(self.dataset_dir / arrays_meta["y_event"], mmap_mode="r")

        sample_rows: list[dict] = []
        with gzip.open(self.dataset_dir / "sample_index.csv.gz", "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row["split"]).strip() == self.split:
                    sample_rows.append(row)

        if len(sample_rows) != x_all.shape[0]:
            raise ValueError(
                f"sample_index rows for split={self.split} do not match X rows: "
                f"{len(sample_rows)} vs {x_all.shape[0]}"
            )

        samples: list[ReplaySample] = []
        for idx, row in enumerate(sample_rows):
            history_end = _parse_timestamp(row["history_end"])
            trace_row = self._trace_by_timestamp.get(history_end)
            if trace_row is None:
                raise KeyError(f"missing canonical observation for history_end={history_end}")
            samples.append(
                ReplaySample(
                    split=self.split,
                    series_id=str(row["series_id"]),
                    history_start=_parse_timestamp(row["history_start"]),
                    history_end=history_end,
                    future_start=_parse_timestamp(row["future_start"]),
                    future_end=_parse_timestamp(row["future_end"]),
                    x_window=np.array(x_all[idx], dtype=np.float32, copy=True),
                    y_uv_raw=np.array(y_uv_raw_all[idx], dtype=np.float32, copy=True),
                    y_event=np.array(y_event_all[idx], dtype=np.float32, copy=True),
                    wind_obs={
                        "ws": float(trace_row.wind_speed_ms),
                        "wd_deg": float(trace_row.wind_dir_deg),
                    },
                    meteo_obs={
                        "air_temp_c": trace_row.air_temp_c,
                        "rel_humidity_pct": trace_row.rel_humidity_pct,
                    },
                )
            )
        return samples

    @property
    def samples(self) -> list[ReplaySample]:
        return self._samples

    @property
    def trace_rows(self) -> list[ReplayTraceRow]:
        return self._trace_rows

    def first_sample_timestamp(self) -> datetime:
        return self._samples[0].history_end

    def first_positive_sample_timestamp(self, event_name: str = "ballast_attention_event") -> datetime:
        try:
            event_idx = self.event_columns.index(event_name)
        except ValueError as exc:
            raise KeyError(f"unsupported event_name={event_name}") from exc
        for sample in self._samples:
            if float(sample.y_event[event_idx]) >= 0.5:
                return sample.history_end
        return self.first_sample_timestamp()

    def sample_for_history_end(self, timestamp: datetime) -> ReplaySample | None:
        return self._sample_by_history_end.get(timestamp)

    def trace_rows_from(self, start_timestamp: datetime, row_count: int | None = None) -> list[ReplayTraceRow]:
        rows = [row for row in self._trace_rows if row.timestamp >= start_timestamp]
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
        for sample in self._samples:
            yield sample.history_end
