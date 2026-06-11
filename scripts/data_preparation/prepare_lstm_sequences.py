#!/usr/bin/env python3
"""Build LSTM-ready wind forecasting sequences.

This converts the cleaned 10-minute supervised table into fixed-length
sliding-window tensors. Windows are kept only when every input timestamp is
strictly continuous at 10-minute spacing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "wind_speed_ms",
    "wind_dir_sin",
    "wind_dir_cos",
    "wind_u_ms",
    "wind_v_ms",
    "wind_speed_delta_10m",
    "wind_dir_delta_10m_deg",
    "hour_sin",
    "hour_cos",
    "dayofyear_sin",
    "dayofyear_cos",
]

TARGET_COLUMNS = [
    "target_wind_u_ms_tplus_30m",
    "target_wind_v_ms_tplus_30m",
]

WINDOWS = {
    "60m": 6,
    "120m": 12,
}

TEN_MINUTES_SECONDS = 600
SPLITS = ["train", "validation", "test"]


def source_table_path(processed_dir: Path) -> Path:
    # Prefer the compressed table because it is the canonical full export.
    # A local uncompressed CSV may be an Excel-sized inspection copy.
    gz_path = processed_dir / "supervised_learning_table_10min.csv.gz"
    if gz_path.exists():
        return gz_path
    return processed_dir / "supervised_learning_table_10min.csv"


def standardizer(values: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats = {}
    for column in values.columns:
        series = pd.to_numeric(values[column], errors="coerce")
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        stats[column] = {"mean": mean, "std": std}
    return stats


def apply_standardizer(frame: pd.DataFrame, stats: dict[str, dict[str, float]], columns: list[str]) -> np.ndarray:
    values = frame[columns].astype("float32").to_numpy(copy=True)
    mean = np.array([stats[col]["mean"] for col in columns], dtype="float32")
    std = np.array([stats[col]["std"] for col in columns], dtype="float32")
    return (values - mean) / std


def window_end_indices(group: pd.DataFrame, window_steps: int) -> np.ndarray:
    if len(group) < window_steps:
        return np.array([], dtype=np.int64)

    timestamps = group["timestamp"].to_numpy(dtype="datetime64[s]").astype("int64")
    row_ok = group[FEATURE_COLUMNS + TARGET_COLUMNS].notna().all(axis=1).to_numpy()
    ends: list[int] = []

    for end in range(window_steps - 1, len(group)):
        start = end - window_steps + 1
        if not row_ok[end] or not row_ok[start : end + 1].all():
            continue
        expected = timestamps[end] - TEN_MINUTES_SECONDS * np.arange(window_steps - 1, -1, -1)
        if np.array_equal(timestamps[start : end + 1], expected):
            ends.append(end)

    return np.asarray(ends, dtype=np.int64)


def count_windows(df: pd.DataFrame, window_steps: int) -> dict[str, int]:
    counts = {split: 0 for split in SPLITS}
    for (_, split), group in df.groupby(["series_id", "split"], sort=False):
        if split in counts:
            counts[split] += len(window_end_indices(group, window_steps))
    return counts


def build_window_dataset(
    df: pd.DataFrame,
    out_dir: Path,
    label: str,
    window_steps: int,
    feature_stats: dict[str, dict[str, float]],
    target_stats: dict[str, dict[str, float]],
    selected_series_ids: list[str],
) -> dict[str, int]:
    window_dir = out_dir / f"window_{label}_target_30m_uv"
    window_dir.mkdir(parents=True, exist_ok=True)

    counts = count_windows(df, window_steps)
    arrays = {}
    offsets = {split: 0 for split in SPLITS}

    for split in SPLITS:
        n = counts[split]
        arrays[(split, "X")] = np.lib.format.open_memmap(
            window_dir / f"X_{split}.npy",
            mode="w+",
            dtype="float32",
            shape=(n, window_steps, len(FEATURE_COLUMNS)),
        )
        arrays[(split, "y")] = np.lib.format.open_memmap(
            window_dir / f"y_{split}.npy",
            mode="w+",
            dtype="float32",
            shape=(n, len(TARGET_COLUMNS)),
        )
        arrays[(split, "y_raw")] = np.lib.format.open_memmap(
            window_dir / f"y_{split}_raw.npy",
            mode="w+",
            dtype="float32",
            shape=(n, len(TARGET_COLUMNS)),
        )

    index_parts = []
    for (series_id, split), group in df.groupby(["series_id", "split"], sort=False):
        if split not in SPLITS:
            continue
        group = group.sort_values("timestamp").reset_index(drop=True)
        end_indices = window_end_indices(group, window_steps)
        if len(end_indices) == 0:
            continue

        X_scaled_rows = apply_standardizer(group, feature_stats, FEATURE_COLUMNS)
        y_raw_rows = group[TARGET_COLUMNS].astype("float32").to_numpy()
        y_scaled_rows = (y_raw_rows - np.array([target_stats[col]["mean"] for col in TARGET_COLUMNS], dtype="float32")) / np.array(
            [target_stats[col]["std"] for col in TARGET_COLUMNS], dtype="float32"
        )

        start_offset = offsets[split]
        for local_idx, end in enumerate(end_indices):
            out_idx = start_offset + local_idx
            start = end - window_steps + 1
            arrays[(split, "X")][out_idx] = X_scaled_rows[start : end + 1]
            arrays[(split, "y")][out_idx] = y_scaled_rows[end]
            arrays[(split, "y_raw")][out_idx] = y_raw_rows[end]

        offsets[split] += len(end_indices)
        index_parts.append(
            pd.DataFrame(
                {
                    "split": split,
                    "series_id": series_id,
                    "window_start": group.loc[end_indices - window_steps + 1, "timestamp"].to_numpy(),
                    "window_end": group.loc[end_indices, "timestamp"].to_numpy(),
                    "target_time": group.loc[end_indices, "timestamp"].to_numpy() + pd.Timedelta(minutes=30),
                }
            )
        )

    for arr in arrays.values():
        arr.flush()

    sample_index = pd.concat(index_parts, ignore_index=True) if index_parts else pd.DataFrame()
    sample_index.to_csv(window_dir / "sample_index.csv.gz", index=False, compression="gzip")

    metadata = {
        "window_label": label,
        "window_steps": window_steps,
        "input_resolution_minutes": 10,
        "nominal_history_minutes": window_steps * 10,
        "timestamp_span_minutes": (window_steps - 1) * 10,
        "forecast_horizon_minutes": 30,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "target_representation": "future meteorological wind vector components u/v at t+30min",
        "series_ids": selected_series_ids,
        "arrays": {
            split: {
                "X": f"X_{split}.npy",
                "y": f"y_{split}.npy",
                "y_raw": f"y_{split}_raw.npy",
                "shape_X": [counts[split], window_steps, len(FEATURE_COLUMNS)],
                "shape_y": [counts[split], len(TARGET_COLUMNS)],
            }
            for split in SPLITS
        },
        "continuity_rule": "Each input window must contain exact 10-minute timestamps; windows never cross series_id or split boundaries.",
        "scaling_rule": "Z-score scaling fitted on training rows only; y_raw keeps unscaled target values for evaluation.",
    }
    (window_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return counts


def write_readme(out_dir: Path, summary: list[dict[str, object]], selected_series_ids: list[str]) -> None:
    text = """# LSTM-ready wind sequence datasets

Generated by `scripts/data_preparation/prepare_lstm_sequences.py`.

These datasets convert the 10-minute wind forecasting table into fixed-length
sliding windows for LSTM/GRU/TCN-style sequence models.

## Processing rules

- Input windows are strict 10-minute sequences.
- Windows do not cross station/sensor series.
- Windows do not cross train/validation/test boundaries.
- Input features and targets are standardized using training rows only.
- `y_raw` files keep unscaled targets for metric calculation and inverse checks.
- Target is the future 30-minute wind vector, `u/v`, because this avoids direct regression on circular wind direction.

## Included series

"""
    for series_id in selected_series_ids:
        text += f"- `{series_id}`\n"

    text += """
## Sequence sets

"""
    for row in summary:
        text += (
            f"- `{row['window_label']}`: history `{row['history_minutes']} min`, "
            f"{row['window_steps']} samples, timestamp span `{(row['window_steps'] - 1) * 10} min`, "
            f"train `{row['train']}`, validation `{row['validation']}`, test `{row['test']}`.\n"
        )
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/wind_ml_10min"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/wind_ml_10min/sequences"))
    parser.add_argument(
        "--series-id",
        action="append",
        help="Limit sequence generation to one or more exact series_id values. Repeat for multiple series.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = source_table_path(args.processed_dir)
    usecols = ["timestamp", "series_id", "split", *FEATURE_COLUMNS, *TARGET_COLUMNS]
    df = pd.read_csv(table, usecols=usecols, parse_dates=["timestamp"])
    df = df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    source_counts = df.groupby("series_id").size().reset_index(name="source_rows")

    if args.series_id:
        requested = set(args.series_id)
        df = df[df["series_id"].isin(requested)].copy()
        if df.empty:
            available = "\n".join(source_counts["series_id"].astype(str).tolist())
            raise SystemExit(f"No rows matched --series-id. Available series_id values:\n{available}")

    selected_series_ids = sorted(df["series_id"].unique().tolist())
    source_counts["included"] = source_counts["series_id"].isin(selected_series_ids)
    source_counts.to_csv(args.out_dir / "source_series_counts.csv", index=False)

    train = df[df["split"].eq("train")]
    feature_stats = standardizer(train[FEATURE_COLUMNS])
    target_stats = standardizer(train[TARGET_COLUMNS])
    scaler = {
        "feature_scaler": feature_stats,
        "target_scaler": target_stats,
        "fit_scope": "training rows only",
        "method": "z_score",
        "series_ids": selected_series_ids,
    }
    (args.out_dir / "scaler_train.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")

    summary = []
    for label, steps in WINDOWS.items():
        counts = build_window_dataset(df, args.out_dir, label, steps, feature_stats, target_stats, selected_series_ids)
        summary.append(
            {
                "window_label": label,
                "window_steps": steps,
                "history_minutes": steps * 10,
                **counts,
            }
        )

    pd.DataFrame(summary).to_csv(args.out_dir / "sequence_summary.csv", index=False)
    write_readme(args.out_dir, summary, selected_series_ids)
    print(f"Wrote LSTM sequence datasets to {args.out_dir}")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
