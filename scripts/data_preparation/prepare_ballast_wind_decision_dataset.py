#!/usr/bin/env python3
"""Build wind forecasting tensors for active-ballast preview decisions.

This dataset extends the basic one-step LSTM table into a multi-step,
decision-oriented sequence dataset:

- past wind history -> future wind-vector sequence;
- future wind-change event labels for an auxiliary classification head;
- multi-resolution historical statistics retained as input features.

The implementation keeps time-series safeguards explicit: chronological splits
are respected, windows never cross a split/series boundary, and all scaling is
fit on training data only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TEN_MINUTES_SECONDS = 600
SPLITS = ["train", "validation", "test"]
DEFAULT_SERIES_ID = "DWD::02115_Helgoland::10min_wind"

FEATURE_COLUMNS = [
    "wind_speed_ms",
    "wind_dir_sin",
    "wind_dir_cos",
    "wind_u_ms",
    "wind_v_ms",
    "wind_speed_delta_10m",
    "wind_dir_delta_10m_deg",
    "wind_u_delta_10m",
    "wind_v_delta_10m",
    "wind_speed_mean_30m",
    "wind_speed_std_30m",
    "wind_u_mean_30m",
    "wind_v_mean_30m",
    "wind_dir_sin_mean_30m",
    "wind_dir_cos_mean_30m",
    "wind_speed_mean_60m",
    "wind_speed_std_60m",
    "wind_u_mean_60m",
    "wind_v_mean_60m",
    "wind_dir_sin_mean_60m",
    "wind_dir_cos_mean_60m",
    "wind_speed_mean_120m",
    "wind_speed_std_120m",
    "wind_u_mean_120m",
    "wind_v_mean_120m",
    "wind_dir_sin_mean_120m",
    "wind_dir_cos_mean_120m",
    "hour_sin",
    "hour_cos",
    "dayofyear_sin",
    "dayofyear_cos",
]

FUTURE_VALUE_COLUMNS = ["wind_u_ms", "wind_v_ms", "wind_speed_ms", "wind_dir_deg"]
TARGET_UV_COLUMNS = ["wind_u_ms", "wind_v_ms"]
SAMPLE_SUMMARY_COLUMNS = [
    "future_speed_max_ms",
    "future_speed_ramp_max_ms",
    "future_dir_shift_abs_max_deg",
    "future_vector_change_max_ms",
]
EVENT_COLUMNS = [
    "speed_ramp_ge_3ms",
    "direction_shift_ge_45deg",
    "vector_change_ge_train_p90",
    "future_speed_ge_train_p95",
    "ballast_attention_event",
]


def source_table_path(processed_dir: Path) -> Path:
    gz_path = processed_dir / "supervised_learning_table_10min.csv.gz"
    if gz_path.exists():
        return gz_path
    return processed_dir / "supervised_learning_table_10min.csv"


def circular_diff_deg(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    return (target - base + 180.0) % 360.0 - 180.0


def standardizer(values: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for column in values.columns:
        series = pd.to_numeric(values[column], errors="coerce")
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        stats[column] = {"mean": mean, "std": std}
    return stats


def apply_standardizer(frame: pd.DataFrame, stats: dict[str, dict[str, float]], columns: list[str]) -> np.ndarray:
    values = frame[columns].astype("float32").to_numpy(copy=True)
    mean = np.array([stats[col]["mean"] for col in columns], dtype="float32")
    std = np.array([stats[col]["std"] for col in columns], dtype="float32")
    return (values - mean) / std


def scale_uv(values: np.ndarray, stats: dict[str, dict[str, float]]) -> np.ndarray:
    mean = np.array([stats[col]["mean"] for col in TARGET_UV_COLUMNS], dtype="float32")
    std = np.array([stats[col]["std"] for col in TARGET_UV_COLUMNS], dtype="float32")
    return (values.astype("float32") - mean) / std


def valid_history_end_indices(group: pd.DataFrame, history_steps: int, future_steps: int) -> np.ndarray:
    n = len(group)
    if n < history_steps + future_steps:
        return np.array([], dtype=np.int64)

    timestamps = group["timestamp"].to_numpy(dtype="datetime64[s]").astype("int64")
    step_ok = np.zeros(n, dtype=np.int8)
    step_ok[1:] = (np.diff(timestamps) == TEN_MINUTES_SECONDS).astype(np.int8)
    step_prefix = np.concatenate([[0], np.cumsum(step_ok, dtype=np.int64)])

    feature_ok = group[FEATURE_COLUMNS].notna().all(axis=1).to_numpy(dtype=np.int8)
    target_ok = group[FUTURE_VALUE_COLUMNS].notna().all(axis=1).to_numpy(dtype=np.int8)
    feature_prefix = np.concatenate([[0], np.cumsum(feature_ok, dtype=np.int64)])
    target_prefix = np.concatenate([[0], np.cumsum(target_ok, dtype=np.int64)])

    ends = np.arange(history_steps - 1, n - future_steps, dtype=np.int64)
    starts = ends - history_steps + 1
    future_ends = ends + future_steps

    continuous = (step_prefix[future_ends + 1] - step_prefix[starts + 1]) == (history_steps + future_steps - 1)
    inputs_valid = (feature_prefix[ends + 1] - feature_prefix[starts]) == history_steps
    future_valid = (target_prefix[future_ends + 1] - target_prefix[ends + 1]) == future_steps
    return ends[continuous & inputs_valid & future_valid]


def sample_summaries(group: pd.DataFrame, ends: np.ndarray, future_steps: int) -> np.ndarray:
    speed = group["wind_speed_ms"].astype("float64").to_numpy()
    direction = group["wind_dir_deg"].astype("float64").to_numpy()
    u = group["wind_u_ms"].astype("float64").to_numpy()
    v = group["wind_v_ms"].astype("float64").to_numpy()

    future_offsets = np.arange(1, future_steps + 1, dtype=np.int64)
    future_idx = ends[:, None] + future_offsets[None, :]

    current_speed = speed[ends][:, None]
    current_dir = direction[ends][:, None]
    current_u = u[ends][:, None]
    current_v = v[ends][:, None]

    future_speed = speed[future_idx]
    future_dir = direction[future_idx]
    future_u = u[future_idx]
    future_v = v[future_idx]

    speed_ramp_max = np.max(future_speed - current_speed, axis=1)
    direction_shift_abs_max = np.max(np.abs(circular_diff_deg(future_dir, current_dir)), axis=1)
    vector_change_max = np.max(np.sqrt((future_u - current_u) ** 2 + (future_v - current_v) ** 2), axis=1)
    future_speed_max = np.max(future_speed, axis=1)

    return np.column_stack(
        [future_speed_max, speed_ramp_max, direction_shift_abs_max, vector_change_max]
    ).astype("float32")


def event_labels(summaries: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    speed_ramp = summaries[:, 1] >= thresholds["speed_ramp_ms"]
    direction_shift = summaries[:, 2] >= thresholds["direction_shift_deg"]
    vector_change = summaries[:, 3] >= thresholds["vector_change_train_p90_ms"]
    high_speed = summaries[:, 0] >= thresholds["future_speed_train_p95_ms"]
    attention = speed_ramp | direction_shift | vector_change | high_speed
    return np.column_stack([speed_ramp, direction_shift, vector_change, high_speed, attention]).astype("int8")


def collect_counts_and_thresholds(
    df: pd.DataFrame, history_steps: int, future_steps: int
) -> tuple[dict[str, int], dict[str, float], list[pd.DataFrame]]:
    counts = {split: 0 for split in SPLITS}
    train_summaries: list[np.ndarray] = []
    index_records: list[pd.DataFrame] = []

    for (series_id, split), group in df.groupby(["series_id", "split"], sort=False):
        if split not in counts:
            continue
        group = group.sort_values("timestamp").reset_index(drop=True)
        ends = valid_history_end_indices(group, history_steps, future_steps)
        counts[split] += len(ends)
        if len(ends) == 0:
            continue

        summaries = sample_summaries(group, ends, future_steps)
        if split == "train":
            train_summaries.append(summaries)
        index_records.append(
            pd.DataFrame(
                {
                    "split": split,
                    "series_id": series_id,
                    "history_start": group.loc[ends - history_steps + 1, "timestamp"].to_numpy(),
                    "history_end": group.loc[ends, "timestamp"].to_numpy(),
                    "future_start": group.loc[ends + 1, "timestamp"].to_numpy(),
                    "future_end": group.loc[ends + future_steps, "timestamp"].to_numpy(),
                    **{col: summaries[:, i] for i, col in enumerate(SAMPLE_SUMMARY_COLUMNS)},
                }
            )
        )

    if not train_summaries:
        raise SystemExit("No valid training samples; cannot derive event thresholds.")

    train_summary = np.vstack(train_summaries)
    thresholds = {
        "speed_ramp_ms": 3.0,
        "direction_shift_deg": 45.0,
        "future_speed_train_p95_ms": float(np.quantile(train_summary[:, 0], 0.95)),
        "vector_change_train_p90_ms": float(np.quantile(train_summary[:, 3], 0.90)),
        "threshold_source": "fixed engineering bins for ramp/direction plus empirical train-only quantiles",
    }
    return counts, thresholds, index_records


def build_dataset(
    df: pd.DataFrame,
    out_dir: Path,
    history_steps: int,
    future_steps: int,
    feature_stats: dict[str, dict[str, float]],
    target_stats: dict[str, dict[str, float]],
    thresholds: dict[str, float],
    counts: dict[str, int],
    selected_series_ids: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    offsets = {split: 0 for split in SPLITS}

    for split in SPLITS:
        n = counts[split]
        arrays[(split, "X")] = np.lib.format.open_memmap(
            out_dir / f"X_{split}.npy",
            mode="w+",
            dtype="float32",
            shape=(n, history_steps, len(FEATURE_COLUMNS)),
        )
        arrays[(split, "y_uv")] = np.lib.format.open_memmap(
            out_dir / f"y_uv_{split}.npy",
            mode="w+",
            dtype="float32",
            shape=(n, future_steps, len(TARGET_UV_COLUMNS)),
        )
        arrays[(split, "y_uv_raw")] = np.lib.format.open_memmap(
            out_dir / f"y_uv_raw_{split}.npy",
            mode="w+",
            dtype="float32",
            shape=(n, future_steps, len(TARGET_UV_COLUMNS)),
        )
        arrays[(split, "y_speed_dir_raw")] = np.lib.format.open_memmap(
            out_dir / f"y_speed_dir_raw_{split}.npy",
            mode="w+",
            dtype="float32",
            shape=(n, future_steps, 2),
        )
        arrays[(split, "y_event")] = np.lib.format.open_memmap(
            out_dir / f"y_event_{split}.npy",
            mode="w+",
            dtype="int8",
            shape=(n, len(EVENT_COLUMNS)),
        )
        arrays[(split, "y_summary_raw")] = np.lib.format.open_memmap(
            out_dir / f"y_summary_raw_{split}.npy",
            mode="w+",
            dtype="float32",
            shape=(n, len(SAMPLE_SUMMARY_COLUMNS)),
        )

    index_parts = []
    for (series_id, split), group in df.groupby(["series_id", "split"], sort=False):
        if split not in SPLITS:
            continue
        group = group.sort_values("timestamp").reset_index(drop=True)
        ends = valid_history_end_indices(group, history_steps, future_steps)
        if len(ends) == 0:
            continue

        x_rows = apply_standardizer(group, feature_stats, FEATURE_COLUMNS)
        uv_rows_raw = group[TARGET_UV_COLUMNS].astype("float32").to_numpy()
        uv_rows_scaled = scale_uv(uv_rows_raw, target_stats)
        speed_dir_rows_raw = group[["wind_speed_ms", "wind_dir_deg"]].astype("float32").to_numpy()
        summaries = sample_summaries(group, ends, future_steps)
        events = event_labels(summaries, thresholds)

        future_offsets = np.arange(1, future_steps + 1, dtype=np.int64)
        start_offset = offsets[split]
        for local_idx, end in enumerate(ends):
            out_idx = start_offset + local_idx
            hist_start = end - history_steps + 1
            future_idx = end + future_offsets
            arrays[(split, "X")][out_idx] = x_rows[hist_start : end + 1]
            arrays[(split, "y_uv")][out_idx] = uv_rows_scaled[future_idx]
            arrays[(split, "y_uv_raw")][out_idx] = uv_rows_raw[future_idx]
            arrays[(split, "y_speed_dir_raw")][out_idx] = speed_dir_rows_raw[future_idx]
            arrays[(split, "y_event")][out_idx] = events[local_idx]
            arrays[(split, "y_summary_raw")][out_idx] = summaries[local_idx]

        offsets[split] += len(ends)
        index_parts.append(
            pd.DataFrame(
                {
                    "split": split,
                    "series_id": series_id,
                    "history_start": group.loc[ends - history_steps + 1, "timestamp"].to_numpy(),
                    "history_end": group.loc[ends, "timestamp"].to_numpy(),
                    "future_start": group.loc[ends + 1, "timestamp"].to_numpy(),
                    "future_end": group.loc[ends + future_steps, "timestamp"].to_numpy(),
                    **{col: summaries[:, i] for i, col in enumerate(SAMPLE_SUMMARY_COLUMNS)},
                    **{col: events[:, i] for i, col in enumerate(EVENT_COLUMNS)},
                }
            )
        )

    for arr in arrays.values():
        arr.flush()

    sample_index = pd.concat(index_parts, ignore_index=True) if index_parts else pd.DataFrame()
    sample_index.to_csv(out_dir / "sample_index.csv.gz", index=False, compression="gzip")

    metadata = {
        "dataset_role": "wind preview dataset for active ballast decision support",
        "series_ids": selected_series_ids,
        "input_resolution_minutes": 10,
        "history_steps": history_steps,
        "nominal_history_minutes": history_steps * 10,
        "history_timestamp_span_minutes": (history_steps - 1) * 10,
        "future_steps": future_steps,
        "future_minutes": future_steps * 10,
        "feature_columns": FEATURE_COLUMNS,
        "target_sequence_columns": TARGET_UV_COLUMNS,
        "raw_speed_direction_sequence_columns": ["wind_speed_ms", "wind_dir_deg"],
        "event_columns": EVENT_COLUMNS,
        "sample_summary_columns": SAMPLE_SUMMARY_COLUMNS,
        "event_thresholds": thresholds,
        "arrays": {
            split: {
                "X": f"X_{split}.npy",
                "y_uv": f"y_uv_{split}.npy",
                "y_uv_raw": f"y_uv_raw_{split}.npy",
                "y_speed_dir_raw": f"y_speed_dir_raw_{split}.npy",
                "y_event": f"y_event_{split}.npy",
                "y_summary_raw": f"y_summary_raw_{split}.npy",
                "shape_X": [counts[split], history_steps, len(FEATURE_COLUMNS)],
                "shape_y_uv": [counts[split], future_steps, len(TARGET_UV_COLUMNS)],
                "shape_y_event": [counts[split], len(EVENT_COLUMNS)],
            }
            for split in SPLITS
        },
        "scaling_rule": "Feature and target z-score scalers are fit on training rows only.",
        "continuity_rule": "Samples require exact 10-minute continuity across both history and future windows; no sample crosses series_id or split boundaries.",
        "literature_mapping": {
            "multi_step_forecast": "Fuentes-Barrios et al. 2022: 10-minute input sequence to multi-step nowcasting output.",
            "wind_direction_encoding": "Sari et al. 2021 and Blazakis et al. 2025: avoid raw angular regression; use vector/circular representation.",
            "dual_output_events": "Mode et al. 2025: regression plus event-classification framing for extreme wind.",
            "multi_resolution_features": "Feng et al. 2025: multi-resolution wind history improves uncertainty-aware forecasting.",
            "leakage_control": "Liu & Chen 2019 plus data-leakage cautions: train-only scaling and chronological splits.",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def write_readme(
    out_dir: Path,
    counts: dict[str, int],
    thresholds: dict[str, float],
    selected_series_ids: list[str],
    history_steps: int,
    future_steps: int,
) -> None:
    lines = [
        "# Ballast-decision wind forecasting dataset",
        "",
        "Generated by `scripts/data_preparation/prepare_ballast_wind_decision_dataset.py`.",
        "",
        "This dataset is intended for the first active-ballast preview model: historical 10-minute wind sequences are used to predict a future wind-vector sequence and classify future wind-change events.",
        "",
        "## Scope",
        "",
        f"- Included series: {', '.join(f'`{s}`' for s in selected_series_ids)}",
        f"- Input history: `{history_steps}` samples, nominal `{history_steps * 10}` minutes.",
        f"- Future output: `{future_steps}` samples, `{future_steps * 10}` minutes.",
        "- Input features include instantaneous wind-vector variables plus 30/60/120 minute rolling history statistics.",
        "- Regression target is future `u/v` sequence. `y_speed_dir_raw` stores future speed and direction for metric reporting.",
        "- Classification target is a compact set of future wind-change events relevant to whether ballast pre-adjustment should be considered.",
        "",
        "## Split counts",
        "",
    ]
    for split in SPLITS:
        lines.append(f"- `{split}`: `{counts[split]}` samples")

    lines += [
        "",
        "## Event thresholds",
        "",
        f"- Speed ramp event: future max speed increase >= `{thresholds['speed_ramp_ms']:.2f}` m/s.",
        f"- Direction shift event: future max absolute direction shift >= `{thresholds['direction_shift_deg']:.1f}` degrees.",
        f"- Vector-change event: future max vector change >= train p90 `{thresholds['vector_change_train_p90_ms']:.3f}` m/s.",
        f"- High-wind event: future max speed >= train p95 `{thresholds['future_speed_train_p95_ms']:.3f}` m/s.",
        "- `ballast_attention_event` is the union of the above events.",
        "",
        "## Literature-informed choices",
        "",
        "- Fuentes-Barrios et al. 2022 supports a 10-minute sliding-window, multi-step nowcasting setup.",
        "- Sari et al. 2021 and Blazakis et al. 2025 support vector/circular wind-direction handling instead of direct 0-360 degree regression.",
        "- Mode et al. 2025 supports adding an event-classification output alongside wind regression for rare/high-impact wind changes.",
        "- Feng et al. 2025 motivates keeping multi-resolution historical wind features for uncertainty-sensitive prediction.",
        "- Data-processing reviews and leakage cautions motivate chronological splits and train-only normalization.",
        "",
        "## Recommended first model",
        "",
        "Use `X_*` as input. Train one recurrent/temporal model with two heads:",
        "",
        "- regression head: `y_uv_*`, evaluated against `y_uv_raw_*` and `y_speed_dir_raw_*`;",
        "- classification head: `y_event_*`, with emphasis on recall and false alarm rate for `ballast_attention_event`.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/wind_ml_10min"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_dwd_helgoland"),
    )
    parser.add_argument("--series-id", action="append", default=[DEFAULT_SERIES_ID])
    parser.add_argument("--history-steps", type=int, default=12, help="10-minute samples in the input history.")
    parser.add_argument("--future-steps", type=int, default=6, help="10-minute samples in the future output sequence.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = source_table_path(args.processed_dir)
    usecols = sorted(set(["timestamp", "series_id", "split", *FEATURE_COLUMNS, *FUTURE_VALUE_COLUMNS]))
    df = pd.read_csv(table, usecols=usecols, parse_dates=["timestamp"])
    df = df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)

    source_counts = df.groupby("series_id").size().reset_index(name="source_rows")
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
    target_stats = standardizer(train[TARGET_UV_COLUMNS])

    counts, thresholds, _index_preview = collect_counts_and_thresholds(df, args.history_steps, args.future_steps)

    scaler = {
        "feature_scaler": feature_stats,
        "target_uv_scaler": target_stats,
        "fit_scope": "training rows only",
        "method": "z_score",
        "series_ids": selected_series_ids,
    }
    (args.out_dir / "scaler_train.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    (args.out_dir / "event_thresholds.json").write_text(
        json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    build_dataset(
        df=df,
        out_dir=args.out_dir,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        feature_stats=feature_stats,
        target_stats=target_stats,
        thresholds=thresholds,
        counts=counts,
        selected_series_ids=selected_series_ids,
    )

    summary = pd.DataFrame(
        [
            {
                "dataset": "history_120m_future_60m_decision",
                "history_steps": args.history_steps,
                "history_minutes": args.history_steps * 10,
                "future_steps": args.future_steps,
                "future_minutes": args.future_steps * 10,
                **counts,
            }
        ]
    )
    summary.to_csv(args.out_dir / "dataset_summary.csv", index=False)
    write_readme(args.out_dir, counts, thresholds, selected_series_ids, args.history_steps, args.future_steps)

    print(f"Wrote ballast-decision wind dataset to {args.out_dir}")
    print(summary.to_string(index=False))
    print(json.dumps(thresholds, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
