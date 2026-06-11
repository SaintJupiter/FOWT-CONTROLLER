#!/usr/bin/env python3
"""Evaluate simple baselines for the ballast-decision wind dataset.

The first baseline is persistence: future wind vector is assumed to stay equal
to the current wind vector at the end of the history window. It is deliberately
simple and dependency-light, but it is essential because any LSTM/GRU/TCN model
should beat this baseline before being useful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ["train", "validation", "test"]
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


def load_current_wind(processed_dir: Path, series_ids: list[str]) -> pd.DataFrame:
    table = source_table_path(processed_dir)
    usecols = ["timestamp", "series_id", "wind_speed_ms", "wind_dir_deg", "wind_u_ms", "wind_v_ms"]
    df = pd.read_csv(table, usecols=usecols, parse_dates=["timestamp"])
    return df[df["series_id"].isin(series_ids)].copy()


def metric_row(name: str, split: str, values: dict[str, float]) -> dict[str, object]:
    return {"baseline": name, "split": split, **values}


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = y_true.astype(bool)
    pred = y_pred.astype(bool)
    tp = int(np.logical_and(true, pred).sum())
    fp = int(np.logical_and(~true, pred).sum())
    fn = int(np.logical_and(true, ~pred).sum())
    tn = int(np.logical_and(~true, ~pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "prevalence": float(true.mean()),
        "predicted_rate": float(pred.mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
    }


def evaluate_split(
    dataset_dir: Path,
    processed_dir: Path,
    split: str,
    current_wind: pd.DataFrame,
    thresholds: dict[str, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sample_index = pd.read_csv(
        dataset_dir / "sample_index.csv.gz",
        usecols=["split", "series_id", "history_end"],
        parse_dates=["history_end"],
    )
    sample_index = sample_index[sample_index["split"].eq(split)].copy()
    sample_index = sample_index.merge(
        current_wind,
        left_on=["series_id", "history_end"],
        right_on=["series_id", "timestamp"],
        how="left",
        validate="many_to_one",
    )
    if sample_index[["wind_speed_ms", "wind_dir_deg", "wind_u_ms", "wind_v_ms"]].isna().any().any():
        raise SystemExit(f"Missing current wind values for split {split}")

    y_uv = np.load(dataset_dir / f"y_uv_raw_{split}.npy", mmap_mode="r")
    y_speed_dir = np.load(dataset_dir / f"y_speed_dir_raw_{split}.npy", mmap_mode="r")
    y_event = np.load(dataset_dir / f"y_event_{split}.npy", mmap_mode="r")

    current_uv = sample_index[["wind_u_ms", "wind_v_ms"]].to_numpy(dtype="float32")
    current_speed = sample_index["wind_speed_ms"].to_numpy(dtype="float32")
    current_dir = sample_index["wind_dir_deg"].to_numpy(dtype="float32")

    pred_uv = current_uv[:, None, :]
    uv_error = y_uv - pred_uv
    speed_error = y_speed_dir[:, :, 0] - current_speed[:, None]
    direction_error = circular_diff_deg(y_speed_dir[:, :, 1], current_dir[:, None])

    rows = [
        metric_row(
            "persistence",
            split,
            {
                "scope": "all_future_steps",
                "vector_mae_ms": float(np.mean(np.abs(uv_error))),
                "vector_rmse_ms": float(np.sqrt(np.mean(uv_error**2))),
                "speed_mae_ms": float(np.mean(np.abs(speed_error))),
                "speed_rmse_ms": float(np.sqrt(np.mean(speed_error**2))),
                "direction_mae_deg": float(np.mean(np.abs(direction_error))),
            },
        )
    ]

    for step in range(y_uv.shape[1]):
        step_uv_error = uv_error[:, step, :]
        step_speed_error = speed_error[:, step]
        step_direction_error = direction_error[:, step]
        rows.append(
            metric_row(
                "persistence",
                split,
                {
                    "scope": f"tplus_{(step + 1) * 10}m",
                    "vector_mae_ms": float(np.mean(np.abs(step_uv_error))),
                    "vector_rmse_ms": float(np.sqrt(np.mean(step_uv_error**2))),
                    "speed_mae_ms": float(np.mean(np.abs(step_speed_error))),
                    "speed_rmse_ms": float(np.sqrt(np.mean(step_speed_error**2))),
                    "direction_mae_deg": float(np.mean(np.abs(step_direction_error))),
                },
            )
        )

    pred_event = np.zeros_like(y_event, dtype="int8")
    pred_event[:, EVENT_COLUMNS.index("future_speed_ge_train_p95")] = (
        current_speed >= thresholds["future_speed_train_p95_ms"]
    ).astype("int8")
    pred_event[:, EVENT_COLUMNS.index("ballast_attention_event")] = pred_event[
        :, EVENT_COLUMNS.index("future_speed_ge_train_p95")
    ]

    event_rows = []
    for idx, event_name in enumerate(EVENT_COLUMNS):
        event_rows.append(
            {
                "baseline": "persistence_event_rule",
                "split": split,
                "event": event_name,
                **binary_metrics(y_event[:, idx], pred_event[:, idx]),
            }
        )
    return rows, event_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/wind_ml_10min"))
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_dwd_helgoland"),
    )
    args = parser.parse_args()

    metadata = json.loads((args.dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    thresholds = json.loads((args.dataset_dir / "event_thresholds.json").read_text(encoding="utf-8"))
    current_wind = load_current_wind(args.processed_dir, metadata["series_ids"])

    metric_rows = []
    event_rows = []
    for split in SPLITS:
        split_metrics, split_events = evaluate_split(args.dataset_dir, args.processed_dir, split, current_wind, thresholds)
        metric_rows.extend(split_metrics)
        event_rows.extend(split_events)

    metrics = pd.DataFrame(metric_rows)
    events = pd.DataFrame(event_rows)
    metrics.to_csv(args.dataset_dir / "baseline_persistence_regression_metrics.csv", index=False)
    events.to_csv(args.dataset_dir / "baseline_persistence_event_metrics.csv", index=False)

    print("Regression metrics")
    print(metrics.to_string(index=False))
    print("\nEvent metrics")
    print(events.to_string(index=False))


if __name__ == "__main__":
    main()
