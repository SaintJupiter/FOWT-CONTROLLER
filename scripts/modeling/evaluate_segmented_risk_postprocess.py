#!/usr/bin/env python3
"""Evaluate segmented wind-change risk from predicted future wind sequences.

This is a lightweight output-design check. It does not retrain a model. It
loads the existing GRU prediction model, converts its 6-step future u/v output
into three lead-time risk bins, then compares those bins with the same risk
definition computed from the observed future wind sequence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_ballast_lstm import (  # noqa: E402
    WindRNN,
    binary_metrics,
    circular_diff_deg,
    choose_device,
    inverse_scale_uv,
    load_dataset,
    predict_split,
    speed_direction_from_uv,
)


SEGMENTS = [
    ("0_20min", slice(0, 2)),
    ("20_40min", slice(2, 4)),
    ("40_60min", slice(4, 6)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/wind_ml_10min/ballast_decision_dwd_helgoland"))
    parser.add_argument("--gru-dir", type=Path, default=Path("outputs/wind_prediction/gru_h128_b2048_e30_dir002"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/wind_prediction/segmented_risk_postprocess"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    return parser.parse_args()


def current_wind(dataset_dir: Path, metadata: dict, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = dataset_dir.parent / "supervised_learning_table_10min.csv.gz"
    index = pd.read_csv(
        dataset_dir / "sample_index.csv.gz",
        usecols=["split", "series_id", "history_end"],
        parse_dates=["history_end"],
    )
    index = index[index["split"].eq(split)].copy()
    obs = pd.read_csv(
        table,
        usecols=["timestamp", "series_id", "wind_speed_ms", "wind_dir_deg", "wind_u_ms", "wind_v_ms"],
        parse_dates=["timestamp"],
    )
    obs = obs[obs["series_id"].isin(metadata["series_ids"])]
    merged = index.merge(
        obs,
        left_on=["series_id", "history_end"],
        right_on=["series_id", "timestamp"],
        how="left",
        validate="many_to_one",
    )
    if merged[["wind_speed_ms", "wind_dir_deg", "wind_u_ms", "wind_v_ms"]].isna().any().any():
        raise SystemExit(f"Missing current wind rows for split={split}.")
    return (
        merged[["wind_u_ms", "wind_v_ms"]].to_numpy(dtype=np.float32),
        merged["wind_speed_ms"].to_numpy(dtype=np.float32),
        merged["wind_dir_deg"].to_numpy(dtype=np.float32),
    )


def load_gru_prediction(model_dir: Path, arrays: dict, metadata: dict, scaler: dict, split: str, batch_size: int, device_name: str) -> np.ndarray:
    cfg = json.loads((model_dir / "lstm_config.json").read_text(encoding="utf-8"))
    device = choose_device(device_name)
    model = WindRNN(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        future_steps=cfg["future_steps"],
        event_count=cfg["event_count"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        model_type=cfg["model_type"],
    ).to(device)
    checkpoint = torch.load(model_dir / "lstm_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    pred_scaled, _ = predict_split(
        model,
        arrays[split],
        device,
        batch_size,
        residual_regression=cfg["residual_regression"],
        uv_feature_indices=tuple(cfg["uv_feature_indices"]),
    )
    return inverse_scale_uv(pred_scaled, scaler)


def segmented_events(
    future_uv: np.ndarray,
    future_speed_dir: np.ndarray,
    current_uv: np.ndarray,
    current_speed: np.ndarray,
    current_dir: np.ndarray,
    thresholds: dict,
) -> dict[str, np.ndarray]:
    speed = future_speed_dir[:, :, 0]
    direction = future_speed_dir[:, :, 1]
    vector_change = np.linalg.norm(future_uv - current_uv[:, None, :], axis=2)
    speed_ramp = np.abs(speed - current_speed[:, None])
    dir_shift = np.abs(circular_diff_deg(direction, current_dir[:, None]))

    out: dict[str, np.ndarray] = {}
    for name, seg in SEGMENTS:
        speed_risk = np.max(speed_ramp[:, seg], axis=1) >= thresholds["speed_ramp_ms"]
        dir_risk = np.max(dir_shift[:, seg], axis=1) >= thresholds["direction_shift_deg"]
        vector_risk = np.max(vector_change[:, seg], axis=1) >= thresholds["vector_change_train_p90_ms"]
        highwind_risk = np.max(speed[:, seg], axis=1) >= thresholds["future_speed_train_p95_ms"]
        out[name] = speed_risk | dir_risk | vector_risk | highwind_risk
    return out


def speed_dir_from_uv_array(uv: np.ndarray) -> np.ndarray:
    speed, direction = speed_direction_from_uv(uv)
    return np.stack([speed, direction], axis=2).astype(np.float32)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, metadata, scaler = load_dataset(args.dataset_dir)
    thresholds = json.loads((args.dataset_dir / "event_thresholds.json").read_text(encoding="utf-8"))

    current_uv, current_speed, current_dir = current_wind(args.dataset_dir, metadata, args.split)
    true_uv = np.asarray(arrays[args.split]["y_uv_raw"])
    true_speed_dir = np.asarray(arrays[args.split]["y_speed_dir_raw"])
    gru_uv = load_gru_prediction(args.gru_dir, arrays, metadata, scaler, args.split, args.batch_size, args.device)
    gru_speed_dir = speed_dir_from_uv_array(gru_uv)
    persistence_uv = np.repeat(current_uv[:, None, :], repeats=metadata["future_steps"], axis=1)
    persistence_speed_dir = speed_dir_from_uv_array(persistence_uv)

    true_events = segmented_events(true_uv, true_speed_dir, current_uv, current_speed, current_dir, thresholds)
    model_events = {
        "persistence_postprocess": segmented_events(persistence_uv, persistence_speed_dir, current_uv, current_speed, current_dir, thresholds),
        "gru_sequence_postprocess": segmented_events(gru_uv, gru_speed_dir, current_uv, current_speed, current_dir, thresholds),
    }

    rows = []
    for model_name, events in model_events.items():
        for segment_name, true_mask in true_events.items():
            pred_mask = events[segment_name]
            rows.append({
                "model": model_name,
                "segment": segment_name,
                "prevalence": float(true_mask.mean()),
                **binary_metrics(true_mask.astype(np.int8), pred_mask.astype(np.int8)),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / f"{args.split}_segmented_risk_metrics.csv", index=False)
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "risk_definition": "segment event is speed ramp, direction shift, vector change, or high wind relative to current wind state",
                "thresholds": thresholds,
                "metrics": str(args.output_dir / f"{args.split}_segmented_risk_metrics.csv"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
