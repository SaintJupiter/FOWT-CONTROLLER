#!/usr/bin/env python3
"""Evaluate wind prediction models by operating regime and event threshold.

The main average metrics hide the behavior that matters for active ballast
preview: high wind, fast ramps, direction shifts, and vector changes. This
script reloads the current GRU and LightGBM models, evaluates them on the same
test split, and writes:

- regression metrics by wind regime;
- ballast-attention classification metrics by wind regime;
- threshold trade-off curves for the ballast-attention event.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
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
    uv_feature_indices,
)
from train_lightgbm_wind_baseline import (  # noqa: E402
    flatten_features,
    full_scaled_prediction,
    predict_events,
    predict_regression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_dwd_helgoland"),
    )
    parser.add_argument(
        "--gru-dir",
        type=Path,
        default=Path("outputs/wind_prediction/gru_h128_b2048_e30_dir002"),
    )
    parser.add_argument(
        "--lightgbm-dir",
        type=Path,
        default=Path("outputs/wind_prediction/lightgbm_tabular_residual_e300"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wind_prediction/regime_threshold_analysis"),
    )
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    return parser.parse_args()


def load_gru_predictions(
    model_dir: Path,
    arrays: dict[str, dict[str, np.ndarray]],
    metadata: dict,
    scaler: dict,
    split: str,
    batch_size: int,
    requested_device: str,
) -> tuple[np.ndarray, np.ndarray]:
    config = json.loads((model_dir / "lstm_config.json").read_text(encoding="utf-8"))
    device = choose_device(requested_device)
    model = WindRNN(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        future_steps=config["future_steps"],
        event_count=config["event_count"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        model_type=config["model_type"],
    ).to(device)
    checkpoint = torch.load(model_dir / "lstm_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    uv_indices = tuple(config["uv_feature_indices"])
    pred_scaled, pred_event_prob = predict_split(
        model,
        arrays[split],
        device,
        batch_size,
        residual_regression=config["residual_regression"],
        uv_feature_indices=uv_indices,
    )
    return inverse_scale_uv(pred_scaled, scaler), pred_event_prob


def load_lightgbm_predictions(
    model_dir: Path,
    arrays: dict[str, dict[str, np.ndarray]],
    metadata: dict,
    scaler: dict,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    config = json.loads((model_dir / "lightgbm_config.json").read_text(encoding="utf-8"))
    bundle = joblib.load(model_dir / "lightgbm_models.joblib")
    x = flatten_features(arrays[split]["X"])
    pred_flat = predict_regression(bundle["regression"], x)
    pred_scaled = full_scaled_prediction(
        pred_flat,
        arrays[split]["X"],
        metadata["future_steps"],
        residual_regression=config["residual_regression"],
        uv_indices=uv_feature_indices(metadata),
    )
    pred_event_prob = predict_events(bundle["events"], x)
    return inverse_scale_uv(pred_scaled, scaler), pred_event_prob


def persistence_predictions(arrays: dict[str, np.ndarray], metadata: dict) -> np.ndarray:
    uv_indices = uv_feature_indices(metadata)
    current_scaled = np.asarray(arrays["X"][:, -1, list(uv_indices)], dtype=np.float32)
    scaler_path = metadata.get("unused", None)
    del scaler_path
    # X is scaled, so use the raw current u/v from y target anchor indirectly:
    # persistence metrics here are computed from the already generated raw
    # baseline files in the data pipeline when possible. For regime masks we can
    # still reconstruct current raw wind from y at t+10 persistence assumption
    # less accurately, so the caller supplies baseline CSV for headline use.
    raise NotImplementedError("Use persistence_raw_predictions_from_sample_index instead.")


def persistence_raw_predictions_from_sample_index(dataset_dir: Path, metadata: dict, split: str) -> np.ndarray:
    processed_dir = dataset_dir.parent
    table_path = processed_dir / "supervised_learning_table_10min.csv.gz"
    sample_index = pd.read_csv(
        dataset_dir / "sample_index.csv.gz",
        usecols=["split", "series_id", "history_end"],
        parse_dates=["history_end"],
    )
    sample_index = sample_index[sample_index["split"].eq(split)].copy()
    current = pd.read_csv(
        table_path,
        usecols=["timestamp", "series_id", "wind_u_ms", "wind_v_ms"],
        parse_dates=["timestamp"],
    )
    current = current[current["series_id"].isin(metadata["series_ids"])]
    merged = sample_index.merge(
        current,
        left_on=["series_id", "history_end"],
        right_on=["series_id", "timestamp"],
        how="left",
        validate="many_to_one",
    )
    if merged[["wind_u_ms", "wind_v_ms"]].isna().any().any():
        raise SystemExit(f"Missing current wind values for {split} persistence predictions.")
    current_uv = merged[["wind_u_ms", "wind_v_ms"]].to_numpy(dtype=np.float32)
    return np.repeat(current_uv[:, None, :], repeats=metadata["future_steps"], axis=1)


def regime_masks(summary: np.ndarray, y_event: np.ndarray, metadata: dict, thresholds: dict) -> dict[str, np.ndarray]:
    summary_cols = metadata["sample_summary_columns"]
    event_cols = metadata["event_columns"]
    speed_max = summary[:, summary_cols.index("future_speed_max_ms")]
    speed_ramp = summary[:, summary_cols.index("future_speed_ramp_max_ms")]
    dir_shift = summary[:, summary_cols.index("future_dir_shift_abs_max_deg")]
    vector_change = summary[:, summary_cols.index("future_vector_change_max_ms")]
    attention = y_event[:, event_cols.index("ballast_attention_event")].astype(bool)

    masks = {
        "all": np.ones(len(summary), dtype=bool),
        "stable_no_attention": ~attention,
        "attention_event": attention,
        "speed_ramp_ge_3ms": speed_ramp >= thresholds["speed_ramp_ms"],
        "direction_shift_ge_45deg": dir_shift >= thresholds["direction_shift_deg"],
        "vector_change_ge_train_p90": vector_change >= thresholds["vector_change_train_p90_ms"],
        "future_speed_ge_train_p95": speed_max >= thresholds["future_speed_train_p95_ms"],
        "compound_change_no_highwind": attention & (speed_max < thresholds["future_speed_train_p95_ms"]),
    }
    return masks


def regression_row(
    model_name: str,
    regime: str,
    mask: np.ndarray,
    pred_uv_raw: np.ndarray,
    true_uv_raw: np.ndarray,
    true_speed_dir: np.ndarray,
) -> dict[str, object]:
    pred_speed, pred_dir = speed_direction_from_uv(pred_uv_raw[mask])
    true_speed = true_speed_dir[mask, :, 0]
    true_dir = true_speed_dir[mask, :, 1]
    uv_error = pred_uv_raw[mask] - true_uv_raw[mask]
    speed_error = pred_speed - true_speed
    direction_error = circular_diff_deg(pred_dir, true_dir)
    return {
        "model": model_name,
        "regime": regime,
        "sample_count": int(mask.sum()),
        "sample_share": float(mask.mean()),
        "vector_mae_ms": float(np.mean(np.abs(uv_error))),
        "speed_mae_ms": float(np.mean(np.abs(speed_error))),
        "direction_mae_deg": float(np.mean(np.abs(direction_error))),
        "tplus60_vector_mae_ms": float(np.mean(np.abs(uv_error[:, -1, :]))),
        "tplus60_speed_mae_ms": float(np.mean(np.abs(speed_error[:, -1]))),
        "tplus60_direction_mae_deg": float(np.mean(np.abs(direction_error[:, -1]))),
    }


def event_row(
    model_name: str,
    regime: str,
    mask: np.ndarray,
    y_true_event: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    metrics = binary_metrics(y_true_event[mask], prob[mask] >= threshold)
    return {
        "model": model_name,
        "regime": regime,
        "sample_count": int(mask.sum()),
        "sample_share": float(mask.mean()),
        "threshold": float(threshold),
        **metrics,
    }


def threshold_rows(
    model_name: str,
    y_true_event: np.ndarray,
    prob: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for threshold in thresholds:
        rows.append({"model": model_name, "threshold": float(threshold), **binary_metrics(y_true_event, prob >= threshold)})
    return rows


def load_attention_threshold(model_dir: Path, prefix: str, metadata: dict) -> float:
    path = model_dir / f"{prefix}_event_thresholds.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["thresholds"]["ballast_attention_event"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, metadata, scaler = load_dataset(args.dataset_dir)
    thresholds = json.loads((args.dataset_dir / "event_thresholds.json").read_text(encoding="utf-8"))

    split_arrays = arrays[args.split]
    y_uv_raw = np.asarray(split_arrays["y_uv_raw"])
    y_speed_dir = np.asarray(split_arrays["y_speed_dir_raw"])
    y_event = np.asarray(split_arrays["y_event"])
    summary = np.asarray(np.load(args.dataset_dir / f"y_summary_raw_{args.split}.npy", mmap_mode="r"))
    attention_idx = metadata["event_columns"].index("ballast_attention_event")
    y_attention = y_event[:, attention_idx]
    masks = regime_masks(summary, y_event, metadata, thresholds)

    predictions = {
        "persistence": {
            "uv": persistence_raw_predictions_from_sample_index(args.dataset_dir, metadata, args.split),
            "prob": None,
            "threshold": None,
        },
        "residual_gru": {
            "uv": None,
            "prob": None,
            "threshold": load_attention_threshold(args.gru_dir, "lstm", metadata),
        },
        "residual_lightgbm": {
            "uv": None,
            "prob": None,
            "threshold": load_attention_threshold(args.lightgbm_dir, "lightgbm", metadata),
        },
    }

    gru_uv, gru_prob = load_gru_predictions(
        args.gru_dir,
        arrays,
        metadata,
        scaler,
        args.split,
        args.batch_size,
        args.device,
    )
    predictions["residual_gru"]["uv"] = gru_uv
    predictions["residual_gru"]["prob"] = gru_prob[:, attention_idx]

    lightgbm_uv, lightgbm_prob = load_lightgbm_predictions(args.lightgbm_dir, arrays, metadata, scaler, args.split)
    predictions["residual_lightgbm"]["uv"] = lightgbm_uv
    predictions["residual_lightgbm"]["prob"] = lightgbm_prob[:, attention_idx]

    reg_rows = []
    event_rows = []
    threshold_tradeoff_rows = []
    for model_name, pred in predictions.items():
        for regime_name, mask in masks.items():
            if mask.sum() == 0:
                continue
            reg_rows.append(regression_row(model_name, regime_name, mask, pred["uv"], y_uv_raw, y_speed_dir))
            if pred["prob"] is not None:
                event_rows.append(
                    event_row(
                        model_name,
                        regime_name,
                        mask,
                        y_attention,
                        pred["prob"],
                        float(pred["threshold"]),
                    )
                )
        if pred["prob"] is not None:
            threshold_tradeoff_rows.extend(
                threshold_rows(model_name, y_attention, pred["prob"], np.linspace(0.05, 0.95, 19, dtype=np.float32))
            )

    reg_frame = pd.DataFrame(reg_rows)
    event_frame = pd.DataFrame(event_rows)
    threshold_frame = pd.DataFrame(threshold_tradeoff_rows)

    reg_frame.to_csv(args.output_dir / f"{args.split}_regression_by_regime.csv", index=False)
    event_frame.to_csv(args.output_dir / f"{args.split}_attention_event_by_regime.csv", index=False)
    threshold_frame.to_csv(args.output_dir / f"{args.split}_attention_threshold_tradeoff.csv", index=False)

    summary_payload = {
        "split": args.split,
        "dataset_dir": str(args.dataset_dir),
        "gru_dir": str(args.gru_dir),
        "lightgbm_dir": str(args.lightgbm_dir),
        "output_files": {
            "regression_by_regime": str(args.output_dir / f"{args.split}_regression_by_regime.csv"),
            "attention_event_by_regime": str(args.output_dir / f"{args.split}_attention_event_by_regime.csv"),
            "attention_threshold_tradeoff": str(args.output_dir / f"{args.split}_attention_threshold_tradeoff.csv"),
        },
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nRegression by regime")
    print(reg_frame.to_string(index=False))
    print("\nAttention event by regime")
    print(event_frame.to_string(index=False))
    print("\nAttention threshold tradeoff")
    print(threshold_frame.to_string(index=False))


if __name__ == "__main__":
    main()
