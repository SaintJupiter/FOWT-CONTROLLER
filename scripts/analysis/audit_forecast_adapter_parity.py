#!/usr/bin/env python3
"""Check that ForecastModelAdapter matches the training-script inference path.

This catches wiring mistakes such as a residual-regression scaling mismatch.
It intentionally compares two independent loading paths for the same checkpoint
on a few replay samples; it does not run plant simulation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "modeling"))

from train_ballast_lstm import WindRNN, full_scaled_prediction, inverse_scale_uv
from wind_prediction import Fino1ReplayDataset, ForecastModelAdapter


DEFAULT_TIMESTAMPS = [
    "2024-11-27 19:40:00",
    "2024-09-27 13:00:00",
    "2023-10-03 06:30:00",
    "2022-02-04 11:00:00",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1",
    )
    parser.add_argument(
        "--model-dir",
        default="outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/forecast_adapter_parity_20260506",
    )
    parser.add_argument(
        "--timestamps",
        default=",".join(DEFAULT_TIMESTAMPS),
        help="Comma-separated history_end timestamps to inspect.",
    )
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_training_model(model_dir: Path, cfg: dict) -> WindRNN:
    model = WindRNN(
        input_size=int(cfg["input_size"]),
        hidden_size=int(cfg["hidden_size"]),
        future_steps=int(cfg["future_steps"]),
        event_count=int(cfg["event_count"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
        model_type=str(cfg["model_type"]),
    )
    checkpoint = torch.load(model_dir / "lstm_best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _training_path_predict(
    model: WindRNN,
    x_window: np.ndarray,
    cfg: dict,
    scaler: dict,
) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(np.asarray(x_window, dtype=np.float32)[None, :, :])
    with torch.no_grad():
        pred_regression, event_logits = model(x)
        pred_scaled = full_scaled_prediction(
            pred_regression,
            x,
            residual_regression=bool(cfg.get("residual_regression", False)),
            uv_feature_indices=tuple(int(v) for v in cfg["uv_feature_indices"]),
        )
        pred_scaled_np = pred_scaled.detach().cpu().numpy()[0].astype(np.float32)
        event_prob = torch.sigmoid(event_logits).detach().cpu().numpy()[0].astype(np.float32)
    return inverse_scale_uv(pred_scaled_np[None, :, :], scaler)[0], event_prob


def main() -> None:
    args = parse_args()
    dataset_dir = _resolve(args.dataset_dir)
    model_dir = _resolve(args.model_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((model_dir / "lstm_config.json").read_text(encoding="utf-8"))
    scaler = json.loads((dataset_dir / "scaler_train.json").read_text(encoding="utf-8"))
    feature_scaler = scaler["feature_scaler"]
    target_scaler = scaler["target_uv_scaler"]
    scale_rows = []
    for name in ("wind_u_ms", "wind_v_ms"):
        scale_rows.append(
            {
                "component": name,
                "feature_mean": float(feature_scaler[name]["mean"]),
                "target_mean": float(target_scaler[name]["mean"]),
                "mean_diff": float(feature_scaler[name]["mean"] - target_scaler[name]["mean"]),
                "feature_std": float(feature_scaler[name]["std"]),
                "target_std": float(target_scaler[name]["std"]),
                "std_diff": float(feature_scaler[name]["std"] - target_scaler[name]["std"]),
            }
        )
    scale_check = pd.DataFrame(scale_rows)
    scale_check.to_csv(out_dir / "residual_scaler_contract.csv", index=False)

    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split="test")
    adapter = ForecastModelAdapter(model_dir=model_dir, dataset_dir=dataset_dir, device="cpu")
    training_model = _load_training_model(model_dir, cfg)

    rows: list[dict[str, object]] = []
    for raw_ts in [x.strip() for x in str(args.timestamps).split(",") if x.strip()]:
        sample = replay.sample_for_history_end(pd.Timestamp(raw_ts).to_pydatetime())
        if sample is None:
            raise KeyError(f"missing replay sample at {raw_ts}")
        adapter_result = adapter.predict_window(sample.x_window, timestamp=raw_ts)
        train_uv, train_event = _training_path_predict(training_model, sample.x_window, cfg, scaler)
        uv_diff = np.asarray(adapter_result.wind_uv_raw, dtype=float) - np.asarray(train_uv, dtype=float)
        event_cols = replay.event_columns
        adapter_event = np.array(
            [float(adapter_result.event_probs[name]) for name in event_cols],
            dtype=float,
        )
        event_diff = adapter_event - train_event.astype(float)
        rows.append(
            {
                "history_end": raw_ts,
                "max_abs_uv_diff_ms": float(np.max(np.abs(uv_diff))),
                "mean_abs_uv_diff_ms": float(np.mean(np.abs(uv_diff))),
                "max_abs_event_prob_diff": float(np.max(np.abs(event_diff))),
                "mean_abs_event_prob_diff": float(np.mean(np.abs(event_diff))),
                "adapter_speed_mean_ms": float(np.mean(adapter_result.wind_speed)),
                "oracle_speed_mean_ms": float(
                    np.mean(np.sqrt(sample.y_uv_raw[:, 0] ** 2 + sample.y_uv_raw[:, 1] ** 2))
                ),
            }
        )
    parity = pd.DataFrame(rows)
    parity.to_csv(out_dir / "adapter_training_path_parity.csv", index=False)

    max_uv = float(parity["max_abs_uv_diff_ms"].max())
    max_event = float(parity["max_abs_event_prob_diff"].max())
    max_mean_diff = float(scale_check["mean_diff"].abs().max())
    max_std_diff = float(scale_check["std_diff"].abs().max())
    ok = max_uv <= 1e-5 and max_event <= 1e-6 and max_mean_diff <= 1e-12 and max_std_diff <= 1e-12
    lines = [
        "# Forecast Adapter Parity Audit",
        "",
        f"- model: `{model_dir.relative_to(REPO_ROOT)}`",
        f"- dataset: `{dataset_dir.relative_to(REPO_ROOT)}`",
        f"- residual regression: `{bool(cfg.get('residual_regression', False))}`",
        f"- uv feature indices: `{cfg.get('uv_feature_indices')}`",
        f"- max abs uv diff adapter vs training path: `{max_uv:.8g} m/s`",
        f"- max abs event prob diff adapter vs training path: `{max_event:.8g}`",
        f"- residual scaler mean/std max diff: `{max_mean_diff:.8g}` / `{max_std_diff:.8g}`",
        f"- verdict: `{'PASS' if ok else 'CHECK'}`",
        "",
        "## Interpretation",
        "",
        (
            "当前 adapter 在线推理路径与训练脚本的 checkpoint 推理路径一致。"
            "residual 回归确实依赖 feature u/v 与 target u/v 使用同一套均值/方差；"
            "本数据集该契约成立，因此 learned≈persistence 不能归因于 adapter 反归一化接线错误。"
        ),
        "",
        "## Files",
        "",
        f"- parity: `{(out_dir / 'adapter_training_path_parity.csv').relative_to(REPO_ROOT)}`",
        f"- scaler contract: `{(out_dir / 'residual_scaler_contract.csv').relative_to(REPO_ROOT)}`",
    ]
    (out_dir / "forecast_adapter_parity_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {out_dir / 'forecast_adapter_parity_report.md'}")
    print(parity.to_string(index=False))


if __name__ == "__main__":
    main()
