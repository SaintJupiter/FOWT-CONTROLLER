#!/usr/bin/env python3
"""Forecast-layer horizon sweep for FINO1 10-60 minute preview quality.

This script does not run the controller. It compares the learned forecast with
the 10-minute persistence baseline on the same test windows and reports error by
lead time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wind_prediction.forecast_adapter import ForecastModelAdapter, PersistenceMeanForecastAdapter  # noqa: E402


def _uv_to_speed_dir(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(uv[..., 0] ** 2 + uv[..., 1] ** 2)
    direction = (np.rad2deg(np.arctan2(-uv[..., 0], -uv[..., 1])) + 360.0) % 360.0
    return speed, direction


def _dir_abs_error_deg(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.abs((pred - true + 180.0) % 360.0 - 180.0)


def _ramp_events(speed: np.ndarray, threshold_ms: float) -> np.ndarray:
    prev = np.concatenate([speed[:, :1], speed[:, :-1]], axis=1)
    return np.abs(speed - prev) >= float(threshold_ms)


def _lead_rows(name: str, pred_uv: np.ndarray, true_uv: np.ndarray, lead_minutes: list[int]) -> list[dict[str, Any]]:
    pred_speed, pred_dir = _uv_to_speed_dir(pred_uv)
    true_speed, true_dir = _uv_to_speed_dir(true_uv)
    uv_err = pred_uv - true_uv
    speed_err = pred_speed - true_speed
    dir_err = _dir_abs_error_deg(pred_dir, true_dir)

    rows: list[dict[str, Any]] = []
    for idx, lead in enumerate(lead_minutes):
        uv_l2 = np.linalg.norm(uv_err[:, idx, :], axis=1)
        rows.append(
            {
                "source": name,
                "lead_min": int(lead),
                "n": int(true_uv.shape[0]),
                "uv_l2_mae": float(np.mean(uv_l2)),
                "uv_l2_rmse": float(math.sqrt(np.mean(uv_l2**2))),
                "speed_mae_ms": float(np.mean(np.abs(speed_err[:, idx]))),
                "speed_rmse_ms": float(math.sqrt(np.mean(speed_err[:, idx] ** 2))),
                "direction_mae_deg": float(np.mean(dir_err[:, idx])),
                "direction_p90_deg": float(np.quantile(dir_err[:, idx], 0.90)),
            }
        )
    return rows


def _block_rows(name: str, pred_uv: np.ndarray, true_uv: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block_idx, start in enumerate((0, 2, 4), start=1):
        pred_block = np.mean(pred_uv[:, start : start + 2, :], axis=1)
        true_block = np.mean(true_uv[:, start : start + 2, :], axis=1)
        pred_speed, pred_dir = _uv_to_speed_dir(pred_block)
        true_speed, true_dir = _uv_to_speed_dir(true_block)
        uv_l2 = np.linalg.norm(pred_block - true_block, axis=1)
        speed_err = pred_speed - true_speed
        dir_err = _dir_abs_error_deg(pred_dir, true_dir)
        rows.append(
            {
                "source": name,
                "block": f"{20 * (block_idx - 1)}_{20 * block_idx}min",
                "n": int(true_uv.shape[0]),
                "uv_l2_mae": float(np.mean(uv_l2)),
                "uv_l2_rmse": float(math.sqrt(np.mean(uv_l2**2))),
                "speed_mae_ms": float(np.mean(np.abs(speed_err))),
                "speed_rmse_ms": float(math.sqrt(np.mean(speed_err**2))),
                "direction_mae_deg": float(np.mean(dir_err)),
                "direction_p90_deg": float(np.quantile(dir_err, 0.90)),
            }
        )
    return rows


def _ramp_rows(name: str, pred_uv: np.ndarray, true_uv: np.ndarray, threshold_ms: float) -> list[dict[str, Any]]:
    pred_speed, _ = _uv_to_speed_dir(pred_uv)
    true_speed, _ = _uv_to_speed_dir(true_uv)
    pred_events = _ramp_events(pred_speed, threshold_ms)
    true_events = _ramp_events(true_speed, threshold_ms)
    rows: list[dict[str, Any]] = []
    for idx, lead in enumerate((10, 20, 30, 40, 50, 60)):
        tp = int(np.sum(pred_events[:, idx] & true_events[:, idx]))
        fp = int(np.sum(pred_events[:, idx] & ~true_events[:, idx]))
        fn = int(np.sum(~pred_events[:, idx] & true_events[:, idx]))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append(
            {
                "source": name,
                "lead_min": int(lead),
                "ramp_threshold_ms": float(threshold_ms),
                "true_ramp_count": int(np.sum(true_events[:, idx])),
                "pred_ramp_count": int(np.sum(pred_events[:, idx])),
                "ramp_precision": float(precision),
                "ramp_recall": float(recall),
                "ramp_f1": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
            }
        )
    return rows


def _plot(out_dir: Path, lead_rows: list[dict[str, Any]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = sorted({r["source"] for r in lead_rows})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)
    metrics = [
        ("speed_mae_ms", "Speed MAE (m/s)"),
        ("direction_mae_deg", "Direction MAE (deg)"),
        ("uv_l2_mae", "UV L2 MAE (m/s)"),
    ]
    for ax, (key, label) in zip(axes, metrics):
        for source in sources:
            rows = [r for r in lead_rows if r["source"] == source]
            rows.sort(key=lambda r: r["lead_min"])
            ax.plot(
                [r["lead_min"] for r in rows],
                [r[key] for r in rows],
                marker="o",
                label=source,
            )
        ax.set_title(label)
        ax.set_xlabel("Lead time (min)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Error")
    axes[-1].legend(loc="best")
    fig.suptitle("FINO1 forecast horizon sweep: learned vs 10-min persistence")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "forecast_horizon_sweep_10_60.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1")
    parser.add_argument("--model-dir", default="outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--ramp-threshold-ms", type=float, default=2.0)
    parser.add_argument("--out-dir", default="outputs/wind_prediction/forecast_horizon_sweep_10_60_v1")
    args = parser.parse_args()

    dataset_dir = (REPO_ROOT / args.dataset_dir).resolve()
    model_dir = (REPO_ROOT / args.model_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    arrays = metadata["arrays"][args.split]
    x_all = np.load(dataset_dir / arrays["X"], mmap_mode="r")
    y_true = np.load(dataset_dir / arrays["y_uv_raw"], mmap_mode="r")
    n_total = int(x_all.shape[0])
    n = min(int(args.max_samples), n_total)
    if n <= 0:
        raise ValueError("--max-samples must be positive")
    indices = np.linspace(0, n_total - 1, num=n, dtype=int)

    learned = ForecastModelAdapter(model_dir=model_dir, dataset_dir=dataset_dir, device="cpu")
    persistence = PersistenceMeanForecastAdapter(dataset_dir=dataset_dir, history_minutes=10.0)

    pred_learned = np.empty((n, int(metadata["future_steps"]), 2), dtype=np.float32)
    pred_persistence = np.empty_like(pred_learned)
    truth = np.empty_like(pred_learned)
    for out_idx, idx in enumerate(indices):
        x = np.asarray(x_all[idx], dtype=np.float32)
        pred_learned[out_idx] = learned.predict_window(x).wind_uv_raw
        pred_persistence[out_idx] = persistence.predict_window(x).wind_uv_raw
        truth[out_idx] = np.asarray(y_true[idx], dtype=np.float32)
        if (out_idx + 1) % 1000 == 0:
            print(f"processed {out_idx + 1}/{n}", flush=True)

    lead_minutes = [10, 20, 30, 40, 50, 60]
    lead_rows = []
    block_rows = []
    ramp_rows = []
    for name, pred in (("learned", pred_learned), ("persistence_10min", pred_persistence)):
        lead_rows.extend(_lead_rows(name, pred, truth, lead_minutes))
        block_rows.extend(_block_rows(name, pred, truth))
        ramp_rows.extend(_ramp_rows(name, pred, truth, float(args.ramp_threshold_ms)))

    _write_csv(out_dir / "lead_metrics.csv", lead_rows)
    _write_csv(out_dir / "block_metrics.csv", block_rows)
    _write_csv(out_dir / "ramp_metrics.csv", ramp_rows)
    fig_path = _plot(out_dir, lead_rows)
    summary = {
        "split": args.split,
        "n_total": n_total,
        "n_sampled": n,
        "indices": "linspace",
        "future_steps": int(metadata["future_steps"]),
        "resolution_minutes": float(metadata.get("input_resolution_minutes", 10.0)),
        "ramp_threshold_ms": float(args.ramp_threshold_ms),
        "figure": str(fig_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"wrote {out_dir / 'lead_metrics.csv'}")
    print(f"wrote {out_dir / 'block_metrics.csv'}")
    print(f"wrote {out_dir / 'ramp_metrics.csv'}")
    print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
