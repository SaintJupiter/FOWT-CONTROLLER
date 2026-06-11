#!/usr/bin/env python3
"""Forecast-source sanity check for the prediction-primary ablation.

The check is intentionally small: it evaluates the same 10-minute bucket
timestamps used by the 5-case, 2h control ablation. Its only job is to verify
whether the learned forecast is meaningfully better than the chosen persistence
baseline before spending control-simulation effort on a model-claim figure.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from wind_prediction import Fino1ReplayDataset, ForecastModelAdapter, PersistenceMeanForecastAdapter


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_CASES = [
    ("onset", "2024-11-27 19:40:00"),
    ("decay", "2024-09-27 13:00:00"),
    ("signflip", "2023-10-03 06:30:00"),
    ("normal_low", "2021-12-20 14:30:00"),
    ("case09", "2022-02-04 11:00:00"),
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
        default="outputs/wind_prediction/forecast_source_sanity_5case",
    )
    parser.add_argument("--duration-s", type=float, default=7200.0)
    parser.add_argument("--persistence-history-min", type=float, default=10.0)
    parser.add_argument(
        "--pass-ratio",
        type=float,
        default=0.85,
        help="Learned vector RMSE must be below this multiple of persistence.",
    )
    return parser.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _metric_row(case_id: str, source: str, err: np.ndarray) -> dict[str, object]:
    return {
        "case_id": case_id,
        "source": source,
        "vector_mae_ms": float(np.mean(np.abs(err))),
        "vector_rmse_ms": float(np.sqrt(np.mean(err**2))),
        "sample_count": int(err.shape[0]),
        "future_steps": int(err.shape[1]),
    }


def main() -> None:
    args = parse_args()
    dataset_dir = _resolve(str(args.dataset_dir))
    model_dir = _resolve(str(args.model_dir))
    out_dir = _resolve(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split="test")
    learned = ForecastModelAdapter(model_dir=model_dir, dataset_dir=dataset_dir, device="cpu")
    persistence = PersistenceMeanForecastAdapter(
        dataset_dir=dataset_dir,
        history_minutes=float(args.persistence_history_min),
    )

    buckets_per_case = int(np.ceil(float(args.duration_s) / replay.update_interval_s))
    rows = []
    detail_rows = []
    all_errors: dict[str, list[np.ndarray]] = {"learned": [], "persistence_mean": []}
    for case_id, ts_str in DEFAULT_CASES:
        start = datetime.strptime(ts_str, TIMESTAMP_FMT)
        case_errors: dict[str, list[np.ndarray]] = {"learned": [], "persistence_mean": []}
        for bucket in range(buckets_per_case):
            ts = replay.simulation_timestamp(start, bucket * replay.update_interval_s)
            sample = replay.sample_for_history_end(ts)
            if sample is None:
                raise KeyError(f"missing sample for {case_id} at {ts.strftime(TIMESTAMP_FMT)}")
            pred_learned = learned.predict_window(
                sample.x_window,
                timestamp=sample.history_end.strftime(TIMESTAMP_FMT),
            ).wind_uv_raw
            pred_persistence = persistence.predict_window(
                sample.x_window,
                timestamp=sample.history_end.strftime(TIMESTAMP_FMT),
            ).wind_uv_raw
            for source, pred in (
                ("learned", pred_learned),
                ("persistence_mean", pred_persistence),
            ):
                err = np.asarray(pred, dtype=float) - np.asarray(sample.y_uv_raw, dtype=float)
                case_errors[source].append(err)
                all_errors[source].append(err)
                detail_rows.append(
                    {
                        "case_id": case_id,
                        "timestamp": sample.history_end.strftime(TIMESTAMP_FMT),
                        "source": source,
                        "vector_rmse_ms": float(np.sqrt(np.mean(err**2))),
                        "vector_mae_ms": float(np.mean(np.abs(err))),
                    }
                )
        for source, parts in case_errors.items():
            rows.append(_metric_row(case_id, source, np.stack(parts, axis=0)))

    for source, parts in all_errors.items():
        rows.append(_metric_row("ALL_5_CASES", source, np.stack(parts, axis=0)))

    metrics = pd.DataFrame(rows)
    details = pd.DataFrame(detail_rows)
    metrics.to_csv(out_dir / "forecast_source_sanity_metrics.csv", index=False)
    details.to_csv(out_dir / "forecast_source_sanity_detail.csv", index=False)

    all_rows = metrics[metrics["case_id"].eq("ALL_5_CASES")]
    learned_rmse = float(all_rows[all_rows["source"].eq("learned")]["vector_rmse_ms"].iloc[0])
    persistence_rmse = float(
        all_rows[all_rows["source"].eq("persistence_mean")]["vector_rmse_ms"].iloc[0]
    )
    ratio = learned_rmse / max(persistence_rmse, 1e-12)
    passed = ratio < float(args.pass_ratio)

    lines = [
        "# Forecast-source sanity check",
        "",
        f"- dataset: `{dataset_dir.relative_to(repo_root) if dataset_dir.is_relative_to(repo_root) else dataset_dir}`",
        f"- model: `{model_dir.relative_to(repo_root) if model_dir.is_relative_to(repo_root) else model_dir}`",
        f"- persistence baseline: `{persistence.model_version}`",
        f"- learned model: `{learned.model_version}`",
        f"- pass rule: learned RMSE < persistence RMSE x `{float(args.pass_ratio):.2f}`",
        f"- result: `{'PASS' if passed else 'FAIL'}`",
        f"- all-case vector RMSE: learned `{learned_rmse:.3f}` vs persistence `{persistence_rmse:.3f}`; ratio `{ratio:.3f}`",
        "",
        "| case | source | vector RMSE | vector MAE | samples |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in metrics.iterrows():
        lines.append(
            f"| {r['case_id']} | {r['source']} | {r['vector_rmse_ms']:.3f} | "
            f"{r['vector_mae_ms']:.3f} | {int(r['sample_count'])} |"
        )
    (out_dir / "forecast_source_sanity_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {out_dir / 'forecast_source_sanity_report.md'}")
    print(f"RESULT {'PASS' if passed else 'FAIL'} ratio={ratio:.3f}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
