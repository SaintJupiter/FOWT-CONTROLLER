#!/usr/bin/env python3
"""Validate forecast-source output against the planner forecast contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from wind_prediction.forecast_adapter import (
    CurrentOnlyForecastAdapter,
    ForecastModelAdapter,
    PersistenceMeanForecastAdapter,
)
from wind_prediction.forecast_contract import (
    ForecastContract,
    forecast_from_oracle_sample,
    validate_forecast_result,
)
from wind_prediction.replay_dataset import Fino1ReplayDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1",
    )
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument(
        "--forecast-source",
        default="current_only",
        choices=("oracle", "persistence_mean", "current_only", "learned"),
    )
    parser.add_argument(
        "--model-dir",
        default="outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1",
        help="Used only with --forecast-source learned.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def main() -> None:
    args = parse_args()
    replay = Fino1ReplayDataset(_resolve(str(args.dataset_dir)), split=str(args.split))
    sample = replay.sample_for_history_end(replay.first_sample_timestamp())
    if sample is None:
        raise RuntimeError("first replay sample could not be loaded")

    source = str(args.forecast_source)
    if source == "oracle":
        forecast = forecast_from_oracle_sample(
            y_uv_raw=sample.y_uv_raw,
            y_event=sample.y_event,
            event_columns=replay.event_columns,
            model_version="oracle_future",
            timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
        )
        provides_future_preview = True
        require_event_probs = True
    elif source == "persistence_mean":
        adapter = PersistenceMeanForecastAdapter(dataset_dir=replay.dataset_dir)
        forecast = adapter.predict_window(sample.x_window, timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S"))
        provides_future_preview = bool(adapter.provides_future_preview)
        require_event_probs = False
    elif source == "current_only":
        adapter = CurrentOnlyForecastAdapter(dataset_dir=replay.dataset_dir)
        forecast = adapter.predict_window(sample.x_window, timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S"))
        provides_future_preview = bool(adapter.provides_future_preview)
        require_event_probs = False
    elif source == "learned":
        adapter = ForecastModelAdapter(
            model_dir=_resolve(str(args.model_dir)),
            dataset_dir=replay.dataset_dir,
            device="cpu",
        )
        forecast = adapter.predict_window(sample.x_window, timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S"))
        provides_future_preview = bool(adapter.provides_future_preview)
        require_event_probs = True
    else:
        raise ValueError(f"unsupported forecast source: {source}")

    contract = ForecastContract(
        future_steps=int(replay.future_steps),
        event_columns=tuple(replay.event_columns),
        require_event_probs=require_event_probs,
    )
    validate_forecast_result(forecast, contract)
    if source == "current_only" and provides_future_preview:
        raise ValueError("current_only must not advertise provides_future_preview=True")

    payload = {
        "ok": True,
        "forecast_source": source,
        "forecast_model_version": str(forecast.model_version),
        "provides_future_preview": bool(provides_future_preview),
        "future_steps": int(replay.future_steps),
        "uv_shape": list(forecast.wind_uv_raw.shape),
        "event_prob_count": len(forecast.event_probs),
        "timestamp": forecast.timestamp,
    }
    if bool(args.json):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"[PASS] forecast contract: source={payload['forecast_source']} "
            f"model={payload['forecast_model_version']} preview={int(payload['provides_future_preview'])} "
            f"uv_shape={tuple(payload['uv_shape'])} event_probs={payload['event_prob_count']}"
        )


if __name__ == "__main__":
    main()
