#!/usr/bin/env python3
"""Forecast-level learned-vs-persistence audit for control-relevant wind events.

This deliberately stays above the controller/planner layer.  It asks whether
the learned forecast itself provides earlier or more accurate signals than a
10-minute persistence baseline on true FINO1 test windows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wind_prediction.ballast_planner import PlannerConfig, compute_pressure_blocks, norm_term  # noqa: E402
from wind_prediction.forecast_adapter import ForecastModelAdapter, PersistenceMeanForecastAdapter  # noqa: E402
from wind_prediction.replay_dataset import Fino1ReplayDataset  # noqa: E402


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_planner_config() -> PlannerConfig:
    sign_path = REPO_ROOT / "outputs/wind_prediction/planner_a1_dryrun/diagnostics/a01_pressure_vec_sign_convention.json"
    if sign_path.exists():
        sign_cfg = json.loads(sign_path.read_text(encoding="utf-8"))
        return PlannerConfig(pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]))
    return PlannerConfig()


def _block_features(uv: np.ndarray, cfg: PlannerConfig) -> dict[str, float]:
    blocks = compute_pressure_blocks(np.asarray(uv, dtype=float), [1.0, 0.85, 0.70], cfg)
    raw_vecs = [np.asarray(b["pressure_vec_raw"], dtype=float) for b in blocks]
    raw_norms = [norm_term(v, cfg) for v in raw_vecs]
    vec0 = raw_vecs[0]
    vec2 = raw_vecs[2]
    denom = float(np.linalg.norm(vec0) * np.linalg.norm(vec2))
    cos02 = float(np.dot(vec0, vec2) / denom) if denom > 1e-9 else float("nan")
    speed = np.sqrt(np.sum(np.asarray(uv, dtype=float) ** 2, axis=1))
    return {
        "norm0": float(raw_norms[0]),
        "norm1": float(raw_norms[1]),
        "norm2": float(raw_norms[2]),
        "max_norm": float(np.max(raw_norms)),
        "relief02": float(raw_norms[0] - raw_norms[2]),
        "growth02": float(raw_norms[2] - raw_norms[0]),
        "cos02": cos02,
        "speed0": float(np.mean(speed[:2])),
        "speed1": float(np.mean(speed[2:4])),
        "speed2": float(np.mean(speed[4:6])),
        "speed_relief02": float(np.mean(speed[:2]) - np.mean(speed[4:6])),
        "speed_growth02": float(np.mean(speed[4:6]) - np.mean(speed[:2])),
    }


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def _scenario_from_oracle(feat: dict[str, float], args: argparse.Namespace) -> str:
    if feat["norm0"] >= args.high_norm and feat["norm2"] <= args.relief_norm:
        return "future_relief"
    if feat["norm0"] <= args.low_norm and feat["norm2"] >= args.high_norm:
        return "future_risk_onset"
    if feat["norm0"] >= args.signflip_norm and feat["norm2"] >= args.signflip_future_norm and feat["cos02"] <= args.signflip_cos:
        return "signflip_high"
    if feat["max_norm"] <= args.low_norm:
        return "lowrisk"
    if feat["norm0"] >= args.high_norm and feat["norm2"] >= args.high_norm:
        return "sustained_high"
    return "other"


def _hit_onset(feat: dict[str, float], args: argparse.Namespace) -> bool:
    return bool(feat["norm2"] >= args.high_norm and feat["growth02"] >= args.event_margin)


def _hit_relief(feat: dict[str, float], args: argparse.Namespace) -> bool:
    return bool(feat["norm0"] >= args.high_norm and feat["norm2"] <= args.relief_norm)


def _hit_signflip(feat: dict[str, float], args: argparse.Namespace) -> bool:
    return bool(
        feat["norm0"] >= args.signflip_norm
        and feat["norm2"] >= args.signflip_future_norm
        and feat["cos02"] <= args.signflip_cos
    )


def _score_source(
    *,
    prefix: str,
    pred_feat: dict[str, float],
    oracle_feat: dict[str, float],
    pred_uv: np.ndarray,
    oracle_uv: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    return {
        f"{prefix}_uv_rmse": _rmse(pred_uv, oracle_uv),
        f"{prefix}_norm0_abs_err": abs(pred_feat["norm0"] - oracle_feat["norm0"]),
        f"{prefix}_norm1_abs_err": abs(pred_feat["norm1"] - oracle_feat["norm1"]),
        f"{prefix}_norm2_abs_err": abs(pred_feat["norm2"] - oracle_feat["norm2"]),
        f"{prefix}_relief_abs_err": abs(pred_feat["relief02"] - oracle_feat["relief02"]),
        f"{prefix}_growth_abs_err": abs(pred_feat["growth02"] - oracle_feat["growth02"]),
        f"{prefix}_cos02_abs_err": (
            abs(pred_feat["cos02"] - oracle_feat["cos02"])
            if np.isfinite(pred_feat["cos02"]) and np.isfinite(oracle_feat["cos02"])
            else float("nan")
        ),
        f"{prefix}_onset_hit": int(_hit_onset(pred_feat, args)),
        f"{prefix}_relief_hit": int(_hit_relief(pred_feat, args)),
        f"{prefix}_signflip_hit": int(_hit_signflip(pred_feat, args)),
    }


def _confusion(rows: pd.DataFrame, scenario: str, signal_col: str) -> dict[str, float | int]:
    sub = rows.copy()
    truth = sub["oracle_scenario"].eq(scenario)
    pred = sub[signal_col].astype(bool)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    tn = int((~truth & ~pred).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "truth_scenario": scenario,
        "signal": signal_col,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_rate": float(pred.mean()),
        "truth_rate": float(truth.mean()),
    }


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_none_"
    if max_rows is not None:
        df = df.head(max_rows)
    lines = [
        "| " + " | ".join(str(c) for c in df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1")
    parser.add_argument("--model-dir", default="outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1")
    parser.add_argument("--out-dir", default="outputs/wind_prediction/forecast_level_prediction_value_v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 means all sampled windows after stride.")
    parser.add_argument("--persistence-history-min", type=float, default=10.0)
    parser.add_argument("--high-norm", type=float, default=1.20)
    parser.add_argument("--relief-norm", type=float, default=0.85)
    parser.add_argument("--low-norm", type=float, default=0.65)
    parser.add_argument("--signflip-norm", type=float, default=1.20)
    parser.add_argument("--signflip-future-norm", type=float, default=0.70)
    parser.add_argument("--signflip-cos", type=float, default=-0.30)
    parser.add_argument("--event-margin", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = Fino1ReplayDataset(dataset_dir=_resolve(args.dataset_dir), split=args.split)
    cfg = _load_planner_config()
    learned = ForecastModelAdapter(model_dir=_resolve(args.model_dir), dataset_dir=_resolve(args.dataset_dir), device="cpu")
    persistence = PersistenceMeanForecastAdapter(dataset_dir=_resolve(args.dataset_dir), history_minutes=args.persistence_history_min)

    samples = dataset.samples[:: max(1, int(args.stride))]
    if int(args.limit) > 0:
        samples = samples[: int(args.limit)]

    rows: list[dict[str, object]] = []
    for idx, sample in enumerate(samples):
        if idx and idx % 10000 == 0:
            print(f"processed {idx}/{len(samples)}", flush=True)
        oracle_uv = np.asarray(sample.y_uv_raw, dtype=float)
        learned_uv = np.asarray(learned.predict_window(sample.x_window, timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S")).wind_uv_raw, dtype=float)
        persistence_uv = np.asarray(persistence.predict_window(sample.x_window, timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S")).wind_uv_raw, dtype=float)
        oracle_feat = _block_features(oracle_uv, cfg)
        learned_feat = _block_features(learned_uv, cfg)
        persistence_feat = _block_features(persistence_uv, cfg)
        row: dict[str, object] = {
            "timestamp": sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
            "oracle_scenario": _scenario_from_oracle(oracle_feat, args),
        }
        for prefix, feat in (("oracle", oracle_feat), ("learned", learned_feat), ("persistence", persistence_feat)):
            for key, value in feat.items():
                row[f"{prefix}_{key}"] = value
        row.update(_score_source(prefix="learned", pred_feat=learned_feat, oracle_feat=oracle_feat, pred_uv=learned_uv, oracle_uv=oracle_uv, args=args))
        row.update(_score_source(prefix="persistence", pred_feat=persistence_feat, oracle_feat=oracle_feat, pred_uv=persistence_uv, oracle_uv=oracle_uv, args=args))
        row["learned_uv_rmse_advantage"] = float(row["persistence_uv_rmse"] - row["learned_uv_rmse"])
        row["learned_norm2_abs_err_advantage"] = float(row["persistence_norm2_abs_err"] - row["learned_norm2_abs_err"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "forecast_level_window_metrics.csv", index=False)

    scenario_summary = (
        df.groupby("oracle_scenario")
        .agg(
            windows=("timestamp", "count"),
            learned_uv_rmse=("learned_uv_rmse", "mean"),
            persistence_uv_rmse=("persistence_uv_rmse", "mean"),
            learned_uv_rmse_advantage=("learned_uv_rmse_advantage", "mean"),
            learned_better_uv_rmse_rate=("learned_uv_rmse_advantage", lambda s: float((s > 0.0).mean())),
            learned_norm2_abs_err=("learned_norm2_abs_err", "mean"),
            persistence_norm2_abs_err=("persistence_norm2_abs_err", "mean"),
            learned_better_norm2_rate=("learned_norm2_abs_err_advantage", lambda s: float((s > 0.0).mean())),
            learned_onset_hit_rate=("learned_onset_hit", "mean"),
            persistence_onset_hit_rate=("persistence_onset_hit", "mean"),
            learned_relief_hit_rate=("learned_relief_hit", "mean"),
            persistence_relief_hit_rate=("persistence_relief_hit", "mean"),
            learned_signflip_hit_rate=("learned_signflip_hit", "mean"),
            persistence_signflip_hit_rate=("persistence_signflip_hit", "mean"),
        )
        .reset_index()
        .sort_values("windows", ascending=False)
    )
    scenario_summary.to_csv(out_dir / "forecast_level_scenario_summary.csv", index=False)

    confusion_rows = []
    for scenario, learned_col, persistence_col in [
        ("future_risk_onset", "learned_onset_hit", "persistence_onset_hit"),
        ("future_relief", "learned_relief_hit", "persistence_relief_hit"),
        ("signflip_high", "learned_signflip_hit", "persistence_signflip_hit"),
    ]:
        confusion_rows.append({**_confusion(df, scenario, learned_col), "source": "learned"})
        confusion_rows.append({**_confusion(df, scenario, persistence_col), "source": "persistence"})
    confusion = pd.DataFrame(confusion_rows)
    confusion.to_csv(out_dir / "forecast_level_event_confusion.csv", index=False)

    lines = [
        "# Forecast-Level Prediction Value Audit v1",
        "",
        f"- split: `{args.split}`",
        f"- windows evaluated: `{len(df)}`",
        f"- stride: `{args.stride}`",
        "- This audit does not run the controller; it compares learned and persistence forecasts against true future/oracle UV.",
        "",
        "## Scenario Summary",
        "",
        _markdown_table(scenario_summary),
        "",
        "## Event Confusion",
        "",
        _markdown_table(confusion),
        "",
        "## Interpretation Rules",
        "",
        "1. If learned is not better than persistence at this forecast layer, controller-layer tuning cannot reliably create forecast-source value.",
        "2. If learned is better at this forecast layer but not in closed-loop holdout, the bottleneck is planner/execution credit assignment.",
        "3. These thresholds are physical probes, not tuned controller gates.",
    ]
    (out_dir / "forecast_level_prediction_value_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {out_dir / 'forecast_level_prediction_value_report.md'}")


if __name__ == "__main__":
    main()
