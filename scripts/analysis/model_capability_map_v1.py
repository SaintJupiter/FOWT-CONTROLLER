#!/usr/bin/env python3
"""Build G001 model capability map for forecast-controller experiments.

Evaluation-only: reads existing guard10 casebook outputs, planner logs, and
 timeseries. It does not run training, closed-loop simulation, or modify
 controller parameters.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wind_prediction.ballast_planner import (  # noqa: E402
    _ALL_SEQUENCES,
    PlannerConfig,
    apply_posture_state_to_blocks,
    compute_pressure_blocks,
    economic_scalar_cost,
    evaluate_sequence,
    select_best_economic,
)
from wind_prediction.forecast_adapter import (  # noqa: E402
    CurrentOnlyForecastAdapter,
    ForecastModelAdapter,
    PersistenceMeanForecastAdapter,
)
from wind_prediction.replay_dataset import Fino1ReplayDataset  # noqa: E402

CASES = [
    "fr_relief_01",
    "fr_relief_09",
    "sf_holdout_02",
    "b_decay_strong",
    "b_signflip_fallback",
    "b_high_pressure_event",
    "b_residual_high",
    "lowrisk_quiet",
    "lowrisk_clean",
    "lowrisk_random_03",
]

RUNS = {
    "f60_relief_e15": "outputs/wind_prediction/pp_learned_relief_e15_guard10",
    "h240_f120_baseline": "outputs/wind_prediction/horizon_guard10_fino1_h240_f120_learned_v1",
    "h240_f120_relief_e15": "outputs/wind_prediction/pp_h240_f120_relief_e15_farlog_guard10_learned_v1",
    "pressure_shape_v1": "outputs/wind_prediction/pp_h240_f120_pressure_shape_supervision_v1_guard10_learned_v1",
    "planner_block_shape_v1": "outputs/wind_prediction/pp_h240_f120_planner_block_shape_supervision_v1_guard10_learned_v1",
    "near_block_eventbalanced_v2": "outputs/wind_prediction/pp_h240_f120_near_block_eventbalanced_v2_guard10_learned_v1",
    "multitask_far_hint_v0": "outputs/wind_prediction/pp_h240_f120_multitask_far_hint_v0_guard10_learned_v1",
    "oracle": "outputs/wind_prediction/pp_h240_f120_oracle_farlog_guard10_v1",
    "persistence": "outputs/wind_prediction/pp_h240_f120_persistence_farlog_guard10_v1",
    "current_only": "outputs/wind_prediction/pp_h240_f120_current_only_farlog_guard10_v1",
}

KEY_BUCKETS = "outputs/wind_prediction/h240_f120_oracle_learned_attribution_v1/oracle_learned_action_diff_buckets.csv"
DEFAULT_DATASET_DIR = "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"

FORECAST_MODEL_DIRS = {
    "h240_f120_baseline": "outputs/wind_prediction/gru_fino1_h240_f120_segmented_head_v1",
    "h240_f120_relief_e15": "outputs/wind_prediction/lstm_h240_f120_synth_relief_t030_e15_v1",
    "pressure_shape_v1": "outputs/wind_prediction/lstm_h240_f120_pressure_shape_supervision_v1",
    "planner_block_shape_v1": "outputs/wind_prediction/lstm_h240_f120_planner_block_shape_supervision_v1",
    "near_block_eventbalanced_v2": "outputs/wind_prediction/lstm_h240_f120_near_block_eventbalanced_v2",
    "multitask_far_hint_v0": "outputs/wind_prediction/lstm_h240_f120_multitask_far_hint_v0",
}


def _run_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(_safe_float(value, 0.0))


def _case_from_name(name: str) -> str:
    for case in CASES:
        if case in name:
            return case
    base = re.sub(r"_(prediction_primary_econ|closed_only|current_only).*", "", name)
    base = re.sub(r"_\d{4}-\d{2}-\d{2}_\d{6}.*", "", base)
    return re.sub(r"^\d+_", "", base)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cfg_from_casebook(casebook_dir: Path) -> tuple[PlannerConfig, list[float]]:
    diagnostics = REPO_ROOT / "outputs" / "wind_prediction" / "planner_a1_dryrun" / "diagnostics"
    sign_cfg = _load_json(diagnostics / "a01_pressure_vec_sign_convention.json")
    discount_cfg = _load_json(diagnostics / "a1_block_discount_config.json")
    summary_path = casebook_dir / "casebook_summary.csv"
    row: dict[str, Any] = {}
    if summary_path.exists():
        summary = pd.read_csv(summary_path, low_memory=False)
        if not summary.empty:
            row = summary.iloc[0].to_dict()

    def value(col: str, default: Any) -> Any:
        raw = row.get(col, default)
        if pd.isna(raw):
            return default
        return raw

    base = PlannerConfig()
    cfg = PlannerConfig(
        pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]),
        pressure_norm_cap=float(value("planner_pressure_norm_cap", base.pressure_norm_cap)),
        envelope_use_discount=_boolish(value("planner_envelope_use_discount", int(base.envelope_use_discount))),
        envelope_barrier_active=_boolish(value("planner_envelope_barrier_active", int(base.envelope_barrier_active))),
        envelope_barrier_const=float(value("planner_envelope_barrier_const", base.envelope_barrier_const)),
        posture_hold_barrier_active=_boolish(value("planner_posture_hold_barrier_active", int(base.posture_hold_barrier_active))),
        posture_hold_pitch_deg=float(value("planner_posture_hold_pitch_deg", base.posture_hold_pitch_deg)),
        posture_hold_roll_deg=float(value("planner_posture_hold_roll_deg", base.posture_hold_roll_deg)),
        posture_hold_barrier_const=float(value("planner_posture_hold_barrier_const", base.posture_hold_barrier_const)),
        posture_hold_relief_margin_norm=float(value("planner_posture_hold_relief_margin_norm", base.posture_hold_relief_margin_norm)),
        posture_hold_forecast_credit=float(value("planner_posture_hold_forecast_credit", base.posture_hold_forecast_credit)),
        posture_hold_action_uses_posture_vec=_boolish(value("planner_posture_hold_action_uses_posture_vec", int(base.posture_hold_action_uses_posture_vec))),
        posture_hold_low_risk_norm_max=float(value("planner_posture_hold_low_risk_norm_max", base.posture_hold_low_risk_norm_max)),
        posture_state_residual_active=_boolish(value("planner_posture_state_residual_active", int(base.posture_state_residual_active))),
        posture_state_gain=float(value("planner_posture_state_gain", base.posture_state_gain)),
        posture_state_decay=float(value("planner_posture_state_decay", base.posture_state_decay)),
        posture_state_clip_norm=float(value("planner_posture_state_clip_norm", base.posture_state_clip_norm)),
        w_attitude_residual=float(value("planner_attitude_residual_weight", base.w_attitude_residual)),
        w_terminal_residual=float(value("planner_terminal_residual_weight", base.w_terminal_residual)),
    )
    return cfg, [float(x) for x in discount_cfg["default_discount_blocks"]]


def _discover_logs(run_dir: Path) -> dict[tuple[str, int], tuple[Path, int]]:
    out: dict[tuple[str, int], tuple[Path, int]] = {}
    for path in sorted((run_dir / "planner_logs").glob("*_planner_log.csv")):
        case = _case_from_name(path.name)
        d = pd.read_csv(path, low_memory=False, usecols=lambda c: c in {"bucket"})
        for idx, row in d.iterrows():
            out[(case, int(row["bucket"]))] = (path, int(idx))
    return out


def _planner_frame(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "planner_logs").glob("*_planner_log.csv")):
        case = _case_from_name(path.name)
        d = pd.read_csv(path, low_memory=False)
        for _, r in d.iterrows():
            rows.append(
                {
                    "case_id": case,
                    "bucket": int(r.get("bucket", -1)),
                    "first_action": str(r.get("first_action", "")),
                    "best_sequence": str(r.get("best_sequence", "")),
                    "best_scalar_cost": _safe_float(r.get("best_scalar_cost")),
                    "b0": _safe_float(r.get("pressure_block0_norm")),
                    "b1": _safe_float(r.get("pressure_block1_norm")),
                    "b2": _safe_float(r.get("pressure_block2_norm")),
                    "raw_b0": _safe_float(r.get("raw_pressure_block0_norm")),
                    "raw_b1": _safe_float(r.get("raw_pressure_block1_norm")),
                    "raw_b2": _safe_float(r.get("raw_pressure_block2_norm")),
                    "far_available": int(r.get("far_horizon_available", 0)),
                    "far_hint_any": int(r.get("far_horizon_hint_any", 0)),
                    "hidden_relief": int(r.get("far_horizon_hidden_relief", 0)),
                    "hidden_intensification": int(r.get("far_horizon_hidden_intensification", 0)),
                    "reversal": int(r.get("far_horizon_reversal", 0)),
                    "direction_shift": int(r.get("far_horizon_direction_shift", 0)),
                    "far_0_20": _safe_float(r.get("far_horizon_norm_0_20")),
                    "far_20_40": _safe_float(r.get("far_horizon_norm_20_40")),
                    "far_40_60": _safe_float(r.get("far_horizon_norm_40_60")),
                    "far_60_80": _safe_float(r.get("far_horizon_norm_60_80")),
                    "far_80_100": _safe_float(r.get("far_horizon_norm_80_100")),
                    "far_100_120": _safe_float(r.get("far_horizon_norm_100_120")),
                }
            )
    return pd.DataFrame(rows)


def _timeseries_metrics(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "timeseries").glob("*_timeseries.csv")):
        case = _case_from_name(path.name)
        ts = pd.read_csv(path, low_memory=False)
        if "pitch_deg" not in ts.columns:
            continue
        pitch = ts["pitch_deg"].abs()
        roll = ts["roll_deg"].abs() if "roll_deg" in ts.columns else pd.Series(0.0, index=ts.index)
        pump = float((ts.get("pump_total_rate_m3_min", 0.0).abs() / 60.0).sum()) if "pump_total_rate_m3_min" in ts else np.nan
        fb_col = next((c for c in ts.columns if "safety_fallback" in c.lower() or "fallback_active" in c.lower()), None)
        fb = float(ts[fb_col].mean()) if fb_col else 0.0
        rows.append(
            {
                "case_id": case,
                "pump_m3": pump,
                "pitch_p95": float(np.percentile(pitch, 95)),
                "roll_p95": float(np.percentile(roll, 95)),
                "fallback_ratio": fb,
                "time_over_3_s": int(((pitch > 3.0) | (roll > 3.0)).sum()),
                "time_over_5_s": int(((pitch > 5.0) | (roll > 5.0)).sum()),
            }
        )
    return pd.DataFrame(rows)


def _direction_agreement(pred: np.ndarray, true: np.ndarray, threshold: float = 0.05) -> float:
    pred_delta = np.diff(pred, axis=1)
    true_delta = np.diff(true, axis=1)
    mask = np.abs(true_delta) >= threshold
    if not mask.any():
        return float("nan")
    return float((np.sign(pred_delta[mask]) == np.sign(true_delta[mask])).mean())


def _event_timing(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values).all(axis=1)
    out = np.full(values.shape[0], "missing", dtype=object)
    if finite.any():
        vmax = values[finite].max(axis=1)
        vmin = values[finite].min(axis=1)
        idx_max = values[finite].argmax(axis=1)
        idx_min = values[finite].argmin(axis=1)
        labels = np.array(["60_80", "80_100", "100_120"], dtype=object)
        chosen = np.where((vmax - values[finite, 0]) >= (values[finite, 0] - vmin), labels[idx_max], labels[idx_min])
        out[np.where(finite)[0]] = chosen
    return out


def _shape_tables(runs: dict[str, Path], oracle_name: str, baseline_name: str, key_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = {name: _planner_frame(path) for name, path in runs.items()}
    oracle = frames[oracle_name][["case_id", "bucket", "first_action", "b0", "b1", "b2", "far_60_80", "far_80_100", "far_100_120"]].rename(
        columns={"first_action": "oracle_action", "b0": "oracle_b0", "b1": "oracle_b1", "b2": "oracle_b2", "far_60_80": "oracle_far_60_80", "far_80_100": "oracle_far_80_100", "far_100_120": "oracle_far_100_120"}
    )
    baseline = frames[baseline_name][["case_id", "bucket", "b0", "b1", "b2"]].rename(columns={"b0": "base_b0", "b1": "base_b1", "b2": "base_b2"})
    key_pairs = set(zip(key_df["case_id"].map(_case_from_name), key_df["bucket"].astype(int))) if not key_df.empty else set()
    summary_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    far_rows: list[dict[str, Any]] = []
    for name, frame in frames.items():
        merged = frame.merge(oracle, on=["case_id", "bucket"], how="inner").merge(baseline, on=["case_id", "bucket"], how="left")
        pred = merged[["b0", "b1", "b2"]].to_numpy(float)
        true = merged[["oracle_b0", "oracle_b1", "oracle_b2"]].to_numpy(float)
        valid = np.isfinite(pred).all(axis=1) & np.isfinite(true).all(axis=1)
        pred_v, true_v = pred[valid], true[valid]
        err = pred_v - true_v
        pred_range = np.nanmax(pred_v, axis=1) - np.nanmin(pred_v, axis=1)
        true_range = np.nanmax(true_v, axis=1) - np.nanmin(true_v, axis=1)
        anti_mask = true_range >= 0.12
        collapsed = (pred_range < 0.5 * true_range) & anti_mask
        key_mask = merged.apply(lambda r: (str(r["case_id"]), int(r["bucket"])) in key_pairs, axis=1)
        key = merged[key_mask].copy()
        key_count = len(key)
        closer = np.nan
        key_action_matches = np.nan
        if key_count:
            kp = key[["b0", "b1", "b2"]].to_numpy(float)
            kt = key[["oracle_b0", "oracle_b1", "oracle_b2"]].to_numpy(float)
            kb = key[["base_b0", "base_b1", "base_b2"]].to_numpy(float)
            kvalid = np.isfinite(kp).all(axis=1) & np.isfinite(kt).all(axis=1) & np.isfinite(kb).all(axis=1)
            if kvalid.any():
                dist = np.linalg.norm(kp[kvalid] - kt[kvalid], axis=1)
                bdist = np.linalg.norm(kb[kvalid] - kt[kvalid], axis=1)
                closer = float((dist < bdist).sum())
            key_action_matches = float((key["first_action"].astype(str) == key["oracle_action"].astype(str)).sum())
            for _, r in key.iterrows():
                pvec = np.array([r["b0"], r["b1"], r["b2"]], dtype=float)
                tvec = np.array([r["oracle_b0"], r["oracle_b1"], r["oracle_b2"]], dtype=float)
                bvec = np.array([r["base_b0"], r["base_b1"], r["base_b2"]], dtype=float)
                key_rows.append(
                    {
                        "run": name,
                        "case_id": r["case_id"],
                        "bucket": int(r["bucket"]),
                        "first_action": r["first_action"],
                        "oracle_action": r["oracle_action"],
                        "action_match_oracle": int(str(r["first_action"]) == str(r["oracle_action"])),
                        "shape_distance_to_oracle": float(np.linalg.norm(pvec - tvec)),
                        "baseline_distance_to_oracle": float(np.linalg.norm(bvec - tvec)) if np.isfinite(bvec).all() else np.nan,
                        "shape_closer_than_baseline": int(np.linalg.norm(pvec - tvec) < np.linalg.norm(bvec - tvec)) if np.isfinite(bvec).all() else 0,
                        "b0": r["b0"], "b1": r["b1"], "b2": r["b2"],
                        "oracle_b0": r["oracle_b0"], "oracle_b1": r["oracle_b1"], "oracle_b2": r["oracle_b2"],
                    }
                )
        far = frame[["case_id", "bucket", "far_available", "far_hint_any", "hidden_relief", "hidden_intensification", "reversal", "direction_shift", "far_60_80", "far_80_100", "far_100_120"]].copy()
        far_vals = far[["far_60_80", "far_80_100", "far_100_120"]].to_numpy(float)
        far["far_range"] = np.nanmax(far_vals, axis=1) - np.nanmin(far_vals, axis=1)
        far["event_timing"] = _event_timing(far_vals)
        far.insert(0, "run", name)
        far_rows.extend(far.to_dict("records"))
        summary_rows.append(
            {
                "run": name,
                "near_b012_mae": float(np.nanmean(np.abs(err))) if err.size else np.nan,
                "near_b012_rmse": float(np.sqrt(np.nanmean(err ** 2))) if err.size else np.nan,
                "near_direction_agreement": _direction_agreement(pred_v, true_v),
                "near_range_mae": float(np.nanmean(np.abs(pred_range - true_range))) if pred_range.size else np.nan,
                "near_range_utilization": float(np.nanmean(pred_range / np.clip(true_range, 1e-9, None))) if pred_range.size else np.nan,
                "anti_persistence_collapse_count": int(collapsed.sum()) if collapsed.size else 0,
                "anti_persistence_collapse_ratio": float(collapsed.mean()) if collapsed.size else np.nan,
                "far_hint_any_count": int(frame.get("far_hint_any", pd.Series(dtype=float)).sum()),
                "hidden_relief_count": int(frame.get("hidden_relief", pd.Series(dtype=float)).sum()),
                "hidden_intensification_count": int(frame.get("hidden_intensification", pd.Series(dtype=float)).sum()),
                "reversal_count": int(frame.get("reversal", pd.Series(dtype=float)).sum()),
                "direction_shift_count": int(frame.get("direction_shift", pd.Series(dtype=float)).sum()),
                "far_range_mean": float(far["far_range"].mean()) if not far.empty else np.nan,
                "oracle_diff_key_buckets": int(key_count),
                "key_shape_closer_than_baseline": closer,
                "key_first_action_match_oracle": key_action_matches,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(key_rows), pd.DataFrame(far_rows)


def _plant_info(log_row: pd.Series, ts_row: pd.Series) -> dict[str, Any]:
    return {
        "tank_masses": np.array([_safe_float(ts_row.get("tank1_kg"), 1108000.0), _safe_float(ts_row.get("tank2_kg"), 1362000.0), _safe_float(ts_row.get("tank3_kg"), 1362000.0)], dtype=float),
        "pump_fullspeed_any": int(_safe_float(ts_row.get("pump_fullspeed_any"), 0.0) > 0.5),
        "pump_total_backlog_kg": _safe_float(ts_row.get("pump_total_backlog_kg"), 0.0),
        "pump_rate_cmd_m3_min": np.array([_safe_float(ts_row.get("pump_rate1_m3min"), 0.0), _safe_float(ts_row.get("pump_rate2_m3min"), 0.0), _safe_float(ts_row.get("pump_rate3_m3min"), 0.0)], dtype=float),
        "posture_vec_deg": np.array([_safe_float(log_row.get("current_pitch_deg"), 0.0), _safe_float(log_row.get("current_roll_deg"), 0.0)], dtype=float),
        "forecast_has_future": 1,
    }


def _adapter_for_run(
    run_name: str,
    dataset_dir: Path,
    adapters: dict[str, Any],
) -> Any | None:
    if run_name == "oracle":
        return None
    if run_name not in adapters:
        if run_name == "persistence":
            adapters[run_name] = PersistenceMeanForecastAdapter(dataset_dir=dataset_dir)
        elif run_name == "current_only":
            adapters[run_name] = CurrentOnlyForecastAdapter(dataset_dir=dataset_dir)
        elif run_name in FORECAST_MODEL_DIRS:
            adapters[run_name] = ForecastModelAdapter(
                model_dir=_run_path(FORECAST_MODEL_DIRS[run_name]),
                dataset_dir=dataset_dir,
                device="cpu",
            )
        else:
            raise ValueError(f"no forecast adapter mapping for run={run_name!r}")
    return adapters[run_name]


def _forecast_uv_for_run(
    run_name: str,
    sample: Any,
    timestamp: str,
    dataset_dir: Path,
    adapters: dict[str, Any],
) -> tuple[np.ndarray, bool, str]:
    if run_name == "oracle":
        return np.asarray(sample.y_uv_raw, dtype=float), True, "oracle"
    adapter = _adapter_for_run(run_name, dataset_dir, adapters)
    result = adapter.predict_window(sample.x_window, timestamp=timestamp)
    return (
        np.asarray(result.wind_uv_raw, dtype=float),
        bool(getattr(adapter, "provides_future_preview", True)),
        str(result.model_version),
    )


def _nearest_ts(ts: pd.DataFrame, time_s: float) -> pd.Series:
    if ts.empty or "t_s" not in ts.columns:
        return pd.Series(dtype=object)
    idx = (pd.to_numeric(ts["t_s"], errors="coerce") - float(time_s)).abs().idxmin()
    return ts.loc[idx]


def _action_margin_for_run(run_name: str, run_dir: Path, key_df: pd.DataFrame, dataset_dir: Path) -> pd.DataFrame:
    if key_df.empty:
        return pd.DataFrame()
    cfg, discounts = _cfg_from_casebook(run_dir)
    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split="test")
    adapters: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for _, key in key_df.iterrows():
        case = _case_from_name(str(key["case_id"]))
        bucket = int(key["bucket"])
        log_path = next((run_dir / "planner_logs").glob(f"*{case}*_planner_log.csv"), None)
        ts_path = next((run_dir / "timeseries").glob(f"*{case}*_timeseries.csv"), None)
        if log_path is None or ts_path is None:
            continue
        log = pd.read_csv(log_path, low_memory=False)
        hit = log[log["bucket"].astype(int).eq(bucket)]
        if hit.empty:
            continue
        log_row = hit.iloc[0]
        ts = pd.read_csv(ts_path, low_memory=False)
        ts_row = _nearest_ts(ts, _safe_float(log_row.get("current_time_s"), bucket * 600.0))
        hist = str(log_row.get("history_end", "")).strip()
        if not hist:
            continue
        from datetime import datetime
        sample = replay.sample_for_history_end(datetime.strptime(hist, TIMESTAMP_FMT))
        if sample is None:
            continue
        plant = _plant_info(log_row, ts_row)
        try:
            uv, forecast_has_future, forecast_version = _forecast_uv_for_run(
                run_name,
                sample,
                hist,
                dataset_dir,
                adapters,
            )
        except Exception as exc:
            rows.append(
                {
                    "run": run_name,
                    "case_id": case,
                    "bucket": bucket,
                    "margin_status": f"forecast_error:{type(exc).__name__}",
                    "margin_error": str(exc),
                }
            )
            continue
        plant["forecast_has_future"] = int(bool(forecast_has_future))
        effective_discounts = discounts if forecast_has_future else [1.0 for _ in discounts]
        raw_blocks = compute_pressure_blocks(uv, effective_discounts, cfg)
        planner_blocks, posture_meta = apply_posture_state_to_blocks(raw_blocks, plant, cfg)
        info = dict(plant)
        info.update(posture_meta)
        evaluated = [evaluate_sequence(seq, planner_blocks, info, cfg) for seq in _ALL_SEQUENCES]
        feasible, scalars = select_best_economic(evaluated, cfg)
        if not feasible:
            rows.append({"run": run_name, "case_id": case, "bucket": bucket, "margin_status": "no_feasible"})
            continue
        action_best: dict[str, tuple[int, float, str]] = {}
        for rank, (item, cost) in enumerate(zip(feasible, scalars), start=1):
            action = str(item["sequence"][0])
            action_best.setdefault(action, (rank, float(cost), ">".join(item["sequence"])))
        chosen = str(log_row.get("first_action", ""))
        oracle_action = str(key.get("oracle_first_action", ""))
        best_cost = float(scalars[0])
        chosen_tuple = action_best.get(chosen)
        oracle_tuple = action_best.get(oracle_action)
        rows.append(
            {
                "run": run_name,
                "case_id": case,
                "bucket": bucket,
                "chosen_action": chosen,
                "oracle_action": oracle_action,
                "best_action": str(feasible[0]["sequence"][0]),
                "chosen_action_rank": chosen_tuple[0] if chosen_tuple else np.nan,
                "chosen_action_cost": chosen_tuple[1] if chosen_tuple else np.nan,
                "oracle_action_rank": oracle_tuple[0] if oracle_tuple else np.nan,
                "oracle_action_cost": oracle_tuple[1] if oracle_tuple else np.nan,
                "oracle_action_cost_gap": (oracle_tuple[1] - best_cost) if oracle_tuple else np.nan,
                "oracle_action_margin_ratio": ((oracle_tuple[1] - best_cost) / max(abs(best_cost), 1e-9)) if oracle_tuple else np.nan,
                "oracle_action_sequence": oracle_tuple[2] if oracle_tuple else "hard_rejected_or_missing",
                "chosen_action_sequence": chosen_tuple[2] if chosen_tuple else "hard_rejected_or_missing",
                "forecast_version": forecast_version,
                "forecast_has_future": int(bool(forecast_has_future)),
                "margin_status": "ok",
            }
        )
    return pd.DataFrame(rows)


def _closed_loop_summary(runs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    case_rows: list[pd.DataFrame] = []
    for name, path in runs.items():
        df = _timeseries_metrics(path)
        if df.empty:
            continue
        tmp = df.copy()
        tmp.insert(0, "run", name)
        case_rows.append(tmp)
        rows.append(
            {
                "run": name,
                "sum_pump": float(df["pump_m3"].sum()),
                "max_p95": float(df["pitch_p95"].max()),
                "sum_fb": float(df["fallback_ratio"].sum()),
                "time_over_3": int(df["time_over_3_s"].sum()),
                "time_over_5": int(df["time_over_5_s"].sum()),
                "lowrisk_pump": float(df[df["case_id"].str.contains("lowrisk", na=False)]["pump_m3"].sum()),
                "lowrisk_fb": float(df[df["case_id"].str.contains("lowrisk", na=False)]["fallback_ratio"].sum()),
            }
        )
    return pd.DataFrame(rows), pd.concat(case_rows, ignore_index=True) if case_rows else pd.DataFrame()


def _md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    if cols:
        df = df[cols]
    if df.empty:
        return "_none_"
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/wind_prediction/model_capability_map_v1")
    parser.add_argument("--key-buckets", default=KEY_BUCKETS)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    for name, path in RUNS.items():
        parser.add_argument(f"--run-{name}", default=path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _run_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {name: _run_path(getattr(args, f"run_{name}")) for name in RUNS}
    key_path = _run_path(args.key_buckets)
    key_df = pd.read_csv(key_path, low_memory=False) if key_path.exists() else pd.DataFrame()
    if not key_df.empty:
        key_df["case_id"] = key_df["case_id"].map(_case_from_name)
    closed, case_detail = _closed_loop_summary(runs)
    capability, key_table, far_table = _shape_tables(runs, "oracle", "h240_f120_baseline", key_df)
    margin_rows = []
    dataset_dir = _run_path(args.dataset_dir)
    # Margin replay is useful only for forecast modes with 120-min dataset support; f60 has a different dataset.
    for name in ["h240_f120_baseline", "h240_f120_relief_e15", "pressure_shape_v1", "planner_block_shape_v1", "near_block_eventbalanced_v2", "multitask_far_hint_v0", "oracle", "persistence", "current_only"]:
        margin_rows.append(_action_margin_for_run(name, runs[name], key_df, dataset_dir))
    margin = pd.concat([m for m in margin_rows if not m.empty], ignore_index=True) if margin_rows else pd.DataFrame()
    if not margin.empty:
        base = margin[margin["run"].eq("h240_f120_baseline")][["case_id", "bucket", "oracle_action_cost_gap", "oracle_action_rank"]].rename(columns={"oracle_action_cost_gap": "baseline_oracle_action_cost_gap", "oracle_action_rank": "baseline_oracle_action_rank"})
        margin = margin.merge(base, on=["case_id", "bucket"], how="left")
        margin["oracle_margin_gap_delta_vs_baseline"] = margin["oracle_action_cost_gap"] - margin["baseline_oracle_action_cost_gap"]
        margin["oracle_rank_delta_vs_baseline"] = margin["oracle_action_rank"] - margin["baseline_oracle_action_rank"]
    merged_summary = capability.merge(closed, on="run", how="left")
    merged_summary.to_csv(out_dir / "model_capability_table.csv", index=False)
    key_table.to_csv(out_dir / "key_bucket_capability_table.csv", index=False)
    margin.to_csv(out_dir / "action_margin_table.csv", index=False)
    far_table.to_csv(out_dir / "far_horizon_event_table.csv", index=False)
    case_detail.to_csv(out_dir / "case_closed_loop_detail.csv", index=False)

    focus = merged_summary[[
        "run", "sum_pump", "max_p95", "sum_fb", "time_over_3", "time_over_5",
        "near_b012_mae", "near_b012_rmse", "near_direction_agreement", "near_range_utilization",
        "anti_persistence_collapse_ratio", "key_shape_closer_than_baseline", "key_first_action_match_oracle",
        "far_hint_any_count", "hidden_relief_count", "hidden_intensification_count",
    ]]
    margin_summary = pd.DataFrame()
    if not margin.empty:
        margin_summary = margin.groupby("run", as_index=False).agg(
            oracle_action_rank_mean=("oracle_action_rank", "mean"),
            oracle_action_cost_gap_mean=("oracle_action_cost_gap", "mean"),
            margin_gap_delta_mean=("oracle_margin_gap_delta_vs_baseline", "mean"),
            rank_delta_mean=("oracle_rank_delta_vs_baseline", "mean"),
        )
    lines = [
        "# G001 Model Capability Map v1",
        "",
        "Evaluation-only capability map. Controller logic and forecast models were not changed by this script.",
        "",
        "## Closed-loop + 0-60min planner-block capability",
        _md_table(focus),
        "",
        "## Action decision margin",
        _md_table(margin_summary),
        "",
        "## Interpretation",
    ]
    pb = merged_summary[merged_summary["run"].eq("planner_block_shape_v1")]
    rel = merged_summary[merged_summary["run"].eq("h240_f120_relief_e15")]
    v2 = merged_summary[merged_summary["run"].eq("near_block_eventbalanced_v2")]
    if not pb.empty:
        r = pb.iloc[0]
        lines.append(f"- `planner_block_shape_v1` shape closer is {int(r['key_shape_closer_than_baseline'])}/18 and action match is {int(r['key_first_action_match_oracle'])}/18.")
    if not rel.empty:
        r = rel.iloc[0]
        lines.append(f"- `h240_f120_relief_e15` shape closer is {int(r['key_shape_closer_than_baseline'])}/18 and action match is {int(r['key_first_action_match_oracle'])}/18.")
    if not v2.empty:
        r = v2.iloc[0]
        lines.append(
            f"- `near_block_eventbalanced_v2` shape closer is {int(r['key_shape_closer_than_baseline'])}/18, "
            f"action match is {int(r['key_first_action_match_oracle'])}/18, "
            f"sum_pump={r['sum_pump']:.1f}, max_p95={r['max_p95']:.3f}, sum_fb={r['sum_fb']:.3f}."
        )
    if not margin_summary.empty and "planner_block_shape_v1" in set(margin_summary["run"]):
        r = margin_summary[margin_summary["run"].eq("planner_block_shape_v1")].iloc[0]
        lines.append(f"- planner_block_shape_v1 oracle-action cost-gap delta vs baseline is {r['margin_gap_delta_mean']:.3f}; negative means oracle action moved closer to selected.")
    if not margin_summary.empty and "near_block_eventbalanced_v2" in set(margin_summary["run"]):
        r = margin_summary[margin_summary["run"].eq("near_block_eventbalanced_v2")].iloc[0]
        lines.append(f"- near_block_eventbalanced_v2 oracle-action cost-gap delta vs baseline is {r['margin_gap_delta_mean']:.3f}; it does not move oracle actions closer to selection.")
    lines.extend([
        "",
        "## Outputs",
        "- `model_capability_table.csv`",
        "- `key_bucket_capability_table.csv`",
        "- `action_margin_table.csv`",
        "- `far_horizon_event_table.csv`",
        "- `case_closed_loop_detail.csv`",
    ])
    (out_dir / "model_capability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'model_capability_summary.md'}")


if __name__ == "__main__":
    main()
