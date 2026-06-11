#!/usr/bin/env python3
"""Evaluate whether a candidate improves sustained attitude recovery behavior.

The script compares a baseline casebook directory against a candidate directory.
It intentionally does not modify the controller; it only reads casebook outputs
(`casebook_summary.csv`, `timeseries/`, and `planner_logs/`) and writes a
unified metric table plus a short markdown review.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ATTENTION_DEG = 3.0
HIGH_DEG = 4.0
BUCKET_S = 600.0
RETURN_HOLD_S = 60.0
RESPONSE_EPS_DEG = 0.3
TARGET_ERR_KG = 1500.0
PUMP_IDLE_M3_MIN = 0.5
FUTURE_RELIEF_NORM_DROP = 0.25
FUTURE_RELIEF_REVERSAL_DOT = 0.0
FALLBACK_EPS = 0.01
PUMP_ABS_WARN_M3 = 50.0
PUMP_FRAC_WARN = 0.25
LOWRISK_SAFE_PUMP_EPS_M3 = 5.0
MIN_MEANINGFUL_TIME_REDUCTION_S = 600.0
MIN_MEANINGFUL_CONTINUOUS_REDUCTION_S = 300.0
MIN_MEANINGFUL_P95_IMPROVEMENT_DEG = 0.2

HOLD_ACTIONS = {"", "nan", "none", "hold", "pump_saving", "pause"}

CASE_TYPE_LABELS = {
    "A": "high-risk / high-pressure / fallback-prone",
    "B": "persistent posture / low-pressure high-attitude",
    "C": "lowrisk safe posture",
    "D": "transition / onset / signflip",
    "UNKNOWN": "unclassified",
}

# This is a script-level configuration, not result-based relabeling. Users can
# override it with --case-type-csv containing case_id,case_type.
DEFAULT_CASE_TYPE_BY_NAME = {
    "fr_relief_09": "A",
    "b_high_pressure_event": "A",
    "high_pressure": "A",
    "fr_relief_01": "D",
    "lowrisk_clean": "B",
    "lowrisk_residual_high": "B",
    "lowrisk_random_03": "C",
    "lowrisk_safe": "C",
    "sf_holdout": "D",
    "signflip": "D",
}


@dataclass(frozen=True)
class Thresholds:
    attention_deg: float
    high_deg: float
    bucket_s: float
    return_hold_s: float
    response_eps_deg: float
    target_err_kg: float
    pump_idle_m3_min: float
    future_relief_norm_drop: float
    future_relief_reversal_dot: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, help="Baseline casebook output dir.")
    parser.add_argument("--candidate-dir", required=True, help="Candidate casebook output dir.")
    parser.add_argument("--out-dir", required=True, help="Directory for metric CSV/markdown outputs.")
    parser.add_argument(
        "--case-type-csv",
        default=None,
        help="Optional CSV with columns case_id,case_type. case_type should be A/B/C/D.",
    )
    parser.add_argument("--attention-deg", type=float, default=ATTENTION_DEG)
    parser.add_argument("--high-deg", type=float, default=HIGH_DEG)
    parser.add_argument("--bucket-s", type=float, default=BUCKET_S)
    parser.add_argument("--return-hold-s", type=float, default=RETURN_HOLD_S)
    parser.add_argument("--response-eps-deg", type=float, default=RESPONSE_EPS_DEG)
    parser.add_argument("--target-err-kg", type=float, default=TARGET_ERR_KG)
    parser.add_argument("--pump-idle-m3-min", type=float, default=PUMP_IDLE_M3_MIN)
    parser.add_argument("--future-relief-norm-drop", type=float, default=FUTURE_RELIEF_NORM_DROP)
    parser.add_argument(
        "--future-relief-reversal-dot",
        type=float,
        default=FUTURE_RELIEF_REVERSAL_DOT,
    )
    return parser.parse_args()


def _num(values: pd.Series | float | int, default: float = np.nan) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").fillna(default)
    return pd.Series([float(values)])


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _first_existing(df: pd.DataFrame, cols: Iterable[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(default, index=df.index, dtype=float)
    for col in cols:
        if col in df.columns:
            candidate = pd.to_numeric(df[col], errors="coerce")
            out = out.where(out.notna(), candidate)
    return out.fillna(default)


def _first_existing_text(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    out = pd.Series("", index=df.index, dtype=object)
    for col in cols:
        if col in df.columns:
            candidate = df[col].astype(str)
            out = out.where(out.astype(str) != "", candidate)
    return out.fillna("").astype(str)


def _case_id_from_timeseries(path: Path) -> str:
    name = path.name
    match = re.match(
        r"(.+?)_\d{4}-\d{2}-\d{2}_\d{6}_.+_timeseries\.csv$",
        name,
    )
    if match:
        return match.group(1)
    suffix = "_prediction_primary_econ_timeseries.csv"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    match = re.match(r"(.+)_\d{4}-\d{2}-\d{2}_\d{6}$", name)
    return match.group(1) if match else name


def _normalized_case_id(case_id: str) -> str:
    out = str(case_id).strip()
    out = re.sub(r"^\d+_", "", out)
    return out


def _case_sort_key(case_id: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)_", str(case_id))
    return (int(match.group(1)) if match else 9999, str(case_id))


def _case_type_for(case_id: str, mapping: dict[str, str]) -> str:
    norm = _normalized_case_id(case_id)
    if norm in mapping:
        return mapping[norm]
    if case_id in mapping:
        return mapping[case_id]
    for key, case_type in DEFAULT_CASE_TYPE_BY_NAME.items():
        if key in norm:
            return case_type
    return "UNKNOWN"


def _load_case_type_mapping(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    df = pd.read_csv(path)
    required = {"case_id", "case_type"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        case_type = str(row["case_type"]).strip().upper()
        if case_type not in CASE_TYPE_LABELS:
            raise SystemExit(f"invalid case_type {case_type!r}; expected A/B/C/D")
        case_id = str(row["case_id"]).strip()
        mapping[case_id] = case_type
        mapping[_normalized_case_id(case_id)] = case_type
    return mapping


def _read_summary(casebook_dir: Path) -> pd.DataFrame:
    path = casebook_dir / "casebook_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "case_id" not in df.columns:
        return pd.DataFrame()
    df["_case_norm"] = df["case_id"].astype(str).map(_normalized_case_id)
    return df


def _summary_value(summary: pd.DataFrame, case_id: str, col: str, default: float = np.nan) -> float:
    if summary.empty or col not in summary.columns:
        return default
    norm = _normalized_case_id(case_id)
    rows = summary[summary["_case_norm"] == norm]
    if rows.empty:
        return default
    return _safe_float(rows.iloc[0][col], default)


def _summary_text(summary: pd.DataFrame, case_id: str, col: str, default: str = "") -> str:
    if summary.empty or col not in summary.columns:
        return default
    norm = _normalized_case_id(case_id)
    rows = summary[summary["_case_norm"] == norm]
    if rows.empty:
        return default
    value = rows.iloc[0][col]
    if pd.isna(value):
        return default
    return str(value)


def _dt_s(df: pd.DataFrame) -> float:
    if "t_s" not in df.columns or len(df) < 2:
        return 1.0
    diff = pd.to_numeric(df["t_s"], errors="coerce").diff().dropna()
    diff = diff[diff > 0]
    if diff.empty:
        return 1.0
    return float(diff.median())


def _max_true_run_count(flags: Iterable[bool]) -> int:
    run = 0
    best = 0
    for flag in flags:
        if bool(flag):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _first_return_under(
    attitude: pd.Series,
    time_s: pd.Series,
    threshold: float,
    return_hold_s: float,
    dt_s: float,
) -> tuple[bool, float, float]:
    high_idx = np.flatnonzero((attitude > threshold).to_numpy())
    if len(high_idx) == 0:
        return True, 0.0, 0.0

    first_high_i = int(high_idx[0])
    first_high_t = float(time_s.iloc[first_high_i])
    required = max(1, int(math.ceil(return_hold_s / max(dt_s, 1e-6))))
    below = (attitude <= threshold).to_numpy()
    run = 0
    run_start_i = first_high_i
    for i in range(first_high_i + 1, len(attitude)):
        if below[i]:
            if run == 0:
                run_start_i = i
            run += 1
            if run >= required:
                return_t = float(time_s.iloc[run_start_i])
                return True, return_t, return_t - first_high_t
        else:
            run = 0
    return False, np.nan, np.nan


def _mode_text(series: pd.Series) -> str:
    cleaned = series.dropna().astype(str)
    if cleaned.empty:
        return ""
    mode = cleaned.mode()
    return str(mode.iloc[0]) if not mode.empty else str(cleaned.iloc[0])


def _bucketize_timeseries(df: pd.DataFrame, thresholds: Thresholds) -> pd.DataFrame:
    work = df.copy()
    work["bucket"] = np.floor(_num(work.get("t_s", 0.0), 0.0) / thresholds.bucket_s).astype(int)
    pitch_abs = _num(work.get("pitch_deg", 0.0), 0.0).abs()
    roll_abs = _num(work.get("roll_deg", 0.0), 0.0).abs()
    work["_pitch_abs"] = pitch_abs
    work["_roll_abs"] = roll_abs
    work["_attitude_abs"] = np.maximum(pitch_abs, roll_abs)
    work["_pump_rate"] = _num(work.get("pump_total_rate_m3_min", 0.0), 0.0)
    work["_fallback"] = _first_existing(
        work,
        ["preview_primary_safety_fallback", "pump_fullspeed_any", "preview_primary_safety_active"],
        0.0,
    )

    rows: list[dict[str, object]] = []
    for bucket, group in work.groupby("bucket", sort=True):
        rows.append(
            {
                "bucket": int(bucket),
                "bucket_start_s": float(group["t_s"].min()) if "t_s" in group else bucket * thresholds.bucket_s,
                "pitch_abs_p95": float(group["_pitch_abs"].quantile(0.95)),
                "roll_abs_p95": float(group["_roll_abs"].quantile(0.95)),
                "attitude_abs_p95": float(group["_attitude_abs"].quantile(0.95)),
                "attitude_abs_mean": float(group["_attitude_abs"].mean()),
                "pump_rate_mean_m3_min": float(group["_pump_rate"].mean()),
                "fallback_ratio": float(group["_fallback"].mean()),
                "preview_primary_action": (
                    _mode_text(group["preview_primary_action"])
                    if "preview_primary_action" in group.columns
                    else ""
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["bucket_over_3"] = out["attitude_abs_p95"] > thresholds.attention_deg
    out["bucket_over_4"] = out["attitude_abs_p95"] > thresholds.high_deg
    out["prev_attitude_abs_p95"] = out["attitude_abs_p95"].shift(1)
    out["next1_attitude_abs_p95"] = out["attitude_abs_p95"].shift(-1)
    out["next2_attitude_abs_p95"] = out["attitude_abs_p95"].shift(-2)
    out["current_bucket_response_deg"] = out["prev_attitude_abs_p95"] - out["attitude_abs_p95"]
    out["next1_bucket_response_deg"] = out["attitude_abs_p95"] - out["next1_attitude_abs_p95"]
    out["next2_bucket_response_deg"] = out["attitude_abs_p95"] - out["next2_attitude_abs_p95"]
    return out


def _is_hold_like(action: str) -> bool:
    action_l = str(action).strip().lower()
    return action_l in HOLD_ACTIONS or action_l.startswith("hold")


def _is_active_like(action: str) -> bool:
    action_l = str(action).strip().lower()
    if _is_hold_like(action_l):
        return False
    return action_l.startswith("active") or action_l in {
        "small",
        "medium",
        "large",
        "posture_directed",
    }


def _future_relief_status(log: pd.DataFrame, thresholds: Thresholds) -> pd.Series:
    norm0_col = "raw_pressure_block0_norm" if "raw_pressure_block0_norm" in log else "pressure_block0_norm"
    norm2_col = "raw_pressure_block2_norm" if "raw_pressure_block2_norm" in log else "pressure_block2_norm"
    dot_col = "raw_pressure_block02_dot" if "raw_pressure_block02_dot" in log else "pressure_block02_dot"
    has_norms = norm0_col in log.columns and norm2_col in log.columns
    has_dot = dot_col in log.columns
    if not has_norms and not has_dot:
        return pd.Series("unknown", index=log.index, dtype=object)

    relief = pd.Series(False, index=log.index)
    known = pd.Series(False, index=log.index)
    if has_norms:
        norm0 = _num(log[norm0_col])
        norm2 = _num(log[norm2_col])
        known = known | (norm0.notna() & norm2.notna())
        relief = relief | ((norm0 - norm2) >= thresholds.future_relief_norm_drop)
    if has_dot:
        dot = _num(log[dot_col])
        known = known | dot.notna()
        relief = relief | (dot < thresholds.future_relief_reversal_dot)

    out = pd.Series("unknown", index=log.index, dtype=object)
    out.loc[known & relief] = "clear"
    out.loc[known & ~relief] = "absent"
    return out


def _planner_behavior_metrics(
    log_path: Path | None,
    bucket_df: pd.DataFrame,
    thresholds: Thresholds,
) -> dict[str, float]:
    empty = {
        "high_hold_bucket_count": 0,
        "high_hold_duration_s": 0.0,
        "max_continuous_high_hold_buckets": 0,
        "high_hold_with_target_reached": 0,
        "high_hold_with_pump_idle": 0,
        "high_hold_without_future_relief": 0,
        "high_hold_unknown_future_relief": 0,
        "unexplained_hold_bucket_count": 0,
        "future_relief_unknown_bucket_count": 0,
        "active_but_weak_bucket_count": 0,
        "non_recovering_bucket_count": 0,
        "current_bucket_response_deg": np.nan,
        "next1_bucket_response_deg": np.nan,
        "next2_bucket_response_deg": np.nan,
    }
    if bucket_df.empty:
        return empty

    behavior = bucket_df.copy()
    behavior["planner_action"] = behavior["preview_primary_action"].astype(str)

    if log_path and log_path.exists():
        log = pd.read_csv(log_path, low_memory=False)
        if "bucket" in log.columns:
            log["bucket"] = _num(log["bucket"], 0.0).astype(int)
            keep = [
                c
                for c in [
                    "bucket",
                    "first_action",
                    "planner_first_action_raw",
                    "current_pitch_deg",
                    "current_roll_deg",
                    "prediction_primary_stall_target_err_mean_kg",
                    "prediction_primary_stall_pump_rate_m3_min",
                    "no_unexplained_hold_target_err_mean_kg",
                    "no_unexplained_hold_pump_rate_m3_min",
                    "no_unexplained_hold_fullspeed_proxy",
                    "recovery_mode_fullspeed_proxy",
                    "pressure_block0_norm",
                    "pressure_block2_norm",
                    "pressure_block02_dot",
                    "raw_pressure_block0_norm",
                    "raw_pressure_block2_norm",
                    "raw_pressure_block02_dot",
                ]
                if c in log.columns
            ]
            log = log[keep].copy()
            log["future_relief_status"] = _future_relief_status(log, thresholds)
            log["planner_action"] = _first_existing_text(
                log, ["first_action", "planner_first_action_raw"]
            )
            log["planner_attitude_abs"] = np.maximum(
                _num(log.get("current_pitch_deg", 0.0), 0.0).abs(),
                _num(log.get("current_roll_deg", 0.0), 0.0).abs(),
            )
            merge_cols = [
                "bucket",
                "planner_action",
                "planner_attitude_abs",
                "prediction_primary_stall_target_err_mean_kg",
                "prediction_primary_stall_pump_rate_m3_min",
                "no_unexplained_hold_target_err_mean_kg",
                "no_unexplained_hold_pump_rate_m3_min",
                "no_unexplained_hold_fullspeed_proxy",
                "recovery_mode_fullspeed_proxy",
                "future_relief_status",
            ]
            for col in merge_cols:
                if col not in log.columns:
                    log[col] = 0.0 if col != "future_relief_status" else "unknown"
            behavior = behavior.merge(
                log[merge_cols],
                on="bucket",
                how="left",
                suffixes=("_bucket", "_planner"),
            )
            behavior["planner_action"] = behavior["planner_action_planner"].where(
                behavior["planner_action_planner"].astype(str) != "nan",
                behavior["planner_action_bucket"],
            )
        else:
            behavior["future_relief_status"] = "unknown"
    else:
        behavior["future_relief_status"] = "unknown"

    behavior = behavior.sort_values("bucket").reset_index(drop=True)
    planner_attitude = _num(
        behavior.get("planner_attitude_abs", behavior["attitude_abs_p95"]),
        np.nan,
    )
    behavior["effective_attitude_abs"] = planner_attitude.where(
        planner_attitude.notna(), behavior["attitude_abs_p95"]
    )
    behavior["high_posture"] = behavior["effective_attitude_abs"] > thresholds.attention_deg
    behavior["hold_like"] = behavior["planner_action"].astype(str).map(_is_hold_like)
    behavior["active_like"] = behavior["planner_action"].astype(str).map(_is_active_like)
    target_err = _first_existing(
        behavior,
        ["no_unexplained_hold_target_err_mean_kg", "prediction_primary_stall_target_err_mean_kg"],
        np.inf,
    )
    pump_rate = _first_existing(
        behavior,
        ["no_unexplained_hold_pump_rate_m3_min", "prediction_primary_stall_pump_rate_m3_min"],
        np.inf,
    )
    fallback_proxy = _first_existing(
        behavior,
        ["no_unexplained_hold_fullspeed_proxy", "recovery_mode_fullspeed_proxy"],
        np.nan,
    )
    fallback_proxy = fallback_proxy.where(fallback_proxy.notna(), behavior["fallback_ratio"])
    behavior["target_reached"] = target_err <= thresholds.target_err_kg
    behavior["pump_idle"] = pump_rate <= thresholds.pump_idle_m3_min
    behavior["fallback_free"] = fallback_proxy <= FALLBACK_EPS
    behavior["future_relief_status"] = behavior.get("future_relief_status", "unknown").fillna(
        "unknown"
    )
    behavior["high_hold"] = behavior["high_posture"] & behavior["hold_like"]
    behavior["future_relief_absent"] = behavior["future_relief_status"].eq("absent")
    behavior["future_relief_unknown"] = behavior["future_relief_status"].eq("unknown")
    behavior["unexplained_hold"] = (
        behavior["high_hold"]
        & behavior["target_reached"]
        & behavior["pump_idle"]
        & behavior["fallback_free"]
        & behavior["future_relief_absent"]
    )

    next_min = behavior[["next1_attitude_abs_p95", "next2_attitude_abs_p95"]].min(axis=1)
    improvement_next2 = behavior["attitude_abs_p95"] - next_min
    next_still_high = (
        (behavior["next1_attitude_abs_p95"] > thresholds.attention_deg)
        & (behavior["next2_attitude_abs_p95"] > thresholds.attention_deg)
    )
    behavior["non_recovering"] = (
        behavior["bucket_over_3"]
        & next_still_high
        & (improvement_next2 < thresholds.response_eps_deg)
    )
    behavior["active_but_weak"] = (
        behavior["bucket_over_3"]
        & behavior["active_like"]
        & next_still_high
        & (improvement_next2 < thresholds.response_eps_deg)
    )

    high_response = behavior[behavior["bucket_over_3"]]
    return {
        "high_hold_bucket_count": int(behavior["high_hold"].sum()),
        "high_hold_duration_s": float(behavior["high_hold"].sum() * thresholds.bucket_s),
        "max_continuous_high_hold_buckets": int(_max_true_run_count(behavior["high_hold"])),
        "high_hold_with_target_reached": int((behavior["high_hold"] & behavior["target_reached"]).sum()),
        "high_hold_with_pump_idle": int((behavior["high_hold"] & behavior["pump_idle"]).sum()),
        "high_hold_without_future_relief": int(
            (behavior["high_hold"] & behavior["future_relief_absent"]).sum()
        ),
        "high_hold_unknown_future_relief": int(
            (behavior["high_hold"] & behavior["future_relief_unknown"]).sum()
        ),
        "unexplained_hold_bucket_count": int(behavior["unexplained_hold"].sum()),
        "future_relief_unknown_bucket_count": int(behavior["future_relief_unknown"].sum()),
        "active_but_weak_bucket_count": int(behavior["active_but_weak"].sum()),
        "non_recovering_bucket_count": int(behavior["non_recovering"].sum()),
        "current_bucket_response_deg": float(high_response["current_bucket_response_deg"].mean()),
        "next1_bucket_response_deg": float(high_response["next1_bucket_response_deg"].mean()),
        "next2_bucket_response_deg": float(high_response["next2_bucket_response_deg"].mean()),
    }


def _metrics_for_case(
    casebook_dir: Path,
    case_id: str,
    ts_path: Path,
    log_path: Path | None,
    summary: pd.DataFrame,
    thresholds: Thresholds,
) -> dict[str, float | str | bool]:
    df = pd.read_csv(ts_path, low_memory=False)
    dt = _dt_s(df)
    time_s = _num(df.get("t_s", pd.Series(np.arange(len(df)))), 0.0)
    pitch_abs = _num(df.get("pitch_deg", 0.0), 0.0).abs()
    roll_abs = _num(df.get("roll_deg", 0.0), 0.0).abs()
    attitude_abs = pd.Series(np.maximum(pitch_abs, roll_abs))
    pump_rate = _num(df.get("pump_total_rate_m3_min", 0.0), 0.0)
    fallback = _first_existing(
        df,
        ["preview_primary_safety_fallback", "pump_fullspeed_any", "preview_primary_safety_active"],
        0.0,
    )

    over3 = attitude_abs > thresholds.attention_deg
    over4 = attitude_abs > thresholds.high_deg
    returned, first_return_s, latency_s = _first_return_under(
        attitude_abs,
        time_s,
        thresholds.attention_deg,
        thresholds.return_hold_s,
        dt,
    )
    bucket_df = _bucketize_timeseries(df, thresholds)
    behavior = _planner_behavior_metrics(log_path, bucket_df, thresholds)

    pump_work_summary = _summary_value(summary, case_id, "primary_pump_work_m3", np.nan)
    if math.isnan(pump_work_summary):
        pump_work_summary = float((pump_rate * dt / 60.0).sum())
    latch_switches = _summary_value(summary, case_id, "primary_latch_switches", np.nan)
    if math.isnan(latch_switches) and "pump_latch_switch_count" in df.columns:
        latch_switches = float(_num(df["pump_latch_switch_count"], 0.0).max())
    fallback_summary = _summary_value(summary, case_id, "primary_safety_fallback_ratio", np.nan)
    if math.isnan(fallback_summary):
        fallback_summary = float(fallback.mean())

    row: dict[str, float | str | bool] = {
        "case_id": case_id,
        "label": _summary_text(summary, case_id, "label", ""),
        "source_dir": str(casebook_dir),
        "pump_work_m3": float(pump_work_summary),
        "pitch_abs_p95": float(pitch_abs.quantile(0.95)),
        "roll_abs_p95": float(roll_abs.quantile(0.95)),
        "attitude_abs_p95": float(attitude_abs.quantile(0.95)),
        "pitch_abs_max": float(pitch_abs.max()),
        "roll_abs_max": float(roll_abs.max()),
        "attitude_abs_max": float(attitude_abs.max()),
        "fallback_ratio": float(fallback_summary),
        "latch_switches": float(latch_switches),
        "time_over_3deg_s": float(over3.sum() * dt),
        "time_over_4deg_s": float(over4.sum() * dt),
        "max_continuous_over_3deg_s": float(_max_true_run_count(over3) * dt),
        "max_continuous_over_4deg_s": float(_max_true_run_count(over4) * dt),
        "attitude_area_over_3_deg_s": float(np.maximum(attitude_abs - thresholds.attention_deg, 0.0).sum() * dt),
        "attitude_area_over_4_deg_s": float(np.maximum(attitude_abs - thresholds.high_deg, 0.0).sum() * dt),
        "bucket_count_over_3deg": int(bucket_df["bucket_over_3"].sum()) if not bucket_df.empty else 0,
        "bucket_count_over_4deg": int(bucket_df["bucket_over_4"].sum()) if not bucket_df.empty else 0,
        "max_continuous_over_3deg_buckets": int(_max_true_run_count(bucket_df["bucket_over_3"])) if not bucket_df.empty else 0,
        "max_continuous_over_4deg_buckets": int(_max_true_run_count(bucket_df["bucket_over_4"])) if not bucket_df.empty else 0,
        "returned_under_3deg": bool(returned),
        "first_return_under_3deg_s": float(first_return_s),
        "recovery_latency_s": float(latency_s),
    }
    row.update(behavior)
    return row


def _discover_cases(casebook_dir: Path) -> dict[str, tuple[str, Path, Path | None]]:
    ts_dir = casebook_dir / "timeseries"
    log_dir = casebook_dir / "planner_logs"
    out: dict[str, tuple[str, Path, Path | None]] = {}
    for ts_path in sorted(ts_dir.glob("*_timeseries.csv")):
        case_id = _case_id_from_timeseries(ts_path)
        norm = _normalized_case_id(case_id)
        log_paths = sorted(log_dir.glob(f"{case_id}_*_planner_log.csv"))
        out[norm] = (case_id, ts_path, log_paths[0] if log_paths else None)
    return out


def _safe_ratio(numer: float, denom: float) -> float:
    if not math.isfinite(numer) or not math.isfinite(denom) or abs(denom) < 1e-9:
        return np.inf
    return numer / denom


def _prefixed(prefix: str, row: dict[str, object]) -> dict[str, object]:
    keep_unprefixed = {"case_id", "label", "source_dir"}
    return {f"{prefix}_{key}": value for key, value in row.items() if key not in keep_unprefixed}


def _verdict_for(row: dict[str, object]) -> tuple[str, str]:
    case_type = str(row["case_type"])
    extra_pump = _safe_float(row["extra_pump_m3"])
    fallback_delta = _safe_float(row["fallback_ratio_delta"])
    max_cont3_reduced = _safe_float(row["max_continuous_over_3deg_reduced_s"])
    max_cont4_reduced = _safe_float(row["max_continuous_over_4deg_reduced_s"])
    p95_improve = _safe_float(row["attitude_abs_p95_improvement_deg"])
    area_improve = _safe_float(row["attitude_improvement_area"])
    high_hold_reduced = _safe_float(row["high_hold_bucket_reduced"])
    unexplained_reduced = _safe_float(row["unexplained_hold_bucket_reduced"])

    baseline_pump = max(_safe_float(row["baseline_pump_work_m3"], 0.0), 1.0)
    pump_large = extra_pump > PUMP_ABS_WARN_M3 or extra_pump > PUMP_FRAC_WARN * baseline_pump
    fallback_worse = fallback_delta > FALLBACK_EPS
    max_cont_worse = max_cont3_reduced < -BUCKET_S
    time3_reduced = _safe_float(row["time_over_3deg_reduced_s"])
    bucket3_reduced = _safe_float(row["bucket_count_over_3deg_reduced"])
    meaningful_duration_gain = (
        max_cont3_reduced >= MIN_MEANINGFUL_CONTINUOUS_REDUCTION_S
        or time3_reduced >= MIN_MEANINGFUL_TIME_REDUCTION_S
        or bucket3_reduced >= 1.0
    )
    meaningful_high_duration_gain = (
        max_cont4_reduced >= MIN_MEANINGFUL_CONTINUOUS_REDUCTION_S
        or _safe_float(row["time_over_4deg_reduced_s"]) >= MIN_MEANINGFUL_TIME_REDUCTION_S
        or _safe_float(row["bucket_count_over_4deg_reduced"]) >= 1.0
    )
    meaningful_p95_gain = p95_improve >= MIN_MEANINGFUL_P95_IMPROVEMENT_DEG
    no_posture_gain = (
        not meaningful_p95_gain
        and not meaningful_duration_gain
        and not meaningful_high_duration_gain
        and area_improve <= 0.0
    )

    reasons: list[str] = []
    verdict = "diagnostic_only"

    if fallback_worse:
        verdict = "fail"
        reasons.append(f"fallback rose by {fallback_delta:.3f}")
    if max_cont_worse:
        verdict = "fail"
        reasons.append("max continuous >3deg got worse")

    if case_type == "C":
        if extra_pump > LOWRISK_SAFE_PUMP_EPS_M3:
            verdict = "fail"
            reasons.append(f"lowrisk safe case added {extra_pump:.1f} m3 pump")
        elif no_posture_gain and abs(extra_pump) <= LOWRISK_SAFE_PUMP_EPS_M3:
            verdict = "pass"
            reasons.append("lowrisk safe case stayed quiet")
        else:
            verdict = "pass" if verdict != "fail" else verdict
            reasons.append("lowrisk safe posture not degraded")

    elif case_type == "A":
        safety_gain = fallback_delta < -FALLBACK_EPS or meaningful_high_duration_gain or meaningful_p95_gain
        if verdict != "fail":
            if safety_gain and not pump_large:
                verdict = "pass"
                reasons.append("high-risk posture/fallback improved without large pump increase")
            elif safety_gain:
                verdict = "partial"
                reasons.append("high-risk posture/fallback improved but pump increase is large")
            elif no_posture_gain and pump_large:
                verdict = "fail"
                reasons.append("large pump increase without high-risk posture gain")
            else:
                verdict = "partial"
                reasons.append("mixed or small high-risk change")

    elif case_type == "B":
        duration_gain = meaningful_duration_gain or area_improve > 0.0
        if verdict != "fail":
            if duration_gain and not pump_large and meaningful_p95_gain:
                verdict = "pass"
                reasons.append("persistent posture improved with acceptable pump")
            elif duration_gain:
                verdict = "partial"
                if pump_large:
                    reasons.append("persistent posture duration improved but pump increase is large")
                if p95_improve <= 0.05:
                    reasons.append("persistent posture duration improved but p95 did not move")
            elif pump_large:
                verdict = "fail"
                reasons.append("persistent posture added pump without duration gain")
            else:
                verdict = "diagnostic_only"
                reasons.append("persistent posture mostly unchanged")

    else:
        duration_gain = meaningful_duration_gain or area_improve > 0.0 or meaningful_p95_gain
        if verdict != "fail":
            if duration_gain and not pump_large:
                verdict = "pass"
                reasons.append("transition/onset posture improved without large pump increase")
            elif duration_gain:
                verdict = "partial"
                reasons.append("transition/onset posture improved but pump increase is large")
            elif pump_large:
                verdict = "fail"
                reasons.append("large pump increase without transition/onset gain")
            else:
                verdict = "diagnostic_only"
                reasons.append("transition/onset mostly unchanged")

    if high_hold_reduced > 0:
        reasons.append(f"high-hold buckets reduced by {high_hold_reduced:.0f}")
    elif high_hold_reduced < 0:
        reasons.append(f"high-hold buckets increased by {-high_hold_reduced:.0f}")

    if unexplained_reduced > 0:
        reasons.append(f"unexplained-hold buckets reduced by {unexplained_reduced:.0f}")
    elif unexplained_reduced < 0:
        reasons.append(f"unexplained-hold buckets increased by {-unexplained_reduced:.0f}")

    if pump_large and no_posture_gain and verdict != "fail":
        verdict = "fail"
        reasons.append("pump increase is not backed by sustained-attitude improvement")

    return verdict, "; ".join(dict.fromkeys(reasons))


def _compare_rows(
    case_id: str,
    case_type: str,
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "case_id": case_id,
        "case_type": case_type,
        "case_type_label": CASE_TYPE_LABELS.get(case_type, "unclassified"),
        "label": candidate.get("label") or baseline.get("label") or "",
    }
    row.update(_prefixed("baseline", baseline))
    row.update(_prefixed("candidate", candidate))

    metric_pairs = [
        ("pump_work_m3", "extra_pump_m3", "delta"),
        ("fallback_ratio", "fallback_ratio_delta", "delta"),
        ("latch_switches", "latch_switch_delta", "delta"),
        ("pitch_abs_p95", "pitch_abs_p95_improvement_deg", "improvement"),
        ("roll_abs_p95", "roll_abs_p95_improvement_deg", "improvement"),
        ("attitude_abs_p95", "attitude_abs_p95_improvement_deg", "improvement"),
        ("time_over_3deg_s", "time_over_3deg_reduced_s", "reduction"),
        ("time_over_4deg_s", "time_over_4deg_reduced_s", "reduction"),
        ("max_continuous_over_3deg_s", "max_continuous_over_3deg_reduced_s", "reduction"),
        ("max_continuous_over_4deg_s", "max_continuous_over_4deg_reduced_s", "reduction"),
        ("bucket_count_over_3deg", "bucket_count_over_3deg_reduced", "reduction"),
        ("bucket_count_over_4deg", "bucket_count_over_4deg_reduced", "reduction"),
        ("max_continuous_over_3deg_buckets", "max_continuous_over_3deg_buckets_reduced", "reduction"),
        ("max_continuous_over_4deg_buckets", "max_continuous_over_4deg_buckets_reduced", "reduction"),
        ("attitude_area_over_3_deg_s", "attitude_improvement_area", "reduction"),
        ("high_hold_bucket_count", "high_hold_bucket_reduced", "reduction"),
        ("unexplained_hold_bucket_count", "unexplained_hold_bucket_reduced", "reduction"),
        ("non_recovering_bucket_count", "non_recovering_bucket_reduced", "reduction"),
        ("active_but_weak_bucket_count", "active_but_weak_bucket_reduced", "reduction"),
    ]
    for metric, out_col, direction in metric_pairs:
        base = _safe_float(baseline.get(metric))
        cand = _safe_float(candidate.get(metric))
        if direction == "delta":
            row[out_col] = cand - base
        else:
            row[out_col] = base - cand

    extra = _safe_float(row["extra_pump_m3"])
    p95_improve = _safe_float(row["attitude_abs_p95_improvement_deg"])
    high_min_reduced = _safe_float(row["time_over_3deg_reduced_s"]) / 60.0
    bucket_reduced = _safe_float(row["bucket_count_over_3deg_reduced"])
    cont_min_reduced = _safe_float(row["max_continuous_over_3deg_reduced_s"]) / 60.0
    area_improve = _safe_float(row["attitude_improvement_area"])
    row["pump_per_deg_p95_improvement"] = _safe_ratio(extra, p95_improve)
    row["pump_per_min_high_duration_reduced"] = _safe_ratio(extra, high_min_reduced)
    row["pump_per_bucket_high_reduced"] = _safe_ratio(extra, bucket_reduced)
    row["pump_per_max_continuous_reduced"] = _safe_ratio(extra, cont_min_reduced)
    row["pump_per_area_improvement"] = _safe_ratio(extra, area_improve)

    verdict, reason = _verdict_for(row)
    row["verdict"] = verdict
    row["verdict_reason"] = reason
    return row


def _format_minutes(seconds: object) -> str:
    value = _safe_float(seconds)
    if not math.isfinite(value):
        return "NA"
    return f"{value / 60.0:.1f}"


def _write_summary(table: pd.DataFrame, out_path: Path, thresholds: Thresholds) -> None:
    lines: list[str] = [
        "# Attitude Recovery Metric Summary",
        "",
        "## Thresholds",
        "",
        f"- Attention / long-hold band: `{thresholds.attention_deg:.2f} deg`.",
        f"- Higher concern band: `{thresholds.high_deg:.2f} deg`.",
        f"- Bucket length: `{thresholds.bucket_s / 60.0:.1f} min`.",
        f"- Non-recovering bucket rule: next 2 buckets remain high and improve by `< {thresholds.response_eps_deg:.2f} deg`.",
        "- 3 deg is treated as an engineering attention band, not a hard safety limit.",
        "",
        "## Case Verdicts",
        "",
        "| case | type | verdict | pump delta m3 | p95 improvement deg | max >3 reduced min | high-hold reduced | unexplained-hold reduced | fallback delta | reason |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in table.sort_values("case_id", key=lambda s: s.map(_case_sort_key)).iterrows():
        lines.append(
            f"| {row['case_id']} | {row['case_type']} | {row['verdict']} | "
            f"{_safe_float(row['extra_pump_m3']):.1f} | "
            f"{_safe_float(row['attitude_abs_p95_improvement_deg']):.2f} | "
            f"{_safe_float(row['max_continuous_over_3deg_reduced_s']) / 60.0:.1f} | "
            f"{_safe_float(row['high_hold_bucket_reduced']):.0f} | "
            f"{_safe_float(row['unexplained_hold_bucket_reduced']):.0f} | "
            f"{_safe_float(row['fallback_ratio_delta']):.3f} | "
            f"{row['verdict_reason']} |"
        )

    improved = table[
        (table["max_continuous_over_3deg_reduced_s"] >= MIN_MEANINGFUL_CONTINUOUS_REDUCTION_S)
        | (table["time_over_3deg_reduced_s"] >= MIN_MEANINGFUL_TIME_REDUCTION_S)
        | (table["bucket_count_over_3deg_reduced"] >= 1)
        | (table["attitude_abs_p95_improvement_deg"] >= MIN_MEANINGFUL_P95_IMPROVEMENT_DEG)
        | (table["fallback_ratio_delta"] < -FALLBACK_EPS)
    ]
    inefficient = table[
        (table["extra_pump_m3"] > PUMP_ABS_WARN_M3)
        & (table["attitude_abs_p95_improvement_deg"] <= 0.05)
    ]
    unexplained = table[table["candidate_unexplained_hold_bucket_count"] > 0]
    active_weak = table[table["candidate_active_but_weak_bucket_count"] > 0]

    def case_list(df: pd.DataFrame) -> str:
        if df.empty:
            return "None."
        return ", ".join(str(x) for x in df["case_id"].tolist())

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Long high-attitude improvement appears in: {case_list(improved)}",
            f"- Extra-pump / weak-p95 cases: {case_list(inefficient)}",
            f"- Candidate still has unexplained-hold buckets in: {case_list(unexplained)}",
            f"- Candidate still has active-but-weak response in: {case_list(active_weak)}",
            "",
            "The verdict is intentionally not a single p95 gate. A candidate can be marked partial when it reduces sustained high attitude but spends too much pump, or when p95 hides a useful late-episode recovery.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    thresholds = Thresholds(
        attention_deg=float(args.attention_deg),
        high_deg=float(args.high_deg),
        bucket_s=float(args.bucket_s),
        return_hold_s=float(args.return_hold_s),
        response_eps_deg=float(args.response_eps_deg),
        target_err_kg=float(args.target_err_kg),
        pump_idle_m3_min=float(args.pump_idle_m3_min),
        future_relief_norm_drop=float(args.future_relief_norm_drop),
        future_relief_reversal_dot=float(args.future_relief_reversal_dot),
    )
    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_type_mapping = _load_case_type_mapping(args.case_type_csv)
    baseline_summary = _read_summary(baseline_dir)
    candidate_summary = _read_summary(candidate_dir)
    baseline_cases = _discover_cases(baseline_dir)
    candidate_cases = _discover_cases(candidate_dir)
    common_keys = sorted(set(baseline_cases) & set(candidate_cases), key=_case_sort_key)
    if not common_keys:
        raise SystemExit("no common cases found between baseline and candidate dirs")

    rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    for norm in common_keys:
        base_case_id, base_ts, base_log = baseline_cases[norm]
        cand_case_id, cand_ts, cand_log = candidate_cases[norm]
        case_id = cand_case_id or base_case_id
        case_type = _case_type_for(case_id, case_type_mapping)
        baseline = _metrics_for_case(
            baseline_dir,
            base_case_id,
            base_ts,
            base_log,
            baseline_summary,
            thresholds,
        )
        candidate = _metrics_for_case(
            candidate_dir,
            cand_case_id,
            cand_ts,
            cand_log,
            candidate_summary,
            thresholds,
        )
        baseline["case_id"] = case_id
        candidate["case_id"] = case_id
        rows.append(_compare_rows(case_id, case_type, baseline, candidate))
        run_rows.append({"run": "baseline", "case_type": case_type, **baseline})
        run_rows.append({"run": "candidate", "case_type": case_type, **candidate})

    table = pd.DataFrame(rows)
    table_path = out_dir / "attitude_recovery_metric_table.csv"
    table.to_csv(table_path, index=False)

    run_table = pd.DataFrame(run_rows)
    run_table.to_csv(out_dir / "attitude_recovery_run_metrics.csv", index=False)

    summary_path = out_dir / "attitude_recovery_metric_summary.md"
    _write_summary(table, summary_path, thresholds)

    display_cols = [
        "case_id",
        "case_type",
        "verdict",
        "extra_pump_m3",
        "attitude_abs_p95_improvement_deg",
        "max_continuous_over_3deg_reduced_s",
        "high_hold_bucket_reduced",
        "unexplained_hold_bucket_reduced",
        "verdict_reason",
    ]
    print(table[display_cols].to_string(index=False))
    print(summary_path)


if __name__ == "__main__":
    main()
