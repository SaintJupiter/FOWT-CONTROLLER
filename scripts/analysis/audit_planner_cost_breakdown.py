#!/usr/bin/env python3
"""Audit planner cost breakdown for high-posture hold buckets.

This is a read-only analysis script. It does not run closed-loop simulation and
does not change controller logic. For each high-posture hold bucket in an
existing casebook, it reconstructs the same oracle forecast blocks, evaluates a
small set of representative action sequences, and writes cost breakdown tables.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wind_prediction.ballast_planner import (  # noqa: E402
    ACTIONS,
    PlannerConfig,
    _ALL_SEQUENCES,
    apply_posture_state_to_blocks,
    compute_pressure_blocks,
    economic_scalar_cost,
    evaluate_sequence,
    select_best_economic,
)
from wind_prediction.replay_dataset import Fino1ReplayDataset  # noqa: E402


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
ATTENTION_DEG = 3.0
DEFAULT_DATASET_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "wind_ml_10min"
    / "ballast_decision_fino1_meteo_aux_v1"
)

REPRESENTATIVE_SEQUENCES = [
    ("hold", "hold", "hold"),
    ("pump_saving", "hold", "hold"),
    ("active_small", "hold", "hold"),
    ("active_small", "active_small", "hold"),
    ("active_medium", "hold", "hold"),
    ("active_medium", "active_medium", "hold"),
]

WEIGHTED_TERMS = [
    ("pump_work", "pump_work_cost", "w_pump_work"),
    ("pump_duration", "pump_duration_cost", "w_pump_duration"),
    ("startstop", "startstop_cost", "w_startstop"),
    ("direction_switch", "direction_switch_cost", "w_direction_switch"),
    ("reverse", "reverse_penalty", "w_reverse_penalty"),
    ("attitude_residual", "attitude_residual_cost", "w_attitude_residual"),
    ("terminal_residual", "terminal_residual_cost", "w_terminal_residual"),
    ("envelope", "envelope_violation_cost", "w_envelope_soft"),
    ("terminal_envelope", "terminal_envelope_violation", "w_terminal_envelope_soft"),
    ("saturation", "saturation_penalty", "w_saturation_hard"),
    ("envelope_barrier", "envelope_barrier_triggered", "envelope_barrier_const"),
    ("posture_hold_barrier", "posture_hold_barrier_cost", "posture_hold_barrier_const"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--casebook-dir",
        required=True,
        help="Existing baseline casebook directory with timeseries/ and planner_logs/.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--split", default="test")
    parser.add_argument("--attention-deg", type=float, default=ATTENTION_DEG)
    parser.add_argument(
        "--case-ids",
        default="",
        help="Optional comma-separated case ids/names to audit, e.g. 01,02,lowrisk_clean.",
    )
    parser.add_argument(
        "--max-buckets-per-case",
        type=int,
        default=12,
        help="Cap high-hold buckets per case to keep the audit compact.",
    )
    parser.add_argument(
        "--bucket-mode",
        choices=(
            "high_hold",
            "high_active",
            "active_but_weak",
            "fallback",
            "all_high",
            "all",
        ),
        default="high_hold",
        help=(
            "Bucket selector. high_hold reproduces the original audit; "
            "high_active/fallback support action-attribution audits for "
            "candidate runs that pump too much or raise fallback."
        ),
    )
    parser.add_argument(
        "--bucket-ids",
        default="",
        help="Optional comma-separated bucket ids to audit after case filtering.",
    )
    return parser.parse_args()


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _num(series: pd.Series | float | int, default: float = np.nan) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series([float(series)])


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(_safe_float(value, 0.0))


def _case_id_from_log(path: Path) -> str:
    name = path.name
    match = re.match(
        r"(.+?)_\d{4}-\d{2}-\d{2}_\d{6}_.+_planner_log\.csv$",
        name,
    )
    if match:
        return match.group(1)
    suffix = "_prediction_primary_econ_planner_log.csv"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    match = re.match(r"(.+)_\d{4}-\d{2}-\d{2}_\d{6}$", name)
    return match.group(1) if match else name


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


def _norm_case(case_id: str) -> str:
    return re.sub(r"^\d+_", "", str(case_id).strip())


def _case_sort_key(case_id: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)_", str(case_id))
    return (int(match.group(1)) if match else 9999, str(case_id))


def _selected(case_id: str, selected_ids: set[str]) -> bool:
    if not selected_ids:
        return True
    norm = _norm_case(case_id)
    prefix = str(case_id).split("_", 1)[0]
    return case_id in selected_ids or norm in selected_ids or prefix in selected_ids


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cfg_from_casebook(casebook_dir: Path) -> tuple[PlannerConfig, list[float]]:
    diagnostics = REPO_ROOT / "outputs" / "wind_prediction" / "planner_a1_dryrun" / "diagnostics"
    sign_cfg = _load_json(diagnostics / "a01_pressure_vec_sign_convention.json")
    discount_cfg = _load_json(diagnostics / "a1_block_discount_config.json")
    summary_path = casebook_dir / "casebook_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, low_memory=False)
        row = summary.iloc[0].to_dict() if not summary.empty else {}
    else:
        row = {}

    def value(col: str, default: Any) -> Any:
        raw = row.get(col, default)
        if pd.isna(raw):
            return default
        return raw

    cfg = PlannerConfig(
        pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]),
        pressure_norm_cap=float(value("planner_pressure_norm_cap", PlannerConfig().pressure_norm_cap)),
        envelope_use_discount=_boolish(
            value("planner_envelope_use_discount", int(PlannerConfig().envelope_use_discount))
        ),
        envelope_barrier_active=_boolish(
            value("planner_envelope_barrier_active", int(PlannerConfig().envelope_barrier_active))
        ),
        envelope_barrier_const=float(
            value("planner_envelope_barrier_const", PlannerConfig().envelope_barrier_const)
        ),
        posture_hold_barrier_active=_boolish(
            value(
                "planner_posture_hold_barrier_active",
                int(PlannerConfig().posture_hold_barrier_active),
            )
        ),
        posture_hold_pitch_deg=float(
            value("planner_posture_hold_pitch_deg", PlannerConfig().posture_hold_pitch_deg)
        ),
        posture_hold_roll_deg=float(
            value("planner_posture_hold_roll_deg", PlannerConfig().posture_hold_roll_deg)
        ),
        posture_hold_barrier_const=float(
            value("planner_posture_hold_barrier_const", PlannerConfig().posture_hold_barrier_const)
        ),
        posture_hold_relief_margin_norm=float(
            value(
                "planner_posture_hold_relief_margin_norm",
                PlannerConfig().posture_hold_relief_margin_norm,
            )
        ),
        posture_hold_forecast_credit=float(
            value(
                "planner_posture_hold_forecast_credit",
                PlannerConfig().posture_hold_forecast_credit,
            )
        ),
        posture_hold_action_uses_posture_vec=_boolish(
            value(
                "planner_posture_hold_action_uses_posture_vec",
                int(PlannerConfig().posture_hold_action_uses_posture_vec),
            )
        ),
        posture_hold_low_risk_norm_max=float(
            value(
                "planner_posture_hold_low_risk_norm_max",
                PlannerConfig().posture_hold_low_risk_norm_max,
            )
        ),
        posture_state_residual_active=_boolish(
            value(
                "planner_posture_state_residual_active",
                int(PlannerConfig().posture_state_residual_active),
            )
        ),
        posture_state_gain=float(
            value("planner_posture_state_gain", PlannerConfig().posture_state_gain)
        ),
        posture_state_decay=float(
            value("planner_posture_state_decay", PlannerConfig().posture_state_decay)
        ),
        posture_state_clip_norm=float(
            value("planner_posture_state_clip_norm", PlannerConfig().posture_state_clip_norm)
        ),
        w_attitude_residual=float(
            value("planner_attitude_residual_weight", PlannerConfig().w_attitude_residual)
        ),
        w_terminal_residual=float(
            value("planner_terminal_residual_weight", PlannerConfig().w_terminal_residual)
        ),
    )
    return cfg, [float(x) for x in discount_cfg["default_discount_blocks"]]


def _discover(casebook_dir: Path) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for path in sorted((casebook_dir / "planner_logs").glob("*_planner_log.csv")):
        case_id = _case_id_from_log(path)
        out.setdefault(_norm_case(case_id), {})["log"] = path
        out[_norm_case(case_id)]["case_id"] = case_id  # type: ignore[assignment]
    for path in sorted((casebook_dir / "timeseries").glob("*_timeseries.csv")):
        case_id = _case_id_from_timeseries(path)
        out.setdefault(_norm_case(case_id), {})["timeseries"] = path
    return out


def _nearest_timeseries_row(ts_df: pd.DataFrame, time_s: float) -> pd.Series:
    if "t_s" not in ts_df.columns or ts_df.empty:
        return pd.Series(dtype=object)
    idx = (_num(ts_df["t_s"], 0.0) - float(time_s)).abs().idxmin()
    return ts_df.loc[idx]


def _plant_info_from_row(log_row: pd.Series, ts_row: pd.Series) -> dict[str, Any]:
    masses = np.array(
        [
            _safe_float(ts_row.get("tank1_kg"), 1108000.0),
            _safe_float(ts_row.get("tank2_kg"), 1362000.0),
            _safe_float(ts_row.get("tank3_kg"), 1362000.0),
        ],
        dtype=float,
    )
    pump_rates = np.array(
        [
            _safe_float(ts_row.get("pump_rate1_m3min"), 0.0, ),
            _safe_float(ts_row.get("pump_rate2_m3min"), 0.0, ),
            _safe_float(ts_row.get("pump_rate3_m3min"), 0.0, ),
        ],
        dtype=float,
    )
    return {
        "tank_masses": masses,
        "pump_fullspeed_any": int(_safe_float(ts_row.get("pump_fullspeed_any"), 0.0) > 0.5),
        "pump_total_backlog_kg": _safe_float(ts_row.get("pump_total_backlog_kg"), 0.0),
        "pump_rate_cmd_m3_min": pump_rates,
        "posture_vec_deg": np.array(
            [
                _safe_float(log_row.get("current_pitch_deg"), 0.0),
                _safe_float(log_row.get("current_roll_deg"), 0.0),
            ],
            dtype=float,
        ),
        "forecast_has_future": 1,
    }


def _weighted_costs(result: dict[str, Any], cfg: PlannerConfig) -> dict[str, float]:
    costs = result.get("costs", {})
    out: dict[str, float] = {}
    for label, cost_key, weight_key in WEIGHTED_TERMS:
        raw = _safe_float(costs.get(cost_key), 0.0)
        weight = _safe_float(getattr(cfg, weight_key), 0.0)
        out[f"raw_{label}"] = raw
        out[f"weighted_{label}"] = raw * weight
    return out


def _sequence_row(
    case_id: str,
    bucket: int,
    sequence: tuple[str, ...],
    result: dict[str, Any],
    cfg: PlannerConfig,
    reference_label: str,
) -> dict[str, Any]:
    hard_reject = str(result.get("hard_reject_reason", ""))
    scalar = np.inf if hard_reject else float(economic_scalar_cost(result, cfg))
    row = {
        "case_id": case_id,
        "bucket": int(bucket),
        "sequence_label": reference_label,
        "sequence": ">".join(sequence),
        "first_action": sequence[0] if sequence else "",
        "hard_reject_reason": hard_reject,
        "scalar_cost": scalar,
        "active_medium_gate_reason": str(result.get("active_medium_gate_reason", "")),
        "reverse_reject_reason": str(result.get("reverse_reject_reason", "")),
        "posture_hold_barrier_reason": str(result.get("posture_hold_barrier_reason", "")),
        "posture_state_credit_reason": str(result.get("posture_state_credit_reason", "")),
        "max_envelope_norm": _safe_float(result.get("costs", {}).get("max_envelope_norm"), 0.0),
        "posture_state_norm": _safe_float(result.get("costs", {}).get("posture_state_norm"), 0.0),
        "posture_state_raw_norm": _safe_float(result.get("costs", {}).get("posture_state_raw_norm"), 0.0),
        "planner_action_pitch_deg": _safe_float(result.get("planner_action_pitch_deg"), np.nan),
        "planner_action_roll_deg": _safe_float(result.get("planner_action_roll_deg"), np.nan),
    }
    row.update(_weighted_costs(result, cfg))
    return row


def _dominant_hold_reason(hold: pd.Series, alt: pd.Series) -> str:
    if str(alt.get("hard_reject_reason", "")):
        return f"{alt['sequence_label']}_rejected:{alt['hard_reject_reason']}"
    scalar_delta = _safe_float(alt.get("scalar_cost")) - _safe_float(hold.get("scalar_cost"))
    if scalar_delta <= 0:
        return f"{alt['sequence_label']}_would_win_or_tie"
    deltas: list[tuple[str, float]] = []
    for label, _, _ in WEIGHTED_TERMS:
        key = f"weighted_{label}"
        deltas.append((label, _safe_float(alt.get(key), 0.0) - _safe_float(hold.get(key), 0.0)))
    positive = [(label, value) for label, value in deltas if value > 1e-9]
    if not positive:
        return f"{alt['sequence_label']}_higher_cost_no_positive_component"
    label, value = max(positive, key=lambda item: item[1])
    return f"{label}_penalty_blocks_{alt['sequence_label']}:+{value:.3f}"


def _audit_one_bucket(
    replay: Fino1ReplayDataset,
    cfg: PlannerConfig,
    discounts: list[float],
    case_id: str,
    log_row: pd.Series,
    ts_row: pd.Series,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    history_end = str(log_row.get("history_end", "")).strip()
    if not history_end:
        return [], None
    sample = replay.sample_for_history_end(datetime.strptime(history_end, TIMESTAMP_FMT))
    if sample is None:
        return [], None

    plant_info = _plant_info_from_row(log_row, ts_row)
    raw_blocks = compute_pressure_blocks(np.asarray(sample.y_uv_raw, dtype=float), discounts, cfg)
    planner_blocks, posture_meta = apply_posture_state_to_blocks(raw_blocks, plant_info, cfg)
    planner_info = dict(plant_info)
    planner_info.update(posture_meta)

    evaluated_all = [evaluate_sequence(seq, planner_blocks, planner_info, cfg) for seq in _ALL_SEQUENCES]
    feasible, scalars = select_best_economic(evaluated_all, cfg)
    selected = feasible[0] if feasible else None
    selected_sequence = tuple(selected["sequence"]) if selected else ("hold", "hold", "hold")

    bucket = int(_safe_float(log_row.get("bucket"), -1))
    wanted = list(REPRESENTATIVE_SEQUENCES)
    logged_seq = tuple(str(log_row.get("best_sequence", "hold>hold>hold")).split(">"))
    for seq in (logged_seq, selected_sequence):
        if seq and seq not in wanted:
            wanted.append(seq)

    rows: list[dict[str, Any]] = []
    for seq in wanted:
        result = evaluate_sequence(seq, planner_blocks, planner_info, cfg)
        label = "representative"
        if seq == ("hold", "hold", "hold"):
            label = "hold"
        elif seq == ("active_small", "hold", "hold"):
            label = "active_small"
        elif seq == ("active_medium", "hold", "hold"):
            label = "active_medium"
        elif seq == logged_seq:
            label = "logged_best"
        elif seq == selected_sequence:
            label = "recomputed_best"
        rows.append(_sequence_row(case_id, bucket, seq, result, cfg, label))

    seq_df = pd.DataFrame(rows)
    hold_rows = seq_df[seq_df["sequence"].eq("hold>hold>hold")]
    small_rows = seq_df[seq_df["sequence"].eq("active_small>hold>hold")]
    medium_rows = seq_df[seq_df["sequence"].eq("active_medium>hold>hold")]
    if hold_rows.empty:
        return rows, None
    hold = hold_rows.iloc[0]
    small = small_rows.iloc[0] if not small_rows.empty else pd.Series(dtype=object)
    medium = medium_rows.iloc[0] if not medium_rows.empty else pd.Series(dtype=object)

    best_idx = seq_df["scalar_cost"].replace([np.inf, -np.inf], np.nan).idxmin()
    best_row = seq_df.loc[best_idx] if pd.notna(best_idx) else hold
    pitch = _safe_float(log_row.get("current_pitch_deg"), 0.0)
    roll = _safe_float(log_row.get("current_roll_deg"), 0.0)
    bucket_summary = {
        "case_id": case_id,
        "bucket": bucket,
        "history_end": history_end,
        "logged_first_action": str(log_row.get("first_action", "")),
        "logged_best_sequence": str(log_row.get("best_sequence", "")),
        "planner_action_pitch_deg": _safe_float(log_row.get("planner_action_pitch_deg"), np.nan),
        "planner_action_roll_deg": _safe_float(log_row.get("planner_action_roll_deg"), np.nan),
        "target_pitch_deg": _safe_float(log_row.get("target_pitch_deg"), np.nan),
        "target_roll_deg": _safe_float(log_row.get("target_roll_deg"), np.nan),
        "prediction_primary_delta_abs_mean_kg": _safe_float(
            log_row.get("prediction_primary_delta_abs_mean_kg"), np.nan
        ),
        "target_err_mean_kg": _safe_float(
            log_row.get("prediction_primary_stall_target_err_mean_kg"),
            _safe_float(log_row.get("no_unexplained_hold_target_err_mean_kg"), np.nan),
        ),
        "pump_rate_mean_m3_min": _safe_float(
            log_row.get("pump_rate_mean_m3_min"),
            _safe_float(log_row.get("prediction_primary_stall_pump_rate_m3_min"), np.nan),
        ),
        "fallback_ratio": _safe_float(log_row.get("fallback_ratio"), np.nan),
        "recomputed_best_sequence": str(best_row.get("sequence", "")),
        "recomputed_best_first_action": str(best_row.get("first_action", "")),
        "logged_best_scalar_cost": _safe_float(log_row.get("best_scalar_cost"), np.nan),
        "recomputed_best_scalar_cost": _safe_float(best_row.get("scalar_cost"), np.nan),
        "current_pitch_deg": pitch,
        "current_roll_deg": roll,
        "attitude_abs_deg": max(abs(pitch), abs(roll)),
        "pressure_block0_norm": _safe_float(log_row.get("pressure_block0_norm"), np.nan),
        "pressure_block1_norm": _safe_float(log_row.get("pressure_block1_norm"), np.nan),
        "pressure_block2_norm": _safe_float(log_row.get("pressure_block2_norm"), np.nan),
        "posture_state_norm": _safe_float(hold.get("posture_state_norm"), 0.0),
        "hold_scalar_cost": _safe_float(hold.get("scalar_cost"), np.inf),
        "active_small_scalar_cost": _safe_float(small.get("scalar_cost"), np.inf),
        "active_medium_scalar_cost": _safe_float(medium.get("scalar_cost"), np.inf),
        "small_minus_hold_scalar": _safe_float(small.get("scalar_cost"), np.inf) - _safe_float(hold.get("scalar_cost"), np.inf),
        "medium_minus_hold_scalar": _safe_float(medium.get("scalar_cost"), np.inf) - _safe_float(hold.get("scalar_cost"), np.inf),
        "small_hard_reject_reason": str(small.get("hard_reject_reason", "")),
        "medium_hard_reject_reason": str(medium.get("hard_reject_reason", "")),
        "dominant_small_blocker": _dominant_hold_reason(hold, small) if not small.empty else "active_small_missing",
        "dominant_medium_blocker": _dominant_hold_reason(hold, medium) if not medium.empty else "active_medium_missing",
    }
    for label, _, _ in WEIGHTED_TERMS:
        key = f"weighted_{label}"
        bucket_summary[f"hold_weighted_{label}"] = _safe_float(hold.get(key), 0.0)
        bucket_summary[f"small_delta_weighted_{label}"] = (
            _safe_float(small.get(key), 0.0) - _safe_float(hold.get(key), 0.0)
            if not small.empty
            else np.nan
        )
        bucket_summary[f"medium_delta_weighted_{label}"] = (
            _safe_float(medium.get(key), 0.0) - _safe_float(hold.get(key), 0.0)
            if not medium.empty
            else np.nan
        )
    return rows, bucket_summary


def _timeseries_bucket_context(ts_df: pd.DataFrame) -> pd.DataFrame:
    if ts_df.empty or "t_s" not in ts_df.columns:
        return pd.DataFrame(columns=["bucket", "fallback_ratio", "pump_rate_mean_m3_min"])
    work = ts_df.copy()
    work["bucket"] = np.floor(_num(work["t_s"], 0.0) / 600.0).astype(int)
    fallback = pd.Series(0.0, index=work.index, dtype=float)
    for col in ("preview_primary_safety_fallback", "pump_fullspeed_any", "preview_primary_safety_active"):
        if col in work.columns:
            fallback = _num(work[col], 0.0)
            break
    pump = _num(work.get("pump_total_rate_m3_min", 0.0), 0.0)
    return (
        pd.DataFrame({"bucket": work["bucket"], "fallback": fallback, "pump_rate": pump})
        .groupby("bucket", as_index=False)
        .agg(fallback_ratio=("fallback", "mean"), pump_rate_mean_m3_min=("pump_rate", "mean"))
    )


def _select_audit_rows(
    log: pd.DataFrame,
    ts_df: pd.DataFrame,
    threshold: float,
    bucket_mode: str,
    bucket_ids: set[int],
) -> pd.DataFrame:
    pitch = _num(log.get("current_pitch_deg", 0.0), 0.0).abs()
    roll = _num(log.get("current_roll_deg", 0.0), 0.0).abs()
    action = log.get("first_action", "").astype(str).str.lower()
    high = (pitch > threshold) | (roll > threshold)
    hold = action.isin(["hold", "pump_saving", "pause"]) | action.str.startswith("hold")
    active = action.str.startswith("active")

    work = log.copy()
    if "bucket" in work.columns:
        work["bucket"] = _num(work["bucket"], 0.0).astype(int)
        bucket_ctx = _timeseries_bucket_context(ts_df)
        if not bucket_ctx.empty:
            work = work.merge(bucket_ctx, on="bucket", how="left")
    if "fallback_ratio" not in work.columns:
        fallback_proxy = _num(work.get("recovery_mode_fullspeed_proxy", 0.0), 0.0)
        work["fallback_ratio"] = fallback_proxy
    if "pump_rate_mean_m3_min" not in work.columns:
        work["pump_rate_mean_m3_min"] = np.nan

    fallback = _num(work.get("fallback_ratio", 0.0), 0.0) > 0.01
    if bucket_mode == "high_hold":
        mask = high & hold & ~fallback
    elif bucket_mode == "high_active":
        mask = high & active
    elif bucket_mode == "active_but_weak":
        attitude = np.maximum(pitch, roll)
        next1 = pd.Series(attitude).shift(-1)
        next2 = pd.Series(attitude).shift(-2)
        next_min = pd.concat([next1, next2], axis=1).min(axis=1)
        improvement = attitude - next_min
        mask = high & active & (next1 > threshold) & (next2 > threshold) & (improvement < 0.3)
    elif bucket_mode == "fallback":
        mask = fallback
    elif bucket_mode == "all_high":
        mask = high
    else:
        mask = pd.Series(True, index=work.index)

    if bucket_ids:
        mask = mask & work["bucket"].isin(bucket_ids)

    out = work[mask].copy()
    out["_attitude_abs_deg"] = np.maximum(pitch.loc[out.index], roll.loc[out.index])
    if bucket_mode in {"fallback", "high_active", "active_but_weak"}:
        return out.sort_values(
            ["fallback_ratio", "_attitude_abs_deg", "bucket"],
            ascending=[False, False, True],
        )
    return out.sort_values(["_attitude_abs_deg", "bucket"], ascending=[False, True])


def _write_summary(bucket_df: pd.DataFrame, seq_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Planner Cost Breakdown Audit",
        "",
        "This is a read-only offline audit. It reconstructs oracle forecast blocks for selected planner buckets and compares representative planner sequences.",
        "",
        "## Case Summary",
        "",
        "| case | audited buckets | hold wins | active_small would win | active_medium would win | top small blocker |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for case_id, group in bucket_df.groupby("case_id", sort=True):
        hold_wins = int((group["recomputed_best_first_action"].astype(str).str.lower().isin(["hold", "pump_saving"])).sum())
        small_wins = int(group["recomputed_best_first_action"].astype(str).eq("active_small").sum())
        medium_wins = int(group["recomputed_best_first_action"].astype(str).eq("active_medium").sum())
        blocker = ""
        if "dominant_small_blocker" in group:
            modes = group["dominant_small_blocker"].astype(str).value_counts()
            blocker = modes.index[0] if not modes.empty else ""
        lines.append(
            f"| {case_id} | {len(group)} | {hold_wins} | {small_wins} | {medium_wins} | {blocker} |"
        )

    lines.extend(["", "## Interpretation Pointers", ""])
    if not bucket_df.empty:
        hold_like = bucket_df[
            bucket_df["recomputed_best_first_action"].astype(str).str.lower().isin(["hold", "pump_saving"])
        ]
        active_like = bucket_df[
            ~bucket_df["recomputed_best_first_action"].astype(str).str.lower().isin(["hold", "pump_saving"])
        ]
        lines.append(f"- Recomputed planner still prefers hold-like actions in `{len(hold_like)}/{len(bucket_df)}` audited buckets.")
        lines.append(f"- Recomputed planner would prefer active actions in `{len(active_like)}/{len(bucket_df)}` audited buckets; inspect these for log/config mismatch or target-lifecycle overlays.")
        if "dominant_small_blocker" in bucket_df:
            blockers = bucket_df["dominant_small_blocker"].astype(str).value_counts().head(5)
            lines.append("- Most common active_small blockers:")
            for name, count in blockers.items():
                lines.append(f"  - `{name}`: {int(count)}")
    lines.extend(
        [
            "",
            "Key columns:",
            "- `small_minus_hold_scalar > 0`: active_small is more expensive than hold under the planner objective.",
            "- `small_delta_weighted_*`: positive means active_small pays more than hold on that term; negative means it improves that term.",
            "- `dominant_small_blocker`: largest positive weighted-cost term blocking active_small, or `active_small_would_win_or_tie`.",
            "",
            f"Bucket table: `{path.parent / 'planner_cost_breakdown_bucket_table.csv'}`",
            f"Sequence table: `{path.parent / 'planner_cost_breakdown_sequence_table.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    casebook_dir = Path(args.casebook_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = {
        item.strip()
        for item in str(args.case_ids).split(",")
        if item.strip()
    }
    selected_buckets = {
        int(item.strip())
        for item in str(args.bucket_ids).split(",")
        if item.strip()
    }
    cfg, discounts = _cfg_from_casebook(casebook_dir)
    replay = Fino1ReplayDataset(dataset_dir=Path(args.dataset_dir), split=str(args.split))
    cases = _discover(casebook_dir)

    seq_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    for norm, paths in sorted(cases.items(), key=lambda item: _case_sort_key(item[0])):
        case_id = str(paths.get("case_id", norm))
        if not _selected(case_id, selected_ids):
            continue
        log_path = paths.get("log")
        ts_path = paths.get("timeseries")
        if not log_path or not ts_path:
            continue
        log = pd.read_csv(log_path, low_memory=False)
        ts_df = pd.read_csv(ts_path, low_memory=False)
        candidates = _select_audit_rows(
            log,
            ts_df,
            float(args.attention_deg),
            str(args.bucket_mode),
            selected_buckets,
        ).head(
            max(1, int(args.max_buckets_per_case))
        )
        for _, log_row in candidates.sort_values("bucket").iterrows():
            ts_row = _nearest_timeseries_row(ts_df, _safe_float(log_row.get("current_time_s"), 0.0))
            rows, summary = _audit_one_bucket(
                replay,
                cfg,
                discounts,
                case_id,
                log_row,
                ts_row,
            )
            seq_rows.extend(rows)
            if summary is not None:
                bucket_rows.append(summary)

    if not bucket_rows:
        raise SystemExit("no high-posture hold buckets audited")

    bucket_df = pd.DataFrame(bucket_rows)
    seq_df = pd.DataFrame(seq_rows)
    bucket_path = out_dir / "planner_cost_breakdown_bucket_table.csv"
    seq_path = out_dir / "planner_cost_breakdown_sequence_table.csv"
    bucket_df.to_csv(bucket_path, index=False)
    seq_df.to_csv(seq_path, index=False)
    summary_path = out_dir / "planner_cost_breakdown_summary.md"
    _write_summary(bucket_df, seq_df, summary_path)

    display = bucket_df[
        [
            "case_id",
            "bucket",
            "current_pitch_deg",
            "current_roll_deg",
            "recomputed_best_sequence",
            "small_minus_hold_scalar",
            "medium_minus_hold_scalar",
            "dominant_small_blocker",
        ]
    ]
    print(display.to_string(index=False))
    print(summary_path)


if __name__ == "__main__":
    main()
