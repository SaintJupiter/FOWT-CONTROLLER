#!/usr/bin/env python3
"""h120 regression-risk root-cause audit v1.

This is an offline diagnostic on top of h120_action_effect_dataset_expansion_v1.
It does not run new casebooks, does not attach a controller gate, and does not
modify v1.6 floor logic.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "outputs/wind_prediction/h120_action_effect_dataset_expansion_v1"
OUT = REPO / "outputs/wind_prediction/h120_regression_risk_rootcause_v1"
RAW = OUT / "raw_tables"
DEBUG = OUT / "debug"
PAPER = OUT / "paper_ready"

EXPANSION_SCRIPT = REPO / "scripts/analysis/run_h120_action_effect_dataset_expansion_v1.py"
spec = importlib.util.spec_from_file_location("h120_action_effect_dataset_expansion_v1", EXPANSION_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import expansion helpers from {EXPANSION_SCRIPT}")
expansion = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = expansion
spec.loader.exec_module(expansion)

WAIT_ACTION = "wait"
MICRO50_ACTION = "micro_prepare_50"
PUMP_ACTION = "pump_saving_prepare"
ACTIVE_ACTION = "active_small_prepare"
FORCED_ACTIONS = [MICRO50_ACTION, PUMP_ACTION, ACTIVE_ACTION]


def ensure_dirs() -> None:
    for path in (OUT, RAW, DEBUG, PAPER):
        path.mkdir(parents=True, exist_ok=True)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_empty_"
    d = df.copy()
    if max_rows is not None:
        d = d.head(max_rows)

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = [
        "| " + " | ".join(str(c) for c in d.columns) + " |",
        "| " + " | ".join(["---"] * len(d.columns)) + " |",
    ]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in d.columns) + " |")
    return "\n".join(lines)


def num_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def token_has(value: Any, pattern: str) -> bool:
    return pattern in str(value or "")


def monotone(values: list[float], *, nondecreasing: bool = True, tol: float = 1e-6) -> bool:
    if len(values) < 2:
        return True
    if nondecreasing:
        return all(b >= a - tol for a, b in zip(values, values[1:]))
    return all(b <= a + tol for a, b in zip(values, values[1:]))


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cf = pd.read_csv(SOURCE / "expanded_counterfactual_table.csv")
    action = pd.read_csv(SOURCE / "raw_tables/expanded_action_labeled_table.csv")
    regime = pd.read_csv(SOURCE / "expanded_regime_table.csv")
    candidates = pd.read_csv(SOURCE / "expanded_candidate_table.csv")
    for frame in (cf, action, regime, candidates):
        frame["dataset"] = frame["dataset"].astype(str)
        frame["case_id"] = frame["case_id"].astype(str)
        frame["bucket"] = pd.to_numeric(frame["bucket"], errors="coerce").fillna(-1).astype(int)
        if "group_id" not in frame:
            frame["group_id"] = frame["dataset"] + "::" + frame["case_id"]
        if "bucket_key" not in frame:
            frame["bucket_key"] = frame["dataset"] + "::" + frame["case_id"] + "::" + frame["bucket"].astype(str)
    action = add_forced_vectors(action, candidates)
    return cf, action, regime, candidates


def add_forced_vectors(action: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    by_probe = candidates.set_index("probe_id")
    rows = []
    for _, row in action.iterrows():
        action_name = str(row["action"])
        if action_name not in expansion.ACTION_DEFS or action_name == WAIT_ACTION:
            rows.append((np.nan, np.nan, np.nan, ""))
            continue
        probe_id = str(row["probe_id"])
        if probe_id not in by_probe.index:
            rows.append((np.nan, np.nan, np.nan, "missing_candidate"))
            continue
        cand = by_probe.loc[probe_id]
        pitch, roll, norm, provider_action = expansion.scaled_vec(cand, expansion.ACTION_DEFS[action_name])
        rows.append((pitch, roll, norm, provider_action))
    vec = pd.DataFrame(rows, columns=["forced_pitch_deg", "forced_roll_deg", "forced_vec_norm", "provider_action"])
    return pd.concat([action.reset_index(drop=True), vec], axis=1)


def action_has_safety_gain(row: pd.Series) -> bool:
    return (
        as_float(row.get("time_gain_s")) >= 30.0
        or as_float(row.get("idle_gain_s")) >= 60.0
        or as_float(row.get("delta_post60_floor_entry")) < 0.0
        or as_float(row.get("delta_post60_medium_delay_rows")) < 0.0
    )


def action_no_meaningful_gain(row: pd.Series) -> bool:
    return (
        as_float(row.get("time_gain_s")) < 10.0
        and as_float(row.get("idle_gain_s")) < 20.0
        and as_float(row.get("delta_post60_floor_entry")) >= 0.0
        and as_float(row.get("delta_post60_medium_delay_rows")) >= 0.0
    )


def action_regresses(row: pd.Series) -> bool:
    return (
        as_float(row.get("delta_post60_time_over5")) > 10.0
        or as_float(row.get("delta_post60_idle_over5")) > 20.0
        or as_float(row.get("delta_full_fallback_pp")) > 0.05
        or as_float(row.get("delta_full_max_p95")) > 0.05
    )


def bool_any(group: pd.DataFrame, col: str) -> bool:
    return bool((num_series(group, col) > 0).any()) if col in group else False


def best_action(group: pd.DataFrame, mask: pd.Series | None = None) -> str:
    sub = group.copy()
    if mask is not None:
        sub = sub[mask]
    if sub.empty:
        return ""
    return str(sub.sort_values("action_utility", ascending=False).iloc[0]["action"])


def build_regression_subtypes(action: pd.DataFrame, regime: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    by_probe_candidate = candidates.set_index("probe_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    forced = action[action["action"].isin(FORCED_ACTIONS)].copy()
    wait = pd.read_csv(SOURCE / "expanded_counterfactual_table.csv")
    wait = wait[wait["action"].eq(WAIT_ACTION)].set_index("probe_id")
    for _, r in regime.iterrows():
        probe_id = str(r["probe_id"])
        group = forced[forced["probe_id"].astype(str).eq(probe_id)].copy()
        cand = by_probe_candidate.get(probe_id, {})
        wait_row = wait.loc[probe_id] if probe_id in wait.index else pd.Series(dtype=object)
        if group.empty:
            continue
        regression_actions = group[
            (num_series(group, "action_regression_risk") > 0) | group.apply(action_regresses, axis=1)
        ]
        negative_actions = group[
            (num_series(group, "action_negative_effect") > 0)
            | group["action_label"].astype(str).eq("negative")
        ]
        any_negative_no_gain = bool(negative_actions.apply(action_no_meaningful_gain, axis=1).any()) if not negative_actions.empty else False
        any_pump_no_gain = bool(
            ((num_series(group, "pump_delta_m3") > 5.0) & group.apply(action_no_meaningful_gain, axis=1)).any()
        )
        double_pump_no_avoid = bool(
            (
                (num_series(group, "pump_delta_m3") > 5.0)
                & (num_series(group, "post60_floor_entry") > 0.0)
                & (num_series(group, "delta_post60_floor_entry") >= 0.0)
            ).any()
        )
        floor_override = bool(
            (
                (num_series(group, "forced_prefix_active_rows") > 0.0)
                & ((num_series(group, "post60_floor_entry") > 0.0) | (num_series(group, "post60_medium_delay_rows") > 0.0))
                & group.apply(action_no_meaningful_gain, axis=1)
            ).any()
        )
        low_posture = as_float(cand.get("max_axis_deg", r.get("max_axis_deg"))) < 3.5
        low_future_exposure = (
            as_float(cand.get("future_60m_time_over5", wait_row.get("post60_time_over5", 999.0))) <= 5.0
            and as_float(cand.get("future_60m_idle_over5", wait_row.get("post60_idle_over5", 999.0))) <= 20.0
        )
        far_persistent_alone = (
            as_int(r.get("far_persistent_high")) > 0
            and as_int(r.get("delayed_intensification")) == 0
            and as_int(r.get("reintensification_after_relief")) == 0
        )
        pump_delta_values = list(num_series(group.sort_values("target_refresh_scale"), "pump_delta_m3"))
        time_gain_values = list(num_series(group.sort_values("target_refresh_scale"), "time_gain_s"))
        idle_gain_values = list(num_series(group.sort_values("target_refresh_scale"), "idle_gain_s"))
        utility_values = list(num_series(group.sort_values("target_refresh_scale"), "action_utility"))
        pump_non_mono = not monotone(pump_delta_values, nondecreasing=True, tol=1e-6)
        time_non_mono = not monotone(time_gain_values, nondecreasing=True, tol=0.5)
        idle_non_mono = not monotone(idle_gain_values, nondecreasing=True, tol=0.5)
        utility_non_mono = not monotone(utility_values, nondecreasing=True, tol=0.5)
        response_anomaly = bool(
            (num_series(group, "pump_delta_m3") < -5.0).any()
            or (
                as_float(group.loc[group["action"].eq(ACTIVE_ACTION), "pump_delta_m3"].iloc[0], np.nan)
                < as_float(group.loc[group["action"].eq(MICRO50_ACTION), "pump_delta_m3"].iloc[0], np.nan) - 10.0
                if not group[group["action"].eq(ACTIVE_ACTION)].empty and not group[group["action"].eq(MICRO50_ACTION)].empty
                else False
            )
            or (
                as_float(r.get("target_error_mean_kg")) < 500.0
                and as_int(r.get("pump_idle")) > 0
                and any_pump_no_gain
            )
        )
        direction_mismatch_context = bool(
            as_int(r.get("direction_mismatch")) > 0
            or as_int(r.get("signflip_or_reversal")) > 0
            or token_has(r.get("candidate_reason"), "direction_mismatch")
        )
        labels = {
            "axis_tradeoff_subtype": bool(as_int(r.get("axis_tradeoff")) > 0 or bool_any(group, "axis_tradeoff")),
            "prepare_direction_mismatch_subtype": direction_mismatch_context and bool(as_int(r.get("negative_effect")) > 0 or not regression_actions.empty),
            "double_pump_no_avoid_subtype": double_pump_no_avoid,
            "floor_or_early_stop_override_subtype": floor_override,
            "low_posture_false_prepare_subtype": bool((low_posture or low_future_exposure) and (any_negative_no_gain or bool(as_int(r.get("anti_trigger"))))),
            "far_persistent_high_alone_false_trigger_subtype": bool(far_persistent_alone and (any_negative_no_gain or bool(as_int(r.get("anti_trigger"))))),
            "non_monotone_target_lifecycle_subtype": bool(as_int(r.get("non_monotone")) > 0 or pump_non_mono or time_non_mono or idle_non_mono),
            "tank_allocation_target_error_anomaly_subtype": response_anomaly,
        }
        priority = [
            "axis_tradeoff_subtype",
            "prepare_direction_mismatch_subtype",
            "double_pump_no_avoid_subtype",
            "floor_or_early_stop_override_subtype",
            "low_posture_false_prepare_subtype",
            "far_persistent_high_alone_false_trigger_subtype",
            "non_monotone_target_lifecycle_subtype",
            "tank_allocation_target_error_anomaly_subtype",
        ]
        subtype_list = [name for name in priority if labels[name]]
        dominant = subtype_list[0] if subtype_list else ("other_regression_risk" if as_int(r.get("regression_risk")) else "not_regression_risk")
        row = {
            "probe_id": probe_id,
            "bucket_key": r["bucket_key"],
            "dataset": r["dataset"],
            "case_id": r["case_id"],
            "group_id": r["group_id"],
            "bucket": int(r["bucket"]),
            "timestamp": r["timestamp"],
            "candidate_reason": r.get("candidate_reason", ""),
            "regression_risk": as_int(r.get("regression_risk")),
            "negative_effect": as_int(r.get("negative_effect")),
            "positive_effect": as_int(r.get("positive_effect")),
            "anti_trigger": as_int(r.get("anti_trigger")),
            "axis_tradeoff": as_int(r.get("axis_tradeoff")),
            "non_monotone": as_int(r.get("non_monotone")),
            "no_effect": as_int(r.get("no_effect")),
            "effectful": as_int(r.get("effectful")),
            "dominant_subtype": dominant,
            "subtype_count": len(subtype_list),
            "subtype_list": ";".join(subtype_list),
            "regression_actions": ";".join(regression_actions["action"].astype(str).tolist()),
            "negative_actions": ";".join(negative_actions["action"].astype(str).tolist()),
            "best_action_by_utility": r.get("best_action_by_utility", ""),
            "micro50_action_label": group.loc[group["action"].eq(MICRO50_ACTION), "action_label"].iloc[0]
            if not group[group["action"].eq(MICRO50_ACTION)].empty
            else "",
            "micro50_pump_delta_m3": as_float(group.loc[group["action"].eq(MICRO50_ACTION), "pump_delta_m3"].iloc[0], np.nan)
            if not group[group["action"].eq(MICRO50_ACTION)].empty
            else np.nan,
            "micro50_time_gain_s": as_float(group.loc[group["action"].eq(MICRO50_ACTION), "time_gain_s"].iloc[0], np.nan)
            if not group[group["action"].eq(MICRO50_ACTION)].empty
            else np.nan,
            "micro50_idle_gain_s": as_float(group.loc[group["action"].eq(MICRO50_ACTION), "idle_gain_s"].iloc[0], np.nan)
            if not group[group["action"].eq(MICRO50_ACTION)].empty
            else np.nan,
            "wait_post60_floor_entry": as_float(wait_row.get("post60_floor_entry"), np.nan),
            "wait_post60_time_over5": as_float(wait_row.get("post60_time_over5"), np.nan),
            "wait_post60_idle_over5": as_float(wait_row.get("post60_idle_over5"), np.nan),
            "max_axis_deg": as_float(r.get("max_axis_deg")),
            "posture_trend_10m_deg": as_float(r.get("posture_trend_10m_deg")),
            "target_age_s": as_float(r.get("target_age_s")),
            "target_error_mean_kg": as_float(r.get("target_error_mean_kg")),
            "pump_idle": as_int(r.get("pump_idle")),
            "near_max_norm": as_float(r.get("near_max_norm")),
            "far_max_norm": as_float(r.get("far_max_norm")),
            "far_persistent_high": as_int(r.get("far_persistent_high")),
            "delayed_intensification": as_int(r.get("delayed_intensification")),
            "reintensification_after_relief": as_int(r.get("reintensification_after_relief")),
            "direction_consistent": as_int(r.get("direction_consistent")),
            "direction_mismatch": as_int(r.get("direction_mismatch")),
            "signflip_or_reversal": as_int(r.get("signflip_or_reversal")),
            "near_safe_far_risky": as_int(r.get("near_safe_far_risky")),
            "near_risky_far_relief": as_int(r.get("near_risky_far_relief")),
            "pump_delta_non_monotone": int(pump_non_mono),
            "time_gain_non_monotone": int(time_non_mono),
            "idle_gain_non_monotone": int(idle_non_mono),
            "utility_non_monotone": int(utility_non_mono),
        }
        row.update({name: int(value) for name, value in labels.items()})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "regression_risk_subtype_table.csv", index=False)
    out.to_csv(RAW / "regression_risk_subtype_table.csv", index=False)
    return out


def build_broader20_false_prepare(action: pd.DataFrame, subtype: pd.DataFrame) -> pd.DataFrame:
    sub_cols = [
        "probe_id",
        "dominant_subtype",
        "subtype_list",
        "axis_tradeoff_subtype",
        "prepare_direction_mismatch_subtype",
        "double_pump_no_avoid_subtype",
        "floor_or_early_stop_override_subtype",
        "low_posture_false_prepare_subtype",
        "far_persistent_high_alone_false_trigger_subtype",
        "non_monotone_target_lifecycle_subtype",
        "tank_allocation_target_error_anomaly_subtype",
    ]
    micro = action[
        action["dataset"].astype(str).eq("broader20")
        & action["action"].eq(MICRO50_ACTION)
        & action["action_label"].astype(str).eq("negative")
    ].copy()
    micro["just_extra_pump_no_gain"] = (
        (num_series(micro, "pump_delta_m3") > 5.0)
        & micro.apply(action_no_meaningful_gain, axis=1)
    ).astype(int)
    micro["low_current_posture"] = (num_series(micro, "max_axis_deg") < 3.5).astype(int)
    micro["low_future_exposure"] = (
        (num_series(micro, "post60_time_over5") <= 5.0)
        & (num_series(micro, "post60_idle_over5") <= 20.0)
    ).astype(int)
    micro["anti_trigger_feature_count"] = (
        micro["low_current_posture"]
        + micro["low_future_exposure"]
        + (num_series(micro, "direction_mismatch") > 0).astype(int)
        + (num_series(micro, "near_safe_far_risky") > 0).astype(int)
        + (
            (num_series(micro, "far_persistent_high") > 0)
            & (num_series(micro, "delayed_intensification") == 0)
            & (num_series(micro, "reintensification_after_relief") == 0)
        ).astype(int)
        + (num_series(micro, "signflip_or_reversal") > 0).astype(int)
    )
    micro["suggested_anti_trigger"] = (micro["anti_trigger_feature_count"] > 0).astype(int)
    out = micro.merge(subtype[[c for c in sub_cols if c in subtype.columns]], on="probe_id", how="left")
    cols = [
        "probe_id",
        "dataset",
        "case_id",
        "bucket",
        "timestamp",
        "action",
        "action_label",
        "pump_delta_m3",
        "time_gain_s",
        "idle_gain_s",
        "delta_full_fallback_pp",
        "delta_full_max_p95",
        "post60_floor_entry",
        "post60_time_over5",
        "post60_idle_over5",
        "max_axis_deg",
        "posture_trend_10m_deg",
        "target_age_s",
        "target_error_mean_kg",
        "pump_idle",
        "near_max_norm",
        "far_max_norm",
        "far_persistent_high",
        "delayed_intensification",
        "reintensification_after_relief",
        "direction_consistent",
        "direction_mismatch",
        "signflip_or_reversal",
        "near_safe_far_risky",
        "near_risky_far_relief",
        "just_extra_pump_no_gain",
        "low_current_posture",
        "low_future_exposure",
        "anti_trigger_feature_count",
        "suggested_anti_trigger",
        "dominant_subtype",
        "subtype_list",
        "candidate_reason",
    ]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(OUT / "broader20_false_prepare_table.csv", index=False)
    out.to_csv(RAW / "broader20_false_prepare_table.csv", index=False)
    return out


def build_axis_tradeoff(action: pd.DataFrame) -> pd.DataFrame:
    forced = action[action["action"].isin(FORCED_ACTIONS)].copy()
    forced["axis_tradeoff_type"] = np.select(
        [
            (num_series(forced, "delta_pitch_p95") <= -0.05) & (num_series(forced, "delta_roll_p95") >= 0.05),
            (num_series(forced, "delta_roll_p95") <= -0.05) & (num_series(forced, "delta_pitch_p95") >= 0.05),
        ],
        ["pitch_improves_roll_worsens", "roll_improves_pitch_worsens"],
        default="none",
    )
    out = forced[forced["axis_tradeoff_type"].ne("none")].copy()
    cols = [
        "probe_id",
        "dataset",
        "case_id",
        "bucket",
        "timestamp",
        "action",
        "axis_tradeoff_type",
        "delta_pitch_p95",
        "delta_roll_p95",
        "delta_full_max_p95",
        "pump_delta_m3",
        "time_gain_s",
        "idle_gain_s",
        "forced_pitch_deg",
        "forced_roll_deg",
        "forced_vec_norm",
        "current_pitch_deg",
        "current_roll_deg",
        "near_max_norm",
        "far_max_norm",
        "direction_consistent",
        "direction_mismatch",
        "signflip_or_reversal",
        "near_safe_far_risky",
        "action_label",
        "action_regression_risk",
    ]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(OUT / "axis_tradeoff_table.csv", index=False)
    out.to_csv(RAW / "axis_tradeoff_table.csv", index=False)
    return out


def build_non_monotone(action: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    forced = action[action["action"].isin(FORCED_ACTIONS)].copy()
    for probe_id, group in forced.groupby("probe_id", dropna=False):
        r = regime[regime["probe_id"].astype(str).eq(str(probe_id))]
        if r.empty:
            continue
        reg = r.iloc[0]
        g = group.sort_values("target_refresh_scale").copy()
        pump_vals = list(num_series(g, "pump_delta_m3"))
        time_vals = list(num_series(g, "time_gain_s"))
        idle_vals = list(num_series(g, "idle_gain_s"))
        utility_vals = list(num_series(g, "action_utility"))
        pump_non = not monotone(pump_vals, nondecreasing=True)
        time_non = not monotone(time_vals, nondecreasing=True, tol=0.5)
        idle_non = not monotone(idle_vals, nondecreasing=True, tol=0.5)
        utility_non = not monotone(utility_vals, nondecreasing=True, tol=0.5)
        response_non_monotone = bool(
            as_int(reg.get("non_monotone")) > 0 or pump_non or time_non or idle_non
        )
        row = {
            "probe_id": probe_id,
            "bucket_key": reg["bucket_key"],
            "dataset": reg["dataset"],
            "case_id": reg["case_id"],
            "bucket": int(reg["bucket"]),
            "timestamp": reg["timestamp"],
            "non_monotone": int(response_non_monotone),
            "pump_delta_non_monotone": int(pump_non),
            "time_gain_non_monotone": int(time_non),
            "idle_gain_non_monotone": int(idle_non),
            "utility_non_monotone": int(utility_non),
            "pump_delta_sequence": ";".join(f"{x:.3f}" for x in pump_vals),
            "time_gain_sequence": ";".join(f"{x:.3f}" for x in time_vals),
            "idle_gain_sequence": ";".join(f"{x:.3f}" for x in idle_vals),
            "utility_sequence": ";".join(f"{x:.3f}" for x in utility_vals),
            "has_floor_takeover": int((num_series(g, "post60_floor_entry") > 0).any()),
            "has_medium_delay_after_prepare": int((num_series(g, "post60_medium_delay_rows") > 0).any()),
            "has_regression_risk": int((num_series(g, "action_regression_risk") > 0).any()),
            "has_axis_tradeoff": int((num_series(g, "axis_tradeoff") > 0).any()),
            "forced_prefix_active_rows_max": float(num_series(g, "forced_prefix_active_rows").max()),
            "range_pump_delta_m3": float(max(pump_vals) - min(pump_vals)) if pump_vals else np.nan,
            "range_time_gain_s": float(max(time_vals) - min(time_vals)) if time_vals else np.nan,
            "range_idle_gain_s": float(max(idle_vals) - min(idle_vals)) if idle_vals else np.nan,
            "max_axis_deg": as_float(reg.get("max_axis_deg")),
            "near_max_norm": as_float(reg.get("near_max_norm")),
            "far_max_norm": as_float(reg.get("far_max_norm")),
            "direction_mismatch": as_int(reg.get("direction_mismatch")),
            "near_safe_far_risky": as_int(reg.get("near_safe_far_risky")),
            "far_persistent_high": as_int(reg.get("far_persistent_high")),
            "candidate_reason": reg.get("candidate_reason", ""),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out[out["non_monotone"] > 0].copy()
    out.to_csv(OUT / "non_monotone_table.csv", index=False)
    out.to_csv(RAW / "non_monotone_table.csv", index=False)
    return out


def proposed_labels(subtype: pd.DataFrame, broader: pd.DataFrame, axis: pd.DataFrame, nonmono: pd.DataFrame) -> pd.DataFrame:
    labels = [
        {
            "label": "regression_risk_subtype_label",
            "definition": "Multi-label bucket subtype for any prepare action that worsens time/idle/fallback/max_p95.",
            "source_columns": "action_regression_risk, delta_post60_time_over5, delta_post60_idle_over5, delta_full_fallback_pp, delta_full_max_p95",
            "positive_count": int((subtype["regression_risk"] > 0).sum()),
            "intended_use": "diagnostic supervision before expected-delta heads",
        },
        {
            "label": "broader20_false_prepare_label",
            "definition": "broader20 micro_prepare_50 negative rows with low/no safety gain or anti-trigger context.",
            "source_columns": "dataset, action, action_label, pump_delta_m3, time_gain_s, idle_gain_s, post60_floor_entry",
            "positive_count": int(len(broader)),
            "intended_use": "anti-trigger and lowrisk/broader false-positive guard",
        },
        {
            "label": "axis_tradeoff_risk_label",
            "definition": "Prepare improves one posture axis p95 while worsening the other by at least 0.05 deg.",
            "source_columns": "delta_pitch_p95, delta_roll_p95",
            "positive_count": int(axis["probe_id"].nunique()) if not axis.empty else 0,
            "intended_use": "axis-aware safety head; prevent single-axis improvement from hiding max-axis risk",
        },
        {
            "label": "double_pump_no_avoid_label",
            "definition": "Prepare adds pump but subsequent floor entry is not avoided.",
            "source_columns": "pump_delta_m3, post60_floor_entry, delta_post60_floor_entry",
            "positive_count": int((subtype["double_pump_no_avoid_subtype"] > 0).sum()),
            "intended_use": "guard against pre-floor pump that does not reduce later recovery cost",
        },
        {
            "label": "low_posture_false_prepare_label",
            "definition": "Current posture is low or future60 exposure is low, while prepare is negative/no-gain.",
            "source_columns": "max_axis_deg, wait_post60_time_over5, wait_post60_idle_over5, action_label",
            "positive_count": int((subtype["low_posture_false_prepare_subtype"] > 0).sum()),
            "intended_use": "watch-only mode instead of prepare for remote-only risk",
        },
        {
            "label": "non_monotone_action_response_label",
            "definition": "Pump/safety/utility response is non-monotone across micro50, pump_saving, active_small scales.",
            "source_columns": "pump_delta_m3, time_gain_s, idle_gain_s, action_utility by action scale",
            "positive_count": int(len(nonmono)),
            "intended_use": "screen out buckets where target_refresh_scale is not a reliable continuous action axis",
        },
        {
            "label": "target_lifecycle_response_anomaly_label",
            "definition": "Forced prepare has unexpected pump response, negative pump delta, or target-quiet no-gain behavior.",
            "source_columns": "pump_delta_m3, target_error_mean_kg, pump_idle, forced_prefix_active_rows",
            "positive_count": int((subtype["tank_allocation_target_error_anomaly_subtype"] > 0).sum()),
            "intended_use": "separate model/controller interface failures from forecast-risk failures",
        },
    ]
    out = pd.DataFrame(labels)
    out.to_csv(OUT / "proposed_risk_labels.csv", index=False)
    out.to_csv(RAW / "proposed_risk_labels.csv", index=False)
    return out


def summarize_subtypes(subtype: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "axis_tradeoff_subtype",
        "prepare_direction_mismatch_subtype",
        "double_pump_no_avoid_subtype",
        "floor_or_early_stop_override_subtype",
        "low_posture_false_prepare_subtype",
        "far_persistent_high_alone_false_trigger_subtype",
        "non_monotone_target_lifecycle_subtype",
        "tank_allocation_target_error_anomaly_subtype",
    ]
    reg = subtype[subtype["regression_risk"] > 0].copy()
    rows = []
    for col in cols:
        rows.append(
            {
                "subtype": col,
                "regression_bucket_count": int((reg[col] > 0).sum()) if col in reg else 0,
                "regression_bucket_rate": float((reg[col] > 0).mean()) if col in reg and len(reg) else np.nan,
                "all_bucket_count": int((subtype[col] > 0).sum()) if col in subtype else 0,
                "all_bucket_rate": float((subtype[col] > 0).mean()) if col in subtype else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("regression_bucket_count", ascending=False)
    out.to_csv(RAW / "regression_risk_subtype_summary.csv", index=False)
    return out


def build_next_feature_recommendations(
    subtype_summary: pd.DataFrame,
    broader: pd.DataFrame,
    axis: pd.DataFrame,
    nonmono: pd.DataFrame,
) -> None:
    md = [
        "# Next Model Feature Recommendations",
        "",
        "The current blocker is not effectful detection.  It is separating useful prepare from regression/false-prepare regimes, especially broader20.",
        "",
        "## Add / Export Features",
        "",
        "- Forced prepare vector components: `forced_pitch_deg`, `forced_roll_deg`, vector norm, and action family.",
        "- Dot products between prepare vector and current posture vector, near pressure vector, and far pressure vector.",
        "- Axis-specific risk: pitch and roll p95 deltas, not only max-axis deltas.",
        "- Floor takeover diagnostics: whether floor enters after prepare, time from prepare to floor, medium escalation, and post-exit early_stop activity.",
        "- Target lifecycle state: target age, axis-wise target error, target reuse/staleness, pump idle, pump backlog/saturation.",
        "- Non-monotone response flags by action scale, because target_refresh_scale is not reliably monotone.",
        "",
        "## Labels To Train Before Expected Delta",
        "",
        "- `broader20_false_prepare_label` and `low_posture_false_prepare_label` as hard anti-trigger heads.",
        "- `regression_risk_subtype_label` as a multi-label head, not one monolithic regression bit.",
        "- `axis_tradeoff_risk_label`, because pitch/roll tradeoffs are a real safety failure mode.",
        "- `double_pump_no_avoid_label`, to catch cases where prepare spends pump but v1.6 floor still has to recover.",
        "- `non_monotone_action_response_label`, to detect buckets where the action family cannot be ranked by scale.",
        "",
        "## Interface Implication",
        "",
        "Do not train another expected-delta head on all prepare rows yet.  First build a risk-screening layer that rejects broader false prepare, axis tradeoff, double-pump/no-avoid, and non-monotone response regimes.",
        "",
        "If these labels remain hard to learn, the next change should be action/interface design, not more threshold tuning.  Candidate interface changes: axis-constrained micro prepare, watch-only plus delayed confirmation, or single-axis prepare chosen by pressure/posture alignment.",
        "",
        "## Key Counts",
        "",
        "### Regression Subtypes",
        markdown_table(subtype_summary),
        "",
        f"- broader20 micro_prepare_50 negative rows: {len(broader)}.",
        f"- axis tradeoff action rows: {len(axis)} across {axis['probe_id'].nunique() if not axis.empty else 0} buckets.",
        f"- non-monotone buckets: {len(nonmono)}.",
    ]
    (OUT / "next_model_feature_recommendations.md").write_text("\n".join(md) + "\n")


def write_reports(
    subtype: pd.DataFrame,
    subtype_summary: pd.DataFrame,
    broader: pd.DataFrame,
    axis: pd.DataFrame,
    nonmono: pd.DataFrame,
    labels: pd.DataFrame,
    action: pd.DataFrame,
) -> None:
    reg = subtype[subtype["regression_risk"] > 0].copy()
    dominant = (
        reg.groupby("dominant_subtype")
        .size()
        .reset_index(name="buckets")
        .sort_values("buckets", ascending=False)
    )
    broader_summary = pd.DataFrame(
        [
            {
                "metric": "broader20_micro50_negative_rows",
                "value": len(broader),
            },
            {
                "metric": "suggested_anti_trigger_rate",
                "value": float(broader["suggested_anti_trigger"].mean()) if not broader.empty else np.nan,
            },
            {
                "metric": "just_extra_pump_no_gain_rate",
                "value": float(broader["just_extra_pump_no_gain"].mean()) if not broader.empty else np.nan,
            },
            {
                "metric": "low_future_exposure_rate",
                "value": float(broader["low_future_exposure"].mean()) if not broader.empty else np.nan,
            },
        ]
    )
    axis_summary = (
        axis.groupby(["dataset", "action", "axis_tradeoff_type"])
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
        if not axis.empty
        else pd.DataFrame()
    )
    axis_rate = (
        action[action["action"].isin(FORCED_ACTIONS)]
        .assign(axis_tradeoff_flag=lambda d: (num_series(d, "axis_tradeoff") > 0).astype(int))
        .groupby(["dataset", "action"])
        .agg(rows=("probe_id", "count"), axis_tradeoff_rate=("axis_tradeoff_flag", "mean"))
        .reset_index()
    )
    nonmono_summary = pd.DataFrame(
        [
            {"label": "pump_delta_non_monotone", "count": int(nonmono["pump_delta_non_monotone"].sum()) if not nonmono.empty else 0},
            {"label": "time_gain_non_monotone", "count": int(nonmono["time_gain_non_monotone"].sum()) if not nonmono.empty else 0},
            {"label": "idle_gain_non_monotone", "count": int(nonmono["idle_gain_non_monotone"].sum()) if not nonmono.empty else 0},
            {"label": "utility_non_monotone", "count": int(nonmono["utility_non_monotone"].sum()) if not nonmono.empty else 0},
            {"label": "floor_takeover_in_nonmonotone", "count": int(nonmono["has_floor_takeover"].sum()) if not nonmono.empty else 0},
            {"label": "axis_tradeoff_in_nonmonotone", "count": int(nonmono["has_axis_tradeoff"].sum()) if not nonmono.empty else 0},
        ]
    )
    for name, table in {
        "regression_risk_subtype_summary.csv": subtype_summary,
        "broader20_false_prepare_summary.csv": broader_summary,
        "axis_tradeoff_summary.csv": axis_summary,
        "axis_tradeoff_rate_by_action.csv": axis_rate,
        "non_monotone_summary.csv": nonmono_summary,
    }.items():
        table.to_csv(RAW / name, index=False)

    paper = [
        "# h120 Regression-Risk Root-Cause Findings",
        "",
        f"- Regression-risk buckets: {len(reg)} / {len(subtype)}.",
        f"- broader20 micro_prepare_50 negative rows: {len(broader)}.",
        f"- axis-tradeoff action rows: {len(axis)} across {axis['probe_id'].nunique() if not axis.empty else 0} buckets.",
        f"- non-monotone buckets: {len(nonmono)}.",
        "",
        "## Regression Subtypes",
        markdown_table(subtype_summary),
        "",
        "## Dominant Subtype",
        markdown_table(dominant),
        "",
        "## Broader20 False Prepare",
        markdown_table(broader_summary),
        "",
        "## Axis Tradeoff Rate By Action",
        markdown_table(axis_rate),
        "",
        "## Non-Monotone Diagnostics",
        markdown_table(nonmono_summary),
        "",
        "## Proposed Labels",
        markdown_table(labels),
    ]
    (PAPER / "regression_risk_findings.md").write_text("\n".join(paper) + "\n")

    decision = [
        "# h120 Regression-Risk Root-Cause Summary",
        "",
        "Scope: offline root-cause audit only.  No controller gate was attached, v1.6 delayed-medium reactive floor was not changed, and no new casebooks were run.",
        "",
        "## Main Findings",
        "",
        f"- Regression-risk buckets: {len(reg)} / {len(subtype)}.",
        f"- The largest subtype counts are: {', '.join(f'{r.subtype}={int(r.regression_bucket_count)}' for r in subtype_summary.head(4).itertuples())}.",
        f"- broader20 micro_prepare_50 negative rows: {len(broader)}; suggested anti-trigger coverage is {float(broader['suggested_anti_trigger'].mean()) if not broader.empty else np.nan:.3f}.",
        f"- Axis tradeoff appears in {axis['probe_id'].nunique() if not axis.empty else 0} buckets and {len(axis)} action rows.",
        f"- Non-monotone action response appears in {len(nonmono)} buckets.",
        "",
        "## Required Answers",
        "",
        "1. **Where does regression_risk mainly come from?**  It is multi-mechanism rather than one scalar error.  The strongest mechanisms are broader/low-posture false prepare, floor/no-gain or double-pump/no-avoid behavior, non-monotone target lifecycle response, and axis tradeoff.  Many buckets have multiple subtype labels.",
        f"2. **Can broader20 false prepare be explained by current features?**  Partly yes.  {len(broader)} broader20 micro50 negatives are mostly identifiable through low/no near-term exposure, low posture, far-persistent-high-alone, direction mismatch/signflip/near-safe-far-risky context, or no-gain pump behavior.  But this is not yet enough for a controller gate because Stage A still failed to learn regression risk robustly.",
        f"3. **Is axis_tradeoff a major safety risk?**  Yes, enough to deserve its own head.  It appears in {axis['probe_id'].nunique() if not axis.empty else 0} buckets; it can hide behind max-axis improvements because one axis gets better while the other worsens.",
        f"4. **Is target_refresh_scale still a good action axis?**  Not by itself.  {len(nonmono)} buckets are non-monotone across the action family, so increasing scale is not a reliable continuous action knob.  It should be treated as a discrete action family or replaced by an axis-aware prepare primitive.",
        "5. **Next: regression-risk head or action/interface redesign?**  Do both in order: first build subtype-specific regression-risk heads and broader false-prepare/axis-tradeoff/non-monotone labels; if those remain weak, redesign the prepare action/state interface before any gate.  Do not train expected-delta on all rows yet.",
        "6. **Does v1.6 remain the mainline?**  Yes.  The root-cause result does not justify shadow gate or controller gate; v1.6 remains the safe control mainline.",
        "",
        "## Next Practical Step",
        "",
        "Train a lightweight multi-label risk screen using the proposed labels, with explicit heads for broader20 false prepare, axis_tradeoff risk, double-pump/no-avoid, low-posture false prepare, and non-monotone response.  Only after those heads pass should expected-delta or ranking be retried on the clean subset.",
    ]
    (OUT / "regression_risk_rootcause_summary.md").write_text("\n".join(decision) + "\n")


def main() -> None:
    ensure_dirs()
    cf, action, regime, candidates = load_tables()
    subtype = build_regression_subtypes(action, regime, candidates)
    broader = build_broader20_false_prepare(action, subtype)
    axis = build_axis_tradeoff(action)
    nonmono = build_non_monotone(action, regime)
    labels = proposed_labels(subtype, broader, axis, nonmono)
    subtype_summary = summarize_subtypes(subtype)
    build_next_feature_recommendations(subtype_summary, broader, axis, nonmono)
    write_reports(subtype, subtype_summary, broader, axis, nonmono, labels, action)


if __name__ == "__main__":
    main()
