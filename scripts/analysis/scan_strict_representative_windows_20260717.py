#!/usr/bin/env python3
"""Screen strict-150 paired cases for defensible representative windows.

The screen is intentionally conservative.  Every candidate includes 20 min of
pre-action history and 60 min after the planner anchor.  Planner choice,
target refresh and physical pump onset are recorded separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "outputs/wind_prediction/frozen_h_holdout150_20260710"
PAIRED = FROZEN / "final_150/paired_results.csv"
OUT = (
    ROOT
    / "outputs/paper_figures/20260717_event_decision_cards"
    / "candidate_rescan_strict_v2"
)

TS_COLS = [
    "t_s",
    "wind_speed",
    "wind_dir_deg",
    "pitch_deg",
    "roll_deg",
    "pump_total_rate_m3_min",
    "target_tank1_kg",
    "target_tank2_kg",
    "target_tank3_kg",
]
PLAN_COLS = [
    "current_time_s",
    "first_action",
    "prediction_primary_target_refreshed",
    "prediction_primary_target_reused",
    "prediction_primary_refresh_owner",
    "forecast_source",
    "forecast_has_future",
    "forecast_pressure_trust_trusted",
    "forecast_pressure_trust_future_speed_max_ms",
    "forecast_pressure_trust_current_speed_ms",
    "event_risk_effective_prob_0_20m",
    "event_risk_effective_prob_20_40m",
    "event_risk_effective_prob_40_60m",
    "preview_lead_action_active",
    "preview_sequence_lead_action_active",
    "active_medium_gate_reason",
]
ACTION_SET = {"active_small", "active_medium", "pump_saving"}


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"{directory}/{pattern}: expected 1 file, got {len(matches)}")
    return matches[0]


def attitude(data: pd.DataFrame) -> np.ndarray:
    return np.maximum(
        np.abs(data.pitch_deg.to_numpy(float)),
        np.abs(data.roll_deg.to_numpy(float)),
    )


def metrics(data: pd.DataFrame, start_s: int, end_s: int) -> dict[str, float]:
    part = data[(data.t_s >= start_s) & (data.t_s < end_s)]
    theta = attitude(part)
    return {
        "pump_m3": float(part.pump_total_rate_m3_min.sum() / 60.0),
        "mean_deg": float(theta.mean()),
        "p95_deg": float(np.quantile(theta, 0.95)),
        "peak_deg": float(theta.max()),
        "gt2_s": float(np.count_nonzero(theta > 2.0)),
    }


def first_pump_onset(
    data: pd.DataFrame, start_s: int, end_s: int
) -> float | None:
    run = data.pump_total_rate_m3_min.gt(0.05)
    onset = run & ~run.shift(fill_value=False)
    hits = data[onset & data.t_s.ge(start_s) & data.t_s.le(end_s)]
    if hits.empty:
        return None
    return float(hits.t_s.iloc[0])


def value_at(data: pd.DataFrame, t_s: int, column: str) -> float:
    idx = (data.t_s - t_s).abs().idxmin()
    return float(data.loc[idx, column])


def scan_case(pair: pd.Series) -> list[dict[str, object]]:
    task_dir = FROZEN / str(pair.source_task)
    case_id = str(pair.case_id)
    ts_dir = task_dir / "timeseries"
    plan_dir = task_dir / "planner_logs"
    pred_path = find_one(
        ts_dir, f"{case_id}_*_prediction_primary_econ_timeseries.csv"
    )
    closed_path = find_one(ts_dir, f"{case_id}_*_closed_only_timeseries.csv")
    planner_path = find_one(
        plan_dir, f"{case_id}_*_prediction_primary_econ_planner_log.csv"
    )
    pred = pd.read_csv(pred_path, usecols=TS_COLS)
    closed = pd.read_csv(closed_path, usecols=TS_COLS)
    available_plan_cols = pd.read_csv(planner_path, nrows=0).columns
    use_plan_cols = [c for c in PLAN_COLS if c in available_plan_cols]
    planner = pd.read_csv(planner_path, usecols=use_plan_cols)

    if not np.array_equal(pred.t_s.to_numpy(), closed.t_s.to_numpy()):
        raise RuntimeError(f"{case_id}: paired second grids differ")

    rows: list[dict[str, object]] = []
    anchors = planner[
        planner.first_action.isin(ACTION_SET)
        & planner.prediction_primary_target_refreshed.fillna(0).eq(1)
        & planner.current_time_s.between(1200, 18000)
    ]
    for _, anchor in anchors.iterrows():
        anchor_s = int(round(float(anchor.current_time_s)))
        window_start_s = anchor_s - 1200
        window_end_s = min(anchor_s + 3600, 21600)
        if window_end_s - window_start_s < 4200:
            continue

        pred_onset = first_pump_onset(pred, anchor_s, anchor_s + 180)
        if pred_onset is None:
            continue
        pred_running_at_anchor = (
            value_at(pred, anchor_s, "pump_total_rate_m3_min") > 0.05
        )
        closed_running_at_anchor = (
            value_at(closed, anchor_s, "pump_total_rate_m3_min") > 0.05
        )
        closed_onset = first_pump_onset(closed, anchor_s, anchor_s + 1800)
        closed_lead_min = (
            (closed_onset - pred_onset) / 60.0
            if closed_onset is not None
            else 30.0
        )

        current_wind = value_at(pred, anchor_s, "wind_speed")
        future = pred[
            pred.t_s.between(anchor_s + 300, min(anchor_s + 1800, 21599))
        ]
        future_wind_max = float(future.wind_speed.max())
        actual_wind_rise = future_wind_max - current_wind

        pre_pred = metrics(pred, window_start_s, anchor_s)
        pre_closed = metrics(closed, window_start_s, anchor_s)
        win_pred = metrics(pred, window_start_s, window_end_s)
        win_closed = metrics(closed, window_start_s, window_end_s)
        post_pred = metrics(pred, anchor_s, window_end_s)
        post_closed = metrics(closed, anchor_s, window_end_s)
        pred_first_10min = metrics(pred, anchor_s, anchor_s + 600)
        full_pred = metrics(pred, 0, 21600)
        full_closed = metrics(closed, 0, 21600)

        plan_window = planner[
            planner.current_time_s.between(window_start_s, window_end_s)
        ]
        actions = plan_window.first_action.astype(str).tolist()
        switches = sum(a != b for a, b in zip(actions, actions[1:]))
        action_types = len(set(actions))
        event_prob = max(
            float(anchor.get("event_risk_effective_prob_0_20m", np.nan)),
            float(anchor.get("event_risk_effective_prob_20_40m", np.nan)),
            float(anchor.get("event_risk_effective_prob_40_60m", np.nan)),
        )

        save_pct = (
            100.0 * (win_closed["pump_m3"] - win_pred["pump_m3"])
            / win_closed["pump_m3"]
            if win_closed["pump_m3"] > 0
            else np.nan
        )
        record: dict[str, object] = {
            "case_id": case_id,
            "timestamp": pair.timestamp,
            "source_task": pair.source_task,
            "anchor_min": anchor_s / 60.0,
            "window_start_min": window_start_s / 60.0,
            "window_end_min": window_end_s / 60.0,
            "action": anchor.first_action,
            "refresh_owner": anchor.get("prediction_primary_refresh_owner", ""),
            "forecast_source": anchor.get("forecast_source", ""),
            "forecast_has_future": anchor.get("forecast_has_future", np.nan),
            "forecast_trusted": anchor.get(
                "forecast_pressure_trust_trusted", np.nan
            ),
            "preview_lead_active": anchor.get(
                "preview_lead_action_active", np.nan
            ),
            "preview_sequence_lead_active": anchor.get(
                "preview_sequence_lead_action_active", np.nan
            ),
            "active_medium_gate_reason": anchor.get(
                "active_medium_gate_reason", ""
            ),
            "current_wind_m_s": current_wind,
            "future_actual_wind_max_m_s": future_wind_max,
            "actual_wind_rise_m_s": actual_wind_rise,
            "forecast_future_speed_max_m_s": anchor.get(
                "forecast_pressure_trust_future_speed_max_ms", np.nan
            ),
            "event_probability_max": event_prob,
            "prediction_pump_onset_min": pred_onset / 60.0,
            "prediction_running_at_anchor": int(pred_running_at_anchor),
            "closed_running_at_anchor": int(closed_running_at_anchor),
            "prediction_first_10min_pump_m3": pred_first_10min["pump_m3"],
            "closed_next_pump_onset_min": (
                closed_onset / 60.0 if closed_onset is not None else np.nan
            ),
            "closed_minus_prediction_onset_min": closed_lead_min,
            "action_switch_count": switches,
            "action_type_count": action_types,
            "action_sequence": ">".join(actions),
            "prediction_window_pump_m3": win_pred["pump_m3"],
            "closed_window_pump_m3": win_closed["pump_m3"],
            "window_pump_saving_pct": save_pct,
            "pre_mean_diff_deg": pre_pred["mean_deg"] - pre_closed["mean_deg"],
            "pre_p95_diff_deg": pre_pred["p95_deg"] - pre_closed["p95_deg"],
            "window_mean_diff_deg": win_pred["mean_deg"] - win_closed["mean_deg"],
            "window_p95_diff_deg": win_pred["p95_deg"] - win_closed["p95_deg"],
            "window_peak_diff_deg": win_pred["peak_deg"] - win_closed["peak_deg"],
            "window_gt2_diff_s": win_pred["gt2_s"] - win_closed["gt2_s"],
            "post_mean_diff_deg": post_pred["mean_deg"] - post_closed["mean_deg"],
            "post_p95_diff_deg": post_pred["p95_deg"] - post_closed["p95_deg"],
            "post_peak_diff_deg": post_pred["peak_deg"] - post_closed["peak_deg"],
            "post_gt2_diff_s": post_pred["gt2_s"] - post_closed["gt2_s"],
            "full_prediction_pump_m3": full_pred["pump_m3"],
            "full_closed_pump_m3": full_closed["pump_m3"],
            "full_mean_diff_deg": full_pred["mean_deg"] - full_closed["mean_deg"],
            "full_p95_diff_deg": full_pred["p95_deg"] - full_closed["p95_deg"],
            "full_gt2_diff_s": full_pred["gt2_s"] - full_closed["gt2_s"],
            "paired_full_pump_reduction_pct": pair.pump_reduction_pct,
            "prediction_timeseries_path": str(pred_path),
            "closed_timeseries_path": str(closed_path),
            "planner_log_path": str(planner_path),
        }

        record["strict_preemptive_pass"] = int(
            str(record["refresh_owner"]) == "event_reset"
            and bool(record["forecast_has_future"])
            and bool(record["forecast_trusted"])
            and not pred_running_at_anchor
            and not closed_running_at_anchor
            and pred_first_10min["pump_m3"] >= 5.0
            and actual_wind_rise >= 1.5
            and closed_lead_min >= 5.0
            and save_pct >= 15.0
            and abs(float(record["pre_mean_diff_deg"])) <= 0.20
            and abs(float(record["pre_p95_diff_deg"])) <= 0.30
            and float(record["window_mean_diff_deg"]) <= 0.10
            and float(record["window_p95_diff_deg"]) <= 0.20
            and float(record["window_peak_diff_deg"]) <= 0.20
            and float(record["window_gt2_diff_s"]) <= 60.0
            and float(record["post_p95_diff_deg"]) <= 0.20
            and float(record["post_gt2_diff_s"]) <= 60.0
            and switches >= 1
            and float(pair.pump_reduction_pct) > 0.0
        )
        record["defensible_window_pass"] = int(
            str(record["refresh_owner"]) == "event_reset"
            and bool(record["forecast_has_future"])
            and save_pct >= 15.0
            and abs(float(record["pre_mean_diff_deg"])) <= 0.30
            and float(record["window_mean_diff_deg"]) <= 0.15
            and float(record["window_p95_diff_deg"]) <= 0.30
            and float(record["window_peak_diff_deg"]) <= 0.30
            and float(record["window_gt2_diff_s"]) <= 90.0
            and switches >= 1
        )
        record["display_score"] = (
            2.0 * record["strict_preemptive_pass"]
            + 1.0 * record["defensible_window_pass"]
            + min(max(save_pct, 0.0), 70.0) / 100.0
            + min(max(actual_wind_rise, 0.0), 5.0) / 10.0
            + min(max(closed_lead_min, 0.0), 20.0) / 40.0
            - max(float(record["window_mean_diff_deg"]), 0.0)
            - max(float(record["window_p95_diff_deg"]), 0.0)
            - max(float(record["window_peak_diff_deg"]), 0.0) / 2.0
            - max(float(record["window_gt2_diff_s"]), 0.0) / 600.0
        )
        rows.append(record)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(PAIRED)
    all_rows: list[dict[str, object]] = []
    errors: list[str] = []
    for _, pair in paired.iterrows():
        try:
            all_rows.extend(scan_case(pair))
        except Exception as exc:  # preserve a complete audit rather than hiding cases
            errors.append(f"{pair.case_id}: {exc}")

    result = pd.DataFrame(all_rows)
    result = result.sort_values(
        [
            "strict_preemptive_pass",
            "defensible_window_pass",
            "display_score",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result.to_csv(OUT / "all_candidate_anchors.csv", index=False)
    result.head(30).to_csv(OUT / "top30_candidate_anchors.csv", index=False)
    (OUT / "scan_errors.txt").write_text("\n".join(errors), encoding="utf-8")
    summary = [
        "# 严格150组代表性决策窗口复筛",
        "",
        f"- 严格配对工况：{len(paired)}",
        f"- 可检查锚点：{len(result)}",
        f"- strict_preemptive_pass：{int(result.strict_preemptive_pass.sum())}",
        f"- defensible_window_pass：{int(result.defensible_window_pass.sum())}",
        f"- 读取错误：{len(errors)}",
        "",
        "所有窗口均包含动作前20 min与动作后60 min；姿态定义为"
        "`max(|pitch|, |roll|)`；泵量由逐秒流量积分。",
    ]
    (OUT / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
