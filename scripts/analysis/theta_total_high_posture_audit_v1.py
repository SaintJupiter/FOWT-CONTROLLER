from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/wind_prediction/theta_total_high_posture_audit_v1"

RUNS = {
    "guard10_baseline": ROOT / "outputs/wind_prediction/tune_baseline_learned_guard10",
    "broader20_baseline": ROOT
    / "outputs/wind_prediction/f60_relief_e15_holdout_validation_v1/baseline_learned_2h",
}

PUMP_IDLE_RATE_M3_MIN = 0.05
HIGH_DEG = 5.0
TARGET_ERR_TINY_KG = 500.0


def _case_key(path: Path) -> str:
    name = path.name
    for suffix in (
        "_prediction_primary_econ_timeseries.csv",
        "_prediction_primary_econ_planner_log.csv",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _case_short(case_key: str) -> str:
    parts = case_key.split("_")
    if parts and parts[0].isdigit():
        return "_".join(parts[1:-2]) if len(parts) > 3 else "_".join(parts[1:])
    return case_key


def theta_total_deg(pitch_deg: pd.Series | np.ndarray, roll_deg: pd.Series | np.ndarray) -> np.ndarray:
    pitch_rad = np.deg2rad(np.asarray(pitch_deg, dtype=float))
    roll_rad = np.deg2rad(np.asarray(roll_deg, dtype=float))
    theta = np.arctan(np.sqrt(np.tan(pitch_rad) ** 2 + np.tan(roll_rad) ** 2))
    return np.rad2deg(theta)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _planner_row(log: pd.DataFrame, bucket: int) -> pd.Series | None:
    if log.empty or "bucket" not in log.columns:
        return None
    rows = log[log["bucket"].astype(int) == int(bucket)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _fallback_col(ts: pd.DataFrame) -> str | None:
    for c in ts.columns:
        lc = c.lower()
        if "safety_fallback" in lc or "fallback_active" in lc:
            return c
    return None


def _target_err(ts: pd.DataFrame) -> pd.Series:
    err_cols = [c for c in ("err_tank1_kg", "err_tank2_kg", "err_tank3_kg") if c in ts.columns]
    if not err_cols:
        return pd.Series(np.zeros(len(ts)), index=ts.index)
    return ts[err_cols].abs().mean(axis=1)


def _future_mean(series: pd.Series, start_s: int, horizon_s: int) -> float:
    end_s = start_s + horizon_s
    rows = series.loc[(series.index >= start_s) & (series.index < end_s)]
    if rows.empty:
        return float("nan")
    return float(rows.mean())


def _prefix_suffix_decline(posture: pd.Series, mask_index: pd.Index) -> tuple[float, bool, bool]:
    if len(mask_index) == 0:
        return 0.0, False, True
    vals = posture.loc[mask_index].astype(float)
    if len(vals) == 1:
        return 0.0, False, True
    n = min(60, max(1, len(vals) // 3))
    start = float(vals.iloc[:n].mean())
    end = float(vals.iloc[-n:].mean())
    decline = start - end
    return decline, decline >= 0.30, end <= start + 0.10


def _forecast_relief(row: pd.Series | None) -> tuple[float, float, bool]:
    if row is None:
        return 0.0, 1.0, False
    b0 = _safe_float(row.get("raw_pressure_block0_norm", row.get("pressure_block0_norm", 0.0)))
    b1 = _safe_float(row.get("raw_pressure_block1_norm", row.get("pressure_block1_norm", 0.0)))
    b2 = _safe_float(row.get("raw_pressure_block2_norm", row.get("pressure_block2_norm", 0.0)))
    dot02 = _safe_float(row.get("raw_pressure_block02_dot", row.get("pressure_block02_dot", 0.0)))
    future_min = min(b1, b2)
    margin = b0 - future_min
    ratio = future_min / max(b0, 1e-6)
    clear = bool((margin >= 0.25) or (ratio < 0.75) or (dot02 < 0.0))
    return margin, ratio, clear


def classify_bucket(
    action: str,
    target_reused: bool,
    target_err_tiny: bool,
    pump_idle: bool,
    pump_active: bool,
    forecast_relief_clear: bool,
    current_not_worsening: bool,
    future_decline: bool,
) -> str:
    action_l = str(action).lower()
    active = action_l in {"active_small", "active_medium"}
    holdlike = action_l in {"hold", "pump_saving", "park", "none"}
    if (holdlike or target_reused or pump_idle) and forecast_relief_clear and current_not_worsening and future_decline:
        return "justified_hold"
    if active and (target_reused or target_err_tiny) and pump_idle:
        return "active_but_execution_stale"
    if pump_active and not future_decline:
        return "pump_active_but_not_recovering"
    if (holdlike or pump_idle) and not forecast_relief_clear:
        return "unjustified_hold_or_reuse"
    if active and pump_active and future_decline:
        return "active_recovering"
    return "logging_insufficient"


def hysteresis_stats(theta: np.ndarray, enter: float, exit_: float, min_duration: int) -> tuple[int, int]:
    active = False
    start = 0
    episodes: list[tuple[int, int]] = []
    for idx, val in enumerate(theta):
        if not active and val > enter:
            active = True
            start = idx
        elif active and val < exit_:
            end = idx
            if end - start >= min_duration:
                episodes.append((start, end))
            active = False
    if active:
        end = len(theta)
        if end - start >= min_duration:
            episodes.append((start, end))
    total = int(sum(end - start for start, end in episodes))
    return len(episodes), total


def analyze_run(run_label: str, run_dir: Path) -> tuple[list[dict], list[dict]]:
    ts_dir = run_dir / "timeseries"
    log_dir = run_dir / "planner_logs"
    bucket_rows: list[dict] = []
    case_rows: list[dict] = []
    logs = {_case_key(p): p for p in log_dir.glob("*_planner_log.csv")} if log_dir.exists() else {}

    for ts_path in sorted(ts_dir.glob("*_timeseries.csv")):
        case = _case_key(ts_path)
        case_short = _case_short(case)
        log_path = logs.get(case)
        ts = pd.read_csv(ts_path, low_memory=False)
        log = pd.read_csv(log_path, low_memory=False) if log_path and log_path.exists() else pd.DataFrame()
        t_col = "t_s" if "t_s" in ts.columns else "time_s"
        ts = ts.copy()
        ts["_t_s"] = ts[t_col].round().astype(int)
        ts = ts.set_index("_t_s", drop=False)
        pitch = ts["pitch_deg"].abs().astype(float)
        roll = ts["roll_deg"].abs().astype(float) if "roll_deg" in ts.columns else pd.Series(0.0, index=ts.index)
        max_axis = np.maximum(pitch.to_numpy(), roll.to_numpy())
        theta = theta_total_deg(pitch, roll)
        theta_approx = np.sqrt(pitch.to_numpy() ** 2 + roll.to_numpy() ** 2)
        ts["_max_axis_deg"] = max_axis
        ts["_theta_total_deg"] = theta
        ts["_theta_approx_deg"] = theta_approx
        pump = ts["pump_total_rate_m3_min"].abs() if "pump_total_rate_m3_min" in ts.columns else pd.Series(0.0, index=ts.index)
        idle = pump < PUMP_IDLE_RATE_M3_MIN
        fb_c = _fallback_col(ts)
        fallback = ts[fb_c].astype(float) > 0.5 if fb_c else pd.Series(False, index=ts.index)
        target_err = _target_err(ts)
        buckets = (ts.index // 600).astype(int)
        ts["_bucket"] = buckets

        max_mask = ts["_max_axis_deg"] > HIGH_DEG
        theta_mask = ts["_theta_total_deg"] > HIGH_DEG
        combined_only = theta_mask & ~max_mask

        hstats = {}
        for exit_ in (4.0, 4.5):
            for min_dur in (60, 120):
                n, total = hysteresis_stats(theta, HIGH_DEG, exit_, min_dur)
                key = f"theta_hyst_enter5_exit{str(exit_).replace('.', 'p')}_min{min_dur}"
                hstats[f"{key}_episodes"] = n
                hstats[f"{key}_time_s"] = total

        class_seconds: dict[str, int] = {}
        for bucket in sorted(ts.loc[theta_mask, "_bucket"].unique()):
            sub = ts[ts["_bucket"] == bucket]
            high = sub[sub["_theta_total_deg"] > HIGH_DEG]
            if high.empty:
                continue
            prow = _planner_row(log, int(bucket))
            action = str(prow.get("first_action", "unknown")) if prow is not None else "unknown"
            raw_action = str(prow.get("planner_first_action_raw", "")) if prow is not None else ""
            target_reused = bool(_safe_float(prow.get("prediction_primary_target_reused", 0.0)) > 0.5) if prow is not None else False
            target_refreshed = bool(_safe_float(prow.get("prediction_primary_target_refreshed", 0.0)) > 0.5) if prow is not None else False
            target_age = _safe_float(prow.get("prediction_primary_target_age_s", 0.0)) if prow is not None else 0.0
            duration = int(len(high))
            idle_fraction = float(idle.loc[high.index].mean()) if duration else 0.0
            pump_active_fraction = float((~idle.loc[high.index]).mean()) if duration else 0.0
            fallback_fraction = float(fallback.loc[high.index].mean()) if duration else 0.0
            target_err_mean = float(target_err.loc[high.index].mean()) if duration else 0.0
            target_err_tiny = bool(target_err_mean <= TARGET_ERR_TINY_KG)
            current_decline, current_declining, current_not_worsening = _prefix_suffix_decline(
                ts["_theta_total_deg"], high.index
            )
            seg_end = int(high.index.max()) + 1
            future10 = _future_mean(ts["_theta_total_deg"], seg_end, 600)
            future20 = _future_mean(ts["_theta_total_deg"], seg_end, 1200)
            future_candidates = [x for x in (future10, future20) if not math.isnan(x)]
            future_best = min(future_candidates) if future_candidates else float("nan")
            theta_mean = float(high["_theta_total_deg"].mean())
            future_decline_amt = theta_mean - future_best if not math.isnan(future_best) else 0.0
            future_decline = bool(future_decline_amt >= 0.30)
            relief_margin, relief_ratio, relief_clear = _forecast_relief(prow)
            cls = classify_bucket(
                action=action,
                target_reused=target_reused,
                target_err_tiny=target_err_tiny,
                pump_idle=idle_fraction >= 0.80,
                pump_active=pump_active_fraction >= 0.20,
                forecast_relief_clear=relief_clear,
                current_not_worsening=current_not_worsening,
                future_decline=future_decline,
            )
            class_seconds[cls] = class_seconds.get(cls, 0) + duration
            bucket_rows.append(
                {
                    "run": run_label,
                    "case": case,
                    "case_short": case_short,
                    "bucket": int(bucket),
                    "seg_start_s": int(high.index.min()),
                    "seg_end_s": int(high.index.max()) + 1,
                    "duration_s": duration,
                    "theta_total_mean_deg": theta_mean,
                    "theta_total_p95_deg": float(np.percentile(high["_theta_total_deg"], 95)),
                    "theta_total_max_deg": float(high["_theta_total_deg"].max()),
                    "theta_approx_mean_deg": float(high["_theta_approx_deg"].mean()),
                    "max_axis_mean_deg": float(high["_max_axis_deg"].mean()),
                    "max_axis_max_deg": float(high["_max_axis_deg"].max()),
                    "combined_only_s": int((high["_max_axis_deg"] <= HIGH_DEG).sum()),
                    "action": action,
                    "raw_action": raw_action,
                    "target_refreshed": int(target_refreshed),
                    "target_reused": int(target_reused),
                    "target_age_s": target_age,
                    "target_err_mean_kg": target_err_mean,
                    "target_error_tiny": int(target_err_tiny),
                    "pump_idle_fraction": idle_fraction,
                    "pump_active_fraction": pump_active_fraction,
                    "fallback_fraction": fallback_fraction,
                    "current_decline_deg": current_decline,
                    "current_not_worsening": int(current_not_worsening),
                    "future_posture_mean_10m": future10,
                    "future_posture_mean_20m": future20,
                    "future_decline_deg": future_decline_amt,
                    "future_actual_decline": int(future_decline),
                    "forecast_relief_margin": relief_margin,
                    "forecast_future_min_ratio": relief_ratio,
                    "forecast_relief_clear": int(relief_clear),
                    "theta_recovery_justification_class": cls,
                }
            )

        row = {
            "run": run_label,
            "case": case,
            "case_short": case_short,
            "max_axis_time_over_5_s": int(max_mask.sum()),
            "theta_total_time_over_5_s": int(theta_mask.sum()),
            "theta_total_idle_time_over_5_s": int((theta_mask & idle).sum()),
            "theta_total_idle_ratio_over_5": float((theta_mask & idle).sum() / max(theta_mask.sum(), 1)),
            "combined_only_time_over_5_s": int(combined_only.sum()),
            "combined_only_idle_time_over_5_s": int((combined_only & idle).sum()),
            "theta_total_mean_deg": float(np.mean(theta)),
            "theta_total_p95_deg": float(np.percentile(theta, 95)),
            "theta_total_max_deg": float(np.max(theta)),
            "theta_approx_p95_deg": float(np.percentile(theta_approx, 95)),
            "max_axis_p95_deg": float(np.percentile(max_axis, 95)),
            "pump_m3": float((pump / 60.0).sum()),
            "fallback_pct": float(fallback.mean() * 100.0),
            **hstats,
        }
        for cls, seconds in class_seconds.items():
            row[f"class_{cls}_s"] = int(seconds)
        case_rows.append(row)

    return bucket_rows, case_rows


def _aggregate(case_df: pd.DataFrame, bucket_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, sub in case_df.groupby("run"):
        bsub = bucket_df[bucket_df["run"] == run]
        theta_time = int(sub["theta_total_time_over_5_s"].sum())
        max_time = int(sub["max_axis_time_over_5_s"].sum())
        row = {
            "run": run,
            "cases": int(sub["case"].nunique()),
            "max_axis_time_over_5_s": max_time,
            "theta_total_time_over_5_s": theta_time,
            "theta_total_idle_time_over_5_s": int(sub["theta_total_idle_time_over_5_s"].sum()),
            "theta_total_idle_ratio_over_5": float(
                sub["theta_total_idle_time_over_5_s"].sum() / max(theta_time, 1)
            ),
            "combined_only_time_over_5_s": int(sub["combined_only_time_over_5_s"].sum()),
            "combined_only_share_of_theta_over_5": float(
                sub["combined_only_time_over_5_s"].sum() / max(theta_time, 1)
            ),
            "theta_total_mean_deg": float(np.average(sub["theta_total_mean_deg"], weights=np.ones(len(sub)))),
            "theta_total_p95_max_case_deg": float(sub["theta_total_p95_deg"].max()),
            "theta_total_max_deg": float(sub["theta_total_max_deg"].max()),
        }
        for cls, csub in bsub.groupby("theta_recovery_justification_class"):
            seconds = int(csub["duration_s"].sum())
            row[f"{cls}_s"] = seconds
            row[f"{cls}_share"] = float(seconds / max(theta_time, 1))
        for col in case_df.columns:
            if col.startswith("theta_hyst_") and col.endswith("_time_s"):
                row[col] = int(sub[col].sum())
            if col.startswith("theta_hyst_") and col.endswith("_episodes"):
                row[col] = int(sub[col].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, cols: list[str], float_digits: int = 3) -> str:
    if df.empty:
        return "(empty)"
    view = df[cols].copy()
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda x: f"{x:.{float_digits}f}")
    headers = [str(c) for c in view.columns]
    rows = [[str(v) for v in row] for row in view.to_numpy().tolist()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def write_reports(case_df: pd.DataFrame, bucket_df: pd.DataFrame, agg: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bucket_df.to_csv(OUT_DIR / "theta_total_bucket_table.csv", index=False)
    case_df.to_csv(OUT_DIR / "theta_total_case_table.csv", index=False)

    summary_cols = [
        "run",
        "cases",
        "max_axis_time_over_5_s",
        "theta_total_time_over_5_s",
        "theta_total_idle_time_over_5_s",
        "theta_total_idle_ratio_over_5",
        "combined_only_time_over_5_s",
        "combined_only_share_of_theta_over_5",
        "theta_total_p95_max_case_deg",
        "theta_total_max_deg",
    ]
    class_cols = [
        c
        for c in [
            "justified_hold_s",
            "justified_hold_share",
            "active_but_execution_stale_s",
            "active_but_execution_stale_share",
            "active_recovering_s",
            "active_recovering_share",
            "pump_active_but_not_recovering_s",
            "pump_active_but_not_recovering_share",
            "unjustified_hold_or_reuse_s",
            "unjustified_hold_or_reuse_share",
        ]
        if c in agg.columns
    ]
    hyst_cols = [
        "run",
        "theta_hyst_enter5_exit4p0_min60_episodes",
        "theta_hyst_enter5_exit4p0_min60_time_s",
        "theta_hyst_enter5_exit4p5_min60_episodes",
        "theta_hyst_enter5_exit4p5_min60_time_s",
        "theta_hyst_enter5_exit4p0_min120_episodes",
        "theta_hyst_enter5_exit4p0_min120_time_s",
        "theta_hyst_enter5_exit4p5_min120_episodes",
        "theta_hyst_enter5_exit4p5_min120_time_s",
    ]
    hyst_cols = [c for c in hyst_cols if c in agg.columns]

    top_cases = case_df.sort_values("theta_total_time_over_5_s", ascending=False).head(12)
    top_cols = [
        "run",
        "case_short",
        "max_axis_time_over_5_s",
        "theta_total_time_over_5_s",
        "combined_only_time_over_5_s",
        "theta_total_idle_time_over_5_s",
        "theta_total_p95_deg",
        "theta_total_max_deg",
    ]

    summary = "\n".join(
        [
            "# Theta-total high-posture audit v1",
            "",
            "Definition:",
            "- `max_axis = max(abs(pitch), abs(roll))`.",
            "- `theta_total = atan(sqrt(tan(pitch)^2 + tan(roll)^2))`, reported in degrees.",
            "- `theta_approx = sqrt(pitch^2 + roll^2)` is also stored for small-angle comparison.",
            "- High posture is counted with `>5 deg`; pump idle means total pump rate below 0.05 m3/min.",
            "",
            "## Aggregate comparison",
            _md_table(agg, summary_cols),
            "",
            "## Theta-total recovery-justification classes",
            _md_table(agg, ["run", *class_cols]),
            "",
            "## Hysteresis candidates",
            _md_table(agg, hyst_cols),
            "",
            "## Top theta_total>5 cases",
            _md_table(top_cases, top_cols),
            "",
            "## Notes",
            "- `combined_only_time_over_5_s` is the time where `theta_total>5` but neither single axis exceeds 5 deg.",
            "- Recovery-justification classes are computed on theta-total high-posture buckets, not on the old max-axis mask.",
        ]
    )
    (OUT_DIR / "theta_total_summary.md").write_text(summary, encoding="utf-8")

    decision_lines = [
        "# Theta-total high-posture decision",
        "",
        "## 1. Should theta_total>5 replace max_axis>5 as the recovery-enter definition?",
        "",
        "Yes for the recovery-enter audit/control concept. `theta_total` is closer to total platform inclination, while `max_axis` only sees the largest single component. The two are similar when one axis dominates, but `theta_total` correctly counts combined pitch+roll tilt.",
        "",
        "## 2. How much combined tilt did the old max-axis metric miss?",
        "",
    ]
    for _, row in agg.iterrows():
        decision_lines.append(
            f"- {row['run']}: theta_total>5 adds {int(row['combined_only_time_over_5_s'])} s "
            f"that max_axis>5 missed ({row['combined_only_share_of_theta_over_5']*100:.1f}% of theta-total high-posture time)."
        )
    decision_lines += [
        "",
        "## 3. Hysteresis recommendation",
        "",
        "Use `enter theta_total > 5 deg`; prefer `exit theta_total < 4.5 deg` for control probing, with `min_duration` 60-120 s. Exit 4.0 deg is cleaner but keeps the recovery state active longer; 4.5 deg is a more practical first probe and still prevents chattering.",
        "",
        "## 4. Paper wording",
        "",
        "State that 5 deg is used as an operating recovery-trigger threshold based on total platform inclination, not as an instantaneous hard safety limit. Higher limits such as 8-10 deg should be treated as high-risk operating bounds, while the controller should start recovering near 5 deg unless forecast relief and observed posture decline justify waiting.",
        "",
        "## 5. What the theta-total audit changes",
        "",
        "The old conclusion remains, but becomes more physically grounded: high posture should be measured as total inclination. The dominant problem is still not mostly justified hold; it is nominal active/reused target/pump-idle behavior during elevated total tilt.",
    ]
    (OUT_DIR / "theta_total_decision.md").write_text("\n".join(decision_lines), encoding="utf-8")


def main() -> None:
    all_buckets: list[dict] = []
    all_cases: list[dict] = []
    for run_label, run_dir in RUNS.items():
        buckets, cases = analyze_run(run_label, run_dir)
        all_buckets.extend(buckets)
        all_cases.extend(cases)
    bucket_df = pd.DataFrame(all_buckets)
    case_df = pd.DataFrame(all_cases).fillna(0)
    agg = _aggregate(case_df, bucket_df)
    write_reports(case_df, bucket_df, agg)
    print(f"Wrote {OUT_DIR}")
    print(_md_table(agg, [
        "run",
        "max_axis_time_over_5_s",
        "theta_total_time_over_5_s",
        "theta_total_idle_time_over_5_s",
        "combined_only_time_over_5_s",
        "combined_only_share_of_theta_over_5",
        "theta_total_max_deg",
    ]))


if __name__ == "__main__":
    main()
