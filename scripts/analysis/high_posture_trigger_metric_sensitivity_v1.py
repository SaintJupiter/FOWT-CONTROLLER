from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/wind_prediction/high_posture_trigger_metric_sensitivity_v1"

RUNS = {
    "guard10_baseline": ROOT / "outputs/wind_prediction/tune_baseline_learned_guard10",
    "broader20_baseline": ROOT
    / "outputs/wind_prediction/f60_relief_e15_holdout_validation_v1/baseline_learned_2h",
    "guard10_refresh_guarded": ROOT
    / "outputs/wind_prediction/stale_active_target_refresh_v1/guard10_refresh_only_guarded_v1",
    "broader20_refresh_guarded": ROOT
    / "outputs/wind_prediction/stale_active_target_refresh_v1/broader20_refresh_only_guarded_v1",
}

PUMP_IDLE_RATE_M3_MIN = 0.05
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


def _hysteresis_mask(raw: np.ndarray, exit_raw: np.ndarray, min_duration: int) -> np.ndarray:
    active = False
    start = 0
    mask = np.zeros(len(raw), dtype=bool)
    for idx in range(len(raw)):
        if not active and raw[idx]:
            active = True
            start = idx
        elif active and exit_raw[idx]:
            end = idx
            if end - start >= min_duration:
                mask[start:end] = True
            active = False
    if active:
        end = len(raw)
        if end - start >= min_duration:
            mask[start:end] = True
    return mask


def metric_masks(max_axis: np.ndarray, theta: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "A_max_axis_gt5": (
            max_axis > 5.0,
            max_axis < 4.5,
        ),
        "B_theta_total_gt5": (
            theta > 5.0,
            theta < 4.5,
        ),
        "C_max_axis_gt5_or_theta_gt5p5": (
            (max_axis > 5.0) | (theta > 5.5),
            (max_axis < 4.5) & (theta < 5.0),
        ),
        "D_max_axis_gt5_or_theta_gt6": (
            (max_axis > 5.0) | (theta > 6.0),
            (max_axis < 4.5) & (theta < 5.5),
        ),
    }


def _classify_segment(
    action: str,
    target_reused: bool,
    target_err_tiny: bool,
    pump_idle_fraction: float,
    pump_active_fraction: float,
) -> str:
    action_l = str(action).lower()
    active = action_l in {"active_small", "active_medium"}
    holdlike = action_l in {"hold", "pump_saving", "park", "none"}
    if active and (target_reused or target_err_tiny) and pump_idle_fraction >= 0.8:
        return "active_but_execution_stale"
    if pump_active_fraction >= 0.2:
        return "pump_active"
    if holdlike or pump_idle_fraction >= 0.8:
        return "hold_or_idle"
    return "other"


def analyze_run(run_label: str, run_dir: Path) -> list[dict]:
    ts_dir = run_dir / "timeseries"
    log_dir = run_dir / "planner_logs"
    logs = {_case_key(p): p for p in log_dir.glob("*_planner_log.csv")} if log_dir.exists() else {}
    rows: list[dict] = []
    for ts_path in sorted(ts_dir.glob("*_timeseries.csv")):
        case = _case_key(ts_path)
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
        pump = ts["pump_total_rate_m3_min"].abs() if "pump_total_rate_m3_min" in ts.columns else pd.Series(0.0, index=ts.index)
        idle = (pump < PUMP_IDLE_RATE_M3_MIN).to_numpy()
        fb_col = _fallback_col(ts)
        fallback = (ts[fb_col].astype(float) > 0.5).to_numpy() if fb_col else np.zeros(len(ts), dtype=bool)
        target_err = _target_err(ts)
        buckets = (ts.index.to_numpy() // 600).astype(int)

        base_metrics = {
            "pump_m3": float((pump / 60.0).sum()),
            "sum_fb_pct": float(fallback.mean() * 100.0),
            "max_pitch_p95": float(np.percentile(pitch, 95)),
            "max_roll_p95": float(np.percentile(roll, 95)),
            "max_axis_p95": float(np.percentile(max_axis, 95)),
            "theta_total_p95": float(np.percentile(theta, 95)),
            "theta_total_max": float(np.max(theta)),
            "lowrisk": int("lowrisk" in case),
        }
        masks = metric_masks(max_axis, theta)
        for metric_name, (raw_mask, exit_raw) in masks.items():
            raw_seconds = int(raw_mask.sum())
            raw_idle = int((raw_mask & idle).sum())
            max_only_s = int((raw_mask & (max_axis > 5.0)).sum())
            theta_only_s = int((raw_mask & ~(max_axis > 5.0) & (theta > 5.0)).sum())
            for min_duration in (60, 120):
                mask = _hysteresis_mask(raw_mask, exit_raw, min_duration)
                trigger_seconds = int(mask.sum())
                trigger_idle = int((mask & idle).sum())
                classes: dict[str, int] = {}
                trigger_buckets = sorted(set(buckets[mask]))
                for bucket in trigger_buckets:
                    idx = np.where((buckets == bucket) & mask)[0]
                    if len(idx) == 0:
                        continue
                    prow = _planner_row(log, int(bucket))
                    action = str(prow.get("first_action", "unknown")) if prow is not None else "unknown"
                    target_reused = bool(_safe_float(prow.get("prediction_primary_target_reused", 0.0)) > 0.5) if prow is not None else False
                    terr = float(target_err.iloc[idx].mean()) if len(idx) else 0.0
                    cls = _classify_segment(
                        action=action,
                        target_reused=target_reused,
                        target_err_tiny=terr <= TARGET_ERR_TINY_KG,
                        pump_idle_fraction=float(idle[idx].mean()),
                        pump_active_fraction=float((~idle[idx]).mean()),
                    )
                    classes[cls] = classes.get(cls, 0) + int(len(idx))
                rows.append(
                    {
                        "run": run_label,
                        "case": case,
                        "case_short": _case_short(case),
                        "metric": metric_name,
                        "min_duration_s": min_duration,
                        "raw_trigger_seconds": raw_seconds,
                        "raw_idle_seconds": raw_idle,
                        "raw_idle_ratio": float(raw_idle / max(raw_seconds, 1)),
                        "raw_max_axis_component_seconds": max_only_s,
                        "raw_theta_supplement_seconds": theta_only_s,
                        "trigger_seconds": trigger_seconds,
                        "trigger_count": int(sum(1 for x in np.diff(np.r_[False, mask, False].astype(int)) if x == 1)),
                        "idle_time_in_trigger": trigger_idle,
                        "idle_ratio_in_trigger": float(trigger_idle / max(trigger_seconds, 1)),
                        "active_but_execution_stale_s": int(classes.get("active_but_execution_stale", 0)),
                        "pump_active_s": int(classes.get("pump_active", 0)),
                        "hold_or_idle_s": int(classes.get("hold_or_idle", 0)),
                        "other_s": int(classes.get("other", 0)),
                        **base_metrics,
                    }
                )
    return rows


def _md_table(df: pd.DataFrame, cols: list[str], digits: int = 3) -> str:
    if df.empty:
        return "(empty)"
    view = df[cols].copy()
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda x: f"{x:.{digits}f}")
    headers = [str(c) for c in view.columns]
    data = [[str(v) for v in row] for row in view.to_numpy().tolist()]
    widths = [max(len(headers[i]), *(len(row[i]) for row in data)) if data else len(headers[i]) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in data)
    return "\n".join(lines)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["run", "metric", "min_duration_s"]
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            cases=("case", "nunique"),
            trigger_seconds=("trigger_seconds", "sum"),
            trigger_count=("trigger_count", "sum"),
            idle_time_in_trigger=("idle_time_in_trigger", "sum"),
            active_but_execution_stale_s=("active_but_execution_stale_s", "sum"),
            pump_active_s=("pump_active_s", "sum"),
            hold_or_idle_s=("hold_or_idle_s", "sum"),
            raw_theta_supplement_seconds=("raw_theta_supplement_seconds", "sum"),
            raw_trigger_seconds=("raw_trigger_seconds", "sum"),
            sum_pump=("pump_m3", "sum"),
            sum_fb=("sum_fb_pct", "sum"),
            max_p95=("max_axis_p95", "max"),
            theta_total_p95=("theta_total_p95", "max"),
            theta_total_max=("theta_total_max", "max"),
            lowrisk_trigger_seconds=("trigger_seconds", lambda x: int(x[df.loc[x.index, "lowrisk"].astype(bool)].sum())),
        )
    )
    agg["idle_ratio_in_trigger"] = agg["idle_time_in_trigger"] / agg["trigger_seconds"].clip(lower=1)
    agg["active_but_execution_stale_share"] = agg["active_but_execution_stale_s"] / agg["trigger_seconds"].clip(lower=1)
    agg["theta_supplement_share_raw"] = agg["raw_theta_supplement_seconds"] / agg["raw_trigger_seconds"].clip(lower=1)
    return agg


def write_reports(detail: pd.DataFrame, agg: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT_DIR / "trigger_metric_table.csv", index=False)

    summary_cols = [
        "run",
        "metric",
        "min_duration_s",
        "trigger_seconds",
        "trigger_count",
        "idle_time_in_trigger",
        "idle_ratio_in_trigger",
        "active_but_execution_stale_s",
        "active_but_execution_stale_share",
        "theta_supplement_share_raw",
        "sum_pump",
        "sum_fb",
        "max_p95",
        "theta_total_p95",
        "lowrisk_trigger_seconds",
    ]
    focus = agg[(agg["min_duration_s"] == 60)].copy()
    focus = focus.sort_values(["run", "metric"])
    lines = [
        "# High-posture trigger metric sensitivity v1",
        "",
        "Scope: guard10/broader20 baseline plus the existing refresh_only_guarded_v1 candidate. No simulation was rerun; this is a read-only trigger-mask sensitivity audit over existing traces.",
        "",
        "Candidate metrics:",
        "- A: `max_axis > 5 deg`",
        "- B: `theta_total > 5 deg`",
        "- C: `max_axis > 5 deg OR theta_total > 5.5 deg`",
        "- D: `max_axis > 5 deg OR theta_total > 6 deg`",
        "",
        "All candidates are evaluated with hysteresis and min duration. The table below shows min_duration=60s; the CSV includes 60s and 120s.",
        "",
        _md_table(focus, summary_cols),
    ]
    (OUT_DIR / "trigger_metric_summary.md").write_text("\n".join(lines), encoding="utf-8")

    def pick(run: str, metric: str) -> pd.Series:
        row = agg[(agg["run"] == run) & (agg["metric"] == metric) & (agg["min_duration_s"] == 60)]
        return row.iloc[0] if not row.empty else pd.Series(dtype=float)

    decision = [
        "# Trigger metric decision",
        "",
        "## 1. Does theta_total>5 over-expand the trigger range?",
        "",
        "Yes. As a standalone trigger, B expands the active high-posture mask substantially, especially on broader20. It catches physically real combined tilt, but at 5 deg it is too broad to use as the only recovery-enter condition without making the controller chase many moderate combined-tilt periods.",
        "",
        "## 2. Does max_axis>5 miss key combined tilt?",
        "",
        "Yes. A is directionally clean but misses combined pitch+roll periods. The missed time is not negligible, especially in broader20 and in cases such as sf_holdout_02 / fr_relief_08 where neither axis alone stays above 5 deg but total inclination does.",
        "",
        "## 3. Recommended enter metric",
        "",
        "Use C as the first default-off probe: `max_axis > 5 deg OR theta_total > 5.5 deg`, with exit requiring `max_axis < 4.5 deg AND theta_total < 5.0 deg`, min_duration 60-120s.",
        "",
        "Why C: it preserves the single-axis recovery trigger while adding a limited combined-tilt supplement. D is safer and narrower but may miss too much combined tilt; B is too broad; A is too blind to combined tilt.",
        "",
        "## 4. Paper wording",
        "",
        "Describe 5 deg as a recovery-reference threshold for directional pitch/roll components, not a hard red line. Total inclination should be reported as a companion metric; combined-tilt supplement can be triggered at a slightly higher threshold such as 5.5-6 deg to avoid overreacting to benign diagonal tilt.",
        "",
        "## 5. Practical implication for reactive floor",
        "",
        "Do not make theta_total>5 the sole floor trigger. Use max_axis for the recovery direction, and use theta_total only as a supplemental enter condition. If implementing a reactive floor candidate next, start with C and keep D as a conservative ablation.",
        "",
        "## Key min-duration=60 snapshot",
        "",
    ]
    for run in sorted(agg["run"].unique()):
        a = pick(run, "A_max_axis_gt5")
        b = pick(run, "B_theta_total_gt5")
        c = pick(run, "C_max_axis_gt5_or_theta_gt5p5")
        d = pick(run, "D_max_axis_gt5_or_theta_gt6")
        if a.empty:
            continue
        decision.append(
            f"- {run}: A={int(a.trigger_seconds)}s, B={int(b.trigger_seconds)}s, "
            f"C={int(c.trigger_seconds)}s, D={int(d.trigger_seconds)}s; "
            f"C adds {int(c.trigger_seconds - a.trigger_seconds)}s over A, "
            f"D adds {int(d.trigger_seconds - a.trigger_seconds)}s over A."
        )
    (OUT_DIR / "trigger_metric_decision.md").write_text("\n".join(decision), encoding="utf-8")


def main() -> None:
    rows: list[dict] = []
    for label, path in RUNS.items():
        rows.extend(analyze_run(label, path))
    detail = pd.DataFrame(rows)
    agg = aggregate(detail)
    write_reports(detail, agg)
    print(f"Wrote {OUT_DIR}")
    show = agg[(agg["min_duration_s"] == 60)][
        [
            "run",
            "metric",
            "trigger_seconds",
            "idle_time_in_trigger",
            "active_but_execution_stale_s",
            "theta_supplement_share_raw",
            "lowrisk_trigger_seconds",
        ]
    ].sort_values(["run", "metric"])
    print(_md_table(show, list(show.columns)))


if __name__ == "__main__":
    main()
