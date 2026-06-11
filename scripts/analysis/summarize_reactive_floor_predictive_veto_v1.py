from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/wind_prediction/reactive_floor_predictive_veto_v1"

RUNS = {
    "guard10_baseline": ROOT / "outputs/wind_prediction/tune_baseline_learned_guard10",
    "guard10_stale_guarded": ROOT
    / "outputs/wind_prediction/stale_active_target_refresh_v1/guard10_refresh_only_guarded_v1",
    "guard10_floor_A_max_axis": OUT / "guard10_floor_A_max_axis",
    "guard10_floor_B_hybrid": OUT / "guard10_floor_B_hybrid",
    "broader20_baseline": ROOT
    / "outputs/wind_prediction/f60_relief_e15_holdout_validation_v1/baseline_learned_2h",
    "broader20_stale_guarded": ROOT
    / "outputs/wind_prediction/stale_active_target_refresh_v1/broader20_refresh_only_guarded_v1",
    "broader20_floor_A_max_axis": OUT / "broader20_floor_A_max_axis",
    "broader20_floor_B_hybrid": OUT / "broader20_floor_B_hybrid",
}


def theta_total_deg(pitch_deg, roll_deg):
    p = np.deg2rad(np.asarray(pitch_deg, dtype=float))
    r = np.deg2rad(np.asarray(roll_deg, dtype=float))
    return np.rad2deg(np.arctan(np.sqrt(np.tan(p) ** 2 + np.tan(r) ** 2)))


def case_key(path: Path) -> str:
    name = path.name
    for suffix in (
        "_prediction_primary_econ_timeseries.csv",
        "_prediction_primary_econ_planner_log.csv",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def case_short(key: str) -> str:
    parts = key.split("_")
    if parts and parts[0].isdigit():
        return "_".join(parts[1:-2]) if len(parts) > 3 else "_".join(parts[1:])
    return key


def fallback_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        lc = c.lower()
        if "fallback_active" in lc or "safety_fallback" in lc:
            return c
    return None


def target_err(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ("err_tank1_kg", "err_tank2_kg", "err_tank3_kg") if c in df.columns]
    if not cols:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return df[cols].abs().mean(axis=1)


def load_logs(run_dir: Path) -> dict[str, pd.DataFrame]:
    log_dir = run_dir / "planner_logs"
    if not log_dir.exists():
        return {}
    out = {}
    for p in log_dir.glob("*_planner_log.csv"):
        out[case_key(p)] = pd.read_csv(p, low_memory=False)
    return out


def summarize_run(label: str, run_dir: Path) -> tuple[list[dict], list[dict]]:
    ts_dir = run_dir / "timeseries"
    if not ts_dir.exists():
        return [], []
    logs = load_logs(run_dir)
    cases = []
    events = []
    for p in sorted(ts_dir.glob("*_timeseries.csv")):
        key = case_key(p)
        ts = pd.read_csv(p, low_memory=False)
        t_col = "t_s" if "t_s" in ts.columns else "time_s"
        ts["_t"] = ts[t_col].round().astype(int)
        pitch = ts["pitch_deg"].abs().astype(float)
        roll = ts["roll_deg"].abs().astype(float) if "roll_deg" in ts.columns else pd.Series(0.0, index=ts.index)
        max_axis = np.maximum(pitch, roll)
        theta = theta_total_deg(pitch, roll)
        pump = ts["pump_total_rate_m3_min"].abs() if "pump_total_rate_m3_min" in ts.columns else pd.Series(0.0, index=ts.index)
        idle = pump < 0.05
        fb = ts[fallback_col(ts)].astype(float) > 0.5 if fallback_col(ts) else pd.Series(False, index=ts.index)
        high = max_axis > 5.0
        theta_high = theta > 5.0
        terr = target_err(ts)
        log = logs.get(key, pd.DataFrame())
        floor_count = 0
        floor_veto = 0
        stale_seconds = 0
        if not log.empty:
            floor_count = int(log.get("reactive_floor_active", pd.Series(dtype=float)).fillna(0).astype(float).sum())
            floor_veto = int(log.get("reactive_floor_veto_active", pd.Series(dtype=float)).fillna(0).astype(float).sum())
            if "bucket" in log.columns:
                bucket_by_t = (ts["_t"] // 600).astype(int)
                for _, row in log.iterrows():
                    b = int(row["bucket"])
                    idx = (bucket_by_t == b) & high
                    if not idx.any():
                        continue
                    action = str(row.get("first_action", ""))
                    active = action in ("active_small", "active_medium")
                    reused = float(row.get("prediction_primary_target_reused", 0) or 0) > 0.5
                    if active and reused and float(terr[idx].mean()) <= 500 and float(idle[idx].mean()) >= 0.8:
                        stale_seconds += int(idx.sum())
        cases.append(
            {
                "run": label,
                "case": key,
                "case_short": case_short(key),
                "lowrisk": int("lowrisk" in key),
                "pump_m3": float((pump / 60.0).sum()),
                "pitch_p95": float(np.percentile(pitch, 95)),
                "roll_p95": float(np.percentile(roll, 95)),
                "max_p95": float(max(np.percentile(pitch, 95), np.percentile(roll, 95))),
                "sum_fb_pct": float(fb.mean() * 100.0),
                "time_over_5": int(high.sum()),
                "idle_time_over_5": int((high & idle).sum()),
                "theta_total_time_over_5": int(theta_high.sum()),
                "theta_total_idle_time_over_5": int((theta_high & idle).sum()),
                "active_but_execution_stale": int(stale_seconds),
                "floor_trigger_count": floor_count,
                "veto_count": floor_veto,
            }
        )
        if not log.empty and "reactive_floor_active" in log.columns:
            cols = [
                c
                for c in log.columns
                if c.startswith("reactive_floor")
                or c in ("bucket", "current_time_s", "first_action", "prediction_primary_target_reused")
            ]
            ev = log[cols].copy()
            ev["run"] = label
            ev["case"] = key
            events.append(ev)
    return cases, events


def md_table(df: pd.DataFrame, cols: list[str], digits: int = 2) -> str:
    if df.empty:
        return "(empty)"
    v = df[cols].copy()
    for c in v.columns:
        if pd.api.types.is_float_dtype(v[c]):
            v[c] = v[c].map(lambda x: f"{x:.{digits}f}")
    headers = [str(c) for c in v.columns]
    data = [[str(x) for x in row] for row in v.to_numpy().tolist()]
    widths = [max(len(headers[i]), *(len(row[i]) for row in data)) if data else len(headers[i]) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in data)
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    case_rows = []
    event_frames = []
    for label, path in RUNS.items():
        cases, events = summarize_run(label, path)
        case_rows.extend(cases)
        event_frames.extend(events)
    case_df = pd.DataFrame(case_rows)
    event_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    case_df.to_csv(OUT / "reactive_floor_case_table.csv", index=False)
    event_df.to_csv(OUT / "reactive_floor_event_table.csv", index=False)
    agg = (
        case_df.groupby("run", as_index=False)
        .agg(
            cases=("case", "nunique"),
            sum_pump=("pump_m3", "sum"),
            max_p95=("max_p95", "max"),
            pitch_p95=("pitch_p95", "max"),
            roll_p95=("roll_p95", "max"),
            sum_fb=("sum_fb_pct", "sum"),
            time_over_5=("time_over_5", "sum"),
            idle_time_over_5=("idle_time_over_5", "sum"),
            theta_total_time_over_5=("theta_total_time_over_5", "sum"),
            theta_total_idle_time_over_5=("theta_total_idle_time_over_5", "sum"),
            active_but_execution_stale=("active_but_execution_stale", "sum"),
            floor_trigger_count=("floor_trigger_count", "sum"),
            veto_count=("veto_count", "sum"),
            lowrisk_pump=("pump_m3", lambda s: float(s[case_df.loc[s.index, "lowrisk"].astype(bool)].sum())),
            lowrisk_fb=("sum_fb_pct", lambda s: float(s[case_df.loc[s.index, "lowrisk"].astype(bool)].sum())),
        )
    )
    order = [
        "guard10_baseline",
        "guard10_stale_guarded",
        "guard10_floor_A_max_axis",
        "guard10_floor_B_hybrid",
        "broader20_baseline",
        "broader20_stale_guarded",
        "broader20_floor_A_max_axis",
        "broader20_floor_B_hybrid",
    ]
    agg["run"] = pd.Categorical(agg["run"], categories=order, ordered=True)
    agg = agg.sort_values("run")
    agg.to_csv(OUT / "reactive_floor_compare_table.csv", index=False)

    cols = [
        "run",
        "sum_pump",
        "max_p95",
        "sum_fb",
        "time_over_5",
        "idle_time_over_5",
        "theta_total_time_over_5",
        "theta_total_idle_time_over_5",
        "active_but_execution_stale",
        "floor_trigger_count",
        "veto_count",
        "lowrisk_pump",
        "lowrisk_fb",
    ]
    summary = [
        "# Reactive floor predictive veto v1",
        "",
        "Default-off probe. Metrics are computed from existing run outputs after running A=max_axis and B=hybrid variants.",
        "",
        md_table(agg, cols),
    ]
    (OUT / "reactive_floor_summary.md").write_text("\n".join(summary), encoding="utf-8")

    decision = [
        "# Reactive floor decision",
        "",
        "This file is generated from aggregate metrics. Interpret A/B against baseline and stale-active-target guarded candidate.",
        "",
        md_table(agg, cols),
        "",
        "Decision checklist:",
        "1. Does reactive floor reduce high-posture idle?",
        "2. Does it reduce active_but_execution_stale?",
        "3. Is pump increase acceptable?",
        "4. Does veto allow only justified relief?",
        "5. Is hybrid steadier than max_axis?",
        "6. Should this become the new PP foundation or remain a default-off probe?",
    ]
    (OUT / "reactive_floor_decision.md").write_text("\n".join(decision), encoding="utf-8")
    print(md_table(agg, cols))


if __name__ == "__main__":
    main()
