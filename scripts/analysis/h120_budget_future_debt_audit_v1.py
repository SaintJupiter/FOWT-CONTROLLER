#!/usr/bin/env python3
"""Read-only audit for forecast-justifying economy-budget holds.

The blind economy budget can produce large pump reduction, but its decisions are
not forecast-attributable.  This audit inspects those hold decisions after the
fact and asks whether a richer "future posture debt" justification could have
explained them without relying on budget exhaustion alone.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


CASE_SUFFIXES = (
    "_prediction_primary_econ_planner_log.csv",
    "_prediction_primary_econ_timeseries.csv",
)


def case_id(path: Path) -> str:
    name = path.name
    for suffix in CASE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return re.sub(r"_prediction_primary_.*$", "", name)


def m3_from_rate(df: pd.DataFrame) -> float:
    if df.empty or "pump_total_rate_m3_min" not in df.columns:
        return 0.0
    return float(np.nansum(np.abs(df["pump_total_rate_m3_min"].to_numpy(dtype=float))) / 60.0)


def max_axis_series(df: pd.DataFrame) -> np.ndarray:
    pitch = df.get("pitch_deg", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    roll = df.get("roll_deg", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    return np.maximum(np.abs(pitch), np.abs(roll))


def window_metrics(ts: pd.DataFrame, start_s: float, horizon_s: int) -> dict[str, float]:
    if ts.empty:
        return {
            "pump_m3": 0.0,
            "time_over5_s": 0.0,
            "idle_over5_s": 0.0,
            "p95_max_axis_deg": 0.0,
            "max_axis_max_deg": 0.0,
            "debt_over3_deg_s": 0.0,
            "debt_over4_deg_s": 0.0,
            "debt_over45_deg_s": 0.0,
            "max_axis_start_deg": 0.0,
            "max_axis_end_deg": 0.0,
            "max_axis_trend_deg": 0.0,
        }
    t = ts["t_s"].to_numpy(dtype=float)
    mask = (t >= float(start_s)) & (t < float(start_s) + float(horizon_s))
    w = ts.loc[mask]
    if w.empty:
        return window_metrics(pd.DataFrame(columns=ts.columns), start_s, horizon_s)
    max_axis = max_axis_series(w)
    pump = (
        w.get("pump_total_rate_m3_min", pd.Series(np.zeros(len(w))))
        .to_numpy(dtype=float)
    )
    return {
        "pump_m3": m3_from_rate(w),
        "time_over5_s": float(np.sum(max_axis > 5.0)),
        "idle_over5_s": float(np.sum((max_axis > 5.0) & (np.abs(pump) < 0.5))),
        "p95_max_axis_deg": float(np.nanpercentile(max_axis, 95)) if max_axis.size else 0.0,
        "max_axis_max_deg": float(np.nanmax(max_axis)) if max_axis.size else 0.0,
        "debt_over3_deg_s": float(np.nansum(np.maximum(max_axis - 3.0, 0.0))),
        "debt_over4_deg_s": float(np.nansum(np.maximum(max_axis - 4.0, 0.0))),
        "debt_over45_deg_s": float(np.nansum(np.maximum(max_axis - 4.5, 0.0))),
        "max_axis_start_deg": float(max_axis[0]) if max_axis.size else 0.0,
        "max_axis_end_deg": float(max_axis[-1]) if max_axis.size else 0.0,
        "max_axis_trend_deg": float(max_axis[-1] - max_axis[0]) if max_axis.size else 0.0,
    }


def load_case_map(run_dir: Path, subdir: str, pattern: str) -> dict[str, Path]:
    base = run_dir / subdir
    return {case_id(p): p for p in sorted(base.glob(pattern))}


def pressure_features(row: pd.Series) -> dict[str, float | int | str]:
    near0 = float(row.get("raw_pressure_block0_norm", np.nan))
    near1 = float(row.get("raw_pressure_block1_norm", np.nan))
    near2 = float(row.get("raw_pressure_block2_norm", np.nan))
    near_max = np.nanmax([near0, near1, near2])
    far60 = float(row.get("far_horizon_norm_0_20", np.nan))
    far80 = float(row.get("far_horizon_norm_20_40", np.nan))
    far100 = float(row.get("far_horizon_norm_40_60", np.nan))
    far_max = np.nanmax([far60, far80, far100])
    relief_decay = bool(np.isfinite(near0) and np.isfinite(near2) and near2 <= 0.7 and (near0 - near2) >= 0.2)
    no_imminent_worsening = bool(np.isfinite(near_max) and near_max < 0.9)
    reintensify = bool(np.isfinite(near0) and np.isfinite(near2) and near2 >= 0.9 and near2 > near0 + 0.05)
    return {
        "near0_norm": near0,
        "near1_norm": near1,
        "near2_norm": near2,
        "near_max_norm": float(near_max) if np.isfinite(near_max) else np.nan,
        "far_0_20_norm": far60,
        "far_20_40_norm": far80,
        "far_40_60_norm": far100,
        "far_0_60_max_norm": float(far_max) if np.isfinite(far_max) else np.nan,
        "pressure_relief_decay": int(relief_decay),
        "pressure_no_imminent_worsening": int(no_imminent_worsening),
        "pressure_reintensify_veto": int(reintensify),
    }


def classify_hold(row: dict[str, float | int | str]) -> tuple[str, str]:
    pump_saved_30 = float(row["delta_pump_m3_1800s"]) > 5.0
    hard_ok_30 = (
        float(row["delta_time_over5_s_1800s"]) <= 0.0
        and float(row["delta_idle_over5_s_1800s"]) <= 0.0
    )
    p95_ok_30 = float(row["delta_p95_max_axis_deg_1800s"]) <= 0.10
    future_a0_declines = float(row["a0_max_axis_trend_deg_1800s"]) <= -0.05
    future_a0_stable = float(row["a0_max_axis_trend_deg_1800s"]) <= 0.05
    no_floor_risk = float(row["budget_time_over5_s_1800s"]) <= 0.0
    pressure_relieves = bool(int(row["pressure_relief_decay"]))
    pressure_safe = bool(int(row["pressure_no_imminent_worsening"]))
    pressure_veto = bool(int(row["pressure_reintensify_veto"]))

    if pump_saved_30 and hard_ok_30 and p95_ok_30:
        if pressure_relieves:
            return "forecast_relief_justified", "pump_saved_hard_ok_p95_ok_pressure_relief"
        if (future_a0_declines or future_a0_stable) and pressure_safe and not pressure_veto:
            return "future_debt_justified", "pump_saved_hard_ok_future_debt_stable_pressure_safe"
        if future_a0_declines or future_a0_stable:
            return "oracle_debt_only", "pump_saved_hard_ok_future_debt_stable_no_pressure_reason"
        return "saved_but_unjustified", "pump_saved_hard_ok_no_forecast_or_debt_reason"
    if pump_saved_30 and not hard_ok_30:
        return "saved_with_hard_safety_cost", "pump_saved_but_time_or_idle_over5_regressed"
    if pump_saved_30 and not p95_ok_30:
        return "saved_with_comfort_cost", "pump_saved_hard_ok_but_p95_regressed"
    if float(row["delta_pump_m3_1800s"]) < -5.0:
        return "catchup_or_more_pump", "budget_arm_used_more_pump_after_hold"
    if no_floor_risk and pressure_safe:
        return "neutral_safe_hold", "little_pump_effect_but_no_floor_risk"
    return "neutral_or_ambiguous", "little_pump_effect_or_mixed_metrics"


def audit_group(root: Path, group: str, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_root = root / group
    a0 = run_root / "a0_baseline"
    b = run_root / arm
    logs = load_case_map(b, "planner_logs", "*planner_log.csv")
    a0_ts_map = load_case_map(a0, "timeseries", "*timeseries.csv")
    b_ts_map = load_case_map(b, "timeseries", "*timeseries.csv")
    rows: list[dict[str, float | int | str]] = []
    for cid, log_path in logs.items():
        if cid not in a0_ts_map or cid not in b_ts_map:
            continue
        log = pd.read_csv(log_path, low_memory=False)
        active = log.get("economy_pump_budget_active", pd.Series(np.zeros(len(log)))).to_numpy(dtype=float) > 0.5
        hold_rows = log.loc[active].copy()
        if hold_rows.empty:
            continue
        a0_ts = pd.read_csv(a0_ts_map[cid], low_memory=False)
        b_ts = pd.read_csv(b_ts_map[cid], low_memory=False)
        for _, r in hold_rows.iterrows():
            t = float(r.get("current_time_s", np.nan))
            if not np.isfinite(t):
                continue
            out: dict[str, float | int | str] = {
                "group": group,
                "arm": arm,
                "case_id": cid,
                "current_time_s": t,
                "budget_reason": str(r.get("economy_pump_budget_reason", "")),
                "budget_used_m3": float(r.get("economy_pump_budget_used_m3", np.nan)),
                "budget_remaining_m3": float(r.get("economy_pump_budget_remaining_m3", np.nan)),
                "blocked_delta_mean_kg": float(r.get("economy_pump_budget_blocked_delta_mean_kg", np.nan)),
                "current_pitch_deg": float(r.get("current_pitch_deg", r.get("planner_action_pitch_deg", np.nan))),
                "current_roll_deg": float(r.get("current_roll_deg", r.get("planner_action_roll_deg", np.nan))),
            }
            out["current_max_axis_deg"] = max(abs(float(out["current_pitch_deg"])), abs(float(out["current_roll_deg"])))
            out.update(pressure_features(r))
            for horizon in (600, 1200, 1800):
                a = window_metrics(a0_ts, t, horizon)
                bb = window_metrics(b_ts, t, horizon)
                for k, v in a.items():
                    out[f"a0_{k}_{horizon}s"] = v
                for k, v in bb.items():
                    out[f"budget_{k}_{horizon}s"] = v
                out[f"delta_pump_m3_{horizon}s"] = a["pump_m3"] - bb["pump_m3"]
                out[f"delta_time_over5_s_{horizon}s"] = bb["time_over5_s"] - a["time_over5_s"]
                out[f"delta_idle_over5_s_{horizon}s"] = bb["idle_over5_s"] - a["idle_over5_s"]
                out[f"delta_p95_max_axis_deg_{horizon}s"] = bb["p95_max_axis_deg"] - a["p95_max_axis_deg"]
                out[f"delta_debt_over3_deg_s_{horizon}s"] = bb["debt_over3_deg_s"] - a["debt_over3_deg_s"]
                out[f"delta_debt_over4_deg_s_{horizon}s"] = bb["debt_over4_deg_s"] - a["debt_over4_deg_s"]
            label, reason = classify_hold(out)
            out["future_debt_label"] = label
            out["future_debt_reason"] = reason
            rows.append(out)
    hold_table = pd.DataFrame(rows)
    if hold_table.empty:
        return hold_table, pd.DataFrame()
    case_summary = (
        hold_table.groupby(["group", "arm", "case_id"], dropna=False)
        .agg(
            hold_rows=("case_id", "size"),
            saved_pump_30m=("delta_pump_m3_1800s", "sum"),
            delta_time_over5_30m=("delta_time_over5_s_1800s", "sum"),
            delta_idle_over5_30m=("delta_idle_over5_s_1800s", "sum"),
            max_delta_p95_30m=("delta_p95_max_axis_deg_1800s", "max"),
            forecast_relief_justified=("future_debt_label", lambda s: int((s == "forecast_relief_justified").sum())),
            future_debt_justified=("future_debt_label", lambda s: int((s == "future_debt_justified").sum())),
            saved_with_hard_safety_cost=("future_debt_label", lambda s: int((s == "saved_with_hard_safety_cost").sum())),
            saved_but_unjustified=("future_debt_label", lambda s: int((s == "saved_but_unjustified").sum())),
        )
        .reset_index()
    )
    return hold_table, case_summary


def write_summary(out_dir: Path, hold_table: pd.DataFrame, case_summary: pd.DataFrame) -> None:
    paper = out_dir / "paper_ready"
    paper.mkdir(parents=True, exist_ok=True)
    if hold_table.empty:
        text = "# Future Posture-Debt Justification Audit\n\nNo active budget holds were found.\n"
        (paper / "future_debt_justification_summary.md").write_text(text)
        (out_dir / "decision.md").write_text(text)
        return
    label_counts = (
        hold_table.groupby(["group", "future_debt_label"]).size().reset_index(name="rows")
    )
    group_summary = (
        hold_table.groupby("group")
        .agg(
            hold_rows=("case_id", "size"),
            cases=("case_id", "nunique"),
            pump_saved_30m=("delta_pump_m3_1800s", "sum"),
            delta_time_over5_30m=("delta_time_over5_s_1800s", "sum"),
            delta_idle_over5_30m=("delta_idle_over5_s_1800s", "sum"),
            mean_delta_p95_30m=("delta_p95_max_axis_deg_1800s", "mean"),
            forecast_relief_rows=("pressure_relief_decay", "sum"),
            no_imminent_rows=("pressure_no_imminent_worsening", "sum"),
            reintensify_veto_rows=("pressure_reintensify_veto", "sum"),
        )
        .reset_index()
    )
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_empty_"
        cols = list(df.columns)
        rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, r in df.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(str(v))
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join(rows)

    justified_mask = hold_table["future_debt_label"].isin(
        ["forecast_relief_justified", "future_debt_justified", "oracle_debt_only"]
    )
    justified_rows = int(justified_mask.sum())
    total_rows = int(len(hold_table))
    justified_pump = float(hold_table.loc[justified_mask, "delta_pump_m3_1800s"].sum())
    total_pump = float(hold_table["delta_pump_m3_1800s"].sum())
    lines = [
        "# Future Posture-Debt Justification Audit",
        "",
        "This is a read-only audit of the blind `budget100` holds. It asks whether the large pump-saving holds can be explained by richer forecast/state evidence, especially whether future posture debt naturally declines or remains bounded after the hold.",
        "",
        "## Key Numbers",
        "",
        f"- Active hold rows inspected: `{total_rows}`.",
        f"- Rows with a plausible forecast/debt justification: `{justified_rows}` (`{justified_rows / max(total_rows, 1):.1%}`).",
        f"- 30-minute pump delta over inspected holds: `{total_pump:.2f} m3`.",
        f"- Pump delta carried by plausibly justified holds: `{justified_pump:.2f} m3`.",
        "",
        "## Group Summary",
        "",
        md_table(group_summary),
        "",
        "## Label Distribution",
        "",
        md_table(label_counts),
        "",
        "## Interpretation",
        "",
    ]
    if justified_rows == 0 or justified_pump <= 0:
        lines += [
            "The current blind 20% budget holds do not appear recoverable by a simple future-debt justification. This means the large saving is mostly a comfort-budget tradeoff rather than a forecast-attributable allocator result.",
        ]
    elif justified_pump < 0.35 * max(total_pump, 1e-9):
        lines += [
            "There is some forecast/debt signal, but it explains only a minority of the blind-budget saving. A next controller should not try to copy the whole 20% budget behavior; it should selectively recover the justified subset and keep the remainder as an explicit comfort tradeoff.",
        ]
    else:
        lines += [
            "A meaningful fraction of the blind-budget saving has a plausible future-debt explanation. This supports a next allocator mode based on predicted posture-debt decline, with budget-only behavior kept as a negative control.",
        ]
    lines += [
        "",
        "## Next Controller Implication",
        "",
        "Do not use `budget_exhausted_hold` as a paper claim. The next candidate should require a logged reason such as `future_debt_declines`, `future_debt_stable_pressure_safe`, or `near_relief_decay`, and it should explicitly veto re-intensification and floor-risk windows.",
        "",
    ]
    text = "\n".join(lines)
    (paper / "future_debt_justification_summary.md").write_text(text)
    (out_dir / "decision.md").write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1",
    )
    ap.add_argument("--groups", nargs="+", default=["guard10_3600", "broader20_3600"])
    ap.add_argument("--arm", default="budget100")
    ap.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1/debt_justification_audit_v1",
    )
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    raw = out_dir / "raw_tables"
    raw.mkdir(parents=True, exist_ok=True)
    all_holds = []
    all_cases = []
    for group in args.groups:
        hold_table, case_summary = audit_group(root, group, args.arm)
        if not hold_table.empty:
            all_holds.append(hold_table)
        if not case_summary.empty:
            all_cases.append(case_summary)
    hold = pd.concat(all_holds, ignore_index=True) if all_holds else pd.DataFrame()
    cases = pd.concat(all_cases, ignore_index=True) if all_cases else pd.DataFrame()
    hold.to_csv(raw / "future_debt_hold_table.csv", index=False)
    cases.to_csv(raw / "future_debt_case_summary.csv", index=False)
    write_summary(out_dir, hold, cases)
    print(f"Wrote {len(hold)} hold rows to {raw / 'future_debt_hold_table.csv'}")
    print(f"Wrote summary to {out_dir / 'decision.md'}")


if __name__ == "__main__":
    main()
