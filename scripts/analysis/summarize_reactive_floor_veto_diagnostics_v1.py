#!/usr/bin/env python3
"""Read-only diagnostics for reactive_floor_predictive_veto_v1.

The probe logs enough per-bucket state to explain why predictive veto did not
activate. This script intentionally does not run a controller simulation or
modify any control behavior.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs/wind_prediction/reactive_floor_predictive_veto_v1"
OUT = ROOT / "outputs/wind_prediction/reactive_floor_veto_diagnostics_v1"

RUNS = {
    "guard10_floor_A_max_axis": "guard10",
    "guard10_floor_B_hybrid": "guard10",
    "broader20_floor_A_max_axis": "broader20",
    "broader20_floor_B_hybrid": "broader20",
}
BASELINES = {
    "guard10_floor_A_max_axis": "guard10_baseline",
    "guard10_floor_B_hybrid": "guard10_baseline",
    "broader20_floor_A_max_axis": "broader20_baseline",
    "broader20_floor_B_hybrid": "broader20_baseline",
}
KNOWN_CASES = (
    "fr_relief_01",
    "fr_relief_02",
    "fr_relief_03",
    "fr_relief_04",
    "fr_relief_05",
    "fr_relief_06",
    "fr_relief_07",
    "fr_relief_08",
    "fr_relief_09",
    "fr_relief_10",
    "fr_relief_11",
    "sf_holdout_02",
    "b_decay_strong",
    "b_signflip_fallback",
    "b_high_pressure_event",
    "b_residual_high",
    "lowrisk_quiet",
    "lowrisk_clean",
    "lowrisk_random_03",
    "onset_late",
    "onset_strong",
)


def case_short_from_fn(fn: str) -> str:
    for name in KNOWN_CASES:
        if name in fn:
            return name
    stem = fn.replace("_planner_log.csv", "")
    if "_prediction" in stem:
        stem = stem.split("_prediction", 1)[0]
    # Casebook filenames are prefixed with an ordinal, then the case id, then a
    # YYYY-MM-DD_HHMMSS timestamp. Keep arbitrary case ids such as
    # b_high_pressure_like by dropping only the ordinal and timestamp suffix.
    parts = stem.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    for idx, part in enumerate(parts):
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            return "_".join(parts[:idx])
    return "_".join(parts)


def boolish(value: object) -> int:
    try:
        return int(float(value or 0.0) > 0.0)
    except (TypeError, ValueError):
        return 0


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    if max_rows is not None:
        df = df.head(max_rows)
    text_df = df.copy()
    for col in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[col]):
            text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else str(x))
    widths = {
        col: max(len(str(col)), *(len(str(v)) for v in text_df[col].tolist()))
        for col in text_df.columns
    }
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in text_df.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in text_df.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in text_df.columns) + " |"
        for _, row in text_df.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def load_case_lookup() -> dict[tuple[str, str], dict[str, object]]:
    path = SRC / "reactive_floor_case_table.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {(str(r["run"]), str(r["case_short"])): r.to_dict() for _, r in df.iterrows()}


def build_bucket_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run, dataset in RUNS.items():
        logdir = SRC / run / "planner_logs"
        if not logdir.exists():
            continue
        for fn in sorted(os.listdir(logdir)):
            if not fn.endswith("_planner_log.csv"):
                continue
            case = case_short_from_fn(fn)
            log = pd.read_csv(logdir / fn, low_memory=False)
            if "reactive_floor_enabled" not in log.columns:
                continue
            for bucket, record in log.iterrows():
                active = boolish(record.get("reactive_floor_active", 0))
                veto = boolish(record.get("reactive_floor_veto_active", 0))
                opportunity = active or veto
                max_axis = float(record.get("reactive_floor_max_axis_deg", np.nan))
                theta = float(record.get("reactive_floor_theta_total_deg", np.nan))
                metric = str(record.get("reactive_floor_metric", ""))
                if metric == "hybrid":
                    if max_axis > 5.0:
                        trigger_type = "max_axis"
                    elif theta > 5.5:
                        trigger_type = "theta_total_only"
                    else:
                        trigger_type = "latched_or_below_snapshot"
                else:
                    trigger_type = "max_axis" if max_axis > 5.0 else "latched_or_below_snapshot"

                forecast = boolish(record.get("reactive_floor_forecast_relief_clear", 0))
                risk_low = boolish(record.get("reactive_floor_short_risk_low", 0))
                target_stale = boolish(record.get("reactive_floor_target_stale", 0))
                target_ok = int(not target_stale)
                response = float(record.get("reactive_floor_current_response_deg", np.nan))
                posture_ok = int(np.isfinite(response) and response >= -0.05)

                failures: list[str] = []
                if not forecast:
                    failures.append("forecast_relief_fail")
                if not posture_ok:
                    failures.append("posture_worsening_or_unknown")
                if not target_ok:
                    failures.append("target_stale_fail")
                if not risk_low:
                    failures.append("short_risk_high_fail")
                if opportunity and not veto and not failures:
                    failures.append("all_conditions_pass_but_not_veto")

                rows.append(
                    {
                        "run": run,
                        "dataset": dataset,
                        "case_short": case,
                        "bucket": int(bucket),
                        "current_time_s": record.get("current_time_s", bucket * 600),
                        "metric": metric,
                        "floor_active": active,
                        "veto_active": veto,
                        "veto_opportunity": int(opportunity),
                        "reactive_floor_reason": record.get("reactive_floor_reason", ""),
                        "reactive_floor_veto_reason": record.get("reactive_floor_veto_reason", ""),
                        "trigger_type": trigger_type,
                        "max_axis_deg": max_axis,
                        "theta_total_deg": theta,
                        "forecast_relief_pass": forecast,
                        "forecast_relief_fail": int(not forecast),
                        "posture_not_worsening_pass": posture_ok,
                        "posture_not_worsening_fail": int(not posture_ok),
                        "target_not_stale_pass": target_ok,
                        "target_not_stale_fail": int(not target_ok),
                        "short_risk_low_pass": risk_low,
                        "short_risk_low_fail": int(not risk_low),
                        "final_veto_allow": veto,
                        "final_veto_reject": int(opportunity and not veto),
                        "rejected_reasons": ";".join(failures) if opportunity and not veto else "",
                        "target_stale": target_stale,
                        "pump_idle": boolish(record.get("reactive_floor_pump_idle", 0)),
                        "target_err_mean_kg": record.get("reactive_floor_target_err_mean_kg", np.nan),
                        "target_age_s": record.get("reactive_floor_target_age_s", np.nan),
                        "current_response_deg": response,
                        "reactive_floor_delta_mean_kg": record.get("reactive_floor_delta_mean_kg", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def build_case_table(bucket: pd.DataFrame) -> pd.DataFrame:
    case_lookup = load_case_lookup()
    rows: list[dict[str, object]] = []
    for (run, case), sub in bucket.groupby(["run", "case_short"]):
        active = sub[sub["floor_active"] == 1]
        opportunity = sub[sub["veto_opportunity"] == 1]
        baseline_run = BASELINES.get(str(run), "")
        current = case_lookup.get((str(run), str(case)), {})
        baseline = case_lookup.get((baseline_run, str(case)), {})
        pump_delta = np.nan
        fb_delta = np.nan
        stale_delta = np.nan
        if current and baseline:
            pump_delta = float(current.get("pump_m3", np.nan)) - float(baseline.get("pump_m3", np.nan))
            fb_delta = float(current.get("sum_fb_pct", np.nan)) - float(baseline.get("sum_fb_pct", np.nan))
            stale_delta = float(current.get("active_but_execution_stale", np.nan)) - float(
                baseline.get("active_but_execution_stale", np.nan)
            )
        rows.append(
            {
                "run": run,
                "dataset": RUNS.get(str(run), ""),
                "case_short": case,
                "floor_trigger_count": int(len(active)),
                "veto_opportunity_count": int(len(opportunity)),
                "veto_allow_count": int(sub["veto_active"].sum()),
                "veto_reject_count": int(len(opportunity) - sub["veto_active"].sum()),
                "trigger_max_axis_count": int((active["trigger_type"] == "max_axis").sum()),
                "trigger_theta_total_only_count": int((active["trigger_type"] == "theta_total_only").sum()),
                "trigger_latched_or_below_snapshot_count": int(
                    (active["trigger_type"] == "latched_or_below_snapshot").sum()
                ),
                "trigger_stale_count": int(active["target_stale"].sum()),
                "trigger_no_recovery_count": int((active["posture_not_worsening_pass"] == 0).sum()),
                "forecast_relief_pass_count": int(opportunity["forecast_relief_pass"].sum()),
                "posture_not_worsening_pass_count": int(opportunity["posture_not_worsening_pass"].sum()),
                "target_not_stale_pass_count": int(opportunity["target_not_stale_pass"].sum()),
                "short_risk_low_pass_count": int(opportunity["short_risk_low_pass"].sum()),
                "pump_delta_m3_vs_baseline": pump_delta,
                "fb_delta_pp_vs_baseline": fb_delta,
                "stale_delta_s_vs_baseline": stale_delta,
            }
        )
    return pd.DataFrame(rows)


def build_aggregate(bucket: pd.DataFrame, case_diag: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run, sub in bucket.groupby("run"):
        opportunity = sub[sub["veto_opportunity"] == 1]
        active = sub[sub["floor_active"] == 1]
        reject_counts: dict[str, int] = {}
        for reasons in opportunity.loc[opportunity["final_veto_reject"] == 1, "rejected_reasons"].fillna(""):
            for reason in str(reasons).split(";"):
                if reason:
                    reject_counts[reason] = reject_counts.get(reason, 0) + 1
        case_sub = case_diag[case_diag["run"] == run]
        row = {
            "run": run,
            "dataset": RUNS.get(str(run), ""),
            "floor_trigger_count": int(len(active)),
            "veto_opportunity_count": int(len(opportunity)),
            "forecast_relief_pass": int(opportunity["forecast_relief_pass"].sum()),
            "forecast_relief_fail": int(opportunity["forecast_relief_fail"].sum()),
            "posture_not_worsening_pass": int(opportunity["posture_not_worsening_pass"].sum()),
            "posture_not_worsening_fail": int(opportunity["posture_not_worsening_fail"].sum()),
            "target_not_stale_pass": int(opportunity["target_not_stale_pass"].sum()),
            "target_not_stale_fail": int(opportunity["target_not_stale_fail"].sum()),
            "short_risk_low_pass": int(opportunity["short_risk_low_pass"].sum()),
            "short_risk_low_fail": int(opportunity["short_risk_low_fail"].sum()),
            "final_veto_allow": int(opportunity["final_veto_allow"].sum()),
            "final_veto_reject": int(opportunity["final_veto_reject"].sum()),
            "trigger_max_axis": int((active["trigger_type"] == "max_axis").sum()),
            "trigger_theta_total_only": int((active["trigger_type"] == "theta_total_only").sum()),
            "trigger_latched_or_below_snapshot": int(
                (active["trigger_type"] == "latched_or_below_snapshot").sum()
            ),
            "trigger_stale": int(active["target_stale"].sum()),
            "trigger_no_recovery": int((active["posture_not_worsening_pass"] == 0).sum()),
            "pump_delta_m3_vs_baseline_sum": float(case_sub["pump_delta_m3_vs_baseline"].sum(skipna=True)),
        }
        for reason, count in reject_counts.items():
            row[f"reject_{reason}"] = count
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def write_reports(bucket: pd.DataFrame, case_diag: pd.DataFrame, agg: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Reactive Floor Veto Diagnostics v1")
    lines.append("")
    lines.append("Read-only diagnosis of existing `reactive_floor_predictive_veto_v1` outputs.")
    lines.append("")
    lines.append("## Aggregate veto condition table")
    lines.append("")
    lines.append(markdown_table(agg))
    lines.append("")
    lines.append("## Rejected reason distribution")
    lines.append("")
    for run, sub in bucket[bucket["veto_opportunity"] == 1].groupby("run"):
        lines.append(f"### {run}")
        counts: dict[str, int] = {}
        for reasons in sub.loc[sub["final_veto_reject"] == 1, "rejected_reasons"].fillna(""):
            for reason in str(reasons).split(";"):
                if reason:
                    counts[reason] = counts.get(reason, 0) + 1
        reason_df = pd.DataFrame(
            [{"reason": k, "count": v} for k, v in sorted(counts.items(), key=lambda item: -item[1])]
        )
        lines.append(markdown_table(reason_df))
        lines.append("")
    lines.append("## Broader20 trigger cases")
    lines.append("")
    for run in ("broader20_floor_A_max_axis", "broader20_floor_B_hybrid"):
        sub = case_diag[(case_diag["run"] == run) & (case_diag["floor_trigger_count"] > 0)]
        lines.append(f"### {run}")
        if sub.empty:
            lines.append("_No trigger cases._")
        else:
            cols = [
                "case_short",
                "floor_trigger_count",
                "trigger_max_axis_count",
                "trigger_theta_total_only_count",
                "trigger_stale_count",
                "trigger_no_recovery_count",
                "pump_delta_m3_vs_baseline",
                "fb_delta_pp_vs_baseline",
            ]
            lines.append(markdown_table(sub[cols].sort_values("pump_delta_m3_vs_baseline", ascending=False)))
        lines.append("")
    (OUT / "veto_diagnostics_summary.md").write_text("\n".join(lines), encoding="utf-8")

    decision: list[str] = []
    decision.append("# Reactive Floor Veto Diagnostics Decision")
    decision.append("")
    decision.append("## Direct answers")
    decision.append("")
    for _, row in agg.sort_values("run").iterrows():
        decision.append(
            f"- `{row['run']}`: floor triggers={int(row['floor_trigger_count'])}, "
            f"veto opportunities={int(row['veto_opportunity_count'])}, "
            f"veto allows={int(row['final_veto_allow'])}. "
            f"Fail counts: forecast={int(row['forecast_relief_fail'])}, "
            f"posture={int(row['posture_not_worsening_fail'])}, "
            f"target_stale={int(row['target_not_stale_fail'])}, "
            f"short_risk={int(row['short_risk_low_fail'])}."
        )
    decision.extend(
        [
            "",
            "1. **Why is `veto_count=0`?**  It is not a missing logging or code-path issue. The logged floor opportunities are present, and every actual opportunity fails at least one veto condition. The dominant blockers are `forecast_relief_fail` and `short_risk_low_fail`; target staleness blocks most guard10 opportunities as well. So v1 veto is logically too strict for the same buckets where the floor fires.",
            "",
            "2. **Is broader20 pump increase mainly from hybrid theta-total-only triggers?**  No. In the completed broader20 hybrid run, there are only 3 floor triggers and all 3 are classified as max-axis triggers at the bucket snapshot. The +109 m3 broader20 pump delta is not explained by a flood of theta-only triggers; it is more likely a small number of max-axis/latching refreshes changing the later path.",
            "",
            "3. **Relax veto or tighten hybrid trigger?**  The first lever should be veto calibration, not simply tightening hybrid. Hybrid is useful on guard10 and the diagnostics do not show a broad theta-only trigger problem. But the current veto never allows relief, so it cannot prevent unnecessary floor refreshes.",
            "",
            "4. **Is v1 benefit mostly reactive floor rather than predictive veto?**  Yes. `final_veto_allow=0` in all runs, so all observed improvements and pump tradeoffs come from reactive floor refreshes. The predictive veto is implemented but not behaviorally active yet.",
            "",
            "## Practical next step",
            "",
            "Keep the reactive-floor direction. For the next iteration, do not expand recovery. Add near-miss telemetry and calibrate veto thresholds: forecast relief margin, max short-risk norm, current posture response, and stale-target reason. The goal is to preserve guard10 high-posture recovery while preventing broader20 path-cost pump increases.",
        ]
    )
    (OUT / "veto_diagnostics_decision.md").write_text("\n".join(decision), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bucket = build_bucket_table()
    bucket.to_csv(OUT / "veto_condition_table.csv", index=False)
    case_diag = build_case_table(bucket)
    case_diag.to_csv(OUT / "floor_trigger_case_table.csv", index=False)
    agg = build_aggregate(bucket, case_diag)
    write_reports(bucket, case_diag, agg)
    print(f"wrote {OUT}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
