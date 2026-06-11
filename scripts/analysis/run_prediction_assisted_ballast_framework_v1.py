#!/usr/bin/env python3
"""Package the final 0-120min prediction-assisted ballast framework.

This script does not run closed-loop simulations. It consolidates the two
accepted artifacts into a reproducible paper-ready framework summary:

* learned 0-60min relief-economy closed-loop pump saving;
* 60-120min h120 supervisory far-event advisory / monitoring.

The key framing is intentional: h120 is retained in the system architecture as
supervisory advisory and far-horizon monitoring, while the quantified closed-loop
pump saving comes from the learned near-horizon relief economy lever.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

RELIEF_DIR = (
    REPO_ROOT
    / "outputs/wind_prediction/h120_relief_envelope_learned_a1_locked_holdout_v1"
)
ADVISORY_DIR = REPO_ROOT / "outputs/wind_prediction/h120_supervisory_advisory_v1"
OUT_DIR = REPO_ROOT / "outputs/wind_prediction/prediction_assisted_ballast_framework_v1"

FINAL_ARM = "A1_learned_near_envelope_medium_axis_guard"
BASELINE_ARM = "A0_learned_v16"


def _ensure_dirs() -> dict[str, Path]:
    dirs = {
        "out": OUT_DIR,
        "raw": OUT_DIR / "raw_tables",
        "paper": OUT_DIR / "paper_ready",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _case_key(path: Path) -> str:
    name = path.name
    suffix = "_prediction_primary_econ_planner_log.csv"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def _timestamp_from_case(case: str) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{6})", case)
    if not match:
        return ""
    hhmmss = match.group(2)
    return f"{match.group(1)} {hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"


def _load_source_metadata() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest_path = RELIEF_DIR / "locked_casebook_manifest.csv"
    if not manifest_path.exists():
        return {}, {}
    manifest = pd.read_csv(manifest_path)
    by_case = manifest.set_index("case_id").to_dict(orient="index")
    by_ts = manifest.set_index(manifest["timestamp"].astype(str)).to_dict(orient="index")
    return by_case, by_ts


def _safe_count(df: pd.DataFrame, column: str, threshold: float, *, op: str) -> int:
    if column not in df.columns:
        return 0
    series = df[column].fillna(0).astype(float)
    if op == ">=":
        return int((series >= threshold).sum())
    if op == "<=":
        return int((series <= threshold).sum())
    if op == ">":
        return int((series > threshold).sum())
    raise ValueError(f"unsupported op={op!r}")


def _far_horizon_monitor_table(tau_on: float) -> pd.DataFrame:
    by_case, by_ts = _load_source_metadata()
    run_dir = RELIEF_DIR / "runs" / FINAL_ARM / "planner_logs"
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*_planner_log.csv")):
        case = _case_key(path)
        parsed_ts = _timestamp_from_case(case)
        meta = by_case.get(case, {}) or by_ts.get(parsed_ts, {})
        df = pd.read_csv(path, low_memory=False)
        row = {
            "case": case,
            "case_source": meta.get("case_source", ""),
            "source_groups": meta.get("source_groups", ""),
            "timestamp": meta.get("timestamp", parsed_ts),
            "rows": int(len(df)),
            "far_risk_advisory_rows_proxy": _safe_count(
                df, "far_horizon_far_max", tau_on, op=">="
            ),
            "far_relief_advisory_rows_proxy": _safe_count(
                df, "far_horizon_far_min", 0.5, op="<="
            ),
            "far_reintensification_warning_rows": _safe_count(
                df, "far_horizon_hidden_intensification", 0.5, op=">"
            ),
            "direction_shift_monitor_rows": _safe_count(
                df, "far_horizon_direction_shift", 0.5, op=">"
            ),
            "signflip_reversal_monitor_rows": _safe_count(
                df, "far_horizon_reversal", 0.5, op=">"
            ),
            "h120_advisory_control_action_change_count": 0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _framework_summary(
    baseline: pd.Series,
    final: pd.Series,
    advisory_test: dict[str, Any],
    monitor: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer": "v1.6 hard safety floor",
                "horizon": "current/reactive",
                "role": "hard recovery safety baseline",
                "changes_pump_or_target": "yes, existing controller baseline",
                "primary_metric": "kept unchanged",
                "value": "reactive floor/fallback/recovery remain authoritative",
            },
            {
                "layer": "learned relief-economy closed-loop",
                "horizon": "0-60min",
                "role": "delay non-hard single-axis economy refresh under predicted relief",
                "changes_pump_or_target": "yes, bounded economy refresh delay only",
                "primary_metric": "pump_saved_m3",
                "value": round(float(baseline.sum_pump - final.sum_pump), 3),
            },
            {
                "layer": "h120 supervisory advisory",
                "horizon": "60-120min",
                "role": "far-risk/far-event monitoring, operator/planning advisory",
                "changes_pump_or_target": "no",
                "primary_metric": "row_precision/recall/FPR",
                "value": (
                    f"P={float(advisory_test['precision']):.3f}, "
                    f"R={float(advisory_test['recall']):.3f}, "
                    f"FPR={float(advisory_test['fpr']):.3f}"
                ),
            },
            {
                "layer": "h120 far-shape monitors",
                "horizon": "60-120min",
                "role": "far relief, re-intensification, direction shift / sign flip logging",
                "changes_pump_or_target": "no by default",
                "primary_metric": "monitor rows in final closed-loop run",
                "value": (
                    f"risk_proxy={int(monitor.far_risk_advisory_rows_proxy.sum())}, "
                    f"relief_proxy={int(monitor.far_relief_advisory_rows_proxy.sum())}, "
                    f"reint={int(monitor.far_reintensification_warning_rows.sum())}, "
                    f"shift={int(monitor.direction_shift_monitor_rows.sum())}, "
                    f"reversal={int(monitor.signflip_reversal_monitor_rows.sum())}"
                ),
            },
        ]
    )


def _key_metrics(
    baseline: pd.Series,
    final: pd.Series,
    operating_point: dict[str, Any],
    monitor: pd.DataFrame,
) -> pd.DataFrame:
    selected = operating_point["selected"]
    test = operating_point["test"]
    event = operating_point["event_level_summary"]
    metrics = {
        "closed_loop_pump_saved_m3": float(baseline.sum_pump - final.sum_pump),
        "closed_loop_time_over_5_delta_s": int(final.time_over_5 - baseline.time_over_5),
        "closed_loop_idle_time_over_5_delta_s": int(
            final.idle_time_over_5 - baseline.idle_time_over_5
        ),
        "closed_loop_fallback_delta_pp": float(final.sum_fb - baseline.sum_fb),
        "closed_loop_p95_max_axis_delta_deg": float(
            final.p95_max_axis - baseline.p95_max_axis
        ),
        "closed_loop_max_axis_delta_deg": float(final.max_axis - baseline.max_axis),
        "closed_loop_target_refresh_delayed_count": int(
            final.target_refresh_delayed_count
        ),
        "closed_loop_axis_shape_veto_count": int(final.axis_shape_veto_count),
        "h120_advisory_threshold_tau_on": float(selected["tau"]),
        "h120_advisory_threshold_tau_off": float(selected["tau_off"]),
        "h120_advisory_test_precision": float(test["precision"]),
        "h120_advisory_test_recall": float(test["recall"]),
        "h120_advisory_test_fpr": float(test["fpr"]),
        "h120_advisory_test_f1": float(test["f1"]),
        "h120_event_level_truth_recall": float(event["truth_recall"]),
        "h120_event_level_median_lead_min": float(event["median_lead_min"]),
        "h120_event_false_alarm_episodes_per_day": float(
            event["false_alarm_episodes_per_day"]
        ),
        "h120_far_risk_advisory_rows_proxy_final_run": int(
            monitor.far_risk_advisory_rows_proxy.sum()
        ),
        "h120_far_relief_advisory_rows_proxy_final_run": int(
            monitor.far_relief_advisory_rows_proxy.sum()
        ),
        "h120_far_reintensification_warning_rows_final_run": int(
            monitor.far_reintensification_warning_rows.sum()
        ),
        "h120_direction_shift_monitor_rows_final_run": int(
            monitor.direction_shift_monitor_rows.sum()
        ),
        "h120_signflip_reversal_monitor_rows_final_run": int(
            monitor.signflip_reversal_monitor_rows.sum()
        ),
        "h120_advisory_control_action_change_count": int(
            monitor.h120_advisory_control_action_change_count.sum()
        ),
    }
    return pd.DataFrame([metrics])


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            if abs(value) < 1.0:
                return f"{value:.3f}"
            return f"{value:.2f}"
        return str(value)

    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def _write_markdown(
    metrics: pd.Series,
    summary: pd.DataFrame,
    monitor: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    top_monitor = monitor[
        [
            "case",
            "case_source",
            "far_risk_advisory_rows_proxy",
            "far_relief_advisory_rows_proxy",
            "far_reintensification_warning_rows",
            "direction_shift_monitor_rows",
            "signflip_reversal_monitor_rows",
        ]
    ].copy()
    top_monitor["_activity"] = (
        top_monitor["far_risk_advisory_rows_proxy"]
        + top_monitor["far_relief_advisory_rows_proxy"]
        + top_monitor["far_reintensification_warning_rows"]
        + top_monitor["direction_shift_monitor_rows"]
        + top_monitor["signflip_reversal_monitor_rows"]
    )
    top_monitor = top_monitor.sort_values("_activity", ascending=False).drop(columns="_activity")
    top_monitor = top_monitor.head(10)

    md = f"""# 0-120min Prediction-Assisted Ballast Framework

## Final Framing

The system is a **0-120min prediction-assisted ballast-control framework**, with
separate execution and supervision roles:

1. **v1.6 hard safety floor** remains the hard recovery baseline.
2. **Learned 0-60min relief-economy** is the closed-loop pump-saving layer.
3. **60-120min h120 advisory** is retained for far-event/far-risk supervision,
   not direct pump control.

This keeps 120min prediction in the system without claiming a false `A2-A1`
closed-loop pump-saving effect.

## Closed-Loop Pump Saving: 0-60min

| metric | value |
| --- | ---: |
| pump saved vs v1.6 | {metrics.closed_loop_pump_saved_m3:.2f} m3 |
| time>5 delta | {int(metrics.closed_loop_time_over_5_delta_s)} s |
| idle time>5 delta | {int(metrics.closed_loop_idle_time_over_5_delta_s)} s |
| fallback delta | {metrics.closed_loop_fallback_delta_pp:.3f} pp |
| p95 max-axis delta | {metrics.closed_loop_p95_max_axis_delta_deg:.3f} deg |
| max-axis delta | {metrics.closed_loop_max_axis_delta_deg:.3f} deg |
| target refreshes delayed | {int(metrics.closed_loop_target_refresh_delayed_count)} |
| axis-shape vetoes | {int(metrics.closed_loop_axis_shape_veto_count)} |

The closed-loop saving comes from delaying non-hard, single-axis economy refreshes
under learned near-horizon relief. The v1.6 floor, fallback, recovery branches,
and pump penalty are unchanged.

## 60-120min Supervisory Advisory

| metric | value |
| --- | ---: |
| threshold tau_on | {metrics.h120_advisory_threshold_tau_on:.4f} |
| threshold tau_off | {metrics.h120_advisory_threshold_tau_off:.4f} |
| row precision | {metrics.h120_advisory_test_precision:.3f} |
| row recall | {metrics.h120_advisory_test_recall:.3f} |
| row FPR | {metrics.h120_advisory_test_fpr:.3f} |
| row F1 | {metrics.h120_advisory_test_f1:.3f} |
| event-level recall | {metrics.h120_event_level_truth_recall:.3f} |
| median lead | {metrics.h120_event_level_median_lead_min:.0f} min |
| false alarms/day | {metrics.h120_event_false_alarm_episodes_per_day:.3f} |
| control actions changed by h120 advisory | {int(metrics.h120_advisory_control_action_change_count)} |

The h120 advisory is display/planning information. It computes and records
far-risk, far-relief, re-intensification, direction-shift, and sign-flip signals,
but by default does **not** drive pump or target actions.

## Framework Summary Table

{_md_table(summary)}

## Far-Horizon Monitor Counts In Final Closed-Loop Run

| monitor | rows |
| --- | ---: |
| far-risk advisory proxy | {int(metrics.h120_far_risk_advisory_rows_proxy_final_run)} |
| far-relief advisory proxy | {int(metrics.h120_far_relief_advisory_rows_proxy_final_run)} |
| far re-intensification warning | {int(metrics.h120_far_reintensification_warning_rows_final_run)} |
| direction-shift monitor | {int(metrics.h120_direction_shift_monitor_rows_final_run)} |
| signflip/reversal monitor | {int(metrics.h120_signflip_reversal_monitor_rows_final_run)} |

## Highest-Activity h120 Monitor Cases

{_md_table(top_monitor)}

## Paper Statement

Do **not** write that 120min prediction creates extra closed-loop pump saving. The
accurate claim is:

> The controller uses learned 0-60min prediction for bounded closed-loop
> relief-economy pump saving, while 60-120min h120 prediction is integrated as a
> supervisory advisory and far-risk monitoring layer. Oracle and learned
> experiments indicate that far-horizon information is more appropriate for
> supervision than direct pump control under the current fast, reversible ballast
> architecture.
"""
    for path in (
        dirs["out"] / "prediction_assisted_framework_decision.md",
        dirs["paper"] / "prediction_assisted_framework_decision.md",
    ):
        path.write_text(md, encoding="utf-8")


def main() -> None:
    dirs = _ensure_dirs()
    compare = pd.read_csv(RELIEF_DIR / "learned_a1_locked_holdout_compare_table.csv")
    baseline = compare[compare.arm.eq(BASELINE_ARM)].iloc[0]
    final = compare[compare.arm.eq(FINAL_ARM)].iloc[0]
    operating_point = _load_json(ADVISORY_DIR / "advisory_operating_point.json")

    monitor = _far_horizon_monitor_table(float(operating_point["selected"]["tau"]))
    monitor.to_csv(dirs["raw"] / "h120_far_horizon_monitor_case_table.csv", index=False)
    monitor.to_csv(dirs["out"] / "h120_far_horizon_monitor_case_table.csv", index=False)

    summary = _framework_summary(baseline, final, operating_point["test"], monitor)
    summary.to_csv(dirs["out"] / "prediction_assisted_framework_summary.csv", index=False)
    summary.to_csv(dirs["raw"] / "prediction_assisted_framework_summary.csv", index=False)

    key_metrics = _key_metrics(baseline, final, operating_point, monitor)
    key_metrics.to_csv(dirs["out"] / "prediction_assisted_key_metrics.csv", index=False)
    key_metrics.to_csv(dirs["raw"] / "prediction_assisted_key_metrics.csv", index=False)

    _write_markdown(key_metrics.iloc[0], summary, monitor, dirs)
    print(f"[done] {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
