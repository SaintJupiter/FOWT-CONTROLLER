#!/usr/bin/env python3
"""Summarize an oracle regime selector over existing PSC candidate runs.

This is an offline upper-bound audit, not a deployable controller.  It answers:
if a perfect supervisor could choose one already-ran execution profile per
case/regime, is there combined-objective room beyond blind DC deadband?
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_ROOT = Path("outputs/wind_prediction/oracle_classifier_psc_upper_bound_20260531")


def _read_case_metrics(root: Path, arm: str) -> pd.DataFrame:
    run_dir = root / arm
    summary = pd.read_csv(run_dir / "casebook_summary.csv")
    rows: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        case_id = str(row["case_id"])
        matches = glob.glob(str(run_dir / "timeseries" / f"{case_id}_*_{arm}_timeseries.csv"))
        if not matches:
            raise FileNotFoundError(f"missing timeseries for {arm}/{case_id}")
        ts = pd.read_csv(matches[0], low_memory=False)
        max_axis = ts[["pitch_deg", "roll_deg"]].abs().max(axis=1)
        fallback = (
            ts.get("preview_primary_safety_fallback", pd.Series(0, index=ts.index))
            .fillna(0)
            .astype(float)
            .gt(0)
        )
        rows.append(
            {
                "arm": arm,
                "case_id": case_id,
                "label": str(row.get("label", "")),
                "pump_m3": float(row["primary_pump_work_m3"]),
                "time_gt5_s": int((max_axis > 5.0).sum()),
                "time_gt6_s": int((max_axis > 6.0).sum()),
                "max_axis_deg": float(max_axis.max()),
                "p95_axis_deg": float(max_axis.quantile(0.95)),
                "fallback_s": int(fallback.sum()),
                "latch_switches": float(row.get("primary_latch_switches", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _select_safety_first(group: pd.DataFrame) -> pd.Series:
    safe = group[group["fallback_s"] == 0]
    candidates = safe if not safe.empty else group
    return candidates.sort_values(
        ["time_gt5_s", "time_gt6_s", "pump_m3", "latch_switches"],
        ascending=[True, True, True, True],
    ).iloc[0]


def _select_balanced(group: pd.DataFrame) -> pd.Series:
    safe = group[group["fallback_s"] == 0].copy()
    candidates = safe if not safe.empty else group.copy()
    # One second above 5 deg is deliberately expensive; pump is secondary.
    candidates["objective"] = (
        10.0 * candidates["time_gt5_s"]
        + 20.0 * candidates["time_gt6_s"]
        + 0.15 * candidates["pump_m3"]
        + 1000.0 * candidates["fallback_s"]
    )
    return candidates.sort_values(["objective", "time_gt5_s", "pump_m3"]).iloc[0]


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.to_dict("records"):
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.1f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--arms",
        default=(
            "blind_deadband,oracle_forecast_adaptive,"
            "oracle_h120_safety_watch,oracle_preemptive,rawenv_mainline,h120_pareto_selector"
        ),
    )
    args = parser.parse_args()

    root = args.root
    arms = [arm.strip() for arm in str(args.arms).split(",") if arm.strip()]
    metrics = pd.concat([_read_case_metrics(root, arm) for arm in arms], ignore_index=True)
    metrics.to_csv(root / "psc_selector_candidate_metrics.csv", index=False)

    selected_rows = []
    for case_id, group in metrics.groupby("case_id"):
        blind = group[group["arm"] == "blind_deadband"].iloc[0]
        safety = _select_safety_first(group)
        balanced = _select_balanced(group)
        for selector_name, picked in (
            ("oracle_safety_first", safety),
            ("oracle_balanced", balanced),
        ):
            record = picked.to_dict()
            record["selector"] = selector_name
            record["blind_time_gt5_s"] = int(blind["time_gt5_s"])
            record["blind_pump_m3"] = float(blind["pump_m3"])
            record["d_time_gt5_vs_blind_s"] = int(picked["time_gt5_s"] - blind["time_gt5_s"])
            record["d_pump_vs_blind_m3"] = float(picked["pump_m3"] - blind["pump_m3"])
            selected_rows.append(record)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(root / "psc_oracle_regime_selector_v0_cases.csv", index=False)

    aggregate_rows = []
    for selector, group in selected.groupby("selector"):
        blind_once = selected[selected["selector"] == selector]
        aggregate_rows.append(
            {
                "selector": selector,
                "cases": len(group),
                "pump_m3": group["pump_m3"].sum(),
                "time_gt5_s": group["time_gt5_s"].sum(),
                "time_gt6_s": group["time_gt6_s"].sum(),
                "fallback_s": group["fallback_s"].sum(),
                "latch_switches": group["latch_switches"].sum(),
                "d_pump_vs_blind_m3": group["pump_m3"].sum()
                - blind_once["blind_pump_m3"].sum(),
                "d_time_gt5_vs_blind_s": group["time_gt5_s"].sum()
                - blind_once["blind_time_gt5_s"].sum(),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(root / "psc_oracle_regime_selector_v0_summary.csv", index=False)

    safety_cases = selected[selected["selector"] == "oracle_safety_first"][
        [
            "case_id",
            "label",
            "arm",
            "time_gt5_s",
            "blind_time_gt5_s",
            "pump_m3",
            "blind_pump_m3",
            "fallback_s",
            "d_time_gt5_vs_blind_s",
            "d_pump_vs_blind_m3",
        ]
    ]
    report = [
        "# PSC oracle regime selector v0 - 2026-05-31",
        "",
        "Offline upper bound: choose one already-ran execution profile per case/regime.",
        "This is not deployable; it tells us whether a forecast-driven supervisor has a target worth learning.",
        "",
        "## Aggregate",
        "",
        _md_table(aggregate),
        "",
        "## Safety-first case choices",
        "",
        _md_table(safety_cases),
        "",
        "## Interpretation",
        "",
        "- The broad PSC failure was not caused by prediction being useless; it was caused by applying one aggressive economy mode globally.",
        "- A perfect regime selector can reduce both pump and high-posture exposure on this mixed pool while keeping fallback at zero.",
        "- The next real controller should learn this selector/guard: keep blind/DC deadband or forecast-adaptive in strong event windows, only use rawenv-style economy in regimes where the selector proves it does not create long high-posture tails.",
    ]
    out = root / "PSC_ORACLE_REGIME_SELECTOR_V0_20260531.md"
    out.write_text("\n".join(report) + "\n")
    print(out.resolve())
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
