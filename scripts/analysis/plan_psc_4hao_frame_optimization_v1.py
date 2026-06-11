#!/usr/bin/env python3
"""Plan regime/frame-specific PSC No.4 optimization from existing runs.

This is a read-only planning pass. It reuses the completed broader 6h
baseline/refresh_on/generalized_gate runs and the older
ULTIMATE_PUMP_SAVING_CONCLUSION regime numbering to estimate whether a
frame-specific selector is worth implementing before any new long simulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
RUNS = BASE / "psc_4hao_broader_6h_test_only_main_v1" / "runs"
OUT = BASE / "psc_4hao_frame_optimization_plan_20260603"

POOLS = ["positive_pool", "negative_pool", "background_pool"]
ARMS = ["baseline", "refresh_on", "generalized_gate"]
ARM_LABEL = {
    "baseline": "current_forecast_adaptive",
    "refresh_on": "refresh_on",
    "generalized_gate": "generalized_gate",
}

OLD_REGIME_MAP = {
    "neutral_mhs_broader": {
        "old_id": "P2/P3",
        "old_name": "Broad neutral/headroom; neutral-MHS nested entry",
        "old_reading": "Primary broad clean saving family; old package reports P2 as the main broad denominator and P3 as nested clean-start evidence.",
        "next_role": "main enable",
    },
    "transient_peak_future_decay": {
        "old_id": "P1",
        "old_name": "Transient peak / future decay automatic specialist",
        "old_reading": "Forecast-causal saving family; old package keeps it as a narrow automatic/Pareto regime with disclosed posture cost.",
        "next_role": "conditional enable",
    },
    "lowrisk_stable_redundant_candidate": {
        "old_id": "Regime 3 / lowrisk redundant",
        "old_name": "Low-risk redundant pump",
        "old_reading": "Clean but sparse redundant-pump opportunity; useful as support, not a broad headline.",
        "next_role": "small enable",
    },
    "quiet_low_opportunity": {
        "old_id": "no-action",
        "old_name": "Low pump / low opportunity",
        "old_reading": "No material pump opportunity; keep baseline/no-op.",
        "next_role": "baseline/no-op",
    },
    "direction_reversal_boundary": {
        "old_id": "W1",
        "old_name": "Direction reversal split branch",
        "old_reading": "Old package later split W1: low-posture reversal can save, high-pressure reversal remains warning/release.",
        "next_role": "split detector",
    },
    "reintensification_boundary": {
        "old_id": "W2",
        "old_name": "Re-intensification after relief warning",
        "old_reading": "Warning/veto family; apparent relief can be temporary.",
        "next_role": "release/veto",
    },
    "sustained_high_safety_event": {
        "old_id": "W3/W7",
        "old_name": "Sustained high / high-attention event watch",
        "old_reading": "Hard safety or warning-owned context, not an economy saving target.",
        "next_role": "baseline/watch",
    },
}


def _find_timeseries(pool: str, arm: str, case_id: str) -> Path:
    label = ARM_LABEL[arm]
    matches = sorted((RUNS / pool / arm / "timeseries").glob(f"{case_id}_*_{label}_timeseries.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{pool}/{arm}/{case_id}: expected one timeseries, got {len(matches)}")
    return matches[0]


def _load_metrics() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pool in POOLS:
        path = RUNS / pool / "case_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["pool"] = pool
        frames.append(df)
        refresh_delta = RUNS / pool / "refresh_on_case_deltas.csv"
        if refresh_delta.exists():
            refresh = pd.read_csv(refresh_delta)
            refresh["pool"] = pool
            keep_cols = [
                "arm",
                "case_id",
                "label",
                "pump_m3",
                "time_gt3_s",
                "time_gt4_s",
                "time_gt5_s",
                "fallback_s",
                "p95_axis_deg",
                "max_axis_deg",
                "stratum",
                "validation_role",
                "pool",
            ]
            frames.append(refresh[[c for c in keep_cols if c in refresh.columns]])
    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics[metrics["arm"].isin(ARMS)].copy()
    metrics = metrics.drop_duplicates(["pool", "arm", "case_id"], keep="last")

    time_gt7: list[float] = []
    for rec in metrics.to_dict("records"):
        ts = pd.read_csv(
            _find_timeseries(str(rec["pool"]), str(rec["arm"]), str(rec["case_id"])),
            usecols=lambda c: c in {"pitch_deg", "roll_deg"},
            low_memory=False,
        )
        pitch = pd.to_numeric(ts["pitch_deg"], errors="coerce").fillna(0.0).to_numpy(float)
        roll = pd.to_numeric(ts["roll_deg"], errors="coerce").fillna(0.0).to_numpy(float)
        axis = np.maximum(np.abs(pitch), np.abs(roll))
        time_gt7.append(float(np.sum(axis > 7.0)))
    metrics["time_gt7_s"] = time_gt7
    metrics["case_max_gt5"] = (pd.to_numeric(metrics["max_axis_deg"], errors="coerce") > 5.0).astype(int)
    metrics["case_max_gt7"] = (pd.to_numeric(metrics["max_axis_deg"], errors="coerce") > 7.0).astype(int)
    return metrics


def _aggregate(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "cases": int(df["case_id"].nunique()),
            "pump_m3": float(df["pump_m3"].sum()),
            "time_gt3_s": float(df["time_gt3_s"].sum()),
            "time_gt4_s": float(df["time_gt4_s"].sum()),
            "time_gt5_s": float(df["time_gt5_s"].sum()),
            "time_gt7_s": float(df["time_gt7_s"].sum()),
            "fallback_s": float(df["fallback_s"].sum()),
            "worst_p95_axis_deg": float(df["p95_axis_deg"].max()),
            "worst_max_axis_deg": float(df["max_axis_deg"].max()),
            "cases_max_gt5": int(df["case_max_gt5"].sum()),
            "cases_max_gt7": int(df["case_max_gt7"].sum()),
        }
    )


def _policy_pick(metrics: pd.DataFrame, policy: str) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, group in metrics.groupby(["pool", "case_id"], dropna=False):
        g = group.set_index("arm", drop=False)
        base = g.loc["baseline"]
        gate = g.loc["generalized_gate"] if "generalized_gate" in g.index else base
        stratum = str(base["stratum"])

        if policy == "baseline":
            chosen = base
        elif policy == "current_generalized_gate":
            chosen = gate
        elif policy == "conservative_p2_lowrisk_only":
            chosen = gate if stratum in {"neutral_mhs_broader", "lowrisk_stable_redundant_candidate"} else base
        elif policy == "balanced_positive_background_only":
            chosen = gate if str(base["pool"]) in {"positive_pool", "background_pool"} else base
        elif policy == "old_regime_balanced":
            chosen = gate if stratum in {"neutral_mhs_broader", "transient_peak_future_decay", "lowrisk_stable_redundant_candidate"} else base
        elif policy == "per_case_min_pump_any_arm":
            chosen = group.sort_values("pump_m3", kind="stable").iloc[0]
        elif policy == "per_case_min_pump_tail_guard":
            candidates = group[
                (group["fallback_s"] <= float(base["fallback_s"]) + 120.0)
                & (group["time_gt7_s"] <= float(base["time_gt7_s"]) + 60.0)
                & (group["p95_axis_deg"] <= 5.0)
            ]
            chosen = candidates.sort_values("pump_m3", kind="stable").iloc[0] if not candidates.empty else base
        elif policy == "per_case_min_pump_no_fallback_regret":
            candidates = group[group["fallback_s"] <= float(base["fallback_s"])]
            chosen = candidates.sort_values("pump_m3", kind="stable").iloc[0] if not candidates.empty else base
        else:
            raise ValueError(policy)

        rec = chosen.copy()
        rec["policy"] = policy
        rec["chosen_arm"] = str(chosen["arm"])
        rec["baseline_pump_m3"] = float(base["pump_m3"])
        rec["saved_m3"] = float(base["pump_m3"]) - float(chosen["pump_m3"])
        rows.append(rec)
    return pd.DataFrame(rows)


def _policy_summary(metrics: pd.DataFrame, policies: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = _policy_pick(metrics, "baseline")
    baseline_total = float(baseline["pump_m3"].sum())
    case_frames = [_policy_pick(metrics, policy) for policy in policies]
    case_table = pd.concat(case_frames, ignore_index=True)

    rows: list[dict[str, object]] = []
    for policy, df in case_table.groupby("policy", dropna=False):
        agg = _aggregate(df)
        rec = agg.to_dict()
        rec["policy"] = policy
        rec["saved_m3"] = baseline_total - float(rec["pump_m3"])
        rec["saving_pct"] = 100.0 * float(rec["saved_m3"]) / max(baseline_total, 1e-9)
        rec["non_baseline_cases"] = int((df["chosen_arm"] != "baseline").sum())
        rec["generalized_cases"] = int((df["chosen_arm"] == "generalized_gate").sum())
        rec["refresh_cases"] = int((df["chosen_arm"] == "refresh_on").sum())
        rows.append(rec)
    summary = pd.DataFrame(rows)
    order = {policy: i for i, policy in enumerate(policies)}
    summary["order"] = summary["policy"].map(order)
    summary = summary.sort_values("order").drop(columns="order")
    return summary, case_table


def _stratum_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    gate = metrics[metrics["arm"].eq("generalized_gate")].copy()
    base = metrics[metrics["arm"].eq("baseline")][["pool", "case_id", "pump_m3", "time_gt5_s", "time_gt7_s", "fallback_s"]]
    merged = gate.merge(base, on=["pool", "case_id"], suffixes=("", "_baseline"))
    merged["saved_m3"] = merged["pump_m3_baseline"] - merged["pump_m3"]
    merged["d_time_gt5_s"] = merged["time_gt5_s"] - merged["time_gt5_s_baseline"]
    merged["d_time_gt7_s"] = merged["time_gt7_s"] - merged["time_gt7_s_baseline"]
    merged["d_fallback_s"] = merged["fallback_s"] - merged["fallback_s_baseline"]
    out = (
        merged.groupby(["pool", "stratum"], dropna=False)
        .agg(
            cases=("case_id", "count"),
            baseline_pump_m3=("pump_m3_baseline", "sum"),
            pump_m3=("pump_m3", "sum"),
            saved_m3=("saved_m3", "sum"),
            time_gt5_s=("time_gt5_s", "sum"),
            time_gt7_s=("time_gt7_s", "sum"),
            fallback_s=("fallback_s", "sum"),
            d_time_gt5_s=("d_time_gt5_s", "sum"),
            d_time_gt7_s=("d_time_gt7_s", "sum"),
            d_fallback_s=("d_fallback_s", "sum"),
            worst_p95_axis_deg=("p95_axis_deg", "max"),
            worst_max_axis_deg=("max_axis_deg", "max"),
            cases_max_gt5=("case_max_gt5", "sum"),
            cases_max_gt7=("case_max_gt7", "sum"),
        )
        .reset_index()
    )
    out["saving_pct"] = 100.0 * out["saved_m3"] / out["baseline_pump_m3"].clip(lower=1e-9)
    for key, data in OLD_REGIME_MAP.items():
        mask = out["stratum"].eq(key)
        for col, value in data.items():
            out.loc[mask, col] = value
    return out


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for rec in df[cols].to_dict("records"):
        values = []
        for col in cols:
            value = rec[col]
            if isinstance(value, float):
                if col.endswith("_pct"):
                    values.append(f"{value:.2f}%")
                elif col.endswith("_deg"):
                    values.append(f"{value:.2f}")
                elif col.endswith("_m3"):
                    values.append(f"{value:.1f}")
                elif col.endswith("_s"):
                    values.append(f"{value:.0f}")
                else:
                    values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(policy_summary: pd.DataFrame, stratum: pd.DataFrame) -> None:
    lines = [
        "# PSC No.4 Frame Optimization Plan - 2026-06-03",
        "",
        "## What This Pass Does",
        "",
        "Read-only planning pass. It combines the completed 80-case broader 6h run with the older `ULTIMATE_PUMP_SAVING_CONCLUSION` regime numbering. No controller code is changed and no new long simulation is run.",
        "",
        "## Old Regime Evidence To Reuse",
        "",
        "- `P2/P3` maps to the current `neutral_mhs_broader`: this remains the clean main saving family.",
        "- `P1` maps to `transient_peak_future_decay`: useful but should be conditional/Pareto because posture/fallback exposure is where the cost concentrates.",
        "- `W1` maps to `direction_reversal_boundary`: old evidence says low-posture reversal can save, but high-pressure reversal should release/veto.",
        "- `W2/W3/W7` warning families map to reintensification/sustained-high contexts: do not treat them as ordinary saving pools.",
        "- Low-risk redundant pump remains a small clean support family, not a broad headline.",
        "",
        "## Offline Policy Upper Bounds From Existing Arms",
        "",
        _md_table(
            policy_summary,
            [
                "policy",
                "cases",
                "non_baseline_cases",
                "pump_m3",
                "saved_m3",
                "saving_pct",
                "time_gt5_s",
                "time_gt7_s",
                "fallback_s",
                "worst_p95_axis_deg",
                "worst_max_axis_deg",
            ],
        ),
        "",
        "## Current Generalized Gate By Frame",
        "",
        _md_table(
            stratum.sort_values(["pool", "stratum"]),
            [
                "pool",
                "stratum",
                "old_id",
                "next_role",
                "cases",
                "saving_pct",
                "time_gt5_s",
                "time_gt7_s",
                "fallback_s",
                "worst_p95_axis_deg",
                "worst_max_axis_deg",
            ],
        ),
        "",
        "## Planning Decision",
        "",
        "Continue optimizing, but as a frame-specific dispatcher rather than a global threshold sweep.",
        "",
        "Recommended next implementation order:",
        "",
        "1. **Frame audit first**: inspect current generalized_gate planner logs for the `W1` split variables: low-posture saving branch vs high-pressure release branch. This is the most likely way to keep the extra boundary saving without accepting all boundary behavior.",
        "2. **P2/P3 mainline stays on**: keep `neutral_mhs_broader` as the default positive branch. It already supplies the cleanest large saving.",
        "3. **P1 transient decay becomes conditional**: keep the current setting as the optimistic/Pareto branch, but test a stricter relief specialist if the paper needs a cleaner comfort point.",
        "4. **Low-risk redundant is a small opportunistic branch**: keep only if it remains zero-fallback/zero-tail in broader validation.",
        "5. **Do not build many independent controllers yet**: implement a single dispatcher that selects among existing strategy profiles by frame, then run 8-12 representative smoke cases before another full 80-case run.",
        "",
        "## Proposed Smoke Case Mix",
        "",
        "- 3 `neutral_mhs_broader` cases: preserve P2/P3 savings.",
        "- 3 `transient_peak_future_decay` cases: include one high-tail case and one clean case.",
        "- 3 `direction_reversal_boundary` cases: one low-posture candidate, two high-pressure/release candidates.",
        "- 2 `reintensification_boundary` cases: verify release/veto.",
        "- 1 `lowrisk_stable_redundant_candidate` case: verify small clean saving.",
    ]
    (OUT / "psc_4hao_frame_optimization_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics()
    policies = [
        "baseline",
        "current_generalized_gate",
        "conservative_p2_lowrisk_only",
        "old_regime_balanced",
        "balanced_positive_background_only",
        "per_case_min_pump_no_fallback_regret",
        "per_case_min_pump_tail_guard",
        "per_case_min_pump_any_arm",
    ]
    policy_summary, policy_cases = _policy_summary(metrics, policies)
    stratum = _stratum_summary(metrics)

    metrics.to_csv(OUT / "current_arm_case_metrics_with_gt7.csv", index=False)
    stratum.to_csv(OUT / "current_generalized_gate_by_frame.csv", index=False)
    policy_summary.to_csv(OUT / "offline_policy_upper_bound_summary.csv", index=False)
    policy_cases.to_csv(OUT / "offline_policy_case_choices.csv", index=False)
    pd.DataFrame(
        [
            {"stratum": key, **value}
            for key, value in OLD_REGIME_MAP.items()
        ]
    ).to_csv(OUT / "old_regime_mapping_to_current_frames.csv", index=False)
    _write_report(policy_summary, stratum)

    print(OUT / "psc_4hao_frame_optimization_plan.md")
    print(policy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
