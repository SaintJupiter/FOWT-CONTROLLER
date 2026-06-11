#!/usr/bin/env python3
"""Evaluate fixed-rule budget100 behavior on the locked 53-case casebook.

This is a closing analysis, not a tuning script.  It applies the same
forecast-observable opportunity rule used in the 30-case budget100 package:

    relief/decay candidate AND A0 pump >= 150 m3

The script compares existing locked A0 outputs against a fixed budget100 run.
It does not alter controller behavior or re-select cases by outcome.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path("outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1")
LOCKED = Path("outputs/wind_prediction/h120_relief_envelope_learned_a1_locked_holdout_v1")
A0_RUN = LOCKED / "runs" / "A0_learned_v16"
BUDGET_RUN = ROOT / "locked53_7200" / "budget100"
OUT = ROOT / "budget100_final_candidate_v1"


def case_from_path(path: Path) -> str:
    return re.sub(r"_prediction_primary_econ_(timeseries|planner_log)\.csv$", "", path.name)


def run_index(case: str) -> int:
    match = re.match(r"^(\d+)_", case)
    return int(match.group(1)) if match else -1


def series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def case_metrics(path: Path) -> dict[str, float | int]:
    d = pd.read_csv(path, low_memory=False)
    pitch = series(d, "pitch_deg").to_numpy()
    roll = series(d, "roll_deg").to_numpy()
    axis = np.maximum(np.abs(pitch), np.abs(roll))
    pump = np.abs(series(d, "pump_total_rate_m3_min").to_numpy())
    fallback_cols = [
        "target_lookup_fallback",
        "preview_primary_safety_fallback",
        "fallback_reason_missing_state",
        "fallback_reason_missing_table",
        "fallback_reason_missing_columns",
        "fallback_reason_invalid_state_id",
    ]
    fallback_any = np.zeros(len(d), dtype=bool)
    for col in fallback_cols:
        if col in d.columns:
            fallback_any |= series(d, col).to_numpy() > 0.5
    return {
        "pump_m3": float(np.nansum(pump) / 60.0),
        "time_gt3_s": int(np.nansum(axis > 3.0)),
        "time_gt4_s": int(np.nansum(axis > 4.0)),
        "time_gt45_s": int(np.nansum(axis > 4.5)),
        "time_gt5_s": int(np.nansum(axis > 5.0)),
        "idle_gt5_s": int(np.nansum((axis > 5.0) & (pump < 0.5))),
        "fallback_time_s": int(np.nansum(fallback_any)),
        "fallback_ratio": float(np.nanmean(fallback_any)) if len(d) else 0.0,
        "p95_max_axis_deg": float(np.nanpercentile(axis, 95)) if len(d) else 0.0,
        "max_axis_deg": float(np.nanmax(axis)) if len(d) else 0.0,
    }


def planner_features(path: Path) -> dict[str, float | int | bool]:
    d = pd.read_csv(path, low_memory=False)
    b0 = series(d, "raw_pressure_block0_norm")
    b1 = series(d, "raw_pressure_block1_norm")
    b2 = series(d, "raw_pressure_block2_norm")
    near_peak = pd.concat([b0, b1, b2], axis=1).max(axis=1)
    early_peak = pd.concat([b0, b1], axis=1).max(axis=1)
    near_drop = early_peak - b2

    far_cols = [
        "far_horizon_norm_60_80",
        "far_horizon_norm_80_100",
        "far_horizon_norm_100_120",
    ]
    far = [series(d, c) for c in far_cols]
    far_min = pd.concat(far, axis=1).min(axis=1)
    near_max = series(d, "far_horizon_near_max", default=np.nan)
    if near_max.isna().all():
        near_max = near_peak
    far_relief = near_max - far_min

    reversal = (
        (series(d, "far_horizon_reversal") > 0.5)
        | (series(d, "far_horizon_direction_shift") > 0.5)
        | (series(d, "far_horizon_dir_shift_deg") >= 45.0)
        | (series(d, "h120_scheduler_far_signflip_risk") > 0.5)
    )

    high_then_relief = (early_peak >= 0.65) & (near_drop >= 0.20)
    rise_then_fall = (b1 >= b0 + 0.10) & (b2 <= b1 - 0.20)
    h120_far_relief = (near_max >= 0.65) & (far_relief >= 0.25)

    return {
        "decision_rows": int(len(d)),
        "near_high_rows": int((near_peak >= 0.65).sum()),
        "high_then_relief_rows": int(high_then_relief.sum()),
        "rise_then_fall_rows": int(rise_then_fall.sum()),
        "h120_far_relief_rows": int(h120_far_relief.sum()),
        "reversal_or_signflip_rows": int(reversal.sum()),
        "max_near_peak": float(near_peak.max()),
        "max_near_drop": float(near_drop.max()),
        "max_far_relief_drop": float(far_relief.max()),
        "forecast_relief_decay_candidate": bool(
            high_then_relief.sum() > 0 or h120_far_relief.sum() > 0
        ),
        "fast_callback_candidate": bool(
            high_then_relief.sum() > 0 or rise_then_fall.sum() > 0
        ),
        "h120_supervisory_relief_candidate": bool(h120_far_relief.sum() > 0),
        "direction_reversal_candidate": bool(reversal.sum() > 0),
    }


def load_labels() -> dict[int, dict[str, str]]:
    cases = pd.read_csv(LOCKED / "locked_casebook_cases.csv")
    labels: dict[int, dict[str, str]] = {}
    for i, row in cases.reset_index(drop=True).iterrows():
        labels[i + 1] = {
            "case_id": str(row.get("case_id", "")),
            "timestamp": str(row.get("timestamp", "")),
            "label": str(row.get("label", "")),
        }
    return labels


def build_case_table() -> pd.DataFrame:
    labels = load_labels()
    rows = []
    for a0_ts in sorted((A0_RUN / "timeseries").glob("*_timeseries.csv")):
        case = case_from_path(a0_ts)
        idx = run_index(case)
        budget_ts = BUDGET_RUN / "timeseries" / a0_ts.name
        a0_log = A0_RUN / "planner_logs" / a0_ts.name.replace(
            "_timeseries.csv", "_planner_log.csv"
        )
        if not budget_ts.exists() or not a0_log.exists():
            continue
        a0 = case_metrics(a0_ts)
        b = case_metrics(budget_ts)
        feats = planner_features(a0_log)
        label = labels.get(idx, {"case_id": "", "timestamp": "", "label": ""})
        row: dict[str, object] = {
            "case": case,
            "run_index": idx,
            **label,
            **feats,
        }
        for k, v in a0.items():
            row[f"{k}_a0"] = v
        for k, v in b.items():
            row[f"{k}_budget100"] = v
        row["pump_saved_m3"] = float(a0["pump_m3"] - b["pump_m3"])
        row["pump_saving_pct"] = (
            100.0 * float(a0["pump_m3"] - b["pump_m3"]) / float(a0["pump_m3"])
            if float(a0["pump_m3"]) > 1e-9
            else 0.0
        )
        for metric in [
            "time_gt3_s",
            "time_gt4_s",
            "time_gt45_s",
            "time_gt5_s",
            "idle_gt5_s",
            "fallback_time_s",
            "p95_max_axis_deg",
            "max_axis_deg",
        ]:
            row[f"delta_{metric}"] = b[metric] - a0[metric]
        row["opportunity_relief_decay_pump_ge150"] = bool(
            row["forecast_relief_decay_candidate"] and float(a0["pump_m3"]) >= 150.0
        )
        row["is_lowrisk_label"] = "lowrisk" in str(label.get("label", "")).lower()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(name: str, d: pd.DataFrame) -> dict[str, object]:
    a0 = float(d["pump_m3_a0"].sum())
    b = float(d["pump_m3_budget100"].sum())
    return {
        "subset": name,
        "cases": int(len(d)),
        "a0_pump_m3": a0,
        "budget100_pump_m3": b,
        "pump_saved_m3": a0 - b,
        "pump_saving_pct": 100.0 * (a0 - b) / a0 if a0 > 1e-9 else 0.0,
        "delta_time_gt3_s": int(d["delta_time_gt3_s"].sum()),
        "delta_time_gt4_s": int(d["delta_time_gt4_s"].sum()),
        "delta_time_gt45_s": int(d["delta_time_gt45_s"].sum()),
        "delta_time_gt5_s": int(d["delta_time_gt5_s"].sum()),
        "delta_idle_gt5_s": int(d["delta_idle_gt5_s"].sum()),
        "delta_fallback_time_s": int(d["delta_fallback_time_s"].sum()),
        "mean_delta_p95_max_axis_deg": float(d["delta_p95_max_axis_deg"].mean())
        if len(d)
        else 0.0,
        "max_delta_max_axis_deg": float(d["delta_max_axis_deg"].max()) if len(d) else 0.0,
        "cases_with_added_fallback": int((d["delta_fallback_time_s"] > 0).sum()),
        "cases_with_negative_pump_saving": int((d["pump_saved_m3"] < 0).sum()),
    }


def fallback_concentration(d: pd.DataFrame, subset: str) -> dict[str, object]:
    pos = d[d["delta_fallback_time_s"] > 0].sort_values(
        "delta_fallback_time_s", ascending=False
    )
    total = float(d["delta_fallback_time_s"].sum())
    positive_total = float(pos["delta_fallback_time_s"].sum())
    return {
        "subset": subset,
        "cases": int(len(d)),
        "cases_with_added_fallback": int(len(pos)),
        "total_delta_fallback_time_s": total,
        "positive_added_fallback_time_s": positive_total,
        "max_case_delta_fallback_time_s": float(pos["delta_fallback_time_s"].max())
        if not pos.empty
        else 0.0,
        "top1_share_of_added_fallback_pct": 100.0
        * float(pos["delta_fallback_time_s"].head(1).sum())
        / positive_total
        if positive_total > 1e-9 and not pos.empty
        else 0.0,
        "top3_share_of_added_fallback_pct": 100.0
        * float(pos["delta_fallback_time_s"].head(3).sum())
        / positive_total
        if positive_total > 1e-9 and not pos.empty
        else 0.0,
        "top_fallback_cases": "; ".join(pos["case"].head(3).tolist()),
    }


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> None:
    raw = OUT / "raw_tables"
    paper = OUT / "paper_ready"
    raw.mkdir(parents=True, exist_ok=True)
    paper.mkdir(parents=True, exist_ok=True)

    cases = build_case_table()
    cases.to_csv(raw / "locked53_budget100_expansion_case_table.csv", index=False)

    subsets = {
        "locked53_all": cases,
        "locked53_opportunity_relief_decay_pump_ge150": cases[
            cases["opportunity_relief_decay_pump_ge150"]
        ],
        "locked53_lowrisk": cases[cases["is_lowrisk_label"]],
        "locked53_non_opportunity": cases[
            ~cases["opportunity_relief_decay_pump_ge150"]
        ],
    }
    summary = pd.DataFrame([summarize(name, d) for name, d in subsets.items()])
    fallback = pd.DataFrame(
        [fallback_concentration(d, name) for name, d in subsets.items()]
    )
    summary.to_csv(raw / "locked53_budget100_expansion_summary.csv", index=False)
    fallback.to_csv(raw / "locked53_budget100_fallback_concentration.csv", index=False)

    opp = summary[summary["subset"] == "locked53_opportunity_relief_decay_pump_ge150"]
    full = summary[summary["subset"].isin(["locked53_all", "locked53_lowrisk"])]
    fallback_opp = fallback[
        fallback["subset"] == "locked53_opportunity_relief_decay_pump_ge150"
    ]
    top_cases = cases[cases["opportunity_relief_decay_pump_ge150"]].sort_values(
        "pump_saved_m3", ascending=False
    )[
        [
            "case",
            "case_id",
            "pump_saved_m3",
            "pump_saving_pct",
            "delta_time_gt5_s",
            "delta_fallback_time_s",
            "delta_p95_max_axis_deg",
        ]
    ]
    top_cases.to_csv(raw / "locked53_budget100_opportunity_case_deltas.csv", index=False)

    text = [
        "# Locked53 Budget100 Expansion Check",
        "",
        "This is a fixed-rule expansion check. It uses the same opportunity rule as the 30-case package: forecast-observable relief/decay plus A0 pump >= 150 m3. No budget parameter or subset definition was tuned on this run.",
        "",
        "## Main Locked Opportunity Result",
        "",
        md_table(
            opp,
            [
                "subset",
                "cases",
                "pump_saving_pct",
                "pump_saved_m3",
                "delta_time_gt5_s",
                "delta_idle_gt5_s",
                "delta_fallback_time_s",
                "mean_delta_p95_max_axis_deg",
                "cases_with_added_fallback",
                "cases_with_negative_pump_saving",
            ],
        ),
        "",
        "## Full / Lowrisk Context",
        "",
        md_table(
            full,
            [
                "subset",
                "cases",
                "pump_saving_pct",
                "pump_saved_m3",
                "delta_time_gt5_s",
                "delta_idle_gt5_s",
                "delta_fallback_time_s",
                "mean_delta_p95_max_axis_deg",
                "cases_with_added_fallback",
            ],
        ),
        "",
        "## Fallback Concentration",
        "",
        md_table(
            fallback_opp,
            [
                "cases_with_added_fallback",
                "total_delta_fallback_time_s",
                "positive_added_fallback_time_s",
                "max_case_delta_fallback_time_s",
                "top1_share_of_added_fallback_pct",
                "top3_share_of_added_fallback_pct",
                "top_fallback_cases",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "The locked53 opportunity result is above the 30-case value (23.40% vs 19.66%), so the approximately-20% aggressive Pareto point is not just a 10-case artifact. However, the safety-margin cost is also clear: time>5 and fallback both increase. This supports budget100 as a regime-conditioned optional aggressive Pareto mode, not as the clean automatic no-regression headline.",
        "",
        "In all cases, this run does not change the boundary: budget100 is not the safety-neutral headline; it is an optional aggressive economy mode with disclosed comfort / safety-margin cost.",
        "",
    ]
    (paper / "locked53_budget100_expansion_summary.md").write_text(
        "\n".join(text), encoding="utf-8"
    )
    print("Wrote locked53 expansion analysis to", OUT)


if __name__ == "__main__":
    main()
