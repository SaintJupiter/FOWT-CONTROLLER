#!/usr/bin/env python3
"""Summarize the wider fixed-rule relief/decay expansion casebook."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path("outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1")
RUN = ROOT / "relief_decay_expansion_v1"
OUT = RUN


def case_from_path(path: Path) -> str:
    return re.sub(r"_prediction_primary_econ_timeseries\.csv$", "", path.name)


def series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def metrics(path: Path) -> dict[str, float | int]:
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
    fallback = np.zeros(len(d), dtype=bool)
    for col in fallback_cols:
        if col in d.columns:
            fallback |= series(d, col).to_numpy() > 0.5
    return {
        "pump_m3": float(np.nansum(pump) / 60.0),
        "time_gt3_s": int(np.nansum(axis > 3.0)),
        "time_gt4_s": int(np.nansum(axis > 4.0)),
        "time_gt45_s": int(np.nansum(axis > 4.5)),
        "time_gt5_s": int(np.nansum(axis > 5.0)),
        "idle_gt5_s": int(np.nansum((axis > 5.0) & (pump < 0.5))),
        "fallback_time_s": int(np.nansum(fallback)),
        "p95_max_axis_deg": float(np.nanpercentile(axis, 95)),
        "max_axis_deg": float(np.nanmax(axis)),
    }


def load() -> pd.DataFrame:
    rows = []
    for arm in ["a0_baseline", "budget100"]:
        ts_dir = RUN / arm / "timeseries"
        for path in sorted(ts_dir.glob("*_timeseries.csv")):
            row = metrics(path)
            row.update({"arm": arm, "case": case_from_path(path)})
            rows.append(row)
    return pd.DataFrame(rows)


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
    raw.mkdir(exist_ok=True, parents=True)
    paper.mkdir(exist_ok=True, parents=True)

    d = load()
    a0 = d[d["arm"] == "a0_baseline"].copy()
    b = d[d["arm"] == "budget100"].copy()
    merged = b.merge(a0, on="case", suffixes=("_budget100", "_a0"))
    rows = []
    for _, r in merged.iterrows():
        row = {"case": r["case"]}
        row["a0_pump_m3"] = r["pump_m3_a0"]
        row["budget100_pump_m3"] = r["pump_m3_budget100"]
        row["pump_saved_m3"] = r["pump_m3_a0"] - r["pump_m3_budget100"]
        row["pump_saving_pct"] = (
            100.0 * row["pump_saved_m3"] / r["pump_m3_a0"]
            if r["pump_m3_a0"] > 1e-9
            else 0.0
        )
        for m in [
            "time_gt3_s",
            "time_gt4_s",
            "time_gt45_s",
            "time_gt5_s",
            "idle_gt5_s",
            "fallback_time_s",
            "p95_max_axis_deg",
            "max_axis_deg",
        ]:
            row[f"delta_{m}"] = r[f"{m}_budget100"] - r[f"{m}_a0"]
        rows.append(row)
    case_delta = pd.DataFrame(rows)
    case_delta.to_csv(raw / "relief_decay_expansion_case_delta.csv", index=False)

    a0_pump = float(a0["pump_m3"].sum())
    b_pump = float(b["pump_m3"].sum())
    summary = pd.DataFrame(
        [
            {
                "cases": int(len(a0)),
                "a0_pump_m3": a0_pump,
                "budget100_pump_m3": b_pump,
                "pump_saved_m3": a0_pump - b_pump,
                "pump_saving_pct": 100.0 * (a0_pump - b_pump) / a0_pump
                if a0_pump > 1e-9
                else 0.0,
                "delta_time_gt3_s": int(
                    b["time_gt3_s"].sum() - a0["time_gt3_s"].sum()
                ),
                "delta_time_gt4_s": int(
                    b["time_gt4_s"].sum() - a0["time_gt4_s"].sum()
                ),
                "delta_time_gt45_s": int(
                    b["time_gt45_s"].sum() - a0["time_gt45_s"].sum()
                ),
                "delta_time_gt5_s": int(
                    b["time_gt5_s"].sum() - a0["time_gt5_s"].sum()
                ),
                "delta_idle_gt5_s": int(
                    b["idle_gt5_s"].sum() - a0["idle_gt5_s"].sum()
                ),
                "delta_fallback_time_s": int(
                    b["fallback_time_s"].sum() - a0["fallback_time_s"].sum()
                ),
                "mean_delta_p95_max_axis_deg": float(
                    b["p95_max_axis_deg"].mean() - a0["p95_max_axis_deg"].mean()
                ),
                "cases_with_added_fallback": int(
                    (case_delta["delta_fallback_time_s"] > 0).sum()
                ),
                "cases_with_negative_saving": int((case_delta["pump_saved_m3"] < 0).sum()),
            }
        ]
    )
    summary.to_csv(raw / "relief_decay_expansion_summary.csv", index=False)

    top = case_delta.sort_values("pump_saved_m3", ascending=False).head(8)
    bad = case_delta.sort_values("pump_saved_m3", ascending=True).head(5)
    fallback = case_delta[case_delta["delta_fallback_time_s"] > 0].sort_values(
        "delta_fallback_time_s", ascending=False
    )
    fallback.to_csv(raw / "relief_decay_expansion_fallback_cases.csv", index=False)

    text = [
        "# Relief/Decay Expansion Result",
        "",
        "This run uses 24 new test-split cases selected by a fixed wind-shape rule: near-term high/rising wind followed by clear 60-120min decay, no large direction-shift flag, and at least 6h away from existing locked53 cases. The controller was not tuned on this casebook.",
        "",
        "## Aggregate Result",
        "",
        md_table(
            summary,
            [
                "cases",
                "pump_saving_pct",
                "pump_saved_m3",
                "delta_time_gt5_s",
                "delta_idle_gt5_s",
                "delta_fallback_time_s",
                "mean_delta_p95_max_axis_deg",
                "cases_with_added_fallback",
                "cases_with_negative_saving",
            ],
        ),
        "",
        "## Top Saving Cases",
        "",
        md_table(
            top,
            [
                "case",
                "pump_saved_m3",
                "pump_saving_pct",
                "delta_time_gt5_s",
                "delta_fallback_time_s",
                "delta_p95_max_axis_deg",
            ],
        ),
        "",
        "## Worst Pump Cases",
        "",
        md_table(
            bad,
            [
                "case",
                "pump_saved_m3",
                "pump_saving_pct",
                "delta_time_gt5_s",
                "delta_fallback_time_s",
                "delta_p95_max_axis_deg",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "This expansion directly tests whether the approximately-20% aggressive Pareto effect appears beyond the original casebook. The correct paper use depends on the aggregate row: if pump saving remains near or above 20%, the relief/decay regime is a broad opportunity class; if safety-margin metrics rise, the result remains an operator-selectable Pareto mode rather than a safety-neutral automatic controller.",
        "",
    ]
    (paper / "relief_decay_expansion_summary.md").write_text(
        "\n".join(text), encoding="utf-8"
    )
    print("Wrote relief/decay expansion summary to", OUT)


if __name__ == "__main__":
    main()
