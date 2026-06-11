#!/usr/bin/env python3
"""Build a wider ex-ante relief/decay casebook from the test split.

The rule uses only future wind-shape information available to a forecast-based
case definition, not controller outcome.  It targets "high or rising wind in
the next 0-60 minutes, followed by a clear decay by 60-120 minutes" and spaces
samples apart so the casebook is not many overlapping windows of one storm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path("data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1")
OUT = Path("outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1/relief_decay_expansion_v1")
CASEBOOK = OUT / "relief_decay_expansion_cases.csv"
RAW = OUT / "raw_tables"
LOCKED_CASEBOOK = Path(
    "outputs/wind_prediction/h120_relief_envelope_learned_a1_locked_holdout_v1/locked_casebook_cases.csv"
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(DATA / "sample_index.csv.gz")
    test = idx[idx["split"] == "test"].reset_index(drop=False).rename(columns={"index": "global_index"})
    y = np.load(DATA / "y_speed_dir_raw_test.npy", mmap_mode="r")
    speed = y[:, :, 0]

    early = speed[:, :6]
    late = speed[:, 6:]
    early_max = early.max(axis=1)
    late_min = late.min(axis=1)
    late_mean = late.mean(axis=1)
    start = speed[:, 0]
    peak_to_late_min_drop = early_max - late_min
    peak_to_late_mean_drop = early_max - late_mean
    early_rise = early_max - start

    cand = test.copy()
    cand["early_max_ms"] = early_max
    cand["late_min_ms"] = late_min
    cand["late_mean_ms"] = late_mean
    cand["early_rise_ms"] = early_rise
    cand["peak_to_late_min_drop_ms"] = peak_to_late_min_drop
    cand["peak_to_late_mean_drop_ms"] = peak_to_late_mean_drop

    # Strong, forecast-observable relief/decay opportunity.  The thresholds are
    # intentionally simple and pre-outcome: enough wind magnitude to matter,
    # a genuine near-term rise/peak, and a later decay.
    mask = (
        (cand["early_max_ms"] >= 16.0)
        & (cand["early_rise_ms"] >= 2.0)
        & (cand["peak_to_late_mean_drop_ms"] >= 3.0)
        & (cand["direction_shift_ge_45deg"] == 0)
    )
    cand = cand[mask].copy()
    cand["score"] = (
        cand["peak_to_late_mean_drop_ms"]
        + 0.35 * cand["early_rise_ms"]
        + 0.10 * cand["early_max_ms"]
    )
    cand["future_start_dt"] = pd.to_datetime(cand["future_start"])
    cand = cand.sort_values("score", ascending=False)

    selected = []
    min_gap = pd.Timedelta(hours=8)
    locked_times = []
    if LOCKED_CASEBOOK.exists():
        locked = pd.read_csv(LOCKED_CASEBOOK)
        locked_times = pd.to_datetime(locked["timestamp"], errors="coerce").dropna().tolist()
    locked_gap = pd.Timedelta(hours=6)
    for _, row in cand.iterrows():
        t = row["future_start_dt"]
        if any(abs(t - prev) < locked_gap for prev in locked_times):
            continue
        if any(abs(t - prev["future_start_dt"]) < min_gap for prev in selected):
            continue
        selected.append(row)
        if len(selected) >= 24:
            break

    cases = pd.DataFrame(selected)
    if cases.empty:
        raise SystemExit("No relief/decay expansion candidates selected")
    out = pd.DataFrame(
        {
            "case_id": [
                f"relief_decay_exp_{i:02d}" for i in range(1, len(cases) + 1)
            ],
            "timestamp": cases["future_start_dt"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "label": [
                (
                    "fixed_rule_relief_decay_expansion"
                    f" | early_max={r.early_max_ms:.1f}"
                    f" | early_rise={r.early_rise_ms:.1f}"
                    f" | drop_mean={r.peak_to_late_mean_drop_ms:.1f}"
                )
                for r in cases.itertuples()
            ],
        }
    )
    out.to_csv(CASEBOOK, index=False)
    cases.drop(columns=["future_start_dt"]).to_csv(
        RAW / "relief_decay_expansion_candidate_table.csv", index=False
    )
    print(f"Wrote {len(out)} cases to {CASEBOOK}")


if __name__ == "__main__":
    main()
