#!/usr/bin/env python3
"""Build a mixed validation casebook for the two-specialist dispatcher."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
OUT = ROOT / "dual_specialist_dispatcher_v1"
CASEBOOKS = OUT / "casebooks"
RAW_DIR = OUT / "raw_tables"

RELIEF = Path(
    "outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1/"
    "relief_decay_expansion_v1/relief_decay_expansion_cases.csv"
)
NEUTRAL = (
    ROOT
    / "neutral_moderate_high_steady_broader_v1"
    / "casebooks"
    / "neutral_mhs_broader_cases.csv"
)
BOUNDARY = (
    ROOT
    / "raw_tables"
    / "casebooks_v2"
    / "direction_reversal_catchup_boundary_cases.csv"
)


def load_cases(path: Path, prefix: str, regime: str, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if limit is not None:
        df = df.head(limit)
    out = df[["timestamp", "label"]].copy()
    out["source_case_id"] = df["case_id"].astype(str)
    out["regime_group"] = regime
    out["case_id"] = [f"{prefix}_{i:02d}" for i in range(1, len(out) + 1)]
    out["label"] = out["label"].astype(str) + f" | mixed_regime={regime}"
    return out[["case_id", "timestamp", "label", "source_case_id", "regime_group"]]


def main() -> None:
    CASEBOOKS.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    parts = [
        load_cases(RELIEF, "dual_relief", "transient_peak_future_decay", 24),
        load_cases(NEUTRAL, "dual_neutral", "neutral_mhs_broader", None),
        load_cases(BOUNDARY, "dual_boundary", "direction_reversal_boundary", 8),
    ]
    mixed = pd.concat(parts, ignore_index=True)
    mixed.to_csv(RAW_DIR / "dual_specialist_mixed_case_table.csv", index=False)
    mixed[["case_id", "timestamp", "label"]].to_csv(
        CASEBOOKS / "dual_specialist_mixed_cases.csv",
        index=False,
    )
    summary = (
        mixed.groupby("regime_group")
        .agg(cases=("case_id", "size"))
        .reset_index()
    )
    summary.to_csv(RAW_DIR / "dual_specialist_mixed_casebook_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(CASEBOOKS / "dual_specialist_mixed_cases.csv")


if __name__ == "__main__":
    main()
