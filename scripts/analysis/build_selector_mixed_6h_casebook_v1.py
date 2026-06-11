#!/usr/bin/env python3
"""Build a mixed 6h selector validation casebook.

The casebook is stratified: it combines positive deadband-opportunity regimes
with boundary and low-opportunity negative controls. It is used to test the
deployment selector before scaling to broader 6h sets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs" / "wind_prediction" / "regime_conditioned_policy_development_v1"
OUT = REPO / "outputs" / "wind_prediction" / "selector_mixed_6h_v1"
CASEBOOKS = OUT / "casebooks"
RAW = OUT / "raw_tables"

SOURCES = [
    {
        "name": "p2_neutral_headroom",
        "role": "positive_allow",
        "gate_decision": "allow_deadband",
        "path": ROOT
        / "clean_neutral_headroom_expansion_v1"
        / "casebooks"
        / "clean_neutral_headroom_expanded_cases.csv",
    },
    {
        "name": "c3_gusty_oscillatory",
        "role": "positive_allow",
        "gate_decision": "allow_deadband",
        "path": ROOT / "hard_24h_pump_gate_v1" / "c3_gusty_24case_6h_casebook.csv",
    },
    {
        "name": "w1_stable_direction_event",
        "role": "positive_allow",
        "gate_decision": "allow_deadband",
        "path": ROOT
        / "hard_24h_pump_gate_v1"
        / "casebooks"
        / "high_attention_dir_stable_event_10_6h_cases.csv",
    },
    {
        "name": "direction_reversal_boundary",
        "role": "negative_abstain",
        "gate_decision": "abstain_deadband",
        "path": ROOT
        / "raw_tables"
        / "casebooks_v2"
        / "direction_reversal_catchup_boundary_cases.csv",
    },
    {
        "name": "reintensification_boundary",
        "role": "negative_abstain",
        "gate_decision": "abstain_deadband",
        "path": ROOT
        / "regime_mining_v3"
        / "raw_tables"
        / "casebooks_test"
        / "reintensification_boundary_cases.csv",
    },
    {
        "name": "low_opportunity_background",
        "role": "background_abstain",
        "gate_decision": "abstain_deadband",
        "path": ROOT
        / "regime_mining_v3"
        / "raw_tables"
        / "casebooks_test"
        / "quiet_low_opportunity_cases.csv",
    },
]


def _load_source(spec: dict[str, object], limit: int) -> pd.DataFrame:
    path = Path(spec["path"])
    df = pd.read_csv(path).head(limit)
    missing = {"case_id", "timestamp", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    out = df[["case_id", "timestamp", "label"]].copy()
    out["source_case_id"] = out["case_id"].astype(str)
    out["source_file"] = str(path.relative_to(REPO))
    out["selector_stratum"] = str(spec["name"])
    out["validation_role"] = str(spec["role"])
    out["gate_decision"] = str(spec["gate_decision"])
    out["requested_limit"] = int(limit)
    out["available_from_source"] = int(len(pd.read_csv(path)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-stratum-limit",
        type=int,
        default=4,
        help="Maximum number of cases to take from each stratum source.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT,
        help="Output root for casebooks and raw tables.",
    )
    parser.add_argument(
        "--casebook-name",
        default="selector_mixed_24case_6h_cases.csv",
        help="Casebook CSV filename written under the casebooks directory.",
    )
    args = parser.parse_args()

    if args.per_stratum_limit <= 0:
        raise ValueError("--per-stratum-limit must be positive")

    casebooks = args.out_dir / "casebooks"
    raw = args.out_dir / "raw_tables"
    casebooks.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    frames = [_load_source(spec, args.per_stratum_limit) for spec in SOURCES]
    manifest = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    manifest["case_id"] = [f"sel6h_{i:02d}" for i in range(1, len(manifest) + 1)]
    manifest["label"] = (
        manifest["label"].astype(str)
        + " | selector_stratum="
        + manifest["selector_stratum"].astype(str)
        + " | validation_role="
        + manifest["validation_role"].astype(str)
        + " | gate_decision="
        + manifest["gate_decision"].astype(str)
        + " | source_case_id="
        + manifest["source_case_id"].astype(str)
    )

    summary = (
        manifest.groupby(["validation_role", "selector_stratum", "gate_decision"], dropna=False)
        .agg(
            cases=("case_id", "size"),
            requested_limit=("requested_limit", "max"),
            available_from_source=("available_from_source", "max"),
        )
        .reset_index()
    )
    summary["source_shortfall"] = (
        summary["cases"].astype(int) < summary["requested_limit"].astype(int)
    )

    manifest.to_csv(raw / "selector_mixed_6h_manifest.csv", index=False)
    summary.to_csv(raw / "selector_mixed_6h_casebook_summary.csv", index=False)
    casebook = manifest[["case_id", "timestamp", "label"]]
    casebook.to_csv(casebooks / args.casebook_name, index=False)

    print(summary.to_string(index=False))
    print(casebooks / args.casebook_name)


if __name__ == "__main__":
    main()
