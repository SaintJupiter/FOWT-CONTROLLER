#!/usr/bin/env python3
"""Build replay-grounded labels for a small PSC mode selector.

The label table is case-level: it compares a candidate economy arm against a
safety/default arm on the same replay ladder and marks when the economy arm is
both safe and useful. These labels are for a selector head/classifier, not for
training another wind forecaster.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/wind_prediction/psc_selector_mixed_pool_v1/degradation_ladder_24case/degradation_ladder_case_metrics.csv"),
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--safety-arm", default="current_forecast_adaptive")
    parser.add_argument("--economy-arm", default="learned_rawenv_mainline")
    parser.add_argument("--pump-margin-m3", type=float, default=50.0)
    parser.add_argument("--time-gt5-tol-s", type=float, default=0.0)
    parser.add_argument("--fallback-tol-s", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.metrics)
    safety = metrics[metrics["arm"].eq(args.safety_arm)].set_index("case_id")
    economy = metrics[metrics["arm"].eq(args.economy_arm)].set_index("case_id")
    common = sorted(set(safety.index) & set(economy.index))
    rows = []
    for case_id in common:
        s = safety.loc[case_id]
        e = economy.loc[case_id]
        d_pump = float(e["pump_m3"] - s["pump_m3"])
        d_time = float(e["time_gt5_s"] - s["time_gt5_s"])
        d_fallback = float(e["fallback_s"] - s["fallback_s"])
        rawenv_safe = d_time <= float(args.time_gt5_tol_s) and d_fallback <= float(args.fallback_tol_s)
        rawenv_pump_saves = d_pump <= -float(args.pump_margin_m3)
        rows.append(
            {
                "case_id": case_id,
                "label": str(s.get("label", "")),
                "safety_arm": args.safety_arm,
                "economy_arm": args.economy_arm,
                "safety_pump_m3": float(s["pump_m3"]),
                "economy_pump_m3": float(e["pump_m3"]),
                "d_pump_m3": d_pump,
                "safety_time_gt5_s": float(s["time_gt5_s"]),
                "economy_time_gt5_s": float(e["time_gt5_s"]),
                "d_time_gt5_s": d_time,
                "safety_fallback_s": float(s["fallback_s"]),
                "economy_fallback_s": float(e["fallback_s"]),
                "d_fallback_s": d_fallback,
                "rawenv_safe": int(rawenv_safe),
                "rawenv_pump_saves": int(rawenv_pump_saves),
                "use_rawenv": int(rawenv_safe and rawenv_pump_saves),
            }
        )
    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    summary = (
        out[["rawenv_safe", "rawenv_pump_saves", "use_rawenv"]]
        .agg(["sum", "mean"])
        .T.reset_index()
        .rename(columns={"index": "label_name"})
    )
    summary.to_csv(args.output_csv.with_name(args.output_csv.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))
    print(args.output_csv)


if __name__ == "__main__":
    main()
