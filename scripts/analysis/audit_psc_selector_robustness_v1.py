#!/usr/bin/env python3
"""Audit PSC structural selector robustness on 24-case diagnostics.

This audit is intentionally conservative and transparent: it applies an already
specified structural selector to leave-one-case-out and leave-one-regime-out
evaluation folds.  It does not tune thresholds inside folds unless a future
caller explicitly adds that mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_psc_structural_selector_v1 import _select, parse_args as parse_selector_args


DEFAULT_ROOT = Path("outputs/wind_prediction/psc_selector_mixed_pool_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-table",
        type=Path,
        default=DEFAULT_ROOT / "psc_case_diagnostic_24case_6h_v1" / "case_diagnostic_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "psc_selector_robustness_audit_v1",
    )
    parser.add_argument("--policy", default="neutral_plus_safe_relief_probe")
    parser.add_argument("--feature-source", choices=["learned", "current_only"], default="learned")
    parser.add_argument("--neutral-regime", default="neutral_mhs_broader")
    parser.add_argument("--max-risk-severity", type=float, default=0.0)
    parser.add_argument("--relief-regime", default="transient_peak_future_decay")
    parser.add_argument("--relief-sustained-high-min", type=float, default=0.220)
    parser.add_argument("--relief-opportunity-mean-min", type=float, default=6.0)
    parser.add_argument("--relief-opportunity-mean-max", type=float, default=14.0)
    parser.add_argument("--relief-low-opp-sustained-high-min", type=float, default=0.400)
    parser.add_argument("--relief-low-opp-mean-min", type=float, default=2.5)
    parser.add_argument("--relief-low-opp-mean-max", type=float, default=4.0)
    parser.add_argument("--relief-late-opp-mean-min", type=float, default=20.0)
    parser.add_argument("--relief-late-opp-mean-max", type=float, default=60.0)
    parser.add_argument("--relief-late-reintensification-mean-min", type=float, default=0.750)
    parser.add_argument("--relief-late-reintensification-max-min", type=float, default=0.960)
    parser.add_argument("--relief-late-ramp-onset-max", type=float, default=0.130)
    parser.add_argument("--use-pump-margin", action="store_true")
    return parser.parse_args()


def _selector_namespace(args: argparse.Namespace) -> argparse.Namespace:
    # Reuse the selector implementation without depending on its CLI parser.
    return argparse.Namespace(
        diagnostic_table=args.diagnostic_table,
        output_dir=args.output_dir,
        policy=args.policy,
        feature_source=args.feature_source,
        neutral_regime=args.neutral_regime,
        use_pump_margin=args.use_pump_margin,
        max_risk_severity=args.max_risk_severity,
        relief_regime=args.relief_regime,
        relief_sustained_high_min=args.relief_sustained_high_min,
        relief_opportunity_mean_min=args.relief_opportunity_mean_min,
        relief_opportunity_mean_max=args.relief_opportunity_mean_max,
        relief_low_opp_sustained_high_min=args.relief_low_opp_sustained_high_min,
        relief_low_opp_mean_min=args.relief_low_opp_mean_min,
        relief_low_opp_mean_max=args.relief_low_opp_mean_max,
        relief_late_opp_mean_min=args.relief_late_opp_mean_min,
        relief_late_opp_mean_max=args.relief_late_opp_mean_max,
        relief_late_reintensification_mean_min=args.relief_late_reintensification_mean_min,
        relief_late_reintensification_max_min=args.relief_late_reintensification_max_min,
        relief_late_ramp_onset_max=args.relief_late_ramp_onset_max,
    )


def _summarize_fold(fold_name: str, eval_df: pd.DataFrame, selected: pd.DataFrame) -> dict[str, Any]:
    opened = selected[selected["use_economy"].astype(int).eq(1)]
    return {
        "fold": fold_name,
        "eval_cases": int(len(eval_df)),
        "economy_cases": int(opened.shape[0]),
        "use_rawenv_opened": int(opened["use_rawenv"].astype(int).sum()) if not opened.empty else 0,
        "risk_marker_opened": int(opened["risk_marker"].astype(int).sum()) if not opened.empty else 0,
        "d_pump_vs_safety_m3": float(opened["d_pump_m3"].sum()) if not opened.empty else 0.0,
        "d_time_gt5_vs_safety_s": float(opened["d_time_gt5_s"].sum()) if not opened.empty else 0.0,
        "d_time_gt6_vs_safety_s": float(opened["d_time_gt6_s"].sum()) if not opened.empty else 0.0,
        "d_fallback_vs_safety_s": float(opened["d_fallback_s"].sum()) if not opened.empty else 0.0,
        "opened_case_ids": ";".join(opened["case_id"].astype(str).tolist()),
        "opened_risk_case_ids": ";".join(opened[opened["risk_marker"].astype(int).eq(1)]["case_id"].astype(str).tolist()),
    }


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for rec in df.to_dict("records"):
        vals = []
        for col in df.columns:
            val = rec[col]
            if isinstance(val, float):
                vals.append(f"{val:.4g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.diagnostic_table)
    selector_args = _selector_namespace(args)

    full_selected = _select(df, selector_args)
    full_summary = _summarize_fold("full_pool", df, full_selected)

    rows: list[dict[str, Any]] = [full_summary]
    for case_id in df["case_id"].astype(str):
        eval_df = df[df["case_id"].astype(str).eq(case_id)].copy()
        selected = _select(eval_df, selector_args)
        rows.append(_summarize_fold(f"loocv_eval_{case_id}", eval_df, selected))

    for regime, eval_df in df.groupby("mixed_regime", sort=False):
        selected = _select(eval_df.copy(), selector_args)
        rows.append(_summarize_fold(f"regime_eval_{regime}", eval_df, selected))

    audit = pd.DataFrame(rows)
    audit["passes_no_risk_opened"] = audit["risk_marker_opened"].astype(int).eq(0).astype(int)
    audit.to_csv(args.output_dir / "selector_robustness_folds.csv", index=False)

    readout = [
        f"# PSC selector robustness audit - {args.feature_source} {args.policy}",
        "",
        "This fixed-rule audit evaluates selected cases in each fold; it does not tune thresholds per fold.",
        "",
        "## Aggregate",
        "",
        f"- folds: {len(audit)}",
        f"- folds with risk opened: {int((audit['risk_marker_opened'] > 0).sum())}",
        f"- worst d_time_gt5: {float(audit['d_time_gt5_vs_safety_s'].max()):.4g}",
        f"- worst fallback delta: {float(audit['d_fallback_vs_safety_s'].max()):.4g}",
        "",
        "## Folds",
        "",
        _md_table(audit),
        "",
    ]
    (args.output_dir / "selector_robustness_readout.md").write_text("\n".join(readout), encoding="utf-8")
    print(audit.to_string(index=False))
    print(args.output_dir / "selector_robustness_readout.md")


if __name__ == "__main__":
    main()
