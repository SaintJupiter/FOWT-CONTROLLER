#!/usr/bin/env python3
"""Run deterministic offline structural PSC selectors on the 24-case pool.

The selector consumes the Stage-0 diagnostic table and chooses between the
safe/default arm and the rawenv economy arm using transparent case-level gates.
It is intentionally offline: no controller logic is changed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("outputs/wind_prediction/psc_selector_mixed_pool_v1")
DEFAULT_DIAG = DEFAULT_ROOT / "psc_case_diagnostic_24case_6h_v1" / "case_diagnostic_table.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-table", type=Path, default=DEFAULT_DIAG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "psc_structural_selector_neutral_v1",
    )
    parser.add_argument(
        "--policy",
        choices=["neutral_only", "neutral_plus_safe_relief_probe", "neutral_plus_relief_v2_probe"],
        default="neutral_only",
    )
    parser.add_argument(
        "--feature-source",
        choices=["learned", "current_only"],
        default="learned",
        help="Feature source used for diagnostic scores and falsifier summaries.",
    )
    parser.add_argument(
        "--neutral-regime",
        default="neutral_mhs_broader",
        help="Casebook regime released by the neutral structural selector.",
    )
    parser.add_argument(
        "--use-pump-margin",
        action="store_true",
        help="If set, only release cases that meet the replay label pump margin. Default releases the full neutral block.",
    )
    parser.add_argument(
        "--max-risk-severity",
        type=float,
        default=0.0,
        help="Require risk_severity_score <= this value for structural release.",
    )
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
    return parser.parse_args()


def _safe_bool(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(bool)


def _select(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    neutral = out["mixed_regime"].astype(str).eq(args.neutral_regime)
    low_risk = pd.to_numeric(out["risk_severity_score"], errors="coerce").fillna(np.inf).le(args.max_risk_severity)
    pump_ok = _safe_bool(out["rawenv_pump_saves"]) if args.use_pump_margin else pd.Series(True, index=out.index)

    relief = out["mixed_regime"].astype(str).eq(args.relief_regime)
    feature_prefix = "learned" if args.feature_source == "learned" else "current"
    sustained_col = f"{feature_prefix}_prob_sustained_high_mean"
    opportunity_col = f"{feature_prefix}_opportunity_mean"
    reintensification_col = f"{feature_prefix}_prob_reintensification_mean"
    reintensification_max_col = f"{feature_prefix}_event_reintensification_max"
    ramp_onset_col = f"{feature_prefix}_prob_ramp_onset_mean"
    relief_signature = (
        relief
        & pd.to_numeric(out[sustained_col], errors="coerce")
        .fillna(-np.inf)
        .ge(float(args.relief_sustained_high_min))
        & pd.to_numeric(out[opportunity_col], errors="coerce")
        .fillna(np.inf)
        .ge(float(args.relief_opportunity_mean_min))
        & pd.to_numeric(out[opportunity_col], errors="coerce")
        .fillna(np.inf)
        .le(float(args.relief_opportunity_mean_max))
    )
    relief_low_opp_signature = (
        relief
        & pd.to_numeric(out[sustained_col], errors="coerce")
        .fillna(-np.inf)
        .ge(float(args.relief_low_opp_sustained_high_min))
        & pd.to_numeric(out[opportunity_col], errors="coerce")
        .fillna(np.inf)
        .ge(float(args.relief_low_opp_mean_min))
        & pd.to_numeric(out[opportunity_col], errors="coerce")
        .fillna(np.inf)
        .le(float(args.relief_low_opp_mean_max))
    )
    relief_late_signature = (
        relief
        & pd.to_numeric(out[opportunity_col], errors="coerce")
        .fillna(np.inf)
        .ge(float(args.relief_late_opp_mean_min))
        & pd.to_numeric(out[opportunity_col], errors="coerce")
        .fillna(np.inf)
        .le(float(args.relief_late_opp_mean_max))
        & pd.to_numeric(out[reintensification_col], errors="coerce")
        .fillna(-np.inf)
        .ge(float(args.relief_late_reintensification_mean_min))
        & pd.to_numeric(out[reintensification_max_col], errors="coerce")
        .fillna(-np.inf)
        .ge(float(args.relief_late_reintensification_max_min))
        & pd.to_numeric(out[ramp_onset_col], errors="coerce")
        .fillna(np.inf)
        .le(float(args.relief_late_ramp_onset_max))
    )

    reasons: list[str] = []
    neutral_release = neutral & low_risk & pump_ok
    if args.policy == "neutral_plus_safe_relief_probe":
        relief_release = relief_signature & low_risk & pump_ok
    elif args.policy == "neutral_plus_relief_v2_probe":
        relief_release = (relief_signature | relief_low_opp_signature | relief_late_signature) & low_risk & pump_ok
    else:
        relief_release = pd.Series(False, index=out.index)
    use_economy = neutral_release | relief_release
    for idx, row in out.iterrows():
        selected = bool(use_economy.loc[idx])
        if selected and row.get("mixed_regime") == args.neutral_regime:
            reasons.append("neutral_low_risk_release")
        elif selected and bool(relief_signature.loc[idx]):
            reasons.append("relief_signature_probe_release")
        elif selected and bool(relief_low_opp_signature.loc[idx]):
            reasons.append("relief_low_opportunity_v2_release")
        elif selected and bool(relief_late_signature.loc[idx]):
            reasons.append("relief_late_reintensification_v2_release")
        elif args.policy in {"neutral_plus_safe_relief_probe", "neutral_plus_relief_v2_probe"} and row.get("mixed_regime") == args.relief_regime:
            reasons.append("relief_signature_or_risk_block")
        elif row.get("mixed_regime") != args.neutral_regime:
            reasons.append("not_neutral_regime")
        elif float(row.get("risk_severity_score", math.inf)) > args.max_risk_severity:
            reasons.append("risk_severity_block")
        elif args.use_pump_margin and not bool(row.get("rawenv_pump_saves", 0)):
            reasons.append("pump_margin_block")
        else:
            reasons.append("default_track")

    out["selector"] = f"{args.feature_source}_{args.policy}"
    out["use_economy"] = use_economy.astype(int)
    out["selected_arm"] = np.where(out["use_economy"].eq(1), "learned_rawenv_mainline", "current_forecast_adaptive")
    out["decision_reason"] = reasons
    out["selected_pump_m3"] = np.where(out["use_economy"].eq(1), out["economy_pump_m3"], out["safety_pump_m3"])
    out["selected_time_gt5_s"] = np.where(out["use_economy"].eq(1), out["economy_time_gt5_s"], out["safety_time_gt5_s"])
    out["selected_time_gt6_s"] = np.where(out["use_economy"].eq(1), out["economy_time_gt6_s"], out["safety_time_gt6_s"])
    out["selected_fallback_s"] = np.where(out["use_economy"].eq(1), out["economy_fallback_s"], out["safety_fallback_s"])
    out["selected_p95_axis_deg"] = np.where(out["use_economy"].eq(1), out["economy_p95_axis_deg"], out["safety_p95_axis_deg"])
    out["selected_max_axis_deg"] = np.where(out["use_economy"].eq(1), out["economy_max_axis_deg"], out["safety_max_axis_deg"])
    out["selected_d_pump_m3"] = out["selected_pump_m3"] - out["safety_pump_m3"]
    out["selected_d_time_gt5_s"] = out["selected_time_gt5_s"] - out["safety_time_gt5_s"]
    out["selected_d_time_gt6_s"] = out["selected_time_gt6_s"] - out["safety_time_gt6_s"]
    out["selected_d_fallback_s"] = out["selected_fallback_s"] - out["safety_fallback_s"]
    return out


def _summarize(name: str, selected: pd.DataFrame) -> dict[str, Any]:
    economy = selected[selected["use_economy"].eq(1)]
    positives_opened = int((economy["use_rawenv"].astype(int) == 1).sum())
    risk_opened = int((economy["risk_marker"].astype(int) == 1).sum())
    return {
        "selector": name,
        "cases": int(len(selected)),
        "economy_cases": int(selected["use_economy"].sum()),
        "use_rawenv_opened": positives_opened,
        "risk_marker_opened": risk_opened,
        "pump_m3": float(selected["selected_pump_m3"].sum()),
        "time_gt5_s": float(selected["selected_time_gt5_s"].sum()),
        "time_gt6_s": float(selected["selected_time_gt6_s"].sum()),
        "fallback_s": float(selected["selected_fallback_s"].sum()),
        "d_pump_vs_safety_m3": float(selected["selected_d_pump_m3"].sum()),
        "d_time_gt5_vs_safety_s": float(selected["selected_d_time_gt5_s"].sum()),
        "d_time_gt6_vs_safety_s": float(selected["selected_d_time_gt6_s"].sum()),
        "d_fallback_vs_safety_s": float(selected["selected_d_fallback_s"].sum()),
        "opened_case_ids": ";".join(economy["case_id"].astype(str).tolist()),
        "opened_risk_case_ids": ";".join(economy[economy["risk_marker"].astype(int).eq(1)]["case_id"].astype(str).tolist()),
    }


def _references(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "selector": "safety_reference",
            "cases": int(len(df)),
            "economy_cases": 0,
            "use_rawenv_opened": 0,
            "risk_marker_opened": 0,
            "pump_m3": float(df["safety_pump_m3"].sum()),
            "time_gt5_s": float(df["safety_time_gt5_s"].sum()),
            "time_gt6_s": float(df["safety_time_gt6_s"].sum()),
            "fallback_s": float(df["safety_fallback_s"].sum()),
            "d_pump_vs_safety_m3": 0.0,
            "d_time_gt5_vs_safety_s": 0.0,
            "d_time_gt6_vs_safety_s": 0.0,
            "d_fallback_vs_safety_s": 0.0,
            "opened_case_ids": "",
            "opened_risk_case_ids": "",
        },
        {
            "selector": "oracle_replay_label_selector",
            "cases": int(len(df)),
            "economy_cases": int(df["use_rawenv"].sum()),
            "use_rawenv_opened": int(df["use_rawenv"].sum()),
            "risk_marker_opened": int(((df["use_rawenv"].astype(int) == 1) & (df["risk_marker"].astype(int) == 1)).sum()),
            "pump_m3": float(np.where(df["use_rawenv"].eq(1), df["economy_pump_m3"], df["safety_pump_m3"]).sum()),
            "time_gt5_s": float(np.where(df["use_rawenv"].eq(1), df["economy_time_gt5_s"], df["safety_time_gt5_s"]).sum()),
            "time_gt6_s": float(np.where(df["use_rawenv"].eq(1), df["economy_time_gt6_s"], df["safety_time_gt6_s"]).sum()),
            "fallback_s": float(np.where(df["use_rawenv"].eq(1), df["economy_fallback_s"], df["safety_fallback_s"]).sum()),
            "d_pump_vs_safety_m3": float(np.where(df["use_rawenv"].eq(1), df["d_pump_m3"], 0.0).sum()),
            "d_time_gt5_vs_safety_s": float(np.where(df["use_rawenv"].eq(1), df["d_time_gt5_s"], 0.0).sum()),
            "d_time_gt6_vs_safety_s": float(np.where(df["use_rawenv"].eq(1), df["d_time_gt6_s"], 0.0).sum()),
            "d_fallback_vs_safety_s": float(np.where(df["use_rawenv"].eq(1), df["d_fallback_s"], 0.0).sum()),
            "opened_case_ids": ";".join(df[df["use_rawenv"].eq(1)]["case_id"].astype(str).tolist()),
            "opened_risk_case_ids": "",
        },
    ]
    return pd.DataFrame(rows)


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
            if isinstance(val, (float, np.floating)):
                vals.append("nan" if math.isnan(float(val)) else f"{float(val):.4g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_readout(out_dir: Path, selected: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    opened = selected[selected["use_economy"].eq(1)].copy()
    lines = [
        f"# PSC structural selector {args.policy}",
        "",
        "Offline deterministic selector over already-run 6h arms.",
        "",
        "## Summary",
        "",
        _md_table(summary),
        "",
        "## Opened Cases",
        "",
        _md_table(
            opened[
                [
                    "case_id",
                    "mixed_regime",
                    "use_rawenv",
                    "risk_marker",
                    "decision_reason",
                    "d_pump_m3",
                    "d_time_gt5_s",
                    "d_time_gt6_s",
                    "d_fallback_s",
                ]
            ]
        ),
        "",
        "## Policy",
        "",
        f"- policy: `{args.policy}`",
        f"- feature_source label: `{args.feature_source}`",
        f"- neutral_regime: `{args.neutral_regime}`",
        f"- max_risk_severity: `{args.max_risk_severity}`",
        f"- use_pump_margin: `{args.use_pump_margin}`",
        f"- relief_sustained_high_min: `{args.relief_sustained_high_min}`",
        f"- relief_opportunity_mean_min/max: `{args.relief_opportunity_mean_min}` / `{args.relief_opportunity_mean_max}`",
        f"- relief_low_opp_sustained_high_min: `{args.relief_low_opp_sustained_high_min}`",
        f"- relief_low_opp_mean_min/max: `{args.relief_low_opp_mean_min}` / `{args.relief_low_opp_mean_max}`",
        f"- relief_late_opp_mean_min/max: `{args.relief_late_opp_mean_min}` / `{args.relief_late_opp_mean_max}`",
        f"- relief_late_reintensification_mean_min: `{args.relief_late_reintensification_mean_min}`",
        f"- relief_late_reintensification_max_min: `{args.relief_late_reintensification_max_min}`",
        f"- relief_late_ramp_onset_max: `{args.relief_late_ramp_onset_max}`",
        "",
    ]
    (out_dir / "structural_selector_readout.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.diagnostic_table)
    selected = _select(df, args)
    summary = pd.concat(
        [
            pd.DataFrame([_summarize(f"{args.feature_source}_{args.policy}", selected)]),
            _references(df),
        ],
        ignore_index=True,
    )
    selected.to_csv(args.output_dir / "structural_selector_case_decisions.csv", index=False)
    summary.to_csv(args.output_dir / "structural_selector_summary.csv", index=False)
    _write_readout(args.output_dir, selected, summary, args)
    (args.output_dir / "structural_selector_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(args.output_dir / "structural_selector_readout.md")


if __name__ == "__main__":
    main()
