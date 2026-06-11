#!/usr/bin/env python3
"""Build a feature/label table for PSC regime selector design.

The table is intentionally offline: it links pre-action forecast/regime features
from existing planner logs to outcomes from already-run candidate modes.  It is
used to design a structural selector, not to tune attitude deadbands.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LADDER = REPO_ROOT / "outputs/wind_prediction/psc_degradation_ladder_20260531"
DEFAULT_ORACLE = REPO_ROOT / "outputs/wind_prediction/oracle_classifier_psc_upper_bound_20260531"


FEATURE_COLS = [
    "pressure_block0_norm",
    "pressure_block1_norm",
    "pressure_block2_norm",
    "pressure_block02_dot",
    "raw_pressure_block0_norm",
    "raw_pressure_block1_norm",
    "raw_pressure_block2_norm",
    "raw_pressure_block02_dot",
    "forecast_speed_start_ms",
    "forecast_speed_early_max_ms",
    "forecast_speed_near_max_ms",
    "forecast_speed_far_max_ms",
    "forecast_speed_late_mean_ms",
    "forecast_speed_early_rise_ms",
    "forecast_speed_peak_to_late_mean_drop_ms",
    "forecast_speed_dir_shift_deg",
    "forecast_speed_oscillation_range_ms",
    "forecast_speed_near_range_ms",
    "forecast_speed_far_range_ms",
    "event_risk_prob_0_20m",
    "event_risk_prob_20_40m",
    "event_risk_prob_40_60m",
    "forecast_advised_economy_candidate",
    "forecast_advised_economy_boundary_veto",
    "forecast_advised_economy_near_rise_norm",
    "forecast_advised_economy_headroom_deg",
    "forecast_advised_economy_speed_range_ms",
    "forecast_advised_current_ws_range_ms",
    "forecast_advised_current_dir_shift_deg",
]


def _first_existing(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[0])


def _case_id_from_log(path: Path, arm: str) -> str:
    suffix = f"_{arm}_planner_log.csv"
    name = path.name
    if name.endswith(suffix):
        stem = name[: -len(suffix)]
    else:
        stem = name.replace("_planner_log.csv", "")
    parts = stem.split("_")
    if len(parts) >= 4 and parts[-2].count("-") == 2 and parts[-1].isdigit():
        return "_".join(parts[:-2])
    return stem


def _summarize_log(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, low_memory=False)
    row: dict[str, Any] = {}
    n = max(len(df), 1)
    for col in FEATURE_COLS:
        if col not in df.columns:
            row[f"{col}_mean"] = 0.0
            row[f"{col}_max"] = 0.0
            continue
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        row[f"{col}_mean"] = float(vals.mean(skipna=True)) if vals.notna().any() else 0.0
        row[f"{col}_max"] = float(vals.max(skipna=True)) if vals.notna().any() else 0.0
    for col in (
        "forecast_speed_relief_decay_candidate",
        "forecast_speed_gusty_oscillation_candidate",
        "forecast_speed_gusty_short_candidate",
        "forecast_advised_economy_confirmed",
        "forecast_advised_current_stability_veto",
    ):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            row[f"{col}_ratio"] = float((vals > 0.5).sum() / n)
        else:
            row[f"{col}_ratio"] = 0.0
    return row


def _load_case_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = [
        "arm",
        "case_id",
        "label",
        "pump_m3",
        "time_gt5_s",
        "time_gt6_s",
        "fallback_s",
        "latch_switches",
    ]
    return df[needed].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder-dir", type=Path, default=DEFAULT_LADDER)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument(
        "--feature-arm",
        default="learned_rawenv_mainline",
        help="Run whose planner logs provide learned forecast/regime features.",
    )
    args = parser.parse_args()

    ladder = args.ladder_dir
    oracle_dir = args.oracle_dir
    case_metrics = _load_case_metrics(ladder / "degradation_ladder_case_metrics.csv")
    oracle_choices = pd.read_csv(oracle_dir / "psc_oracle_regime_selector_v0_cases.csv")
    oracle_choices = oracle_choices[oracle_choices["selector"] == "oracle_safety_first"]
    oracle_choices = oracle_choices[["case_id", "arm"]].rename(
        columns={"arm": "oracle_selected_arm"}
    )

    feature_dir = ladder / args.feature_arm / "planner_logs"
    rows: list[dict[str, Any]] = []
    for log_path in sorted(feature_dir.glob("*_planner_log.csv")):
        case_id = _case_id_from_log(log_path, args.feature_arm)
        row = {"case_id": case_id, "feature_arm": args.feature_arm}
        row.update(_summarize_log(log_path))
        rows.append(row)
    features = pd.DataFrame(rows)

    wide: dict[str, dict[str, Any]] = {}
    for rec in case_metrics.to_dict("records"):
        cid = str(rec["case_id"])
        entry = wide.setdefault(
            cid,
            {
                "case_id": cid,
                "label": str(rec.get("label", "")),
            },
        )
        arm = str(rec["arm"])
        for metric in ("pump_m3", "time_gt5_s", "time_gt6_s", "fallback_s", "latch_switches"):
            entry[f"{arm}_{metric}"] = rec[metric]
    outcomes = pd.DataFrame(wide.values())
    table = features.merge(outcomes, on="case_id", how="left").merge(
        oracle_choices, on="case_id", how="left"
    )

    def _metric(row: pd.Series, arm: str, metric: str) -> float:
        return float(row.get(f"{arm}_{metric}", np.nan))

    labels: list[dict[str, Any]] = []
    for row in table.to_dict("records"):
        blind_t5 = float(row.get("blind_deadband_time_gt5_s", np.nan))
        blind_pump = float(row.get("blind_deadband_pump_m3", np.nan))
        raw_t5 = float(row.get("learned_rawenv_mainline_time_gt5_s", np.nan))
        raw_fb = float(row.get("learned_rawenv_mainline_fallback_s", np.nan))
        raw_pump = float(row.get("learned_rawenv_mainline_pump_m3", np.nan))
        fa_t5 = float(row.get("learned_forecast_adaptive_time_gt5_s", np.nan))
        fa_fb = float(row.get("learned_forecast_adaptive_fallback_s", np.nan))
        labels.append(
            {
                "case_id": row["case_id"],
                "rawenv_safe": int(raw_fb == 0 and raw_t5 <= blind_t5),
                "rawenv_danger": int(raw_fb > 0 or raw_t5 > blind_t5 + 60),
                "rawenv_pump_saves": int(raw_pump < blind_pump - 50),
                "forecast_adaptive_safe": int(fa_fb == 0),
                "forecast_adaptive_posture_win": int(fa_fb == 0 and fa_t5 < blind_t5),
            }
        )
    label_df = pd.DataFrame(labels)
    table = table.merge(label_df, on="case_id", how="left")

    out_csv = ladder / "psc_selector_feature_table_v0.csv"
    table.to_csv(out_csv, index=False)

    view_cols = [
        "case_id",
        "label",
        "oracle_selected_arm",
        "rawenv_safe",
        "rawenv_danger",
        "rawenv_pump_saves",
        "forecast_adaptive_posture_win",
        "raw_pressure_block0_norm_mean",
        "raw_pressure_block2_norm_mean",
        "raw_pressure_block02_dot_mean",
        "forecast_speed_dir_shift_deg_max",
        "forecast_speed_peak_to_late_mean_drop_ms_mean",
        "event_risk_prob_0_20m_max",
        "event_risk_prob_40_60m_max",
    ]
    view = table[[c for c in view_cols if c in table.columns]].copy()
    report = [
        "# PSC selector feature table v0 - 2026-05-31",
        "",
        "Purpose: diagnose structural regime features for a prediction supervisor.",
        "Labels are outcome-derived from the existing degradation ladder; features come from learned planner logs.",
        "",
        "## Compact view",
        "",
        view.to_markdown(index=False) if _has_tabulate() else _md_table(view),
        "",
        "## Readout",
        "",
        "- `rawenv_danger=1` marks cases where learned rawenv/economy creates fallback or a large high-posture tail.",
        "- `forecast_adaptive_posture_win=1` marks cases where learned forecast-adaptive improves time>5 without fallback.",
        "- Use this table to design a mode selector; do not tune deadband thresholds from it.",
    ]
    out_md = ladder / "PSC_SELECTOR_FEATURE_TABLE_V0_20260531.md"
    out_md.write_text("\n".join(report) + "\n")
    print(out_csv.resolve())
    print(out_md.resolve())
    print(view.to_string(index=False))


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except Exception:
        return False
    return True


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for rec in df.to_dict("records"):
        vals = []
        for col in cols:
            val = rec[col]
            vals.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
