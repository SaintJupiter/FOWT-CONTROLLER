#!/usr/bin/env python3
"""Audit whether the 10-case prediction-primary calibration set is outcome-biased.

This script is deliberately descriptive.  It does not run the plant/controller
simulation and it does not change any controller code.  It records where the
current 10-case list came from, compares it with the earlier physical
mechanism-selection script output, and summarizes calibration-vs-holdout
forecast-source pump outcomes when those CSVs exist.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


BUILTIN_CASES = [
    ("01_onset_strong", "2024-11-27 19:40:00", "Strong onset: calm now, very strong future wind"),
    ("02_onset_signflip", "2023-10-31 06:20:00", "Onset with direction flip"),
    ("03_onset_moderate", "2024-10-10 03:50:00", "Moderate onset"),
    ("04_decay_strong", "2024-09-27 13:00:00", "Strong decay: high now, weak future wind"),
    ("05_decay_signflip", "2022-03-20 19:00:00", "Decay with direction flip"),
    ("06_signflip_high", "2023-10-03 06:30:00", "High-pressure sign-flip"),
    ("07_signflip_sustained", "2023-03-14 04:40:00", "Sustained high sign-flip"),
    ("08_lowrisk_quiet", "2021-12-20 14:30:00", "Low-risk quiet window"),
    ("09_high_pressure_event", "2022-02-04 11:00:00", "High-pressure high-event normal window"),
    ("10_residual_high", "2024-09-05 18:10:00", "Residual-high normal window"),
]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _read_summary(path: Path, source: str, cohort: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source"] = source
    df["cohort"] = cohort
    return df


def _pump_outcome_table(run_specs: list[tuple[str, str, Path]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cohort, source, path in run_specs:
        if path.exists():
            rows.append(_read_summary(path, source, cohort))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    pivot = df.pivot_table(
        index=["cohort", "case_id"],
        columns="source",
        values=["primary_pump_work_m3", "closed_pump_work_m3", "primary_pitch_p95", "primary_roll_p95"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{source}" for metric, source in pivot.columns]
    pivot = pivot.reset_index()
    for source in ("learned", "persistence", "oracle"):
        col = f"primary_pump_work_m3_{source}"
        closed_col = f"closed_pump_work_m3_{source}"
        if col in pivot.columns and closed_col in pivot.columns:
            pivot[f"{source}_vs_closed_pct"] = (
                (pivot[col] / pivot[closed_col].clip(lower=1e-9) - 1.0) * 100.0
            )
    if "primary_pump_work_m3_learned" in pivot.columns and "primary_pump_work_m3_persistence" in pivot.columns:
        pivot["learned_minus_persistence_m3"] = (
            pivot["primary_pump_work_m3_learned"] - pivot["primary_pump_work_m3_persistence"]
        )
        pivot["learned_vs_persistence_pct"] = (
            pivot["learned_minus_persistence_m3"]
            / pivot["primary_pump_work_m3_persistence"].clip(lower=1e-9)
            * 100.0
        )
    if "primary_pump_work_m3_oracle" in pivot.columns and "primary_pump_work_m3_persistence" in pivot.columns:
        pivot["oracle_minus_persistence_m3"] = (
            pivot["primary_pump_work_m3_oracle"] - pivot["primary_pump_work_m3_persistence"]
        )
        pivot["oracle_vs_persistence_pct"] = (
            pivot["oracle_minus_persistence_m3"]
            / pivot["primary_pump_work_m3_persistence"].clip(lower=1e-9)
            * 100.0
        )
    return pivot


def _cohort_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return outcomes
    rows = []
    for cohort, sub in outcomes.groupby("cohort", sort=True):
        row = {"cohort": cohort, "cases": int(len(sub))}
        for source in ("learned", "persistence", "oracle"):
            col = f"primary_pump_work_m3_{source}"
            closed_col = f"closed_pump_work_m3_{source}"
            if col in sub.columns:
                row[f"{source}_pump_m3"] = float(sub[col].sum())
            if col in sub.columns and closed_col in sub.columns:
                row[f"{source}_vs_closed_pct"] = float(
                    (sub[col].sum() / max(float(sub[closed_col].sum()), 1e-9) - 1.0) * 100.0
                )
        for col in ("learned_minus_persistence_m3", "oracle_minus_persistence_m3"):
            if col in sub.columns:
                vals = sub[col].dropna()
                row[f"{col}_sum"] = float(vals.sum())
                row[f"{col}_median"] = float(vals.median()) if len(vals) else float("nan")
                row[f"{col}_wins_lt0"] = int((vals < 0.0).sum())
                row[f"{col}_losses_gt0"] = int((vals > 0.0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_none_"
    if max_rows is not None:
        df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/wind_prediction/calibration_selection_audit_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builtin = pd.DataFrame(BUILTIN_CASES, columns=["case_id", "timestamp", "label"])
    builtin["timestamp"] = pd.to_datetime(builtin["timestamp"])

    mechanism_path = REPO_ROOT / "outputs/wind_prediction/prediction_primary_mechanism_selection_v2/mechanism_validation_cases.csv"
    mechanism = pd.read_csv(mechanism_path) if mechanism_path.exists() else pd.DataFrame()
    if not mechanism.empty:
        mechanism["timestamp"] = pd.to_datetime(mechanism["timestamp"])

    if not mechanism.empty:
        match = builtin.merge(
            mechanism[["timestamp", "selection_group", "first_action", "best_sequence", "selection_score"]],
            on="timestamp",
            how="left",
        )
    else:
        match = builtin.copy()
    match["in_mechanism_selection_v2"] = match.get("selection_group", pd.Series([None] * len(match))).notna()
    match.to_csv(out_dir / "builtin_10case_selection_trace.csv", index=False)

    run_specs = [
        (
            "calibration_10case",
            "learned",
            REPO_ROOT / "outputs/wind_prediction/forecast_value_rawenv_learned_holdpause_v2_10case_2h/casebook_summary.csv",
        ),
        (
            "calibration_10case",
            "persistence",
            REPO_ROOT / "outputs/wind_prediction/forecast_value_rawenv_persistence_holdpause_v2_10case_2h/casebook_summary.csv",
        ),
        (
            "calibration_10case",
            "oracle",
            REPO_ROOT / "outputs/wind_prediction/forecast_value_rawenv_oracle_holdpause_v2_10case_2h/casebook_summary.csv",
        ),
        (
            "holdout_relief_12case",
            "learned",
            REPO_ROOT / "outputs/wind_prediction/holdout_relief_learned_rawenv_holdpause_v2_12case_2h/casebook_summary.csv",
        ),
        (
            "holdout_relief_12case",
            "persistence",
            REPO_ROOT / "outputs/wind_prediction/holdout_relief_persistence_rawenv_holdpause_v2_12case_2h/casebook_summary.csv",
        ),
        (
            "holdout_relief_12case",
            "oracle",
            REPO_ROOT / "outputs/wind_prediction/holdout_relief_oracle_rawenv_holdpause_v2_12case_2h/casebook_summary.csv",
        ),
        (
            "holdout_signflip_2case",
            "learned",
            REPO_ROOT / "outputs/wind_prediction/holdout_signflip_learned_rawenv_holdpause_v2_2case_2h/casebook_summary.csv",
        ),
        (
            "holdout_signflip_2case",
            "persistence",
            REPO_ROOT / "outputs/wind_prediction/holdout_signflip_persistence_rawenv_holdpause_v2_2case_2h/casebook_summary.csv",
        ),
        (
            "holdout_signflip_2case",
            "oracle",
            REPO_ROOT / "outputs/wind_prediction/holdout_signflip_oracle_rawenv_holdpause_v2_2case_2h/casebook_summary.csv",
        ),
    ]
    outcomes = _pump_outcome_table(run_specs)
    outcomes.to_csv(out_dir / "forecast_source_outcome_by_case.csv", index=False)
    cohort_summary = _cohort_summary(outcomes)
    cohort_summary.to_csv(out_dir / "forecast_source_outcome_by_cohort.csv", index=False)

    lines = [
        "# Calibration Selection Audit v1",
        "",
        "Purpose: check whether the current 10-case prediction-primary calibration set can be treated as an independent forecast-value sample.",
        "",
        "## Source Of The 10 Cases",
        "",
        "- The current `run_prediction_primary_casebook.py` contains a hard-coded 10-case list intended to cover physical wind regimes: onset, decay, signflip, low-risk, high-pressure, and residual-high.",
        "- The list is not generated from learned-vs-persistence or oracle pump outcomes inside that script.",
        "- However, several timestamps overlap earlier mechanism scans and prior manual studies, so it should be treated as a calibration/mechanism set, not a statistical holdout.",
        "",
        "## Built-In 10 Case Trace",
        "",
        _markdown_table(match.assign(timestamp=match["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"))),
        "",
        "## Forecast-Source Pump Outcomes",
        "",
        _markdown_table(cohort_summary),
        "",
        "## Audit Verdict",
        "",
        "1. The 10-case set is defensible as a physical mechanism/calibration set because case labels are wind-regime labels, not forecast-source outcome labels.",
        "2. It is not defensible as an independent statistical proof that learned forecast generally beats persistence, because the set was used repeatedly during mechanism development.",
        "3. The oracle-vs-persistence reversal between calibration and holdout is real and must be discussed: oracle is useful as a diagnostic source, not a monotonic pump upper bound.",
        "4. For paper claims, use the 10-case set for mechanism figures and architecture-vs-closed evidence; use holdout/random/forecast-level audits for forecast-source generalization.",
    ]
    (out_dir / "calibration_selection_audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {out_dir / 'calibration_selection_audit_report.md'}")


if __name__ == "__main__":
    main()
