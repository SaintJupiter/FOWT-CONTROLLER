#!/usr/bin/env python3
"""Summarize where prediction value is expressed or blocked.

This script combines:

1. planner-only mechanism audit rows, and
2. raw 1 Hz signal-flow rows from closed-loop casebook runs.

It is diagnostic only. Thresholds here classify evidence strength; they are
not controller parameters.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SIGNAL_RUNS = {
    "default": {
        "signal_dir": "outputs/wind_prediction/prediction_signal_flow_default_10case_2h",
        "mechanism_envelope": "discounted",
    },
    "rawenv": {
        "signal_dir": "outputs/wind_prediction/prediction_signal_flow_rawenv_10case_2h",
        "mechanism_envelope": "raw",
    },
    "rawenv_holdpause_v2": {
        "signal_dir": "outputs/wind_prediction/prediction_signal_flow_rawenv_holdpause_v2_10case_2h",
        "mechanism_envelope": "raw",
    },
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def case_id_from_key(case_key: str) -> str:
    """Strip replay timestamp suffix from signal-flow case keys."""
    return re.sub(r"_\d{4}-\d{2}-\d{2}_\d{6}$", "", str(case_key))


def bool_count(series: pd.Series) -> int:
    return int(np.nansum(series.to_numpy(dtype=float)))


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f", max_rows: int | None = None) -> str:
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
        vals: list[str] = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(format(float(val), floatfmt) if np.isfinite(float(val)) else "")
            elif isinstance(val, (int, np.integer)):
                vals.append(str(int(val)))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def classify_bucket(row: pd.Series, pump_eps_m3: float, pump_sig_m3: float) -> str:
    delta = float(row["learned_minus_persistence_pump_m3"])
    action_diff = bool(int(row.get("lp_action_diff", 0)))
    if delta <= -pump_sig_m3:
        return "learned_saves_raw_pump_action_diff" if action_diff else "learned_saves_raw_pump_state_memory"
    if delta >= pump_sig_m3:
        return "learned_spends_extra_raw_pump_action_diff" if action_diff else "learned_spends_extra_raw_pump_state_memory"
    if abs(delta) <= pump_eps_m3:
        return "execution_washout_action_diff" if action_diff else "blocked_same_action"
    return "small_raw_effect_action_diff" if action_diff else "small_raw_effect_state_memory"


def load_signal_mode(
    *,
    mode: str,
    signal_dir: Path,
    mechanism_envelope: str,
    pump_eps_m3: float,
    pump_sig_m3: float,
) -> pd.DataFrame:
    path = signal_dir / "prediction_signal_flow_source_compare.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.insert(0, "mode", mode)
    df["mechanism_envelope"] = mechanism_envelope
    df["case_id"] = df["case_key"].map(case_id_from_key)
    df = df.rename(
        columns={
            "oracle_action": "oracle_action_signal",
            "learned_action": "learned_action_signal",
            "persistence_action": "persistence_action_signal",
        }
    )
    df["lp_action_diff"] = 1 - df["learned_vs_persistence_same_planner_action"].fillna(0).astype(int)
    df["op_action_diff"] = 1 - df["oracle_vs_persistence_same_planner_action"].fillna(0).astype(int)
    df["lo_action_diff"] = 1 - df["learned_vs_oracle_same_planner_action"].fillna(0).astype(int)
    df["bucket_class"] = df.apply(classify_bucket, axis=1, pump_eps_m3=pump_eps_m3, pump_sig_m3=pump_sig_m3)
    df["learned_saves_sig"] = df["learned_minus_persistence_pump_m3"] <= -pump_sig_m3
    df["learned_spends_sig"] = df["learned_minus_persistence_pump_m3"] >= pump_sig_m3
    df["oracle_saves_sig"] = df["oracle_minus_persistence_pump_m3"] <= -pump_sig_m3
    df["oracle_spends_sig"] = df["oracle_minus_persistence_pump_m3"] >= pump_sig_m3
    df["oracle_value_missed_by_learned"] = (
        df["oracle_saves_sig"] & (df["learned_minus_persistence_pump_m3"] > -pump_sig_m3)
    )
    return df


def load_all_signal(
    signal_runs: dict[str, dict[str, str]],
    pump_eps_m3: float,
    pump_sig_m3: float,
) -> pd.DataFrame:
    rows = []
    for mode, spec in signal_runs.items():
        rows.append(
            load_signal_mode(
                mode=mode,
                signal_dir=_resolve(spec["signal_dir"]),
                mechanism_envelope=str(spec["mechanism_envelope"]),
                pump_eps_m3=pump_eps_m3,
                pump_sig_m3=pump_sig_m3,
            )
        )
    return pd.concat(rows, ignore_index=True)


def load_mechanisms(path: Path) -> pd.DataFrame:
    mech_path = path / "prediction_value_mechanism_buckets.csv"
    if not mech_path.exists():
        raise FileNotFoundError(mech_path)
    df = pd.read_csv(mech_path)
    keep = [
        "envelope_mode",
        "case_id",
        "bucket",
        "oracle_regime",
        "mechanism",
        "oracle_action",
        "learned_action",
        "persistence_action",
        "oracle_n0",
        "oracle_n1",
        "oracle_n2",
        "learned_n0",
        "learned_n1",
        "learned_n2",
        "persistence_n0",
        "persistence_n1",
        "persistence_n2",
        "learned_uv_rmse_vs_oracle_ms",
        "persistence_uv_rmse_vs_oracle_ms",
        "learned_uv_rmse_vs_persistence_ms",
        "learned_speed_block2_minus_block0_ms",
        "oracle_speed_block2_minus_block0_ms",
        "persistence_speed_block2_minus_block0_ms",
    ]
    return df[[c for c in keep if c in df.columns]].copy()


def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("mode", sort=True)
    out = grouped.agg(
        buckets=("bucket", "count"),
        lp_action_diff_buckets=("lp_action_diff", bool_count),
        op_action_diff_buckets=("op_action_diff", bool_count),
        learned_save_buckets=("learned_saves_sig", bool_count),
        learned_extra_buckets=("learned_spends_sig", bool_count),
        oracle_save_buckets=("oracle_saves_sig", bool_count),
        oracle_missed_buckets=("oracle_value_missed_by_learned", bool_count),
        learned_pump_m3=("learned_pump_work_m3", "sum"),
        persistence_pump_m3=("persistence_pump_work_m3", "sum"),
        oracle_pump_m3=("oracle_pump_work_m3", "sum"),
        learned_minus_persistence_m3=("learned_minus_persistence_pump_m3", "sum"),
        oracle_minus_persistence_m3=("oracle_minus_persistence_pump_m3", "sum"),
        learned_minus_oracle_m3=("learned_minus_oracle_pump_m3", "sum"),
    ).reset_index()
    out["lp_action_diff_ratio"] = out["lp_action_diff_buckets"] / out["buckets"].clip(lower=1)
    out["learned_vs_persistence_pct"] = (
        out["learned_minus_persistence_m3"] / out["persistence_pump_m3"].clip(lower=1e-9) * 100.0
    )
    out["oracle_vs_persistence_pct"] = (
        out["oracle_minus_persistence_m3"] / out["persistence_pump_m3"].clip(lower=1e-9) * 100.0
    )
    return out


def summarize_case(df: pd.DataFrame, case_sig_m3: float) -> pd.DataFrame:
    grouped = df.groupby(["mode", "case_id"], sort=True)
    out = grouped.agg(
        buckets=("bucket", "count"),
        lp_action_diff_buckets=("lp_action_diff", bool_count),
        learned_save_buckets=("learned_saves_sig", bool_count),
        learned_extra_buckets=("learned_spends_sig", bool_count),
        oracle_save_buckets=("oracle_saves_sig", bool_count),
        oracle_missed_buckets=("oracle_value_missed_by_learned", bool_count),
        learned_pump_m3=("learned_pump_work_m3", "sum"),
        persistence_pump_m3=("persistence_pump_work_m3", "sum"),
        oracle_pump_m3=("oracle_pump_work_m3", "sum"),
        learned_minus_persistence_m3=("learned_minus_persistence_pump_m3", "sum"),
        oracle_minus_persistence_m3=("oracle_minus_persistence_pump_m3", "sum"),
        learned_minus_oracle_m3=("learned_minus_oracle_pump_m3", "sum"),
        learned_pitch_delta_max=("learned_minus_persistence_pitch_p95", "max"),
        learned_pitch_delta_min=("learned_minus_persistence_pitch_p95", "min"),
        learned_roll_delta_max=("learned_minus_persistence_roll_p95", "max"),
        learned_roll_delta_min=("learned_minus_persistence_roll_p95", "min"),
    ).reset_index()

    def verdict(delta: float) -> str:
        if delta <= -case_sig_m3:
            return "learned_positive"
        if delta >= case_sig_m3:
            return "learned_negative"
        return "near_tie"

    def oracle_verdict(delta: float) -> str:
        if delta <= -case_sig_m3:
            return "oracle_positive"
        if delta >= case_sig_m3:
            return "oracle_negative"
        return "oracle_near_tie"

    out["learned_case_verdict"] = out["learned_minus_persistence_m3"].map(verdict)
    out["oracle_case_verdict"] = out["oracle_minus_persistence_m3"].map(oracle_verdict)
    out["learned_vs_persistence_pct"] = (
        out["learned_minus_persistence_m3"] / out["persistence_pump_m3"].clip(lower=1e-9) * 100.0
    )
    out["oracle_vs_persistence_pct"] = (
        out["oracle_minus_persistence_m3"] / out["persistence_pump_m3"].clip(lower=1e-9) * 100.0
    )
    return out


def summarize_classes(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["mode", "bucket_class"], sort=True)
        .agg(
            buckets=("bucket", "count"),
            learned_minus_persistence_m3=("learned_minus_persistence_pump_m3", "sum"),
            oracle_minus_persistence_m3=("oracle_minus_persistence_pump_m3", "sum"),
            mean_pitch_delta=("learned_minus_persistence_pitch_p95", "mean"),
            mean_roll_delta=("learned_minus_persistence_roll_p95", "mean"),
        )
        .reset_index()
    )
    return out


def make_report(
    *,
    out_dir: Path,
    buckets: pd.DataFrame,
    overall: pd.DataFrame,
    cases: pd.DataFrame,
    classes: pd.DataFrame,
    pump_eps_m3: float,
    pump_sig_m3: float,
    case_sig_m3: float,
) -> str:
    lines: list[str] = []
    lines.append("# Prediction Value Blocker Summary")
    lines.append("")
    lines.append("Combines planner-only mechanism rows with raw 1 Hz closed-loop bucket metrics.")
    lines.append("")
    lines.append(f"- execution washout threshold: `<= {pump_eps_m3:g} m3 / 10-min bucket`")
    lines.append(f"- significant bucket pump threshold: `>= {pump_sig_m3:g} m3 / 10-min bucket`")
    lines.append(f"- significant case pump threshold: `>= {case_sig_m3:g} m3 / 2h case`")
    lines.append("")

    overall_show = overall[
        [
            "mode",
            "buckets",
            "lp_action_diff_buckets",
            "lp_action_diff_ratio",
            "learned_save_buckets",
            "learned_extra_buckets",
            "oracle_save_buckets",
            "oracle_missed_buckets",
            "learned_pump_m3",
            "persistence_pump_m3",
            "oracle_pump_m3",
            "learned_minus_persistence_m3",
            "learned_vs_persistence_pct",
            "oracle_minus_persistence_m3",
            "oracle_vs_persistence_pct",
        ]
    ]
    lines.append("## Overall")
    lines.append("")
    lines.append(markdown_table(overall_show, floatfmt=".3f"))
    lines.append("")

    lines.append("## Bucket Classes")
    lines.append("")
    lines.append(markdown_table(classes, floatfmt=".3f"))
    lines.append("")

    case_cols = [
        "mode",
        "case_id",
        "learned_case_verdict",
        "oracle_case_verdict",
        "lp_action_diff_buckets",
        "learned_save_buckets",
        "learned_extra_buckets",
        "oracle_missed_buckets",
        "learned_pump_m3",
        "persistence_pump_m3",
        "oracle_pump_m3",
        "learned_minus_persistence_m3",
        "learned_vs_persistence_pct",
        "oracle_minus_persistence_m3",
        "oracle_vs_persistence_pct",
    ]
    lines.append("## Case Summary")
    lines.append("")
    lines.append(markdown_table(cases[case_cols], floatfmt=".3f"))
    lines.append("")

    high_cols = [
        "mode",
        "case_id",
        "bucket",
        "bucket_class",
        "oracle_regime",
        "mechanism",
        "oracle_action_mech",
        "learned_action_mech",
        "persistence_action_mech",
        "learned_pump_work_m3",
        "persistence_pump_work_m3",
        "oracle_pump_work_m3",
        "learned_minus_persistence_pump_m3",
        "oracle_minus_persistence_pump_m3",
        "learned_minus_persistence_pitch_p95",
        "learned_minus_persistence_roll_p95",
    ]

    pos = buckets[buckets["learned_saves_sig"]].copy()
    pos = pos.sort_values("learned_minus_persistence_pump_m3")
    lines.append("## Learned Positive Raw 1Hz Buckets")
    lines.append("")
    lines.append(markdown_table(pos[[c for c in high_cols if c in pos.columns]], floatfmt=".3f", max_rows=30))
    lines.append("")

    neg = buckets[buckets["learned_spends_sig"]].copy()
    neg = neg.sort_values("learned_minus_persistence_pump_m3", ascending=False)
    lines.append("## Learned Negative Raw 1Hz Buckets")
    lines.append("")
    lines.append(markdown_table(neg[[c for c in high_cols if c in neg.columns]], floatfmt=".3f", max_rows=30))
    lines.append("")

    missed = buckets[buckets["oracle_value_missed_by_learned"]].copy()
    missed = missed.sort_values("oracle_minus_persistence_pump_m3")
    lines.append("## Oracle Opportunities Not Captured By Learned")
    lines.append("")
    lines.append(markdown_table(missed[[c for c in high_cols if c in missed.columns]], floatfmt=".3f", max_rows=30))
    lines.append("")

    lines.append("## Working Interpretation")
    lines.append("")
    lines.append("- Default planner path mostly blocks forecast-source value: learned and persistence choose the same first action in most buckets.")
    lines.append("- Raw-envelope mode exposes additional oracle value and learned-vs-persistence separation, so the planner structure is one real compression point.")
    lines.append("- Raw-envelope plus hold-pause target lifecycle removes the main late extra-pumping artifact in case07; this is a prediction-primary execution issue, not a closed-baseline tuning issue.")
    lines.append("- Learned does produce pump-side value in relief/sign-flip style buckets. Remaining gaps are forecast reliability and missed oracle opportunities, especially future relief/onset buckets where learned and persistence still look similar.")
    lines.append("- Future-risk early action remains a design objective, but current learned UV does not yet provide enough clean onset lead in these cases without event-floor style rule injection.")
    lines.append("")

    lines.append("## Files")
    lines.append("")
    for name in [
        "prediction_value_blocker_buckets.csv",
        "prediction_value_blocker_overall.csv",
        "prediction_value_blocker_cases.csv",
        "prediction_value_blocker_classes.csv",
    ]:
        lines.append(f"- `{out_dir / name}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mechanism-dir",
        default="outputs/wind_prediction/prediction_value_mechanism_audit_10case_2h",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/prediction_value_blockers_10case_2h",
    )
    parser.add_argument("--pump-eps-m3", type=float, default=1.0)
    parser.add_argument("--pump-sig-m3", type=float, default=5.0)
    parser.add_argument("--case-sig-m3", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _resolve(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    signal = load_all_signal(
        DEFAULT_SIGNAL_RUNS,
        pump_eps_m3=float(args.pump_eps_m3),
        pump_sig_m3=float(args.pump_sig_m3),
    )
    mech = load_mechanisms(_resolve(str(args.mechanism_dir)))
    buckets = signal.merge(
        mech,
        left_on=["mechanism_envelope", "case_id", "bucket"],
        right_on=["envelope_mode", "case_id", "bucket"],
        how="left",
        suffixes=("", "_mech"),
    )
    buckets = buckets.rename(
        columns={
            "oracle_action": "oracle_action_mech",
            "learned_action": "learned_action_mech",
            "persistence_action": "persistence_action_mech",
        }
    )
    overall = summarize_overall(buckets)
    cases = summarize_case(buckets, case_sig_m3=float(args.case_sig_m3))
    classes = summarize_classes(buckets)

    buckets.to_csv(out_dir / "prediction_value_blocker_buckets.csv", index=False)
    overall.to_csv(out_dir / "prediction_value_blocker_overall.csv", index=False)
    cases.to_csv(out_dir / "prediction_value_blocker_cases.csv", index=False)
    classes.to_csv(out_dir / "prediction_value_blocker_classes.csv", index=False)
    report = make_report(
        out_dir=out_dir,
        buckets=buckets,
        overall=overall,
        cases=cases,
        classes=classes,
        pump_eps_m3=float(args.pump_eps_m3),
        pump_sig_m3=float(args.pump_sig_m3),
        case_sig_m3=float(args.case_sig_m3),
    )
    report_path = out_dir / "prediction_value_blocker_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report: {report_path}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
