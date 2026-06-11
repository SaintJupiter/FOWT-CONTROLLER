#!/usr/bin/env python3
"""Oracle relief-economy pump-saving audit for h120.

This audit asks whether 120min *relief* information can save pump work beyond
what a 0-60min relief signal already captures.

Arms:
    A0: v1.6 baseline
    A1: v1.6 + oracle relief-economy using near [0,60] blocks
    A2: v1.6 + oracle relief-economy using near+far [0,120] blocks

The lever is default-off and economy-path-only in
BallastPlannerPreviewProvider.  It never touches reactive-floor, recovery,
safety fallback, high-posture actions, or pump penalty.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv312/bin/python"
if not PYTHON.exists():
    PYTHON = REPO_ROOT / ".venv/bin/python"
OUT_DIR = REPO_ROOT / "outputs/wind_prediction/h120_relief_economy_oracle_audit_v1"
F120_DATASET = (
    REPO_ROOT / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"
)
CASE_SOURCES = {
    "guard10": REPO_ROOT / "outputs/wind_prediction/h120_oracle_controller_value_audit_v1/guard10_cases.csv",
    "broader20": REPO_ROOT / "outputs/wind_prediction/h120_oracle_controller_value_audit_v1/broader20_cases.csv",
    "relief": REPO_ROOT / "outputs/wind_prediction/prediction_value_mechanism_grid_v2/future_relief_cases.csv",
    "lowrisk": REPO_ROOT / "outputs/wind_prediction/prediction_value_mechanism_grid_v2/lowrisk_quiet_cases.csv",
}


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _mkdirs(out: Path) -> dict[str, Path]:
    dirs = {
        "out": out,
        "runs": out / "runs",
        "raw": out / "raw_tables",
        "paper": out / "paper_ready",
        "debug": out / "debug",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _case_key(path: Path) -> str:
    name = path.name
    for suffix in ("_prediction_primary_econ_timeseries.csv", "_prediction_primary_econ_planner_log.csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _fallback_series(df: pd.DataFrame) -> pd.Series:
    for col in (
        "preview_primary_safety_fallback",
        "primary_safety_fallback",
        "primary_safety_fallback_active",
    ):
        if col in df.columns:
            return df[col].astype(float) > 0.5
    return pd.Series(False, index=df.index)


def _edge_count(series: pd.Series) -> int:
    arr = series.fillna(0).astype(float).to_numpy() > 0.5
    if arr.size == 0:
        return 0
    return int(np.logical_and(arr, np.r_[True, ~arr[:-1]]).sum())


def _read_log(run_dir: Path, key: str) -> pd.DataFrame:
    log_dir = run_dir / "planner_logs"
    direct = log_dir / f"{key}_prediction_primary_econ_planner_log.csv"
    if direct.exists():
        return pd.read_csv(direct, low_memory=False)
    matches = list(log_dir.glob(f"{key}*_planner_log.csv"))
    if not matches:
        return pd.DataFrame()
    return pd.read_csv(matches[0], low_memory=False)


def _base_cmd(out_dir: Path, cases_csv: Path) -> list[str]:
    return [
        str(PYTHON),
        str(REPO_ROOT / "scripts/analysis/run_prediction_primary_casebook.py"),
        "--out-dir",
        str(out_dir),
        "--primary-only",
        "--skip-figures",
        "--duration-s",
        "7200",
        "--cases-csv",
        str(cases_csv),
        "--forecast-source",
        "oracle",
        "--dataset-dir",
        str(F120_DATASET),
        "--primary-control-profile",
        "rawenv_holdpause_barrier_reliefcap_adaptive_v1",
        "--reactive-floor-predictive-veto",
        "on",
        "--high-posture-metric",
        "max_axis",
        "--reactive-floor-action",
        "active_small",
        "--reactive-floor-medium-delay-s",
        "1200",
        "--reactive-floor-post-exit-mode",
        "early_stop",
        "--far-horizon",
    ]


def _run_casebook(
    run_dir: Path,
    cases_csv: Path,
    *,
    arm: str,
    force: bool,
    relief_safe_deg: float,
    relief_low_norm: float,
    relief_high_norm: float,
) -> None:
    summary = run_dir / "casebook_summary.csv"
    if summary.exists() and not force:
        print(f"[skip] {_rel(run_dir)}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = _base_cmd(run_dir, cases_csv)
    if arm in ("A1_near_relief", "A2_h120_relief"):
        horizon = "near" if arm == "A1_near_relief" else "far"
        cmd.extend(
            [
                "--relief-economy",
                "--relief-economy-horizon",
                horizon,
                "--relief-economy-safe-deg",
                str(float(relief_safe_deg)),
                "--relief-economy-low-norm",
                str(float(relief_low_norm)),
                "--relief-economy-high-norm",
                str(float(relief_high_norm)),
            ]
        )
    print(f"[run] {_rel(run_dir)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _case_metrics(run_dir: Path, dataset: str, arm: str, labels: pd.DataFrame) -> list[dict[str, Any]]:
    label_map = dict(zip(labels["case_id"].astype(str), labels["label"].astype(str)))
    rows: list[dict[str, Any]] = []
    for ts_path in sorted((run_dir / "timeseries").glob("*_timeseries.csv")):
        key = _case_key(ts_path)
        df = pd.read_csv(ts_path, low_memory=False)
        pitch = df["pitch_deg"].astype(float).abs()
        roll = df["roll_deg"].astype(float).abs()
        max_axis = np.maximum(pitch, roll)
        pump_rate = df.get("pump_total_rate_m3_min", pd.Series(0.0, index=df.index)).astype(float).abs()
        high = max_axis > 5.0
        idle = pump_rate < 0.05
        fb = _fallback_series(df)
        log = _read_log(run_dir, key)
        zeros = pd.Series(dtype=float)
        case_id_no_ts = key.rsplit("_", 2)[0]
        case_short = (
            case_id_no_ts[3:]
            if len(case_id_no_ts) > 3 and case_id_no_ts[:2].isdigit() and case_id_no_ts[2] == "_"
            else case_id_no_ts
        )
        floor_active = log.get("reactive_floor_active", zeros).fillna(0).astype(float) if not log.empty else zeros
        medium_active = (
            log.get("reactive_floor_medium_delay_active", zeros).fillna(0).astype(float)
            if not log.empty
            else zeros
        )
        relief_active = (
            log.get("relief_economy_active", zeros).fillna(0).astype(float)
            if not log.empty
            else zeros
        )
        delay_count = (
            log.get("relief_economy_delay_count", zeros).fillna(0).astype(float)
            if not log.empty
            else zeros
        )
        near_low = (
            log.get("relief_economy_near_low", zeros).fillna(0).astype(float)
            if not log.empty
            else zeros
        )
        far_low = (
            log.get("relief_economy_far_low", zeros).fillna(0).astype(float)
            if not log.empty
            else zeros
        )
        rows.append(
            {
                "dataset": dataset,
                "arm": arm,
                "case": key,
                "case_short": case_short,
                "label": label_map.get(case_short, label_map.get(str(case_short).replace("future_", ""), "")),
                "pump": float(pump_rate.sum() / 60.0),
                "sum_fb": float(fb.mean() * 100.0),
                "max_p95": float(max(np.percentile(pitch, 95), np.percentile(roll, 95))),
                "pitch_p95": float(np.percentile(pitch, 95)),
                "roll_p95": float(np.percentile(roll, 95)),
                "time_over_5": int(high.sum()),
                "idle_time_over_5": int((high & idle).sum()),
                "floor_trigger_count": _edge_count(floor_active),
                "floor_active_rows": int(floor_active.sum()),
                "medium_escalation_count": _edge_count(medium_active),
                "relief_economy_active_rows": int(relief_active.sum()),
                "relief_economy_delay_count_final": int(delay_count.max()) if len(delay_count) else 0,
                "relief_economy_delay_edges": _edge_count(delay_count.diff().fillna(delay_count)),
                "relief_economy_near_low_rows": int(near_low.sum()),
                "relief_economy_far_low_rows": int(far_low.sum()),
            }
        )
    return rows


def _aggregate(case_table: pd.DataFrame) -> pd.DataFrame:
    agg = (
        case_table.groupby(["dataset", "arm"], as_index=False)
        .agg(
            cases=("case", "count"),
            sum_pump=("pump", "sum"),
            sum_fb=("sum_fb", "sum"),
            max_p95=("max_p95", "max"),
            time_over_5=("time_over_5", "sum"),
            idle_time_over_5=("idle_time_over_5", "sum"),
            floor_trigger_count=("floor_trigger_count", "sum"),
            floor_active_rows=("floor_active_rows", "sum"),
            medium_escalation_count=("medium_escalation_count", "sum"),
            relief_economy_active_rows=("relief_economy_active_rows", "sum"),
            relief_economy_delay_count_final=("relief_economy_delay_count_final", "sum"),
            relief_economy_delay_edges=("relief_economy_delay_edges", "sum"),
            relief_economy_near_low_rows=("relief_economy_near_low_rows", "sum"),
            relief_economy_far_low_rows=("relief_economy_far_low_rows", "sum"),
        )
        .sort_values(["dataset", "arm"])
    )
    baselines = agg[agg["arm"].eq("A0_v16")].set_index("dataset")
    near = agg[agg["arm"].eq("A1_near_relief")].set_index("dataset")
    for idx, row in agg.iterrows():
        ds = str(row["dataset"])
        if ds in baselines.index:
            for col in ("sum_pump", "sum_fb", "max_p95", "time_over_5", "idle_time_over_5"):
                agg.loc[idx, f"delta_vs_A0_{col}"] = float(row[col]) - float(baselines.loc[ds, col])
        if ds in near.index:
            for col in ("sum_pump", "sum_fb", "max_p95", "time_over_5", "idle_time_over_5"):
                agg.loc[idx, f"delta_vs_A1_{col}"] = float(row[col]) - float(near.loc[ds, col])
    return agg


def _reason_table(run_map: dict[tuple[str, str], Path]) -> pd.DataFrame:
    reason_rows: list[dict[str, Any]] = []
    total_rows: list[dict[str, Any]] = []
    for (dataset, arm), run_dir in run_map.items():
        if arm == "A0_v16":
            continue
        active_total = 0
        delay_total = 0
        log_cases = 0
        for log_path in sorted((run_dir / "planner_logs").glob("*_planner_log.csv")):
            log = pd.read_csv(log_path, low_memory=False)
            if "relief_economy_reason" not in log.columns:
                continue
            log_cases += 1
            active = log.get("relief_economy_active", pd.Series(0, index=log.index)).fillna(0).astype(float)
            delay_count = log.get("relief_economy_delay_count", pd.Series(0, index=log.index)).fillna(0).astype(float)
            delay_increments = delay_count.diff().fillna(delay_count).clip(lower=0)
            active_total += int((active > 0.5).sum())
            delay_total += int(delay_increments.sum())
            counts = log["relief_economy_reason"].astype(str).value_counts()
            for reason, count in counts.items():
                reason_rows.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "reason": reason,
                        "rows": int(count),
                    }
                )
        total_rows.append(
            {
                "dataset": dataset,
                "arm": arm,
                "planner_log_cases": log_cases,
                "total_active_rows": active_total,
                "total_delay_increments": delay_total,
            }
        )
    if not reason_rows:
        return pd.DataFrame()
    reasons = pd.DataFrame(reason_rows).groupby(
        ["dataset", "arm", "reason"], as_index=False
    )["rows"].sum()
    totals = pd.DataFrame(total_rows)
    return (
        reasons.merge(totals, on=["dataset", "arm"], how="left")
        .sort_values(["dataset", "arm", "reason"])
        .reset_index(drop=True)
    )


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"

    def fmt(v: Any) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, float):
            if abs(v) <= 1.0:
                return f"{v:.3f}"
            return f"{v:.2f}"
        return str(v)

    lines = [
        "| " + " | ".join(str(c) for c in df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def _write_docs(
    out: Path,
    compare: pd.DataFrame,
    case_table: pd.DataFrame,
    reason_table: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    key_cols = [
        "dataset",
        "arm",
        "cases",
        "sum_pump",
        "sum_fb",
        "max_p95",
        "time_over_5",
        "idle_time_over_5",
        "relief_economy_delay_count_final",
        "delta_vs_A0_sum_pump",
        "delta_vs_A1_sum_pump",
        "delta_vs_A0_time_over_5",
        "delta_vs_A1_time_over_5",
    ]
    key = compare[[c for c in key_cols if c in compare.columns]].copy()
    pass_rows = []
    for ds, group in compare.groupby("dataset"):
        a1 = group[group["arm"].eq("A1_near_relief")]
        a2 = group[group["arm"].eq("A2_h120_relief")]
        if a1.empty or a2.empty:
            continue
        a1r = a1.iloc[0]
        a2r = a2.iloc[0]
        pump_gain = float(a1r["sum_pump"] - a2r["sum_pump"])
        safety_clean = all(
            float(a2r[col]) <= float(a1r[col]) + 1e-9
            for col in ("sum_fb", "max_p95", "time_over_5", "idle_time_over_5")
        )
        pass_rows.append(
            {
                "dataset": ds,
                "A2_minus_A1_pump": float(a2r["sum_pump"] - a1r["sum_pump"]),
                "A2_pump_saved_vs_A1": pump_gain,
                "A2_safety_not_worse_than_A1": int(safety_clean),
                "A2_delay_count": int(a2r.get("relief_economy_delay_count_final", 0)),
            }
        )
    pass_table = pd.DataFrame(pass_rows)
    pass_table.to_csv(out / "oracle_relief_pass_table.csv", index=False)
    pass_table.to_csv(out / "raw_tables/oracle_relief_pass_table.csv", index=False)
    verdict = "NO-GO"
    reason = "No dataset showed material 120-unique pump saving over A1 with all safety metrics non-worse."
    if not pass_table.empty:
        good = pass_table[
            (pass_table["A2_pump_saved_vs_A1"] > 1e-6)
            & (pass_table["A2_safety_not_worse_than_A1"] > 0)
        ]
        if not good.empty:
            verdict = "CONDITIONAL-GO"
            reason = "At least one dataset showed positive A2-vs-A1 pump saving with non-worse aggregate safety."
    lines = [
        "# h120 Relief-Economy Oracle Audit v1",
        "",
        "Scope: oracle-only pump-saving audit. The lever is default-off and economy-path-only; it delays non-urgent target refreshes under safe posture and clear relief. It does not touch v1.6 reactive floor, recovery, safety fallback, or pump penalty.",
        "",
        "## Arms",
        "",
        "- `A0_v16`: v1.6 delayed-medium baseline.",
        "- `A1_near_relief`: v1.6 + oracle 0-60min relief-economy.",
        "- `A2_h120_relief`: v1.6 + oracle 0-120min relief-economy.",
        "",
        "The decisive comparison is `A2_h120_relief - A1_near_relief`, not just A2 vs A0.",
        "",
        "## Parameters",
        "",
        f"- relief safe deg: `{args.relief_safe_deg}`",
        f"- relief low norm: `{args.relief_low_norm}`",
        f"- relief high norm: `{args.relief_high_norm}`",
        "",
        "## Aggregate Results",
        _md_table(key),
        "",
        "## A2-vs-A1 Pass Table",
        _md_table(pass_table),
        "",
        "## Trigger Diagnostics",
        "",
        "The lever may detect relief without saving pump. `total_active_rows` counts rows where the relief guard was true; `total_delay_increments` counts actual delayed non-urgent target refreshes. Pump saving requires delayed refreshes, not merely relief detection.",
        "",
        _md_table(
            reason_table[
                [
                    c
                    for c in (
                        "dataset",
                        "arm",
                        "reason",
                        "rows",
                        "total_active_rows",
                        "total_delay_increments",
                    )
                    if c in reason_table.columns
                ]
            ]
            if not reason_table.empty
            else pd.DataFrame()
        ),
        "",
        "## Decision",
        "",
        f"**{verdict}.** {reason}",
        "",
        "If A2 does not beat A1, relief-economy may exist but the 120min far blocks do not add unique pump-saving value beyond the 0-60min relief signal under this lever.",
    ]
    doc = "\n".join(lines) + "\n"
    (out / "h120_relief_economy_oracle_decision.md").write_text(doc, encoding="utf-8")
    (out / "paper_ready/relief_economy_oracle_findings.md").write_text(doc, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=["relief", "broader20", "lowrisk", "guard10"])
    parser.add_argument("--relief-safe-deg", type=float, default=3.0)
    parser.add_argument("--relief-low-norm", type=float, default=0.5)
    parser.add_argument("--relief-high-norm", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = _mkdirs(OUT_DIR)
    if not PYTHON.exists():
        raise FileNotFoundError(f"missing project python: {PYTHON}")
    run_map: dict[tuple[str, str], Path] = {}
    arms = ("A0_v16", "A1_near_relief", "A2_h120_relief")
    for ds in args.datasets:
        if ds not in CASE_SOURCES:
            raise ValueError(f"unknown dataset={ds!r}; choices={sorted(CASE_SOURCES)}")
        cases_csv = CASE_SOURCES[ds]
        if not cases_csv.exists():
            raise FileNotFoundError(cases_csv)
        for arm in arms:
            run_dir = dirs["runs"] / ds / arm
            run_map[(ds, arm)] = run_dir
            _run_casebook(
                run_dir,
                cases_csv,
                arm=arm,
                force=bool(args.force),
                relief_safe_deg=float(args.relief_safe_deg),
                relief_low_norm=float(args.relief_low_norm),
                relief_high_norm=float(args.relief_high_norm),
            )
    rows: list[dict[str, Any]] = []
    for (ds, arm), run_dir in run_map.items():
        rows.extend(_case_metrics(run_dir, ds, arm, pd.read_csv(CASE_SOURCES[ds])))
    case_table = pd.DataFrame(rows).sort_values(["dataset", "arm", "case"])
    case_table.to_csv(OUT_DIR / "relief_economy_case_table.csv", index=False)
    case_table.to_csv(dirs["raw"] / "relief_economy_case_table.csv", index=False)
    compare = _aggregate(case_table)
    compare.to_csv(OUT_DIR / "relief_economy_compare_table.csv", index=False)
    compare.to_csv(dirs["raw"] / "relief_economy_compare_table.csv", index=False)
    reason_table = _reason_table(run_map)
    reason_table.to_csv(OUT_DIR / "relief_economy_reason_table.csv", index=False)
    reason_table.to_csv(dirs["raw"] / "relief_economy_reason_table.csv", index=False)
    _write_docs(OUT_DIR, compare, case_table, reason_table, args)
    print(f"[done] {_rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
