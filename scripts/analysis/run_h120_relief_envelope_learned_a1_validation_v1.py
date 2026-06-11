#!/usr/bin/env python3
"""Learned-forecast validation for the relief-conditioned economy envelope.

This is the make-or-break follow-up to the oracle relief-envelope audit:
verify whether the A1 near-horizon pump-saving signal survives when the
forecast source is the learned h240/f120 model instead of oracle future wind.

No controller defaults are changed.  The relief envelope remains default-off
and is enabled only for the B1 learned-near-envelope arm.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv312/bin/python"
if not PYTHON.exists():
    PYTHON = REPO_ROOT / ".venv/bin/python"

OUT_DIR = REPO_ROOT / "outputs/wind_prediction/h120_relief_envelope_learned_a1_validation_v1"
F120_DATASET = (
    REPO_ROOT / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"
)
H120_MODEL = REPO_ROOT / "outputs/wind_prediction/lstm_h240_f120_near_block_eventbalanced_v2"
CASE_SOURCES = {
    "relief": REPO_ROOT / "outputs/wind_prediction/prediction_value_mechanism_grid_v2/future_relief_cases.csv",
    "lowrisk": REPO_ROOT / "outputs/wind_prediction/prediction_value_mechanism_grid_v2/lowrisk_quiet_cases.csv",
    "broader20": REPO_ROOT / "outputs/wind_prediction/h120_oracle_controller_value_audit_v1/broader20_cases.csv",
    "guard10": REPO_ROOT / "outputs/wind_prediction/h120_oracle_controller_value_audit_v1/guard10_cases.csv",
}
PROFILES: dict[str, dict[str, float]] = {
    "conservative": {
        "allowed_0_20": 4.50,
        "allowed_20_40": 4.20,
        "allowed_40_60": 3.90,
        "allowed_60_120": 3.50,
        "duration_s": 600.0,
        "debt_budget": 300.0,
        "near_limit_budget_s": 0.0,
        "worsening_eps": 0.0,
    },
    "medium": {
        "allowed_0_20": 4.65,
        "allowed_20_40": 4.40,
        "allowed_40_60": 4.10,
        "allowed_60_120": 3.70,
        "duration_s": 900.0,
        "debt_budget": 600.0,
        "near_limit_budget_s": 300.0,
        "worsening_eps": 0.05,
    },
}


def _mkdirs() -> dict[str, Path]:
    dirs = {
        "out": OUT_DIR,
        "runs": OUT_DIR / "runs",
        "raw": OUT_DIR / "raw_tables",
        "paper": OUT_DIR / "paper_ready",
        "diag": OUT_DIR / "diagnostics",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


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
        "learned",
        "--model-dir",
        str(H120_MODEL),
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
    profile: str,
    force: bool,
) -> None:
    if (run_dir / "casebook_summary.csv").exists() and not force:
        print(f"[skip] {run_dir.relative_to(REPO_ROOT)}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = _base_cmd(run_dir, cases_csv)
    if arm == "B1_learned_near_envelope":
        p = PROFILES[profile]
        cmd.extend(
            [
                "--relief-envelope",
                "--relief-envelope-horizon",
                "near",
                "--relief-envelope-allowed-0-20-deg",
                str(p["allowed_0_20"]),
                "--relief-envelope-allowed-20-40-deg",
                str(p["allowed_20_40"]),
                "--relief-envelope-allowed-40-60-deg",
                str(p["allowed_40_60"]),
                "--relief-envelope-allowed-60-120-deg",
                str(p["allowed_60_120"]),
                "--relief-envelope-max-duration-s",
                str(p["duration_s"]),
                "--relief-envelope-debt-budget-deg-s",
                str(p["debt_budget"]),
                "--relief-envelope-near-limit-budget-s",
                str(p["near_limit_budget_s"]),
                "--relief-envelope-worsening-eps-deg",
                str(p["worsening_eps"]),
            ]
        )
    print(f"[run] {run_dir.relative_to(REPO_ROOT)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


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
    direct = run_dir / "planner_logs" / f"{key}_prediction_primary_econ_planner_log.csv"
    if direct.exists():
        return pd.read_csv(direct, low_memory=False)
    matches = list((run_dir / "planner_logs").glob(f"{key}*_planner_log.csv"))
    if not matches:
        return pd.DataFrame()
    return pd.read_csv(matches[0], low_memory=False)


def _last_metric(log: pd.DataFrame, name: str, default: float = 0.0) -> float:
    if name not in log.columns or log.empty:
        return float(default)
    s = log[name].fillna(default).astype(float)
    return float(s.iloc[-1]) if len(s) else float(default)


def _case_metrics(run_dir: Path, dataset: str, profile: str, arm: str) -> list[dict[str, Any]]:
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
        floor = log.get("reactive_floor_active", pd.Series(dtype=float)).fillna(0).astype(float)
        medium = log.get("reactive_floor_medium_delay_active", pd.Series(dtype=float)).fillna(0).astype(float)
        rows.append(
            {
                "dataset": dataset,
                "profile": profile,
                "arm": arm,
                "case": key,
                "pump": float(pump_rate.sum() / 60.0),
                "sum_fb": float(fb.mean() * 100.0),
                "max_axis": float(max_axis.max()),
                "p95_max_axis": float(np.percentile(max_axis, 95)),
                "time_over_5": int(high.sum()),
                "idle_time_over_5": int((high & idle).sum()),
                "floor_trigger_count": _edge_count(floor),
                "floor_active_rows": int(floor.sum()) if len(floor) else 0,
                "medium_escalation_count": _edge_count(medium),
                "eligible_count": int(_last_metric(log, "relief_envelope_eligible_count")),
                "pump_opportunity_count": int(_last_metric(log, "relief_envelope_pump_opportunity_count")),
                "target_refresh_delayed_count": int(
                    _last_metric(log, "relief_envelope_target_refresh_delayed_count")
                ),
                "relaxation_active_time": float(
                    _last_metric(log, "relief_envelope_relaxation_active_time_s")
                ),
                "safety_debt_used": float(_last_metric(log, "relief_envelope_safety_debt_used")),
            }
        )
    return rows


def _aggregate(case_table: pd.DataFrame) -> pd.DataFrame:
    agg = (
        case_table.groupby(["dataset", "profile", "arm"], as_index=False)
        .agg(
            cases=("case", "count"),
            sum_pump=("pump", "sum"),
            sum_fb=("sum_fb", "sum"),
            max_axis=("max_axis", "max"),
            p95_max_axis=("p95_max_axis", "max"),
            time_over_5=("time_over_5", "sum"),
            idle_time_over_5=("idle_time_over_5", "sum"),
            floor_trigger_count=("floor_trigger_count", "sum"),
            floor_active_rows=("floor_active_rows", "sum"),
            medium_escalation_count=("medium_escalation_count", "sum"),
            eligible_count=("eligible_count", "sum"),
            pump_opportunity_count=("pump_opportunity_count", "sum"),
            target_refresh_delayed_count=("target_refresh_delayed_count", "sum"),
            relaxation_active_time=("relaxation_active_time", "sum"),
            safety_debt_used=("safety_debt_used", "sum"),
        )
        .sort_values(["profile", "dataset", "arm"])
        .reset_index(drop=True)
    )
    baseline = agg[agg["arm"].eq("B0_learned_v16")].set_index(["dataset", "profile"])
    for idx, row in agg.iterrows():
        key = (row["dataset"], row["profile"])
        if key in baseline.index:
            for col in ("sum_pump", "sum_fb", "p95_max_axis", "time_over_5", "idle_time_over_5"):
                agg.loc[idx, f"delta_vs_B0_{col}"] = float(row[col]) - float(baseline.loc[key, col])
    return agg


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


def _write_docs(compare: pd.DataFrame, case_table: pd.DataFrame, profiles: list[str]) -> None:
    summary_rows: list[dict[str, Any]] = []
    for (dataset, profile), group in compare.groupby(["dataset", "profile"]):
        b0 = group[group["arm"].eq("B0_learned_v16")]
        b1 = group[group["arm"].eq("B1_learned_near_envelope")]
        if b0.empty or b1.empty:
            continue
        b0r = b0.iloc[0]
        b1r = b1.iloc[0]
        safety_non_worse = (
            float(b1r["time_over_5"]) <= float(b0r["time_over_5"])
            and float(b1r["idle_time_over_5"]) <= float(b0r["idle_time_over_5"])
            and float(b1r["sum_fb"]) <= float(b0r["sum_fb"]) + 1e-9
            and float(b1r["p95_max_axis"]) <= float(b0r["p95_max_axis"]) + 0.10
        )
        summary_rows.append(
            {
                "dataset": dataset,
                "profile": profile,
                "pump_saved_m3": float(b0r["sum_pump"] - b1r["sum_pump"]),
                "time5_delta_s": int(b1r["time_over_5"] - b0r["time_over_5"]),
                "idle5_delta_s": int(b1r["idle_time_over_5"] - b0r["idle_time_over_5"]),
                "fallback_delta_pp": float(b1r["sum_fb"] - b0r["sum_fb"]),
                "p95_delta_deg": float(b1r["p95_max_axis"] - b0r["p95_max_axis"]),
                "eligible_count": int(b1r["eligible_count"]),
                "target_refresh_delayed_count": int(b1r["target_refresh_delayed_count"]),
                "safety_non_worse": int(safety_non_worse),
                "passes_bar": int((float(b0r["sum_pump"] - b1r["sum_pump"]) > 1e-6) and safety_non_worse),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["profile", "dataset"])
    summary.to_csv(OUT_DIR / "learned_a1_pass_table.csv", index=False)
    summary.to_csv(OUT_DIR / "paper_ready/learned_a1_pass_table.csv", index=False)

    case_rows: list[dict[str, Any]] = []
    pivot = case_table.pivot_table(
        index=["profile", "dataset", "case"],
        columns="arm",
        values=["pump", "time_over_5", "idle_time_over_5", "sum_fb", "p95_max_axis"],
        aggfunc="first",
    )
    for idx, row in pivot.iterrows():
        try:
            pump_delta = row[("pump", "B1_learned_near_envelope")] - row[("pump", "B0_learned_v16")]
            if abs(float(pump_delta)) > 1e-6:
                case_rows.append(
                    {
                        "profile": idx[0],
                        "dataset": idx[1],
                        "case": idx[2],
                        "B1_minus_B0_pump_m3": float(pump_delta),
                        "B1_minus_B0_time5_s": int(
                            row[("time_over_5", "B1_learned_near_envelope")]
                            - row[("time_over_5", "B0_learned_v16")]
                        ),
                        "B1_minus_B0_idle5_s": int(
                            row[("idle_time_over_5", "B1_learned_near_envelope")]
                            - row[("idle_time_over_5", "B0_learned_v16")]
                        ),
                    }
                )
        except Exception:
            continue
    case_delta = pd.DataFrame(case_rows)
    case_delta.to_csv(OUT_DIR / "learned_a1_case_delta_table.csv", index=False)
    case_delta.to_csv(OUT_DIR / "raw_tables/learned_a1_case_delta_table.csv", index=False)

    any_pass = bool((summary["passes_bar"] > 0).any()) if not summary.empty else False
    broad_fail = bool(
        (
            summary["dataset"].eq("broader20")
            & (
                (summary["pump_saved_m3"] < -1e-6)
                | (summary["time5_delta_s"] > 0)
                | (summary["idle5_delta_s"] > 0)
                | (summary["fallback_delta_pp"] > 1e-9)
                | (summary["p95_delta_deg"] > 0.10)
            )
        ).any()
    ) if not summary.empty else False
    verdict = (
        "CONDITIONAL-GO"
        if any_pass and not broad_fail
        else "NO-GO"
    )
    reason = (
        "At least one learned near-relief envelope setting saves pump with no broader20 safety regression."
        if verdict == "CONDITIONAL-GO"
        else "The learned near-relief envelope does not yet reproduce the oracle pump-saving result cleanly enough to freeze as a controller result."
    )
    lines = [
        "# h120 Relief Envelope Learned A1 Validation v1",
        "",
        "Scope: validate the 0-60min relief-envelope pump-saving signal using the existing learned h240/f120 model. This does not test a new 120min closed-loop lever; it checks whether the oracle A1 savings survive without oracle future wind.",
        "",
        f"Profiles run: `{', '.join(profiles)}`.",
        "",
        "## Pass Table",
        _md_table(summary),
        "",
        "## Aggregate Metrics",
        _md_table(compare),
        "",
        "## Case-Level Pump Deltas",
        _md_table(case_delta if not case_delta.empty else pd.DataFrame()),
        "",
        "## Decision",
        "",
        f"**{verdict}.** {reason}",
        "",
        "Interpretation:",
        "",
        "- This experiment gates whether the near-horizon relief economy lever can move from oracle ceiling to learned-forecast controller candidate.",
        "- The h120 60-120min closed-loop increment remains unsupported by the prior A2-A1 oracle result; this validation is about A1 only.",
        "- A learned controller claim requires pump reduction with non-worse `time>5`, `idle>5`, fallback, and p95 max-axis, especially on broader20 and lowrisk sanity sets.",
    ]
    doc = "\n".join(lines) + "\n"
    (OUT_DIR / "learned_a1_validation_decision.md").write_text(doc, encoding="utf-8")
    (OUT_DIR / "paper_ready/learned_a1_validation_findings.md").write_text(doc, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--profiles", nargs="*", default=["conservative", "medium"])
    parser.add_argument("--datasets", nargs="*", default=["relief", "lowrisk", "broader20", "guard10"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = _mkdirs()
    profiles = [str(x) for x in args.profiles]
    for profile in profiles:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile={profile!r}; choices={sorted(PROFILES)}")
    arms = ("B0_learned_v16", "B1_learned_near_envelope")
    run_map: dict[tuple[str, str, str], Path] = {}
    for profile in profiles:
        for dataset in args.datasets:
            if dataset not in CASE_SOURCES:
                raise ValueError(f"unknown dataset={dataset!r}; choices={sorted(CASE_SOURCES)}")
            cases_csv = CASE_SOURCES[dataset]
            if not cases_csv.exists():
                raise FileNotFoundError(cases_csv)
            for arm in arms:
                run_dir = dirs["runs"] / profile / dataset / arm
                run_map[(dataset, profile, arm)] = run_dir
                _run_casebook(
                    run_dir,
                    cases_csv,
                    arm=arm,
                    profile=profile,
                    force=bool(args.force),
                )
    rows: list[dict[str, Any]] = []
    for (dataset, profile, arm), run_dir in run_map.items():
        rows.extend(_case_metrics(run_dir, dataset, profile, arm))
    case_table = pd.DataFrame(rows).sort_values(["profile", "dataset", "arm", "case"])
    case_table.to_csv(OUT_DIR / "learned_a1_case_table.csv", index=False)
    case_table.to_csv(dirs["raw"] / "learned_a1_case_table.csv", index=False)
    compare = _aggregate(case_table)
    compare.to_csv(OUT_DIR / "learned_a1_compare_table.csv", index=False)
    compare.to_csv(dirs["raw"] / "learned_a1_compare_table.csv", index=False)
    _write_docs(compare, case_table, profiles)
    print(f"[done] {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
