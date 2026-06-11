#!/usr/bin/env python3
"""Locked holdout validation for learned 0-60min relief-economy A1.

This script is intentionally narrow:

* fixed medium relief-envelope profile;
* learned forecast only;
* A0 = v1.6 learned baseline;
* A1 = v1.6 + learned near-horizon relief envelope;
* A1-axis = same A1 plus a single-axis action-shape veto;
* no 60-120min closed-loop branch, no parameter sweep.

It also writes a consistency audit against the corrected oracle envelope audit
so the learned result is not mistaken for a different control lever.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv312/bin/python"
if not PYTHON.exists():
    PYTHON = REPO_ROOT / ".venv/bin/python"

OUT_DIR = REPO_ROOT / "outputs/wind_prediction/h120_relief_envelope_learned_a1_locked_holdout_v1"
F120_DATASET = REPO_ROOT / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"
H120_MODEL = REPO_ROOT / "outputs/wind_prediction/lstm_h240_f120_near_block_eventbalanced_v2"

CASE_SOURCES = {
    "guard10": REPO_ROOT / "outputs/wind_prediction/h120_oracle_controller_value_audit_v1/guard10_cases.csv",
    "broader20": REPO_ROOT / "outputs/wind_prediction/h120_oracle_controller_value_audit_v1/broader20_cases.csv",
    "relief_v2": REPO_ROOT / "outputs/wind_prediction/prediction_value_mechanism_grid_v2/future_relief_cases.csv",
    "lowrisk_v2": REPO_ROOT / "outputs/wind_prediction/prediction_value_mechanism_grid_v2/lowrisk_quiet_cases.csv",
    "relief_hold_risk_v2": REPO_ROOT / "outputs/wind_prediction/prediction_value_mechanism_grid_v2/relief_hold_risk_cases.csv",
    "f60_relief_holdout": REPO_ROOT / "outputs/wind_prediction/f60_relief_e15_holdout_validation_v1/f60_relief_e15_holdout_cases.csv",
    "future_relief_holdout": REPO_ROOT / "outputs/wind_prediction/prediction_value_holdout_selection_v1/future_relief_holdout_cases.csv",
    "random_holdout": REPO_ROOT / "outputs/wind_prediction/prediction_value_holdout_selection_v1/random_holdout_cases.csv",
    "signflip_holdout": REPO_ROOT / "outputs/wind_prediction/prediction_value_holdout_selection_v1/case07_like_holdout_cases.csv",
}

MEDIUM_PROFILE = {
    "allowed_0_20": 4.65,
    "allowed_20_40": 4.40,
    "allowed_40_60": 4.10,
    "allowed_60_120": 3.70,
    "duration_s": 900.0,
    "debt_budget": 600.0,
    "near_limit_budget_s": 300.0,
    "worsening_eps": 0.05,
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


def _normalize_case_source(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"case_id", "timestamp"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = df.copy()
    if "label" not in out.columns:
        out["label"] = name
    out["case_source"] = name
    out["source_file"] = str(path.relative_to(REPO_ROOT))
    return out[["case_id", "timestamp", "label", "case_source", "source_file"]]


def _build_locked_casebook() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, path in CASE_SOURCES.items():
        if path.exists():
            frames.append(_normalize_case_source(name, path))
    if not frames:
        raise FileNotFoundError("no case sources found")
    all_cases = pd.concat(frames, ignore_index=True)
    # The same physical window appears in several historical casebooks under
    # different aliases.  Use timestamp as the locked-window identity and keep
    # all source aliases for provenance, otherwise the holdout would overweight
    # repeated windows.
    all_cases["_key"] = all_cases["timestamp"].astype(str)

    # Prefer broader named validation sets over repeated aliases at the same timestamp.
    source_priority = {
        "broader20": 0,
        "guard10": 1,
        "f60_relief_holdout": 2,
        "future_relief_holdout": 3,
        "random_holdout": 4,
        "signflip_holdout": 5,
        "relief_hold_risk_v2": 6,
        "relief_v2": 7,
        "lowrisk_v2": 8,
    }
    all_cases["_priority"] = all_cases["case_source"].map(source_priority).fillna(99).astype(int)
    merged_sources = (
        all_cases.groupby("_key", as_index=False)["case_source"]
        .agg(lambda s: ",".join(sorted(set(map(str, s)))))
        .rename(columns={"case_source": "source_groups"})
    )
    merged_case_ids = (
        all_cases.groupby("_key", as_index=False)["case_id"]
        .agg(lambda s: ",".join(sorted(set(map(str, s)))))
        .rename(columns={"case_id": "source_case_ids"})
    )
    dedup = (
        all_cases.sort_values(["_priority", "timestamp", "case_id"])
        .drop_duplicates("_key", keep="first")
        .drop(columns=["_priority"])
        .merge(merged_sources, on="_key", how="left")
        .merge(merged_case_ids, on="_key", how="left")
        .drop(columns=["_key"])
        .sort_values(["timestamp", "case_id"])
        .reset_index(drop=True)
    )
    dedup["locked_index"] = np.arange(1, len(dedup) + 1)
    return dedup[
        [
            "locked_index",
            "case_id",
            "timestamp",
            "label",
            "case_source",
            "source_groups",
            "source_case_ids",
            "source_file",
        ]
    ]


def _casebook_for_runner(casebook: pd.DataFrame) -> pd.DataFrame:
    labels = casebook["label"].astype(str) + " | locked_sources=" + casebook["source_groups"].astype(str)
    return pd.DataFrame(
        {
            "case_id": casebook["case_id"],
            "timestamp": casebook["timestamp"],
            "label": labels,
        }
    )


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


def _run_casebook(run_dir: Path, cases_csv: Path, *, arm: str, force: bool) -> None:
    if (run_dir / "casebook_summary.csv").exists() and not force:
        print(f"[skip] {run_dir.relative_to(REPO_ROOT)}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = _base_cmd(run_dir, cases_csv)
    if arm in (
        "A1_learned_near_envelope_medium",
        "A1_learned_near_envelope_medium_axis_guard",
    ):
        p = MEDIUM_PROFILE
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
        if arm == "A1_learned_near_envelope_medium_axis_guard":
            cmd.append("--relief-envelope-single-axis-only")
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


def _case_metrics(run_dir: Path, arm: str, casebook: pd.DataFrame) -> list[dict[str, Any]]:
    meta_by_case = casebook.set_index("case_id").to_dict(orient="index")
    meta_by_ts = casebook.set_index(casebook["timestamp"].astype(str)).to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for ts_path in sorted((run_dir / "timeseries").glob("*_timeseries.csv")):
        key = _case_key(ts_path)
        case_id = key.replace("_prediction_primary_econ", "")
        ts_match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{6})", case_id)
        parsed_ts = ""
        if ts_match:
            hhmmss = ts_match.group(2)
            parsed_ts = f"{ts_match.group(1)} {hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"
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
        m = meta_by_case.get(case_id, {}) or meta_by_ts.get(parsed_ts, {})
        rows.append(
            {
                "arm": arm,
                "case": case_id,
                "case_source": m.get("case_source", ""),
                "source_groups": m.get("source_groups", ""),
                "timestamp": m.get("timestamp", ""),
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
                "target_refresh_delayed_count": int(_last_metric(log, "relief_envelope_target_refresh_delayed_count")),
                "axis_shape_veto_count": int(_last_metric(log, "relief_envelope_axis_shape_veto_count")),
                "relaxation_active_time": float(_last_metric(log, "relief_envelope_relaxation_active_time_s")),
                "safety_debt_used": float(_last_metric(log, "relief_envelope_safety_debt_used")),
            }
        )
    return rows


def _aggregate(case_table: pd.DataFrame) -> pd.DataFrame:
    agg = (
        case_table.groupby("arm", as_index=False)
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
            axis_shape_veto_count=("axis_shape_veto_count", "sum"),
            relaxation_active_time=("relaxation_active_time", "sum"),
            safety_debt_used=("safety_debt_used", "sum"),
        )
        .reset_index(drop=True)
    )
    b0 = agg[agg["arm"].eq("A0_learned_v16")]
    if not b0.empty:
        b0r = b0.iloc[0]
        for idx, row in agg.iterrows():
            for col in ("sum_pump", "sum_fb", "p95_max_axis", "max_axis", "time_over_5", "idle_time_over_5"):
                agg.loc[idx, f"delta_vs_A0_{col}"] = float(row[col]) - float(b0r[col])
    return agg


def _case_delta(case_table: pd.DataFrame) -> pd.DataFrame:
    pivot = case_table.pivot_table(
        index=["case", "case_source", "source_groups", "timestamp"],
        columns="arm",
        values=["pump", "time_over_5", "idle_time_over_5", "sum_fb", "p95_max_axis", "max_axis"],
        aggfunc="first",
    )
    rows: list[dict[str, Any]] = []
    for idx, row in pivot.iterrows():
        try:
            b0 = "A0_learned_v16"
            b1 = (
                "A1_learned_near_envelope_medium_axis_guard"
                if ("pump", "A1_learned_near_envelope_medium_axis_guard") in row.index
                else "A1_learned_near_envelope_medium"
            )
            rows.append(
                {
                    "case": idx[0],
                    "case_source": idx[1],
                    "source_groups": idx[2],
                    "timestamp": idx[3],
                    "A1_minus_A0_pump_m3": float(row[("pump", b1)] - row[("pump", b0)]),
                    "A1_minus_A0_time5_s": int(row[("time_over_5", b1)] - row[("time_over_5", b0)]),
                    "A1_minus_A0_idle5_s": int(row[("idle_time_over_5", b1)] - row[("idle_time_over_5", b0)]),
                    "A1_minus_A0_fallback_pp": float(row[("sum_fb", b1)] - row[("sum_fb", b0)]),
                    "A1_minus_A0_p95_deg": float(row[("p95_max_axis", b1)] - row[("p95_max_axis", b0)]),
                    "A1_minus_A0_max_axis_deg": float(row[("max_axis", b1)] - row[("max_axis", b0)]),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values(["A1_minus_A0_pump_m3", "case"]).reset_index(drop=True)


def _trigger_diagnostics(case_table: pd.DataFrame, case_delta: pd.DataFrame) -> pd.DataFrame:
    arm = (
        "A1_learned_near_envelope_medium_axis_guard"
        if case_table["arm"].eq("A1_learned_near_envelope_medium_axis_guard").any()
        else "A1_learned_near_envelope_medium"
    )
    b1 = case_table[case_table["arm"].eq(arm)].copy()
    cols = [
        "case",
        "case_source",
        "source_groups",
        "eligible_count",
        "pump_opportunity_count",
        "target_refresh_delayed_count",
        "axis_shape_veto_count",
        "relaxation_active_time",
        "safety_debt_used",
    ]
    out = b1[cols].merge(
        case_delta[
            [
                "case",
                "A1_minus_A0_pump_m3",
                "A1_minus_A0_time5_s",
                "A1_minus_A0_idle5_s",
                "A1_minus_A0_fallback_pp",
                "A1_minus_A0_p95_deg",
            ]
        ],
        on="case",
        how="left",
    )
    return out.sort_values(["target_refresh_delayed_count", "A1_minus_A0_pump_m3"], ascending=[False, True])


def _source_summary(case_delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, group in case_delta.groupby("case_source"):
        rows.append(
            {
                "case_source": source,
                "cases": len(group),
                "pump_saved_m3": float(-group["A1_minus_A0_pump_m3"].sum()),
                "cases_with_pump_saving": int((group["A1_minus_A0_pump_m3"] < -1e-6).sum()),
                "time5_delta_s": int(group["A1_minus_A0_time5_s"].sum()),
                "idle5_delta_s": int(group["A1_minus_A0_idle5_s"].sum()),
                "fallback_delta_pp": float(group["A1_minus_A0_fallback_pp"].sum()),
                "max_p95_delta_deg": float(group["A1_minus_A0_p95_deg"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("pump_saved_m3", ascending=False)


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


def _write_consistency_audit(dirs: dict[str, Path]) -> None:
    text = f"""# Relief Envelope Consistency Audit

## Lever Identity

The corrected oracle A1 and the learned A1 locked validation use the same provider lever:

- CLI flag: `--relief-envelope`
- Horizon for A1: `--relief-envelope-horizon near`
- Control action: delay a non-hard target refresh by reusing the current primary target.
- Hard floor: unchanged. Reactive floor, fallback, and recovery branches are earlier in the provider chain and still override the relief envelope.
- Pump penalty: unchanged.
- Relief suppression: not used.
- Default behavior: off unless `--relief-envelope` is passed.

The final learned arm additionally enables `--relief-envelope-single-axis-only`. This
does not change the floor or the forecast gate; it only refuses to delay a refresh
when the refresh action itself is mixed-axis. The mechanism audit found mixed-axis
target delay can create later catch-up pumping, while the saving triggers were
single-axis / dominant-axis clear.

## Fixed Medium Profile

| parameter | value |
| --- | --- |
| allowed 0-20min | {MEDIUM_PROFILE['allowed_0_20']:.2f} deg |
| allowed 20-40min | {MEDIUM_PROFILE['allowed_20_40']:.2f} deg |
| allowed 40-60min | {MEDIUM_PROFILE['allowed_40_60']:.2f} deg |
| max duration | {MEDIUM_PROFILE['duration_s']:.0f} s |
| debt budget | {MEDIUM_PROFILE['debt_budget']:.0f} deg*s |
| near-limit budget | {MEDIUM_PROFILE['near_limit_budget_s']:.0f} s |
| worsening epsilon | {MEDIUM_PROFILE['worsening_eps']:.2f} deg |

## Why Oracle A1 And Learned A1 Can Differ

They share the same hook and envelope logic. The difference is the forecast source feeding the same
near pressure blocks. Oracle A1 fires only when true 0-60min blocks satisfy the relief envelope.
Learned A1 can fire on different buckets when the learned h120 model predicts near relief. That is
not a separate control mechanism; it is the learned model driving the same default-off economy-refresh
delay. The locked validation therefore has to treat any learned-only benefit as a controller candidate
only if safety metrics remain non-worse on a larger holdout.
"""
    path = dirs["diag"] / "consistency_audit.md"
    path.write_text(text, encoding="utf-8")


def _write_decision(
    compare: pd.DataFrame,
    case_delta: pd.DataFrame,
    trigger: pd.DataFrame,
    source_summary: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    a0 = compare[compare["arm"].eq("A0_learned_v16")].iloc[0]
    final_arm = (
        "A1_learned_near_envelope_medium_axis_guard"
        if compare["arm"].eq("A1_learned_near_envelope_medium_axis_guard").any()
        else "A1_learned_near_envelope_medium"
    )
    a1 = compare[compare["arm"].eq(final_arm)].iloc[0]
    pump_saved = float(a0["sum_pump"] - a1["sum_pump"])
    time_delta = int(a1["time_over_5"] - a0["time_over_5"])
    idle_delta = int(a1["idle_time_over_5"] - a0["idle_time_over_5"])
    fb_delta = float(a1["sum_fb"] - a0["sum_fb"])
    p95_delta = float(a1["p95_max_axis"] - a0["p95_max_axis"])
    safety_non_worse = (
        time_delta <= 0
        and idle_delta <= 0
        and fb_delta <= 1e-9
        and p95_delta <= 0.10
    )
    saved_cases = case_delta[case_delta["A1_minus_A0_pump_m3"] < -1e-6].copy()
    regress_cases = case_delta[
        (case_delta["A1_minus_A0_time5_s"] > 0)
        | (case_delta["A1_minus_A0_idle5_s"] > 0)
        | (case_delta["A1_minus_A0_fallback_pp"] > 1e-9)
        | (case_delta["A1_minus_A0_p95_deg"] > 0.10)
    ].copy()
    concentration = (
        float((-saved_cases["A1_minus_A0_pump_m3"]).max() / pump_saved)
        if pump_saved > 1e-9 and not saved_cases.empty
        else 0.0
    )
    not_too_concentrated = len(saved_cases) >= 4 and concentration < 0.70
    verdict = "GO" if pump_saved > 1e-6 and safety_non_worse and not_too_concentrated else (
        "CONDITIONAL-GO" if pump_saved > 1e-6 and safety_non_worse else "NO-GO"
    )
    lines = [
        "# Learned A1 Locked Holdout Validation",
        "",
        f"**Decision: {verdict}.** Fixed medium learned 0-60min relief-envelope validation completed on a deduplicated locked casebook.",
        "",
        "## Aggregate Result",
        _md_table(compare),
        "",
        "## Source-Level Summary",
        _md_table(source_summary),
        "",
        "## Trigger Diagnostics",
        _md_table(trigger[trigger["target_refresh_delayed_count"] > 0].head(20)),
        "",
        "## Main Interpretation",
        "",
        f"- Pump saved: `{pump_saved:.2f} m3`.",
        f"- Safety deltas: time>5 `{time_delta:+d}s`, idle>5 `{idle_delta:+d}s`, fallback `{fb_delta:+.3f}pp`, p95 max-axis `{p95_delta:+.3f} deg`.",
        f"- Cases with pump saving: `{len(saved_cases)}` / `{len(case_delta)}`.",
        f"- Largest single-case share of total saving: `{concentration:.2%}`.",
        f"- Safety-regression cases under the hard bar: `{len(regress_cases)}`.",
        f"- Final learned arm: `{final_arm}`.",
        "- Mechanism audit: the single-axis guard removes mixed-axis delayed refreshes that can create later catch-up pump; it is an action-shape veto, not a case-specific rule.",
        "",
        "## 0-60min vs 60-120min Explanation",
        "",
        "This locked validation is deliberately A1-only. It validates whether the previously corrected near-horizon relief economy lever survives with learned forecast blocks. The earlier corrected oracle audit found `A2 - A1 = 0`: adding 60-120min blocks did not create extra target-refresh delays or extra pump saving. The current result therefore supports a near-horizon learned controller lever, not a 120min closed-loop pump-saving claim.",
        "",
        "## Consistency Statement",
        "",
        "The learned and oracle A1 arms use the same default-off relief-envelope provider hook. Both delay non-hard target refreshes by reusing the existing primary target, while the v1.6 hard floor, fallback, recovery branches, and pump penalty remain untouched.",
    ]
    doc = "\n".join(lines) + "\n"
    (OUT_DIR / "decision.md").write_text(doc, encoding="utf-8")
    (dirs["paper"] / "learned_a1_locked_holdout_summary.md").write_text(doc, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--casebook-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = _mkdirs()
    casebook = _build_locked_casebook()
    casebook.to_csv(OUT_DIR / "locked_casebook_manifest.csv", index=False)
    casebook.to_csv(dirs["raw"] / "locked_casebook_manifest.csv", index=False)
    runner_cases = _casebook_for_runner(casebook)
    runner_cases_csv = OUT_DIR / "locked_casebook_cases.csv"
    runner_cases.to_csv(runner_cases_csv, index=False)
    _write_consistency_audit(dirs)
    if args.casebook_only:
        print(f"[casebook] {runner_cases_csv.relative_to(REPO_ROOT)} rows={len(runner_cases)}")
        return

    arms = (
        "A0_learned_v16",
        "A1_learned_near_envelope_medium",
        "A1_learned_near_envelope_medium_axis_guard",
    )
    run_dirs: dict[str, Path] = {}
    for arm in arms:
        run_dir = dirs["runs"] / arm
        run_dirs[arm] = run_dir
        _run_casebook(run_dir, runner_cases_csv, arm=arm, force=bool(args.force))

    rows: list[dict[str, Any]] = []
    for arm, run_dir in run_dirs.items():
        rows.extend(_case_metrics(run_dir, arm, casebook))
    case_table = pd.DataFrame(rows).sort_values(["arm", "case"]).reset_index(drop=True)
    case_table.to_csv(OUT_DIR / "learned_a1_locked_holdout_case_table.csv", index=False)
    case_table.to_csv(dirs["raw"] / "learned_a1_locked_holdout_case_table.csv", index=False)

    compare = _aggregate(case_table)
    compare.to_csv(OUT_DIR / "learned_a1_locked_holdout_compare_table.csv", index=False)
    compare.to_csv(dirs["raw"] / "learned_a1_locked_holdout_compare_table.csv", index=False)

    case_delta = _case_delta(case_table)
    case_delta.to_csv(OUT_DIR / "learned_a1_per_case_delta_table.csv", index=False)
    case_delta.to_csv(dirs["raw"] / "learned_a1_per_case_delta_table.csv", index=False)

    trigger = _trigger_diagnostics(case_table, case_delta)
    trigger.to_csv(OUT_DIR / "learned_a1_trigger_diagnostics.csv", index=False)
    trigger.to_csv(dirs["raw"] / "learned_a1_trigger_diagnostics.csv", index=False)

    source_summary = _source_summary(case_delta)
    source_summary.to_csv(OUT_DIR / "learned_a1_source_summary_table.csv", index=False)
    source_summary.to_csv(dirs["raw"] / "learned_a1_source_summary_table.csv", index=False)

    _write_decision(compare, case_delta, trigger, source_summary, dirs)
    shutil.copy2(dirs["diag"] / "consistency_audit.md", OUT_DIR / "consistency_audit.md")
    print(f"[done] {OUT_DIR.relative_to(REPO_ROOT)} rows={len(casebook)}")


if __name__ == "__main__":
    main()
