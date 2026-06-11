#!/usr/bin/env python3
"""Diagnose PSC No.4 failures and baseline posture health from casebook runs.

This script is deliberately read-only with respect to controller logic.  It
recomputes a common set of posture / pump / fallback metrics from existing
casebook timeseries so old mixed-pool evidence and new matched runs can be
compared without relying on partially-overlapping summary CSV columns.
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
RUNNER = REPO_ROOT / "scripts" / "analysis" / "run_prediction_primary_casebook.py"
BASE = REPO_ROOT / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
DEFAULT_OUT = BASE / "psc_4hao_failure_diagnosis_v1"
MATCHED10_CASES = DEFAULT_OUT / "casebooks" / "matched10_current_failure_probe_cases.csv"

DATASET_DIR = "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1"
MODEL_DIR = "outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1"
DURATION_S = 21600.0

MATCHED10_RAW_CASE_IDS = [
    "dual_relief_01",
    "dual_relief_06",
    "dual_relief_07",
    "dual_relief_08",
    "dual_neutral_01",
    "dual_neutral_03",
    "dual_boundary_02",
    "dual_boundary_05",
    "dual_boundary_06",
    "dual_boundary_07",
]

RUNS: dict[str, dict[str, str]] = {
    "baseline": {
        "label": "current_forecast_adaptive",
        "profile": "dc_preserving_deadband_forecast_adaptive_v1",
        "forecast_source": "current_only",
    },
    "mild_strict_off": {
        "label": "mild_strict_off",
        "profile": "psc_4hao_mild_fraction_v1",
        "forecast_source": "learned",
    },
    "mild_strict_on": {
        "label": "mild_strict_on",
        "profile": "psc_4hao_mild_fraction_v1",
        "forecast_source": "learned",
        "active_posture_refresh": "1",
    },
    "guarded_v2": {
        "label": "guarded_v2",
        "profile": "psc_4hao_guarded_v2",
        "forecast_source": "learned",
    },
    "guarded_v3": {
        "label": "guarded_v3",
        "profile": "psc_4hao_guarded_v3",
        "forecast_source": "learned",
    },
    "regime_gated_v1": {
        "label": "regime_gated_v1",
        "profile": "psc_4hao_regime_gated_v1",
        "forecast_source": "learned",
    },
    "guarded_v2_current": {
        "label": "guarded_v2_current",
        "profile": "psc_4hao_guarded_v2",
        "forecast_source": "current_only",
    },
    "chain_guard_v1": {
        "label": "chain_guard_v1",
        "profile": "psc_4hao_chain_guard_v1",
        "forecast_source": "learned",
    },
    "exec_guard_v1": {
        "label": "exec_guard_v1",
        "profile": "psc_4hao_exec_guard_v1",
        "forecast_source": "learned",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--mode",
        choices=("summarize", "run-matched10", "all"),
        default="summarize",
        help="summarize existing outputs; run-matched10 executes the focused current strict probe; all does both.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--duration-s", type=float, default=DURATION_S)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _extract_tag(label: object, *keys: str) -> str:
    text = str(label)
    for key in keys:
        marker = f"{key}="
        if marker in text:
            return text.split(marker, 1)[1].split("|", 1)[0].strip()
    if "dual_relief" in text or "transient_peak_future_decay" in text:
        return "transient_peak_future_decay"
    if "dual_neutral" in text or "neutral_mhs_broader" in text:
        return "neutral_mhs_broader"
    if "dual_boundary" in text or "direction_reversal_boundary" in text:
        return "direction_reversal_boundary"
    return "unknown"


def _find_timeseries(run_dir: Path, case_id: str, label: str) -> Path:
    matches = sorted((run_dir / "timeseries").glob(f"{case_id}_*_{label}_timeseries.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        matches = sorted((run_dir / "timeseries").glob(f"{case_id}_*_timeseries.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{run_dir}: expected one timeseries for {case_id}/{label}, found {len(matches)}")
    return matches[0]


def _summary_pump(row: dict[str, Any]) -> float:
    for key in ("primary_pump_work_m3", "pump_m3", "pump_work_m3"):
        if key in row and pd.notna(row[key]):
            return float(row[key])
    return float("nan")


def _max_run(flags: np.ndarray) -> int:
    best = 0
    cur = 0
    for flag in flags.astype(bool):
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _metrics_from_ts(scope: str, arm: str, run_dir: Path, label: str) -> pd.DataFrame:
    summary_path = run_dir / "casebook_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    rows: list[dict[str, Any]] = []
    for rec in summary.to_dict("records"):
        case_id = str(rec["case_id"])
        ts_path = _find_timeseries(run_dir, case_id, label)
        ts = pd.read_csv(ts_path, low_memory=False)
        pitch = _num(ts, "pitch_deg").abs().to_numpy(float)
        roll = _num(ts, "roll_deg").abs().to_numpy(float)
        axis = np.maximum(pitch, roll)
        fallback = _num(ts, "preview_primary_safety_fallback").to_numpy(float)
        active = _num(ts, "preview_primary_active").to_numpy(float)
        pump_rate = _num(ts, "pump_total_rate_m3_min").abs().to_numpy(float)
        dt_s = 1.0
        if "t_s" in ts.columns and len(ts) > 1:
            t = _num(ts, "t_s").to_numpy(float)
            diffs = np.diff(t)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                dt_s = float(np.median(diffs))
        rows.append(
            {
                "scope": scope,
                "arm": arm,
                "case_id": case_id,
                "label": str(rec.get("label", "")),
                "regime": _extract_tag(rec.get("label", ""), "mixed_regime", "broader6h_stratum"),
                "role": _extract_tag(rec.get("label", ""), "validation_role"),
                "pump_m3": _summary_pump(rec),
                "pump_m3_from_ts": float(np.trapezoid(pump_rate, dx=dt_s) / 60.0),
                "time_gt3_s": float(np.sum(axis > 3.0) * dt_s),
                "time_gt4_s": float(np.sum(axis > 4.0) * dt_s),
                "time_gt5_s": float(np.sum(axis > 5.0) * dt_s),
                "time_gt6_s": float(np.sum(axis > 6.0) * dt_s),
                "fallback_s": float(np.sum(fallback > 0.0) * dt_s),
                "fallback_ratio": float(np.mean(fallback > 0.0)) if len(fallback) else 0.0,
                "active_ratio": float(np.mean(active > 0.0)) if len(active) else 0.0,
                "pitch_p95_deg": float(np.quantile(pitch, 0.95)),
                "roll_p95_deg": float(np.quantile(roll, 0.95)),
                "axis_mean_deg": float(np.mean(axis)),
                "axis_p95_deg": float(np.quantile(axis, 0.95)),
                "axis_p99_deg": float(np.quantile(axis, 0.99)),
                "axis_max_deg": float(np.max(axis)),
                "max_consecutive_gt5_s": float(_max_run(axis > 5.0) * dt_s),
            }
        )
    return pd.DataFrame(rows)


def _failure_class(row: pd.Series) -> str:
    if float(row.get("d_fallback_s", 0.0)) > 0.0:
        return "fallback_added"
    if float(row.get("d_time_gt5_s", 0.0)) >= 60.0:
        return "gt5_debt_added"
    if float(row.get("d_axis_p95_deg", 0.0)) > 0.75:
        return "p95_debt_added"
    if float(row.get("saved_m3", 0.0)) <= 0.0:
        return "no_pump_saving"
    if float(row.get("baseline_time_gt5_s", 0.0)) > 0.0:
        return "safe_delta_but_baseline_tail"
    return "clean_saving"


def _operating_class(row: pd.Series) -> str:
    saved = float(row.get("saved_m3", 0.0))
    d_fallback = float(row.get("d_fallback_s", 0.0))
    d_gt5 = float(row.get("d_time_gt5_s", 0.0))
    d_p95 = float(row.get("d_axis_p95_deg", 0.0))
    p95 = float(row.get("axis_p95_deg", 0.0))
    if saved <= 0.0:
        return "reject_no_saving"
    if d_fallback == 0.0 and d_gt5 <= 0.0 and d_p95 <= 0.50:
        return "strict_no_debt_accept"
    if d_fallback > 60.0 or d_gt5 > 60.0 or p95 >= 4.5:
        return "reject_hard_posture_debt"
    if d_fallback > 0.0:
        return "light_fallback_debt_candidate"
    if d_gt5 > 0.0:
        return "light_gt5_debt_candidate"
    if d_p95 > 0.50:
        return "p95_debt_candidate"
    return "soft_pareto_candidate"


def _deltas(metrics: pd.DataFrame, baseline_arm: str = "baseline") -> pd.DataFrame:
    base = metrics[metrics["arm"].eq(baseline_arm)].set_index(["scope", "case_id"])
    rows = []
    for _, row in metrics[~metrics["arm"].eq(baseline_arm)].iterrows():
        key = (row["scope"], row["case_id"])
        if key not in base.index:
            continue
        b = base.loc[key]
        out = row.to_dict()
        for col in [
            "pump_m3",
            "time_gt3_s",
            "time_gt4_s",
            "time_gt5_s",
            "time_gt6_s",
            "fallback_s",
            "axis_mean_deg",
            "axis_p95_deg",
            "axis_p99_deg",
            "axis_max_deg",
            "max_consecutive_gt5_s",
        ]:
            out[f"baseline_{col}"] = float(b[col])
            out[f"d_{col}"] = float(row[col]) - float(b[col])
        out["saved_m3"] = float(b["pump_m3"]) - float(row["pump_m3"])
        out["saving_pct"] = 100.0 * out["saved_m3"] / max(float(b["pump_m3"]), 1e-9)
        out["baseline_tail_class"] = _baseline_tail_class(b)
        rows.append(out)
    d = pd.DataFrame(rows)
    if not d.empty:
        d["failure_class"] = d.apply(_failure_class, axis=1)
        d["operating_class"] = d.apply(_operating_class, axis=1)
    return d


def _baseline_tail_class(row: pd.Series) -> str:
    gt5 = float(row.get("time_gt5_s", 0.0))
    p95 = float(row.get("axis_p95_deg", 0.0))
    max_axis = float(row.get("axis_max_deg", 0.0))
    if gt5 >= 600.0 or p95 >= 5.0:
        return "baseline_tail_heavy"
    if gt5 > 0.0 or p95 >= 4.0 or max_axis >= 7.5:
        return "baseline_tail_present"
    return "baseline_stable"


def _existing_run_specs() -> list[tuple[str, str, Path, str]]:
    return [
        (
            "mixed24_dynamic_default",
            "baseline",
            BASE / "psc_4hao_refresh_validation_6h_fresh_20260601" / "mixed24" / "baseline",
            "current_forecast_adaptive",
        ),
        (
            "mixed24_dynamic_default",
            "refresh_off",
            BASE / "psc_4hao_refresh_validation_6h_fresh_20260601" / "mixed24" / "refresh_off",
            "refresh_off",
        ),
        (
            "mixed24_dynamic_default",
            "refresh_on",
            BASE / "psc_4hao_refresh_validation_6h_fresh_20260601" / "mixed24" / "refresh_on",
            "refresh_on",
        ),
        (
            "risky20_dynamic_default",
            "baseline",
            BASE / "psc_4hao_refresh_validation_6h_fresh_20260601" / "risky20" / "baseline",
            "current_forecast_adaptive",
        ),
        (
            "risky20_dynamic_default",
            "refresh_off",
            BASE / "psc_4hao_refresh_validation_6h_fresh_20260601" / "risky20" / "refresh_off",
            "refresh_off",
        ),
        (
            "risky20_dynamic_default",
            "refresh_on",
            BASE / "psc_4hao_refresh_validation_6h_fresh_20260601" / "risky20" / "refresh_on",
            "refresh_on",
        ),
        (
            "broader_negative_dynamic_default",
            "baseline",
            BASE / "psc_4hao_broader_6h_test_only_main_v1" / "runs" / "negative_pool" / "baseline",
            "current_forecast_adaptive",
        ),
        (
            "broader_negative_dynamic_default",
            "refresh_on",
            BASE / "psc_4hao_broader_6h_test_only_main_v1" / "runs" / "negative_pool" / "refresh_on",
            "refresh_on",
        ),
        (
            "broader_background_dynamic_default",
            "baseline",
            BASE / "psc_4hao_broader_6h_test_only_main_v1" / "runs" / "background_pool" / "baseline",
            "current_forecast_adaptive",
        ),
        (
            "broader_background_dynamic_default",
            "refresh_on",
            BASE / "psc_4hao_broader_6h_test_only_main_v1" / "runs" / "background_pool" / "refresh_on",
            "refresh_on",
        ),
        (
            "degradation96_rawenv_reference",
            "baseline",
            BASE / "degradation_ladder_96case_pair" / "current_forecast_adaptive",
            "current_forecast_adaptive",
        ),
        (
            "degradation96_rawenv_reference",
            "learned_rawenv_mainline",
            BASE / "degradation_ladder_96case_pair" / "learned_rawenv_mainline",
            "learned_rawenv_mainline",
        ),
    ]


def write_matched10_casebook(out_root: Path) -> Path:
    out_path = out_root / "casebooks" / "matched10_current_failure_probe_cases.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mixed = pd.read_csv(BASE / "casebooks" / "psc_mixed_24_cases.csv")
    part = mixed[mixed["case_id"].astype(str).isin(MATCHED10_RAW_CASE_IDS)].copy()
    order = {case_id: i for i, case_id in enumerate(MATCHED10_RAW_CASE_IDS)}
    part["order"] = part["case_id"].map(order)
    part = part.sort_values("order").drop(columns=["order"])
    if len(part) != len(MATCHED10_RAW_CASE_IDS):
        found = set(part["case_id"].astype(str))
        missing = [case_id for case_id in MATCHED10_RAW_CASE_IDS if case_id not in found]
        raise ValueError(f"matched10 casebook missing cases: {missing}")
    part.to_csv(out_path, index=False)
    return out_path


def _run_command(cmd: list[str], env_patch: dict[str, str], cwd: Path) -> None:
    env = None
    if env_patch:
        import os

        env = os.environ.copy()
        env.update(env_patch)
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def run_matched10(args: argparse.Namespace) -> None:
    out_root = _resolve(args.output_root)
    cases = write_matched10_casebook(out_root)
    for arm, spec in RUNS.items():
        run_dir = out_root / "matched10_current_strict" / arm
        summary = run_dir / "casebook_summary.csv"
        if args.skip_existing and summary.exists():
            print(f"skip existing {arm}: {summary}")
            continue
        cmd = [
            sys.executable,
            str(RUNNER),
            "--out-dir",
            str(run_dir),
            "--cases-csv",
            str(cases),
            "--duration-s",
            str(float(args.duration_s)),
            "--dataset-dir",
            str(args.dataset_dir),
            "--replay-split",
            "test",
            "--model-dir",
            str(args.model_dir),
            "--primary-label",
            spec["label"],
            "--primary-only",
            "--skip-figures",
            "--primary-control-profile",
            spec["profile"],
            "--forecast-source",
            spec["forecast_source"],
        ]
        if spec["profile"].startswith("psc_4hao_"):
            cmd.extend(["--primary-safety-profile", "strict"])
        env_patch = {}
        if "active_posture_refresh" in spec:
            env_patch["FOWT_4HAO_ACTIVE_POSTURE_REFRESH"] = spec["active_posture_refresh"]
        _run_command(cmd, env_patch, REPO_ROOT)


def _collect_metrics(out_root: Path) -> pd.DataFrame:
    frames = []
    for scope, arm, run_dir, label in _existing_run_specs():
        if (run_dir / "casebook_summary.csv").exists():
            frames.append(_metrics_from_ts(scope, arm, run_dir, label))
    matched_root = out_root / "matched10_current_strict"
    for arm, spec in RUNS.items():
        run_dir = matched_root / arm
        if (run_dir / "casebook_summary.csv").exists():
            frames.append(_metrics_from_ts("matched10_mild_strict_current", arm, run_dir, spec["label"]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _group_summary(metrics: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, arm, regime), part in metrics.groupby(["scope", "arm", "regime"], dropna=False):
        rows.append(
            {
                "scope": scope,
                "arm": arm,
                "regime": regime,
                "cases": int(len(part)),
                "pump_m3": float(part["pump_m3"].sum()),
                "time_gt5_s": float(part["time_gt5_s"].sum()),
                "time_gt6_s": float(part["time_gt6_s"].sum()),
                "fallback_s": float(part["fallback_s"].sum()),
                "max_axis_p95_deg": float(part["axis_p95_deg"].max()),
                "tail_heavy_cases": int(part.apply(_baseline_tail_class, axis=1).eq("baseline_tail_heavy").sum()),
                "tail_present_cases": int(part.apply(_baseline_tail_class, axis=1).ne("baseline_stable").sum()),
            }
        )
    out = pd.DataFrame(rows)
    if deltas.empty:
        return out
    d_rows = []
    for (scope, arm, regime), part in deltas.groupby(["scope", "arm", "regime"], dropna=False):
        d_rows.append(
            {
                "scope": scope,
                "arm": arm,
                "regime": regime,
                "saved_m3": float(part["saved_m3"].sum()),
                "saving_pct_vs_baseline": 100.0
                * float(part["saved_m3"].sum())
                / max(float(part["baseline_pump_m3"].sum()), 1e-9),
                "d_time_gt5_s": float(part["d_time_gt5_s"].sum()),
                "d_time_gt6_s": float(part["d_time_gt6_s"].sum()),
                "d_fallback_s": float(part["d_fallback_s"].sum()),
                "max_d_axis_p95_deg": float(part["d_axis_p95_deg"].max()),
                "clean_saving_cases": int(part["failure_class"].eq("clean_saving").sum()),
                "failed_cases": int((~part["failure_class"].isin(["clean_saving", "safe_delta_but_baseline_tail"])).sum()),
            }
        )
    d = pd.DataFrame(d_rows)
    return out.merge(d, on=["scope", "arm", "regime"], how="left")


def _fmt_num(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.4g}"
    return str(value)


def _md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_empty_"
    if max_rows is not None:
        df = df.head(max_rows)
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for rec in df.to_dict("records"):
        lines.append("| " + " | ".join(_fmt_num(rec[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def _write_readout(out_root: Path, metrics: pd.DataFrame, deltas: pd.DataFrame, group: pd.DataFrame) -> None:
    readout = out_root / "psc_4hao_failure_diagnosis_readout.md"
    base = metrics[metrics["arm"].eq("baseline")].copy()
    baseline_tail = (
        base.assign(tail_class=base.apply(_baseline_tail_class, axis=1))
        .groupby(["scope", "regime", "tail_class"], dropna=False)
        .agg(cases=("case_id", "count"), time_gt5_s=("time_gt5_s", "sum"), fallback_s=("fallback_s", "sum"), max_p95_axis_deg=("axis_p95_deg", "max"))
        .reset_index()
        .sort_values(["scope", "regime", "tail_class"])
    )
    worst = (
        deltas.sort_values(["d_fallback_s", "d_time_gt5_s", "d_axis_p95_deg"], ascending=False)
        if not deltas.empty
        else deltas
    )
    failure_counts = (
        deltas.groupby(["scope", "arm", "regime", "failure_class"], dropna=False)
        .agg(cases=("case_id", "count"), saved_m3=("saved_m3", "sum"), d_time_gt5_s=("d_time_gt5_s", "sum"), d_fallback_s=("d_fallback_s", "sum"))
        .reset_index()
        .sort_values(["scope", "arm", "regime", "failure_class"])
        if not deltas.empty
        else pd.DataFrame()
    )
    operating_counts = (
        deltas.groupby(["scope", "arm", "regime", "operating_class"], dropna=False)
        .agg(
            cases=("case_id", "count"),
            saved_m3=("saved_m3", "sum"),
            d_time_gt5_s=("d_time_gt5_s", "sum"),
            d_fallback_s=("d_fallback_s", "sum"),
            max_axis_p95_deg=("axis_p95_deg", "max"),
        )
        .reset_index()
        .sort_values(["scope", "arm", "regime", "operating_class"])
        if not deltas.empty
        else pd.DataFrame()
    )
    matched = group[group["scope"].eq("matched10_mild_strict_current")].copy()
    lines = [
        "# PSC No.4 failure diagnosis v1",
        "",
        "Purpose: diagnose why the No.4 economy arm fails, and whether the baseline posture itself is already unstable.",
        "",
        "## Key readout",
        "",
        "- The baseline has non-zero posture tails in selected relief/boundary windows, but it is not globally in fallback-driven loss of control: baseline fallback is zero in these casebook runs.",
        "- The unsafe No.4 pattern is concentrated where economy actions save pump while adding `time_gt5_s` or fallback on top of an already thin baseline posture margin.",
        "- Neutral moderate-high steady windows are the cleanest saving region; direction-reversal/boundary and some relief windows are the rejection region.",
        "- Current `mild_fraction_v1 + strict` evidence should be read separately from older `dynamic_refresh_v1` evidence because their safety/profile defaults differ.",
        "",
        "## Baseline Tail Health",
        "",
        _md_table(baseline_tail),
        "",
        "## Group Summary",
        "",
        _md_table(group.sort_values(["scope", "arm", "regime"])),
        "",
        "## Failure Classes",
        "",
        _md_table(failure_counts),
        "",
        "## Operating Classes",
        "",
        _md_table(operating_counts),
        "",
        "## Matched10 Current Strict Summary",
        "",
        _md_table(matched.sort_values(["arm", "regime"])),
        "",
        "## Worst Delta Cases",
        "",
        _md_table(
            worst[
                [
                    "scope",
                    "arm",
                    "case_id",
                    "regime",
                    "baseline_tail_class",
                    "saved_m3",
                    "d_time_gt5_s",
                    "d_time_gt6_s",
                    "d_fallback_s",
                    "d_axis_p95_deg",
                    "failure_class",
                    "operating_class",
                ]
            ],
            max_rows=30,
        ),
        "",
        "## Artifacts",
        "",
        f"- Metrics: `{(out_root / 'case_metrics_long.csv').relative_to(REPO_ROOT)}`",
        f"- Deltas: `{(out_root / 'case_deltas_vs_baseline.csv').relative_to(REPO_ROOT)}`",
        f"- Group summary: `{(out_root / 'group_summary.csv').relative_to(REPO_ROOT)}`",
        f"- Matched10 casebook: `{MATCHED10_CASES.relative_to(REPO_ROOT)}`",
    ]
    readout.write_text("\n".join(lines), encoding="utf-8")
    print(readout)


def summarize(args: argparse.Namespace) -> None:
    out_root = _resolve(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    metrics = _collect_metrics(out_root)
    if metrics.empty:
        raise SystemExit("no casebook outputs found")
    deltas = _deltas(metrics)
    group = _group_summary(metrics, deltas)
    metrics.to_csv(out_root / "case_metrics_long.csv", index=False)
    deltas.to_csv(out_root / "case_deltas_vs_baseline.csv", index=False)
    group.to_csv(out_root / "group_summary.csv", index=False)
    _write_readout(out_root, metrics, deltas, group)


def main() -> None:
    args = parse_args()
    out_root = _resolve(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "duration_s": float(args.duration_s),
        "dataset_dir": str(args.dataset_dir),
        "model_dir": str(args.model_dir),
        "matched10_raw_case_ids": MATCHED10_RAW_CASE_IDS,
    }
    (out_root / "diagnosis_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_matched10_casebook(out_root)
    if args.mode in {"run-matched10", "all"}:
        run_matched10(args)
    if args.mode in {"summarize", "all"}:
        summarize(args)


if __name__ == "__main__":
    main()
