#!/usr/bin/env python3
"""Run/summarize broader 6h PSC No.4 validation pools."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "analysis" / "run_prediction_primary_casebook.py"
BASE = REPO / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1" / "psc_4hao_broader_6h_test_only_main_v1"
CASEBOOKS = BASE / "casebooks"
OUT = BASE / "runs"

POOLS = {
    "positive_pool": CASEBOOKS / "positive_pool_cases.csv",
    "negative_pool": CASEBOOKS / "negative_pool_cases.csv",
    "background_pool": CASEBOOKS / "background_pool_cases.csv",
    "stress_pool": CASEBOOKS / "stress_pool_cases.csv",
    "main_broader_pool": CASEBOOKS / "main_broader_pool_cases.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", default="positive_pool,negative_pool,background_pool")
    parser.add_argument(
        "--arms",
        default="baseline,refresh_on",
        help=(
            "Comma-separated arms to run/summarize. Available: baseline, "
            "refresh_on, generalized_gate."
        ),
    )
    parser.add_argument("--mode", choices=("dry-run", "run", "summarize"), default="dry-run")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--duration-s", type=float, default=21600.0)
    parser.add_argument("--dataset-dir", default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1")
    parser.add_argument("--replay-split", choices=("test", "validation", "train"), default="test")
    parser.add_argument("--model-dir", default="outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _arm(arm: str) -> tuple[str, list[str], dict[str, str]]:
    if arm == "baseline":
        return (
            "current_forecast_adaptive",
            [
                "--primary-control-profile",
                "dc_preserving_deadband_forecast_adaptive_v1",
                "--forecast-source",
                "current_only",
            ],
            {},
        )
    if arm == "refresh_on":
        return (
            "refresh_on",
            [
                "--primary-control-profile",
                "psc_4hao_dynamic_refresh_v1",
                "--forecast-source",
                "learned",
            ],
            {"FOWT_4HAO_ACTIVE_POSTURE_REFRESH": "1"},
        )
    if arm == "generalized_gate":
        return (
            "generalized_gate",
            [
                "--primary-control-profile",
                "psc_4hao_generalized_gate_v1",
                "--forecast-source",
                "learned",
            ],
            {"FOWT_4HAO_ACTIVE_POSTURE_REFRESH": "1"},
        )
    raise ValueError(arm)


def _cmd(args: argparse.Namespace, pool: str, arm: str) -> tuple[list[str], dict[str, str], Path]:
    label, profile, env = _arm(arm)
    out_dir = args.output_root / pool / arm
    return (
        [
            sys.executable,
            str(RUNNER),
            "--out-dir",
            str(out_dir),
            "--cases-csv",
            str(POOLS[pool]),
            "--duration-s",
            str(args.duration_s),
            "--dataset-dir",
            str(args.dataset_dir),
            "--replay-split",
            str(args.replay_split),
            "--model-dir",
            str(args.model_dir),
            "--primary-label",
            label,
            "--primary-only",
            "--skip-figures",
            *profile,
        ],
        env,
        out_dir,
    )


def _run(args: argparse.Namespace, pool: str, arm: str) -> None:
    cmd, env_patch, out_dir = _cmd(args, pool, arm)
    summary = out_dir / "casebook_summary.csv"
    if args.skip_existing and summary.exists():
        print(f"skip existing {pool}/{arm}: {summary}")
        return
    prefix = " ".join(f"{k}={v}" for k, v in env_patch.items())
    print(("+ " + prefix + " " if prefix else "+ ") + " ".join(cmd))
    if args.mode == "run":
        env = os.environ.copy()
        env.update(env_patch)
        subprocess.run(cmd, cwd=REPO, env=env, check=True)


def _find_ts(run_dir: Path, case_id: str, label: str) -> Path:
    matches = sorted((run_dir / "timeseries").glob(f"{case_id}_*_{label}_timeseries.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{run_dir}: {case_id}/{label} matches={len(matches)}")
    return matches[0]


def _metrics(run_dir: Path, arm: str) -> pd.DataFrame:
    label, _, _ = _arm(arm)
    summary = pd.read_csv(run_dir / "casebook_summary.csv")
    rows: list[dict[str, Any]] = []
    for rec in summary.to_dict("records"):
        case_id = str(rec["case_id"])
        ts = pd.read_csv(_find_ts(run_dir, case_id, label), low_memory=False)
        pitch = pd.to_numeric(ts.get("pitch_deg", 0.0), errors="coerce").fillna(0.0)
        roll = pd.to_numeric(ts.get("roll_deg", 0.0), errors="coerce").fillna(0.0)
        axis = np.maximum(np.abs(pitch), np.abs(roll))
        fallback = pd.to_numeric(ts.get("preview_primary_safety_fallback", pd.Series(0.0, index=ts.index)), errors="coerce").fillna(0.0)
        rows.append(
            {
                "arm": arm,
                "case_id": case_id,
                "label": str(rec.get("label", "")),
                "pump_m3": float(rec["primary_pump_work_m3"]),
                "time_gt3_s": float(np.sum(axis > 3.0)),
                "time_gt4_s": float(np.sum(axis > 4.0)),
                "time_gt5_s": float(np.sum(axis > 5.0)),
                "fallback_s": float(np.sum(fallback > 0.0)),
                "p95_axis_deg": float(np.quantile(axis, 0.95)),
                "max_axis_deg": float(np.max(axis)),
            }
        )
    return pd.DataFrame(rows)


def _extract(label: str, key: str) -> str:
    marker = f"{key}="
    text = str(label)
    if marker not in text:
        return ""
    return text.split(marker, 1)[1].split("|", 1)[0].strip()


def summarize(args: argparse.Namespace, pools: list[str], arms: list[str]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    compare_arms = [arm for arm in arms if arm != "baseline"]
    if "baseline" not in arms:
        raise ValueError("summarize requires the baseline arm")
    if not compare_arms:
        raise ValueError("summarize requires at least one non-baseline arm")
    for pool in pools:
        pool_dir = args.output_root / pool
        m = pd.concat(
            [_metrics(pool_dir / arm, arm) for arm in arms],
            ignore_index=True,
        )
        m["stratum"] = m["label"].map(lambda x: _extract(x, "broader6h_stratum"))
        m["validation_role"] = m["label"].map(lambda x: _extract(x, "validation_role"))
        m.to_csv(pool_dir / "case_metrics.csv", index=False)
        base = m[m["arm"].eq("baseline")].set_index("case_id")
        for arm in compare_arms:
            on = m[m["arm"].eq(arm)].copy().join(base, on="case_id", rsuffix="_baseline")
            for col in ["pump_m3", "time_gt3_s", "time_gt4_s", "time_gt5_s", "fallback_s", "p95_axis_deg", "max_axis_deg"]:
                on[f"d_{col}"] = on[col] - on[f"{col}_baseline"]
            on["saved_m3"] = on["pump_m3_baseline"] - on["pump_m3"]
            on["saving_pct"] = 100.0 * on["saved_m3"] / on["pump_m3_baseline"].clip(lower=1e-9)
            on.to_csv(pool_dir / f"{arm}_case_deltas.csv", index=False)
            group = (
                on.groupby(["validation_role", "stratum"], dropna=False)
                .agg(
                    cases=("case_id", "count"),
                    baseline_pump_m3=("pump_m3_baseline", "sum"),
                    pump_m3=("pump_m3", "sum"),
                    saved_m3=("saved_m3", "sum"),
                    d_time_gt3_s=("d_time_gt3_s", "sum"),
                    d_time_gt4_s=("d_time_gt4_s", "sum"),
                    d_time_gt5_s=("d_time_gt5_s", "sum"),
                    d_fallback_s=("d_fallback_s", "sum"),
                    max_d_p95_axis_deg=("d_p95_axis_deg", "max"),
                    mean_d_p95_axis_deg=("d_p95_axis_deg", "mean"),
                    worst_p95_axis_deg=("p95_axis_deg", "max"),
                )
                .reset_index()
            )
            group["pool"] = pool
            group["compare_arm"] = arm
            group["saving_pct"] = 100.0 * group["saved_m3"] / group["baseline_pump_m3"].clip(lower=1e-9)
            cols = ["pool", "compare_arm", "validation_role", "stratum", "cases", "baseline_pump_m3", "pump_m3", "saved_m3", "saving_pct", "d_time_gt3_s", "d_time_gt4_s", "d_time_gt5_s", "d_fallback_s", "max_d_p95_axis_deg", "mean_d_p95_axis_deg", "worst_p95_axis_deg"]
            group = group[cols]
            group.to_csv(pool_dir / f"{arm}_summary_by_stratum.csv", index=False)
            if arm == "refresh_on":
                # Backward-compatible path consumed by the generalization report.
                group.drop(columns=["compare_arm"]).to_csv(
                    pool_dir / "pool_summary_by_stratum.csv",
                    index=False,
                )
            summaries.append(group)
            print(f"\n== {pool}/{arm} ==")
            print(group.to_string(index=False))
    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(args.output_root / "broader6h_summary_by_stratum.csv", index=False)


def main() -> None:
    args = parse_args()
    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [p for p in pools if p not in POOLS]
    if unknown:
        raise SystemExit(f"unknown pools: {unknown}")
    for arm in arms:
        _arm(arm)
    for pool in pools:
        for arm in arms:
            _run(args, pool, arm)
    if args.mode in {"run", "summarize"}:
        summarize(args, pools, arms)


if __name__ == "__main__":
    main()
