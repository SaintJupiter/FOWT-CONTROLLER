#!/usr/bin/env python3
"""Run and summarize the PSC 18% bridge validation casebook."""

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
BASE = REPO / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
PLAN = BASE / "psc_18pct_saving_20260602" / "bridge_plan_v1"
OUT = PLAN / "bridge18_validation_run_v1"
CASEBOOK = PLAN / "bridge18_next_validation_cases.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "run", "summarize"), default="dry-run")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--cases-csv", type=Path, default=CASEBOOK)
    parser.add_argument("--duration-s", type=float, default=21600.0)
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1",
    )
    parser.add_argument("--replay-split", choices=("test", "validation", "train"), default="test")
    parser.add_argument("--model-dir", default="outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _arm_spec(arm: str) -> tuple[str, list[str], dict[str, str]]:
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
    raise ValueError(arm)


def _command(args: argparse.Namespace, arm: str) -> tuple[list[str], dict[str, str], Path]:
    label, profile_args, env = _arm_spec(arm)
    out_dir = args.output_root / arm
    cmd = [
        sys.executable,
        str(RUNNER),
        "--out-dir",
        str(out_dir),
        "--cases-csv",
        str(args.cases_csv),
        "--duration-s",
        str(float(args.duration_s)),
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
        *profile_args,
    ]
    return cmd, env, out_dir


def _print_command(cmd: list[str], env_patch: dict[str, str]) -> None:
    prefix = " ".join(f"{key}={value}" for key, value in sorted(env_patch.items()))
    print(("+ " + prefix + " " if prefix else "+ ") + " ".join(cmd))


def _run_arm(args: argparse.Namespace, arm: str) -> None:
    cmd, env_patch, out_dir = _command(args, arm)
    summary = out_dir / "casebook_summary.csv"
    if args.skip_existing and summary.exists():
        print(f"skip existing {arm}: {summary}")
        return
    _print_command(cmd, env_patch)
    if args.mode == "run":
        env = os.environ.copy()
        env.update(env_patch)
        subprocess.run(cmd, cwd=REPO, env=env, check=True)


def _extract(label: object, key: str) -> str:
    marker = f"{key}="
    text = str(label)
    if marker not in text:
        return ""
    return text.split(marker, 1)[1].split("|", 1)[0].strip()


def _find_timeseries(run_dir: Path, case_id: str, label: str) -> Path:
    matches = sorted((run_dir / "timeseries").glob(f"{case_id}_*_{label}_timeseries.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{run_dir.name}/{case_id}/{label}: {len(matches)} matches")
    return matches[0]


def _case_metrics(run_dir: Path, arm: str) -> pd.DataFrame:
    label, _, _ = _arm_spec(arm)
    summary = pd.read_csv(run_dir / "casebook_summary.csv")
    rows: list[dict[str, Any]] = []
    for rec in summary.to_dict("records"):
        case_id = str(rec["case_id"])
        ts = pd.read_csv(_find_timeseries(run_dir, case_id, label), low_memory=False)
        pitch = pd.to_numeric(ts.get("pitch_deg", 0.0), errors="coerce").fillna(0.0)
        roll = pd.to_numeric(ts.get("roll_deg", 0.0), errors="coerce").fillna(0.0)
        axis = np.maximum(np.abs(pitch), np.abs(roll))
        fallback = pd.to_numeric(
            ts.get("preview_primary_safety_fallback", pd.Series(0.0, index=ts.index)),
            errors="coerce",
        ).fillna(0.0)
        raw_label = str(rec.get("label", ""))
        rows.append(
            {
                "arm": arm,
                "case_id": case_id,
                "label": raw_label,
                "bridge18_role": _extract(raw_label, "bridge18_role"),
                "bridge18_source_pool": _extract(raw_label, "bridge18_source_pool"),
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
            value = rec[col]
            if isinstance(value, float):
                if col.endswith("_pct"):
                    vals.append(f"{value:.2f}%")
                elif col.endswith("_s"):
                    vals.append(f"{value:.0f}")
                elif col.endswith("_deg"):
                    vals.append(f"{value:.2f}")
                else:
                    vals.append(f"{value:.1f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat(
        [
            _case_metrics(args.output_root / "baseline", "baseline"),
            _case_metrics(args.output_root / "refresh_on", "refresh_on"),
        ],
        ignore_index=True,
    )
    metrics.to_csv(args.output_root / "bridge18_case_metrics.csv", index=False)

    base = metrics[metrics["arm"].eq("baseline")].set_index("case_id")
    on = metrics[metrics["arm"].eq("refresh_on")].copy().join(base, on="case_id", rsuffix="_baseline")
    for col in ["pump_m3", "time_gt3_s", "time_gt4_s", "time_gt5_s", "fallback_s", "p95_axis_deg", "max_axis_deg"]:
        on[f"d_{col}"] = on[col] - on[f"{col}_baseline"]
    on["saved_m3"] = on["pump_m3_baseline"] - on["pump_m3"]
    on["saving_pct"] = 100.0 * on["saved_m3"] / on["pump_m3_baseline"].clip(lower=1e-9)
    on["strict_accept"] = (
        (on["saved_m3"] > 0.0)
        & (on["d_fallback_s"] <= 0.0)
        & (on["d_time_gt5_s"] <= 0.0)
        & (on["d_time_gt4_s"] <= 0.0)
        & (on["p95_axis_deg"] < 3.75)
        & ((on["p95_axis_deg"] - on["p95_axis_deg_baseline"]) <= 1.25)
    )
    on.to_csv(args.output_root / "bridge18_refresh_on_case_deltas.csv", index=False)

    role_summary = (
        on.groupby(["bridge18_role"], dropna=False)
        .agg(
            cases=("case_id", "count"),
            strict_accept_cases=("strict_accept", "sum"),
            baseline_pump_m3=("pump_m3_baseline", "sum"),
            refresh_on_pump_m3=("pump_m3", "sum"),
            saved_m3=("saved_m3", "sum"),
            d_time_gt4_s=("d_time_gt4_s", "sum"),
            d_time_gt5_s=("d_time_gt5_s", "sum"),
            d_fallback_s=("d_fallback_s", "sum"),
            max_p95_axis_deg=("p95_axis_deg", "max"),
            max_d_p95_axis_deg=("d_p95_axis_deg", "max"),
        )
        .reset_index()
    )
    role_summary["saving_pct"] = 100.0 * role_summary["saved_m3"] / role_summary["baseline_pump_m3"].clip(lower=1e-9)
    strict = on[on["strict_accept"]].copy()
    strict_summary = pd.DataFrame(
        [
            {
                "cases": int(len(strict)),
                "baseline_pump_m3": float(strict["pump_m3_baseline"].sum()) if len(strict) else 0.0,
                "saved_m3": float(strict["saved_m3"].sum()) if len(strict) else 0.0,
                "saving_pct": 100.0
                * float(strict["saved_m3"].sum())
                / max(float(strict["pump_m3_baseline"].sum()), 1e-9)
                if len(strict)
                else 0.0,
                "d_time_gt4_s": float(strict["d_time_gt4_s"].sum()) if len(strict) else 0.0,
                "d_time_gt5_s": float(strict["d_time_gt5_s"].sum()) if len(strict) else 0.0,
                "d_fallback_s": float(strict["d_fallback_s"].sum()) if len(strict) else 0.0,
                "case_ids": ";".join(strict["case_id"].astype(str).tolist()) if len(strict) else "",
            }
        ]
    )
    role_summary.to_csv(args.output_root / "bridge18_role_summary.csv", index=False)
    strict_summary.to_csv(args.output_root / "bridge18_strict_accept_summary.csv", index=False)

    lines = [
        "# PSC 18% Bridge Validation v1",
        "",
        "## Role summary",
        "",
        _md_table(
            role_summary[
                [
                    "bridge18_role",
                    "cases",
                    "strict_accept_cases",
                    "saved_m3",
                    "saving_pct",
                    "d_time_gt4_s",
                    "d_time_gt5_s",
                    "d_fallback_s",
                    "max_p95_axis_deg",
                    "max_d_p95_axis_deg",
                ]
            ]
        ),
        "",
        "## Strict accepted subset",
        "",
        _md_table(strict_summary),
        "",
        "Strict rule: saved_m3 > 0, fallback delta <= 0, time>5 delta <= 0, time>4 delta <= 0, p95 axis < 3.75 deg, d_p95 <= 1.25 deg.",
    ]
    (args.output_root / "bridge18_validation_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_root / "bridge18_validation_readout.md")
    print(role_summary.to_string(index=False))
    print(strict_summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    for arm in ["baseline", "refresh_on"]:
        _run_arm(args, arm)
    if args.mode in {"run", "summarize"}:
        summarize(args)


if __name__ == "__main__":
    main()
