#!/usr/bin/env python3
"""Run and summarize Stage A 6h refresh ON/OFF validation for No.4.

This script is intentionally limited to Stage A from the 2026-06-01 reviewed
plan: determine whether active-posture refresh ON dominates OFF on 20+ case
6h pools.  It does not implement a deployable debt gate.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "analysis" / "run_prediction_primary_casebook.py"
BASE = REPO_ROOT / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
OUT_ROOT = BASE / "psc_4hao_refresh_validation_6h_v1"

POOLS: dict[str, Path] = {
    "mixed24": BASE / "casebooks" / "psc_mixed_24_cases.csv",
    "risky20": BASE / "psc_4hao_25pct_overnight_v1" / "expanded_top20_risky_gain_cases.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument(
        "--pools",
        default="mixed24,risky20",
        help="Comma-separated pool names from: mixed24,risky20",
    )
    parser.add_argument(
        "--mode",
        choices=("run", "summarize", "dry-run"),
        default="dry-run",
        help="dry-run prints commands; run executes missing arms then summarizes; summarize only reads existing outputs.",
    )
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
    if arm == "refresh_off":
        return (
            "refresh_off",
            [
                "--primary-control-profile",
                "psc_4hao_dynamic_refresh_v1",
                "--forecast-source",
                "learned",
            ],
            {"FOWT_4HAO_ACTIVE_POSTURE_REFRESH": "0"},
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
    raise ValueError(f"unknown arm: {arm}")


def _command(args: argparse.Namespace, pool: str, arm: str) -> tuple[list[str], dict[str, str], Path]:
    label, profile_args, env = _arm_spec(arm)
    out_dir = args.output_root / pool / arm
    cmd = [
        sys.executable,
        str(RUNNER),
        "--out-dir",
        str(out_dir),
        "--cases-csv",
        str(POOLS[pool]),
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


def _print_command(cmd: list[str], env: dict[str, str]) -> None:
    prefix = " ".join(f"{k}={v}" for k, v in sorted(env.items()))
    print(("+ " + prefix + " " if prefix else "+ ") + " ".join(cmd))


def _run_arm(args: argparse.Namespace, pool: str, arm: str) -> None:
    cmd, env_patch, out_dir = _command(args, pool, arm)
    summary = out_dir / "casebook_summary.csv"
    if args.skip_existing and summary.exists():
        print(f"skip existing {pool}/{arm}: {summary}")
        return
    _print_command(cmd, env_patch)
    if args.mode == "run":
        env = os.environ.copy()
        env.update(env_patch)
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _find_timeseries(run_dir: Path, case_id: str, label: str) -> Path:
    matches = sorted((run_dir / "timeseries").glob(f"{case_id}_*_{label}_timeseries.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one timeseries for {run_dir.name}/{case_id}/{label}, found {len(matches)}")
    return matches[0]


def _finite_sum(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").sum(min_count=1))


def _finite_mean(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").mean())


def _finite_max(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").max())


def _finite_min(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").min())


def _row_float(row: dict[str, Any], *names: str, default: float = np.nan) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    return float(default)


def _case_metrics_from_summary(run_dir: Path, arm: str, duration_s: float) -> pd.DataFrame:
    """Best-effort metrics when 1 Hz timeseries have been pruned.

    Existing stage-A outputs keep a richer ``stage_a_case_metrics.csv`` cache.
    This fallback is only for directories that have casebook summaries but no
    retained timeseries/cache, so it intentionally leaves unavailable exposure
    bands as NaN instead of inventing exact safety debt.
    """
    summary_path = run_dir / "casebook_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    rows: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        pitch_p95 = _row_float(row, "primary_pitch_p95")
        roll_p95 = _row_float(row, "primary_roll_p95")
        p95_axis = float(np.nanmax([pitch_p95, roll_p95]))
        fallback_ratio = _row_float(row, "primary_safety_fallback_ratio", default=0.0)
        rows.append(
            {
                "arm": arm,
                "case_id": str(row["case_id"]),
                "label": str(row.get("label", "")),
                "pump_m3": _row_float(row, "primary_pump_work_m3"),
                "time_gt3_s": _row_float(row, "primary_time_over_3deg_s"),
                "time_gt4_s": _row_float(row, "primary_time_over_4deg_s"),
                "time_gt5_s": _row_float(row, "primary_time_over_5deg_s"),
                "fallback_s": float(fallback_ratio * duration_s),
                "p95_axis_deg": p95_axis,
                "max_axis_deg": _row_float(row, "primary_max_axis_deg"),
                "p95_margin_to_5deg": float(5.0 - p95_axis),
            }
        )
    return pd.DataFrame(rows)


def _validate_cached_metrics(metrics: pd.DataFrame, pool_dir: Path) -> bool:
    required = {"arm", "case_id", "pump_m3"}
    if not required.issubset(metrics.columns):
        return False
    for arm in ("baseline", "refresh_off", "refresh_on"):
        summary_path = pool_dir / arm / "casebook_summary.csv"
        if not summary_path.exists():
            return False
        summary = pd.read_csv(summary_path, usecols=["case_id", "primary_pump_work_m3"])
        cached = metrics[metrics["arm"].eq(arm)][["case_id", "pump_m3"]].copy()
        if len(summary) != len(cached):
            return False
        joined = summary.merge(cached, on="case_id", how="outer", indicator=True)
        if not joined["_merge"].eq("both").all():
            return False
        if not np.allclose(
            joined["primary_pump_work_m3"].to_numpy(dtype=float),
            joined["pump_m3"].to_numpy(dtype=float),
            rtol=1e-8,
            atol=1e-6,
            equal_nan=True,
        ):
            return False
    return True


def _pool_metrics(args: argparse.Namespace, pool_dir: Path) -> pd.DataFrame:
    try:
        return pd.concat(
            [
                _case_metrics(pool_dir / arm, arm)
                for arm in ("baseline", "refresh_off", "refresh_on")
            ],
            ignore_index=True,
        )
    except FileNotFoundError as exc:
        cached_path = pool_dir / "stage_a_case_metrics.csv"
        if cached_path.exists():
            cached = pd.read_csv(cached_path)
            if _validate_cached_metrics(cached, pool_dir):
                print(f"using cached metrics: {cached_path} ({exc})")
                return cached
            print(f"cached metrics did not match current summaries: {cached_path}")
        print(f"timeseries unavailable; using casebook_summary fallback for {pool_dir.name}: {exc}")
        return pd.concat(
            [
                _case_metrics_from_summary(pool_dir / arm, arm, float(args.duration_s))
                for arm in ("baseline", "refresh_off", "refresh_on")
            ],
            ignore_index=True,
        )


def _case_metrics(run_dir: Path, arm: str) -> pd.DataFrame:
    label, _, _ = _arm_spec(arm)
    summary_path = run_dir / "casebook_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    rows: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        case_id = str(row["case_id"])
        ts = pd.read_csv(_find_timeseries(run_dir, case_id, label), low_memory=False)
        axis = ts[["pitch_deg", "roll_deg"]].abs().max(axis=1)
        fallback = pd.to_numeric(
            ts.get("preview_primary_safety_fallback", pd.Series(0.0, index=ts.index)),
            errors="coerce",
        ).fillna(0.0)
        rows.append(
            {
                "arm": arm,
                "case_id": case_id,
                "label": str(row.get("label", "")),
                "pump_m3": float(row["primary_pump_work_m3"]),
                "time_gt3_s": float((axis > 3.0).sum()),
                "time_gt4_s": float((axis > 4.0).sum()),
                "time_gt5_s": float((axis > 5.0).sum()),
                "fallback_s": float((fallback > 0.0).sum()),
                "p95_axis_deg": float(axis.quantile(0.95)),
                "max_axis_deg": float(axis.max()),
                "p95_margin_to_5deg": float(5.0 - axis.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def _summarize_pool(args: argparse.Namespace, pool: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool_dir = args.output_root / pool
    metrics = _pool_metrics(args, pool_dir)
    metrics.to_csv(pool_dir / "stage_a_case_metrics.csv", index=False)

    base = metrics[metrics["arm"].eq("baseline")].set_index("case_id")
    rows = []
    for arm in ("refresh_off", "refresh_on"):
        part = metrics[metrics["arm"].eq(arm)].copy()
        joined = part.join(base, on="case_id", rsuffix="_baseline")
        for col in ("pump_m3", "time_gt3_s", "time_gt4_s", "time_gt5_s", "fallback_s", "p95_axis_deg"):
            joined[f"d_{col}"] = joined[col] - joined[f"{col}_baseline"]
        joined["saved_m3"] = joined["pump_m3_baseline"] - joined["pump_m3"]
        joined["saving_pct"] = 100.0 * joined["saved_m3"] / joined["pump_m3_baseline"].clip(lower=1e-9)
        joined.to_csv(pool_dir / f"stage_a_{arm}_case_deltas.csv", index=False)
        rows.append(
            {
                "pool": pool,
                "arm": arm,
                "cases": int(len(joined)),
                "baseline_pump_m3": float(joined["pump_m3_baseline"].sum()),
                "pump_m3": float(joined["pump_m3"].sum()),
                "saved_m3": float(joined["saved_m3"].sum()),
                "saving_pct": 100.0
                * float(joined["saved_m3"].sum())
                / max(float(joined["pump_m3_baseline"].sum()), 1e-9),
                "d_time_gt3_s": _finite_sum(joined["d_time_gt3_s"]),
                "d_time_gt4_s": _finite_sum(joined["d_time_gt4_s"]),
                "d_time_gt5_s": _finite_sum(joined["d_time_gt5_s"]),
                "d_fallback_s": _finite_sum(joined["d_fallback_s"]),
                "max_d_p95_axis_deg": _finite_max(joined["d_p95_axis_deg"]),
                "mean_d_p95_axis_deg": _finite_mean(joined["d_p95_axis_deg"]),
                "worst_p95_axis_deg": _finite_max(joined["p95_axis_deg"]),
                "min_p95_margin_to_5deg": _finite_min(joined["p95_margin_to_5deg"]),
            }
        )
    summary = pd.DataFrame(rows)
    off = summary[summary["arm"].eq("refresh_off")].iloc[0]
    on = summary[summary["arm"].eq("refresh_on")].iloc[0]
    on_saves_at_least = float(on["saved_m3"]) >= float(off["saved_m3"]) - 1e-6
    debt_values = [
        off["d_fallback_s"],
        on["d_fallback_s"],
        off["d_time_gt5_s"],
        on["d_time_gt5_s"],
        off["max_d_p95_axis_deg"],
        on["max_d_p95_axis_deg"],
    ]
    has_debt_detail = bool(np.isfinite(np.asarray(debt_values, dtype=float)).all())
    on_lower_debt = has_debt_detail and (
        float(on["d_fallback_s"]) <= float(off["d_fallback_s"])
        and float(on["d_time_gt5_s"]) <= float(off["d_time_gt5_s"])
        and float(on["max_d_p95_axis_deg"]) <= float(off["max_d_p95_axis_deg"])
    )
    if on_saves_at_least and on_lower_debt:
        verdict = "refresh_on_dominates_off"
    elif not has_debt_detail:
        verdict = "insufficient_safety_detail"
    else:
        verdict = "off_may_have_clean_subset"
    summary["pool_verdict"] = verdict
    summary.to_csv(pool_dir / "stage_a_summary.csv", index=False)
    return metrics, summary


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
                vals.append(f"{float(val):.4g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def summarize(args: argparse.Namespace, pools: list[str]) -> None:
    summaries = []
    for pool in pools:
        _, summary = _summarize_pool(args, pool)
        summaries.append(summary)
    all_summary = pd.concat(summaries, ignore_index=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_summary.to_csv(args.output_root / "stage_a_summary.csv", index=False)
    lines = [
        "# PSC 4hao refresh validation 6h v1 - Stage A",
        "",
        "Question: does active-posture refresh ON dominate refresh OFF on 20+ case 6h pools?",
        "",
        "All saving percentages are relative to the same-pool current-forecast-adaptive baseline denominator. Pools are reported separately and not merged.",
        "",
        "## Summary",
        "",
        _md_table(all_summary),
        "",
        "## Interpretation",
        "",
        "- `refresh_off`: No.4 dynamic refresh profile with active-posture refresh disabled.",
        "- `refresh_on`: same profile with active-posture refresh enabled.",
        "- `refresh_on_dominates_off`: ON saves at least as much water and has no worse fallback, time>5, or max d_p95 debt at aggregate pool level.",
        "- If ON dominates OFF, abandon OFF-gate work and evaluate ON-only No.4 against 3hao.",
    ]
    (args.output_root / "stage_a_readout.md").write_text("\n".join(lines), encoding="utf-8")
    print(all_summary.to_string(index=False))
    print(args.output_root / "stage_a_readout.md")


def main() -> None:
    args = parse_args()
    pools = [p.strip() for p in str(args.pools).split(",") if p.strip()]
    unknown = [p for p in pools if p not in POOLS]
    if unknown:
        raise SystemExit(f"unknown pools: {unknown}")

    for pool in pools:
        for arm in ("baseline", "refresh_off", "refresh_on"):
            _run_arm(args, pool, arm)

    if args.mode in {"run", "summarize"}:
        summarize(args, pools)


if __name__ == "__main__":
    main()
