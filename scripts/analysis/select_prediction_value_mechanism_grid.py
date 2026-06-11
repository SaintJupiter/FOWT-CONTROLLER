#!/usr/bin/env python3
"""Select broad prediction-value candidate windows without running control.

The selector reads the forecast-level audit CSV and builds a stratified case
grid from physical/oracle wind features plus learned-vs-persistence forecast
feature differences.  It deliberately does not use closed-loop pump, attitude,
or controller outcome metrics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
CALIBRATION_TIMESTAMPS = {
    "2024-11-27 19:40:00",
    "2023-10-31 06:20:00",
    "2024-10-10 03:50:00",
    "2024-09-27 13:00:00",
    "2022-03-20 19:00:00",
    "2023-10-03 06:30:00",
    "2023-03-14 04:40:00",
    "2021-12-20 14:30:00",
    "2022-02-04 11:00:00",
    "2024-09-05 18:10:00",
}


GROUPS: list[dict[str, str]] = [
    {
        "group": "future_relief",
        "label": "oracle future relief; tests whether forecast suppresses over-action",
        "score": "future relief severity plus learned RMSE advantage",
    },
    {
        "group": "relief_hold_risk",
        "label": "relief-like window where learned predicts less near-term pressure than persistence",
        "score": "learned extra relief and low near-term norm",
    },
    {
        "group": "stronger_active_like_09",
        "label": "case09-like stronger active candidate where learned predicts higher pressure than persistence",
        "score": "learned extra near-term norm",
    },
    {
        "group": "sustained_signflip_like_07",
        "label": "case07-like sustained signflip by oracle wind geometry",
        "score": "strong opposite-direction high-pressure future",
    },
    {
        "group": "future_onset",
        "label": "future risk onset from quiet/current low pressure",
        "score": "future growth and learned forecast advantage",
    },
    {
        "group": "decay",
        "label": "high current pressure with future pressure decay",
        "score": "oracle pressure relief",
    },
    {
        "group": "sustained_high",
        "label": "persistent high pressure with stable direction",
        "score": "high raw norms and direction consistency",
    },
    {
        "group": "lowrisk_quiet",
        "label": "low-risk quiet reference windows",
        "score": "lowest maximum oracle norm",
    },
    {
        "group": "oracle_event_learned_missed",
        "label": "oracle event exists but learned event-threshold detector misses it",
        "score": "event severity under learned miss",
    },
    {
        "group": "learned_persistence_similar",
        "label": "learned and persistence forecast features are nearly identical",
        "score": "small learned-persistence feature distance",
    },
]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _parse_ts(value: object) -> datetime:
    return datetime.strptime(str(value), TIMESTAMP_FMT)


def _far_from_calibration(ts: datetime, guard_min: float) -> bool:
    guard = timedelta(minutes=float(guard_min))
    for raw in CALIBRATION_TIMESTAMPS:
        if abs(ts - _parse_ts(raw)) <= guard:
            return False
    return True


def _with_eligibility(
    df: pd.DataFrame,
    *,
    duration_s: float,
    calibration_guard_min: float,
    update_interval_s: float,
) -> pd.DataFrame:
    buckets = int(np.ceil(float(duration_s) / float(update_interval_s)))
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], format=TIMESTAMP_FMT)
    mask = pd.Series(True, index=out.index)

    guard = pd.Timedelta(minutes=float(calibration_guard_min))
    for raw in CALIBRATION_TIMESTAMPS:
        cal_ts = pd.Timestamp(_parse_ts(raw))
        mask &= (ts - cal_ts).abs() > guard

    available = set(ts)
    step = pd.Timedelta(seconds=float(update_interval_s))
    for idx in range(buckets):
        mask &= (ts + idx * step).isin(available)
    return out.loc[mask].reset_index(drop=True)


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for prefix in ("oracle", "learned", "persistence"):
        out[f"{prefix}_norm_sum"] = out[[f"{prefix}_norm0", f"{prefix}_norm1", f"{prefix}_norm2"]].sum(axis=1)

    out["learned_extra_norm0"] = out["learned_norm0"] - out["persistence_norm0"]
    out["learned_extra_norm1"] = out["learned_norm1"] - out["persistence_norm1"]
    out["learned_extra_norm2"] = out["learned_norm2"] - out["persistence_norm2"]
    out["learned_extra_relief02"] = out["learned_relief02"] - out["persistence_relief02"]
    out["learned_extra_growth02"] = out["learned_growth02"] - out["persistence_growth02"]
    out["learned_persistence_norm_l1"] = (
        out["learned_extra_norm0"].abs()
        + out["learned_extra_norm1"].abs()
        + out["learned_extra_norm2"].abs()
    )
    out["oracle_event_severity"] = out[["oracle_norm0", "oracle_norm1", "oracle_norm2"]].max(axis=1)
    out["oracle_relief_score"] = out["oracle_relief02"].clip(lower=0.0)
    out["oracle_growth_score"] = out["oracle_growth02"].clip(lower=0.0)
    out["oracle_signflip_score"] = out["oracle_norm0"] + out["oracle_norm2"] + (1.0 - out["oracle_cos02"].clip(-1.0, 1.0))
    return out


def _group_mask_and_score(df: pd.DataFrame, group: str) -> tuple[pd.Series, pd.Series]:
    true = pd.Series(True, index=df.index)
    if group == "future_relief":
        mask = df["oracle_scenario"].eq("future_relief")
        score = df["oracle_relief_score"] + df["learned_uv_rmse_advantage"].clip(lower=-1.0)
    elif group == "relief_hold_risk":
        mask = df["oracle_scenario"].eq("future_relief") & (
            (df["learned_extra_relief02"] >= 0.10)
            | ((df["persistence_norm0"] - df["learned_norm0"]) >= 0.10)
        )
        score = df["learned_extra_relief02"] + (df["persistence_norm0"] - df["learned_norm0"]) + df["oracle_norm0"]
    elif group == "stronger_active_like_09":
        mask = (df["oracle_event_severity"] >= 0.85) & (
            (df["learned_extra_norm0"] >= 0.10)
            | (df["learned_extra_norm1"] >= 0.10)
        )
        score = df["learned_extra_norm0"].clip(lower=0.0) + df["learned_extra_norm1"].clip(lower=0.0) + df["oracle_event_severity"]
    elif group == "sustained_signflip_like_07":
        mask = (df["oracle_norm0"] >= 1.20) & (df["oracle_norm2"] >= 0.70) & (df["oracle_cos02"] <= -0.30)
        score = df["oracle_signflip_score"]
    elif group == "future_onset":
        mask = df["oracle_scenario"].eq("future_risk_onset")
        score = df["oracle_growth_score"] + df["learned_uv_rmse_advantage"].clip(lower=-1.0)
    elif group == "decay":
        mask = (df["oracle_norm0"] >= 1.20) & (df["oracle_norm2"] <= 0.85)
        score = df["oracle_relief_score"]
    elif group == "sustained_high":
        mask = df["oracle_scenario"].eq("sustained_high") & (df["oracle_cos02"] >= 0.50)
        score = df["oracle_norm_sum"] + df["oracle_cos02"].fillna(0.0)
    elif group == "lowrisk_quiet":
        mask = df["oracle_scenario"].eq("lowrisk")
        score = -df["oracle_max_norm"]
    elif group == "oracle_event_learned_missed":
        event_mask = df["oracle_scenario"].isin(["future_relief", "future_risk_onset", "signflip_high"])
        learned_hit = (
            df["learned_relief_hit"].astype(bool)
            | df["learned_onset_hit"].astype(bool)
            | df["learned_signflip_hit"].astype(bool)
        )
        mask = event_mask & ~learned_hit
        score = df["oracle_event_severity"] + df["oracle_relief_score"] + df["oracle_growth_score"]
    elif group == "learned_persistence_similar":
        mask = true & (df["learned_persistence_norm_l1"] <= 0.05) & (df["oracle_event_severity"] >= 0.65)
        score = -df["learned_persistence_norm_l1"] + 0.01 * df["oracle_event_severity"]
    else:
        raise KeyError(f"unsupported group: {group}")
    return mask.fillna(False), score.fillna(-1e9)


def _thin_by_time(rows: pd.DataFrame, min_separation_min: float, limit: int) -> pd.DataFrame:
    selected = []
    used: list[datetime] = []
    sep = timedelta(minutes=float(min_separation_min))
    for _, row in rows.iterrows():
        ts = _parse_ts(row["timestamp"])
        if all(abs(ts - prev) >= sep for prev in used):
            selected.append(row)
            used.append(ts)
        if len(selected) >= int(limit):
            break
    if not selected:
        return rows.head(0).copy()
    return pd.DataFrame(selected)


def _case_rows(rows: pd.DataFrame, group: str, label: str) -> pd.DataFrame:
    out_rows = []
    keep_cols = [
        "timestamp",
        "oracle_scenario",
        "selection_score",
        "oracle_norm0",
        "oracle_norm1",
        "oracle_norm2",
        "oracle_relief02",
        "oracle_growth02",
        "oracle_cos02",
        "learned_norm0",
        "learned_norm1",
        "learned_norm2",
        "persistence_norm0",
        "persistence_norm1",
        "persistence_norm2",
        "learned_uv_rmse_advantage",
        "learned_persistence_norm_l1",
    ]
    for idx, row in enumerate(rows.to_dict("records"), start=1):
        case = {
            "case_id": f"{group}_{idx:02d}",
            "timestamp": row["timestamp"],
            "label": f"{group}: {label}",
            "selection_group": group,
            "selection_rank": idx,
        }
        for col in keep_cols:
            if col in row:
                case[col] = row[col]
        out_rows.append(case)
    return pd.DataFrame(out_rows)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
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
                vals.append(f"{val:.4g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-csv",
        default="outputs/wind_prediction/forecast_level_prediction_value_full_v1/forecast_level_window_metrics.csv",
        help="Forecast-level audit CSV. Reuse this file to avoid rerunning learned forecasts.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1",
        help="Reserved for compatibility; this selector now validates follow-up availability from the metrics CSV timestamps.",
    )
    parser.add_argument("--out-dir", default="outputs/wind_prediction/prediction_value_mechanism_grid_v1")
    parser.add_argument("--duration-s", type=float, default=7200.0)
    parser.add_argument("--update-interval-s", type=float, default=600.0)
    parser.add_argument("--limit-per-group", type=int, default=5)
    parser.add_argument("--min-separation-min", type=float, default=180.0)
    parser.add_argument("--calibration-guard-min", type=float, default=180.0)
    parser.add_argument(
        "--write-all-features",
        action="store_true",
        help="Also write the full eligible feature table. Disabled by default to keep selection fast.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_csv = _resolve(args.metrics_csv)
    if not metrics_csv.exists():
        raise FileNotFoundError(
            f"forecast-level metrics not found: {metrics_csv}. "
            "Run audit_forecast_level_prediction_value.py first."
        )
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(metrics_csv)
    eligible = _with_eligibility(
        raw,
        duration_s=float(args.duration_s),
        calibration_guard_min=float(args.calibration_guard_min),
        update_interval_s=float(args.update_interval_s),
    )
    candidates = _add_derived_columns(eligible)
    if bool(args.write_all_features):
        candidates.to_csv(out_dir / "all_eligible_forecast_features.csv", index=False)

    selected_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = [
        {"group": "all_input_windows", "pool_count": int(len(raw)), "selected_count": ""},
        {"group": "eligible_after_guards", "pool_count": int(len(candidates)), "selected_count": ""},
    ]
    for group_def in GROUPS:
        group = group_def["group"]
        mask, score = _group_mask_and_score(candidates, group)
        pool = candidates[mask].copy()
        pool["selection_score"] = score.loc[pool.index]
        pool = pool.sort_values("selection_score", ascending=False)
        selected = _thin_by_time(
            pool,
            min_separation_min=float(args.min_separation_min),
            limit=int(args.limit_per_group),
        )
        cases = _case_rows(selected, group, group_def["label"])
        cases.to_csv(out_dir / f"{group}_cases.csv", index=False)
        selected_tables.append(cases)
        summary_rows.append(
            {
                "group": group,
                "pool_count": int(len(pool)),
                "selected_count": int(len(cases)),
                "rule": group_def["label"],
                "score": group_def["score"],
            }
        )

    combined = pd.concat(selected_tables, ignore_index=True) if selected_tables else pd.DataFrame()
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
        combined["case_id"] = [
            f"mechanism_{idx:02d}_{str(row.selection_group)}"
            for idx, row in enumerate(combined.itertuples(index=False), start=1)
        ]
    combined.to_csv(out_dir / "mechanism_grid_cases.csv", index=False)

    candidate_pool = pd.concat(selected_tables, ignore_index=True) if selected_tables else pd.DataFrame()
    candidate_pool.to_csv(out_dir / "mechanism_candidate_pool.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "mechanism_grid_summary.csv", index=False)
    preview_cols = [
        "case_id",
        "timestamp",
        "selection_group",
        "oracle_scenario",
        "selection_score",
        "oracle_norm0",
        "oracle_norm2",
        "oracle_cos02",
        "learned_uv_rmse_advantage",
    ]
    preview = combined[[c for c in preview_cols if c in combined.columns]].head(30) if not combined.empty else combined
    report = [
        "# Prediction Value Mechanism Grid v1",
        "",
        "This is a cheap selector, not a closed-loop result. It uses forecast-level/oracle UV features only.",
        "It does not use pump totals, attitude metrics, planner actions, or learned-vs-persistence closed-loop outcomes.",
        "",
        "## Why This Exists",
        "",
        "- Avoid running long 1Hz simulations before we know which windows are informative.",
        "- Cover 07/09-like windows and broad counterexamples in the same grid.",
        "- Separate forecast-signal availability from planner/execution behavior.",
        "",
        "## Settings",
        "",
        f"- metrics CSV: `{metrics_csv.relative_to(REPO_ROOT) if metrics_csv.is_relative_to(REPO_ROOT) else metrics_csv}`",
        f"- duration guard: `{float(args.duration_s) / 60.0:.0f} min`",
        f"- calibration guard: `{float(args.calibration_guard_min):.0f} min` around the 10 manual casebook timestamps",
        f"- per-group limit: `{int(args.limit_per_group)}`",
        f"- per-group thinning: `{float(args.min_separation_min):.0f} min`",
        f"- wrote full eligible table: `{int(bool(args.write_all_features))}`",
        "",
        "## Counts",
        "",
        _markdown_table(summary.fillna("")),
        "",
        "## Combined Candidate Preview",
        "",
        _markdown_table(preview),
        "",
        "## Outputs",
        "",
        f"- combined case CSV: `{out_dir / 'mechanism_grid_cases.csv'}`",
        f"- candidate pool before timestamp de-duplication: `{out_dir / 'mechanism_candidate_pool.csv'}`",
        f"- per-group case CSVs: `{out_dir}/*_cases.csv`",
    ]
    (out_dir / "mechanism_grid_selection_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Combined cases: {len(combined)}")
    print(f"Report: {out_dir / 'mechanism_grid_selection_report.md'}")


if __name__ == "__main__":
    main()
