#!/usr/bin/env python3
"""Mine pressure-trust validation cases from existing casebook planner logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "outputs" / "wind_prediction"
DEFAULT_OUT_DIR = OUT_ROOT / "forecast_pressure_trust_gate_20260604"
DEFAULT_FP_CANDIDATES = (
    OUT_ROOT / "event_gate_control_sensitive_fp_candidates_20260604.csv"
)
DEFAULT_SOURCE_DIRS = [
    OUT_ROOT / "event_gate_control_sensitive_direct_raw_20260604" / "planner_logs",
    OUT_ROOT / "event_gate_control_sensitive_floor_raw_20260604" / "planner_logs",
    OUT_ROOT / "event_gate_ab_raw_20260604" / "planner_logs",
    OUT_ROOT / "event_gate_ab_conservative_raw_20260604" / "planner_logs",
    OUT_ROOT / "pp_learned_emph5_vchange_guard10" / "planner_logs",
    OUT_ROOT / "pp_learned_synth_relief_guard10" / "planner_logs",
    OUT_ROOT / "barrier_const0_sensitivity_v1_learned_12case_2h" / "planner_logs",
    OUT_ROOT / "dc_deadband_forecast_adaptive_5case_6h_v1" / "planner_logs",
]

ACTIVE_ACTIONS = {
    "active_small",
    "active_medium",
    "pump_saving",
    "active",
}
HIGHWIND_LABEL_TOKENS = (
    "highwind",
    "high_wind",
    "high-pressure",
    "high_pressure",
    "onset_strong",
    "strong_onset",
    "storm",
)
STABLE_LABEL_TOKENS = (
    "stable",
    "fp",
    "false_positive",
    "far_floor",
    "mid_lead",
    "quiet",
    "lowrisk",
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(np.full(len(df), float(default)), index=df.index)


def _str_series(df: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    if name in df.columns:
        return df[name].fillna(default).astype(str)
    return pd.Series([default for _ in range(len(df))], index=df.index, dtype=str)


def _mean_bool(values: pd.Series) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(values.to_numpy(dtype=bool)))


def _safe_mean(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else 0.0


def _safe_max(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else 0.0


def _parse_case_from_path(path: Path) -> dict[str, str]:
    stem = path.name
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{6})", stem)
    if not match:
        case_id = stem.replace("_planner_log.csv", "")
        return {"case_id": case_id, "timestamp": "", "label": case_id}
    date_s, time_s = match.group(1), match.group(2)
    prefix = stem[: match.start()].strip("_")
    label = prefix
    raw_case_id = re.sub(r"^\d+_", "", prefix)
    if not raw_case_id:
        raw_case_id = prefix or "case"
    timestamp = (
        f"{date_s} {time_s[0:2]}:{time_s[2:4]}:{time_s[4:6]}"
    )
    return {"case_id": raw_case_id, "timestamp": timestamp, "label": label}


def _pressure_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    b0 = _series(df, "raw_pressure_block0_norm")
    b1 = _series(df, "raw_pressure_block1_norm")
    b2 = _series(df, "raw_pressure_block2_norm")
    if not (b0.any() or b1.any() or b2.any()):
        b0 = _series(df, "pressure_block0_norm")
        b1 = _series(df, "pressure_block1_norm")
        b2 = _series(df, "pressure_block2_norm")
    return b0, b1, b2


def _summarize_log(path: Path) -> dict[str, Any] | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    meta = _parse_case_from_path(path)
    b0, b1, b2 = _pressure_columns(df)
    future = pd.concat([b1, b2], axis=1).max(axis=1)
    event_cols = [
        _series(df, "event_risk_raw_prob_0_20m"),
        _series(df, "event_risk_raw_prob_20_40m"),
        _series(df, "event_risk_raw_prob_40_60m"),
    ]
    event_max = pd.concat(event_cols, axis=1).max(axis=1)
    highwind = _series(df, "trusted_event_highwind_prob")
    attention_hits = _series(df, "trusted_event_attention_hits")
    trusted_event_enabled = _series(df, "trusted_event_gate_enabled")
    trusted_event_pass = _series(df, "trusted_event_gate_trusted", default=1.0)
    active = _str_series(df, "first_action").isin(ACTIVE_ACTIONS)
    pump_work = _series(df, "pump_work_cost")
    label_text = " ".join(
        [
            str(meta["case_id"]),
            str(meta["label"]),
            str(path.parent.parent.name),
        ]
    ).lower()
    stable_label = any(token in label_text for token in STABLE_LABEL_TOKENS)
    highwind_label = any(token in label_text for token in HIGHWIND_LABEL_TOKENS)

    current_support = b0 >= 0.65
    event_support = (
        (event_max >= 0.75)
        & (
            (trusted_event_enabled < 0.5)
            | (trusted_event_pass > 0.5)
            | (highwind >= 0.90)
            | (attention_hits >= 2.0)
        )
    )
    weak_event = event_support < 0.5
    pressure_future_high = future >= 1.0
    pressure_rise = future - b0

    stable_score = 0.0
    stable_score += 3.0 if stable_label else 0.0
    stable_score += 2.0 * _mean_bool(active)
    stable_score += min(_safe_mean(pump_work) / 50.0, 3.0)
    stable_score += _mean_bool(pressure_future_high)
    stable_score += _mean_bool(weak_event)
    stable_score += _mean_bool(current_support < 0.5)
    stable_score += 0.5 if "control_sensitive" in str(path) else 0.0

    falsifier_score = 0.0
    falsifier_score += 3.0 if highwind_label else 0.0
    falsifier_score += 2.0 * _mean_bool(highwind >= 0.90)
    falsifier_score += 1.5 * _mean_bool(attention_hits >= 2.0)
    falsifier_score += _mean_bool(pressure_future_high)
    falsifier_score += 0.5 * _mean_bool(active)

    row: dict[str, Any] = {
        **meta,
        "source_log": str(path.relative_to(REPO_ROOT)),
        "source_group": path.parent.parent.name,
        "stable_label_hint": int(stable_label),
        "highwind_label_hint": int(highwind_label),
        "row_count": int(len(df)),
        "active_first_action_ratio": _mean_bool(active),
        "pump_work_cost_mean": _safe_mean(pump_work),
        "raw_pressure_block0_norm_mean": _safe_mean(b0),
        "raw_pressure_future_max_norm_mean": _safe_mean(future),
        "raw_pressure_future_max_norm_max": _safe_max(future),
        "raw_pressure_rise_norm_mean": _safe_mean(pressure_rise),
        "raw_pressure_block02_dot_mean": _safe_mean(
            _series(df, "raw_pressure_block02_dot")
            if "raw_pressure_block02_dot" in df.columns
            else _series(df, "pressure_block02_dot")
        ),
        "event_risk_raw_prob_max_mean": _safe_mean(event_max),
        "event_risk_raw_prob_max_max": _safe_max(event_max),
        "trusted_event_gate_enabled_ratio": _safe_mean(trusted_event_enabled),
        "trusted_event_gate_pass_ratio": _safe_mean(trusted_event_pass),
        "trusted_event_highwind_prob_mean": _safe_mean(highwind),
        "trusted_event_attention_hits_mean": _safe_mean(attention_hits),
        "current_pressure_support_ratio": _mean_bool(current_support),
        "event_support_ratio": _mean_bool(event_support),
        "future_pressure_high_ratio": _mean_bool(pressure_future_high),
        "stable_candidate_score": float(stable_score),
        "highwind_falsifier_score": float(falsifier_score),
    }
    return row


def _load_seed_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"case_id", "timestamp"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    out = df.copy()
    out["label"] = out.get("label", out.get("selection_group", out["case_id"]))
    out["seed_candidate"] = 1
    return out


def _merge_seed_metrics(stable: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    if seeds.empty:
        return stable
    if stable.empty:
        return seeds
    merge_cols = ["case_id", "timestamp"]
    stable = stable.copy()
    seeds = seeds.copy()
    stable["_join_case_id"] = stable["case_id"].astype(str).str.replace(
        r"^\d+_", "",
        regex=True,
    )
    stable = stable.sort_values(
        ["stable_candidate_score", "pump_work_cost_mean"],
        ascending=[False, False],
    ).drop_duplicates(["_join_case_id", "timestamp"], keep="first")
    seeds["_join_case_id"] = seeds["case_id"].astype(str).str.replace(
        r"^\d+_",
        "",
        regex=True,
    )
    merged = seeds.merge(
        stable.drop(columns=["case_id"], errors="ignore"),
        left_on=["_join_case_id", "timestamp"],
        right_on=["_join_case_id", "timestamp"],
        how="left",
        suffixes=("", "_log"),
    )
    merged["case_id"] = seeds["case_id"].to_numpy()
    merged["label"] = seeds["label"].to_numpy()
    for col in merge_cols:
        if col not in merged.columns:
            merged[col] = seeds[col].to_numpy()
    return pd.concat(
        [
            merged.drop(columns=["_join_case_id"], errors="ignore"),
            stable.assign(seed_candidate=0).drop(
                columns=["_join_case_id"],
                errors="ignore",
            ),
        ],
        ignore_index=True,
        sort=False,
    )


def _write_readout(
    path: Path,
    source_dirs: list[Path],
    stable: pd.DataFrame,
    falsifiers: pd.DataFrame,
) -> None:
    lines = [
        "# Forecast Pressure Trust Candidate Readout",
        "",
        "## Sources",
        "",
    ]
    for source in source_dirs:
        lines.append(f"- `{source.relative_to(REPO_ROOT)}`")
    lines.extend(
        [
            "",
            "## Stable Pressure-Untrusted Candidates",
            "",
            "| case_id | timestamp | active | pump mean | future pressure | event support | score |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stable.head(12).to_dict("records"):
        lines.append(
            "| {case_id} | {timestamp} | {active:.2f} | {pump:.2f} | {future:.2f} | {event:.2f} | {score:.2f} |".format(
                case_id=row.get("case_id", ""),
                timestamp=row.get("timestamp", ""),
                active=float(row.get("active_first_action_ratio", 0.0) or 0.0),
                pump=float(row.get("pump_work_cost_mean", 0.0) or 0.0),
                future=float(row.get("raw_pressure_future_max_norm_mean", 0.0) or 0.0),
                event=float(row.get("event_support_ratio", 0.0) or 0.0),
                score=float(row.get("stable_candidate_score", 0.0) or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Highwind / Attention Falsifiers",
            "",
            "| case_id | timestamp | highwind prob | attention hits | future pressure | score |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in falsifiers.head(10).to_dict("records"):
        lines.append(
            "| {case_id} | {timestamp} | {highwind:.2f} | {attention:.2f} | {future:.2f} | {score:.2f} |".format(
                case_id=row.get("case_id", ""),
                timestamp=row.get("timestamp", ""),
                highwind=float(row.get("trusted_event_highwind_prob_mean", 0.0) or 0.0),
                attention=float(row.get("trusted_event_attention_hits_mean", 0.0) or 0.0),
                future=float(row.get("raw_pressure_future_max_norm_mean", 0.0) or 0.0),
                score=float(row.get("highwind_falsifier_score", 0.0) or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Stable candidates prefer active first actions, high pump work, high learned future pressure, and weak event/current support.",
            "- Falsifiers prefer highwind probability or multihead attention support and should stay trusted under default thresholds.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--fp-candidates", default=str(DEFAULT_FP_CANDIDATES))
    parser.add_argument(
        "--source-dir",
        action="append",
        default=[],
        help="Planner log directory to scan. May be repeated.",
    )
    parser.add_argument("--stable-top-n", type=int, default=12)
    parser.add_argument("--falsifier-top-n", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dirs = (
        [_resolve(p) for p in args.source_dir]
        if args.source_dir
        else [p for p in DEFAULT_SOURCE_DIRS if p.exists()]
    )
    rows = []
    for source in source_dirs:
        for path in sorted(source.glob("*_planner_log.csv")):
            row = _summarize_log(path)
            if row is not None:
                rows.append(row)
    mined = pd.DataFrame(rows)
    if mined.empty:
        raise SystemExit("No planner logs found for pressure-trust mining.")

    seeds = _load_seed_candidates(_resolve(args.fp_candidates))
    stable_pool = mined[
        (mined["stable_candidate_score"] > 0.0)
        & (mined["active_first_action_ratio"] >= 0.25)
    ].copy()
    stable_pool = _merge_seed_metrics(stable_pool, seeds)
    stable_pool = stable_pool.sort_values(
        ["seed_candidate", "stable_candidate_score", "pump_work_cost_mean"],
        ascending=[False, False, False],
    )
    stable_cols = [
        "case_id",
        "timestamp",
        "label",
        "source_group",
        "source_log",
        "active_first_action_ratio",
        "pump_work_cost_mean",
        "raw_pressure_block0_norm_mean",
        "raw_pressure_future_max_norm_mean",
        "raw_pressure_future_max_norm_max",
        "raw_pressure_rise_norm_mean",
        "event_risk_raw_prob_max_mean",
        "trusted_event_highwind_prob_mean",
        "trusted_event_attention_hits_mean",
        "current_pressure_support_ratio",
        "event_support_ratio",
        "future_pressure_high_ratio",
        "stable_candidate_score",
    ]
    for col in stable_cols:
        if col not in stable_pool.columns:
            stable_pool[col] = np.nan
    stable_out = stable_pool[stable_cols].head(int(args.stable_top_n)).copy()

    falsifier_pool = mined[
        (mined["highwind_falsifier_score"] > 0.0)
        & (
            (mined["highwind_label_hint"] > 0)
            | (
                (mined["trusted_event_highwind_prob_mean"] >= 0.98)
                & (mined["stable_label_hint"] < 1)
            )
            | (
                (mined["trusted_event_attention_hits_mean"] >= 2.0)
                & (mined["stable_label_hint"] < 1)
            )
        )
    ].copy()
    falsifier_pool = falsifier_pool.sort_values(
        ["highwind_falsifier_score", "trusted_event_highwind_prob_mean"],
        ascending=[False, False],
    ).drop_duplicates(["case_id", "timestamp"], keep="first")
    falsifier_cols = [
        "case_id",
        "timestamp",
        "label",
        "source_group",
        "source_log",
        "active_first_action_ratio",
        "pump_work_cost_mean",
        "raw_pressure_block0_norm_mean",
        "raw_pressure_future_max_norm_mean",
        "raw_pressure_future_max_norm_max",
        "event_risk_raw_prob_max_mean",
        "trusted_event_highwind_prob_mean",
        "trusted_event_attention_hits_mean",
        "event_support_ratio",
        "future_pressure_high_ratio",
        "highwind_falsifier_score",
    ]
    for col in falsifier_cols:
        if col not in falsifier_pool.columns:
            falsifier_pool[col] = np.nan
    falsifier_out = falsifier_pool[falsifier_cols].head(
        int(args.falsifier_top_n)
    ).copy()

    stable_path = out_dir / "pressure_trust_stable_candidates.csv"
    falsifier_path = out_dir / "pressure_trust_highwind_falsifiers.csv"
    readout_path = out_dir / "pressure_trust_candidate_readout.md"
    stable_out.to_csv(stable_path, index=False)
    falsifier_out.to_csv(falsifier_path, index=False)
    _write_readout(readout_path, source_dirs, stable_out, falsifier_out)
    print(f"Wrote {stable_path.relative_to(REPO_ROOT)} ({len(stable_out)} rows)")
    print(f"Wrote {falsifier_path.relative_to(REPO_ROOT)} ({len(falsifier_out)} rows)")
    print(f"Wrote {readout_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
