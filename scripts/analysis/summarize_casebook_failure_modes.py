#!/usr/bin/env python3
"""Summarize attitude recovery failure modes from existing casebook logs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BUCKET_S = 600.0
PITCH_CONCERN_DEG = 5.0
ROLL_CONCERN_DEG = 5.0
SAFE_PITCH_P95_DEG = 3.2
SAFE_ROLL_P95_DEG = 2.4
IMPROVE_DEG = 0.5
PRESSURE_RECOVERY_NORM = 0.95
FALLBACK_DOMINATED_RATIO = 0.05
PUMP_MOVE_M3 = 1.0
TARGET_MOVE_KG = 1000.0
LOW_ACTIONS = {"hold", "pump_saving", "active_small"}
ACTIVE_ACTIONS = {"pump_saving", "active_small", "active_medium"}


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _case_file(out_dir: Path, subdir: str, case_id: str, suffix: str) -> Path | None:
    matches = sorted((out_dir / subdir).glob(f"*{case_id}*{suffix}"))
    return matches[0] if matches else None


def _numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _max_run(flags: list[bool]) -> int:
    best = 0
    cur = 0
    for flag in flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _group_label(case_id: str, label: str) -> str:
    text = f"{case_id} {label}".lower()
    if "lowrisk" in text:
        return "lowrisk"
    if "future_relief" in text or "fr_relief" in text:
        return "future_relief"
    if "onset" in text:
        return "onset"
    if "signflip" in text or "sf_" in text:
        return "signflip"
    if "high" in text or "pressure" in text or "residual" in text:
        return "high_boundary"
    return "other"


def _bucket_metrics(ts: pd.DataFrame, planner: pd.DataFrame) -> pd.DataFrame:
    if ts.empty:
        return pd.DataFrame()
    work = ts.copy()
    work["bucket"] = np.floor(_numeric(work, "t_s") / BUCKET_S).astype(int)
    work["pitch_abs"] = np.abs(_numeric(work, "pitch_deg"))
    work["roll_abs"] = np.abs(_numeric(work, "roll_deg"))
    work["pump_abs"] = np.abs(_numeric(work, "pump_total_rate_m3_min"))
    rows: list[dict] = []
    for bucket, g in work.groupby("bucket", sort=True):
        rows.append(
            {
                "bucket": int(bucket),
                "pitch_p95": float(np.percentile(g["pitch_abs"], 95)),
                "roll_p95": float(np.percentile(g["roll_abs"], 95)),
                "pitch_mean": float(g["pitch_abs"].mean()),
                "roll_mean": float(g["roll_abs"].mean()),
                "fallback_ratio": float(_numeric(g, "preview_primary_safety_fallback").mean()),
                "primary_active_ratio": float(_numeric(g, "preview_primary_active").mean()),
                "primary_delta_mean_kg": float(
                    _numeric(g, "preview_primary_delta_mean_kg").mean()
                ),
                "pump_work_m3": float(np.trapezoid(g["pump_abs"].to_numpy(dtype=float), dx=1.0) / 60.0),
            }
        )
    buckets = pd.DataFrame(rows)
    if planner.empty:
        buckets["first_action"] = ""
        buckets["pressure_norm"] = 0.0
        buckets["future_pressure_norm"] = 0.0
        buckets["planner_delta_mean_kg"] = 0.0
        return buckets
    keep = planner.copy()
    keep["bucket"] = _numeric(keep, "bucket").astype(int)
    keep["first_action"] = keep.get("first_action", "").astype(str)
    keep["pressure_norm"] = _numeric(keep, "pressure_block0_norm")
    keep["future_pressure_norm"] = keep[
        ["pressure_block0_norm", "pressure_block1_norm", "pressure_block2_norm"]
    ].apply(pd.to_numeric, errors="coerce").fillna(0.0).max(axis=1)
    keep["planner_delta_mean_kg"] = _numeric(
        keep,
        "prediction_primary_delta_abs_mean_kg",
    )
    return buckets.merge(
        keep[
            [
                "bucket",
                "first_action",
                "pressure_norm",
                "future_pressure_norm",
                "planner_delta_mean_kg",
            ]
        ],
        on="bucket",
        how="left",
    ).fillna(
        {
            "first_action": "",
            "pressure_norm": 0.0,
            "future_pressure_norm": 0.0,
            "planner_delta_mean_kg": 0.0,
        }
    )


def _flag_non_recovering(buckets: pd.DataFrame) -> tuple[int, int]:
    high = (
        (buckets["pitch_p95"] >= PITCH_CONCERN_DEG)
        | (buckets["roll_p95"] >= ROLL_CONCERN_DEG)
    ).to_numpy(dtype=bool)
    attitude = np.maximum(
        buckets["pitch_p95"].to_numpy(dtype=float),
        buckets["roll_p95"].to_numpy(dtype=float),
    )
    non_recover = np.zeros(len(buckets), dtype=bool)
    for i, is_high in enumerate(high):
        if not is_high:
            continue
        look = attitude[i + 1 : i + 3]
        if look.size == 0:
            continue
        non_recover[i] = float(np.min(look)) >= float(attitude[i] - IMPROVE_DEG)
    return int(non_recover.sum()), int(non_recover.any())


def _mode_flags(buckets: pd.DataFrame, row: pd.Series) -> dict:
    if buckets.empty:
        return {
            "continuous_high_pitch": 0,
            "continuous_high_roll": 0,
            "non_recovering_attitude": 0,
            "active_but_weak_response": 0,
            "no_recovery_intent": 0,
            "fallback_dominated": 0,
            "lowrisk_safe": 0,
            "dominant_failure_mode": "missing_logs",
        }
    pitch_high = (buckets["pitch_p95"] >= PITCH_CONCERN_DEG).to_list()
    roll_high = (buckets["roll_p95"] >= ROLL_CONCERN_DEG).to_list()
    high = (
        (buckets["pitch_p95"] >= PITCH_CONCERN_DEG)
        | (buckets["roll_p95"] >= ROLL_CONCERN_DEG)
    )
    high_nonfallback = high & (buckets["fallback_ratio"] < 0.5)
    nonrecover_count, nonrecover_flag = _flag_non_recovering(buckets)
    active_moved = (
        buckets["first_action"].astype(str).isin(ACTIVE_ACTIONS)
        & (
            (buckets["pump_work_m3"] > PUMP_MOVE_M3)
            | (buckets["primary_delta_mean_kg"] > TARGET_MOVE_KG)
            | (buckets["planner_delta_mean_kg"] > TARGET_MOVE_KG)
        )
    )
    attitude = np.maximum(
        buckets["pitch_p95"].to_numpy(dtype=float),
        buckets["roll_p95"].to_numpy(dtype=float),
    )
    weak_bucket = np.zeros(len(buckets), dtype=bool)
    for i in range(len(buckets) - 1):
        if bool(high_nonfallback.iloc[i]) and bool(active_moved.iloc[i]):
            weak_bucket[i] = float(attitude[i + 1]) >= float(attitude[i] - IMPROVE_DEG)
    low_intent = high_nonfallback & buckets["first_action"].astype(str).isin(LOW_ACTIONS)
    fallback_ratio = float(row.get("primary_safety_fallback_ratio", 0.0) or 0.0)
    high_fallback_share = (
        float((buckets.loc[high, "fallback_ratio"] >= 0.5).mean())
        if bool(high.any())
        else 0.0
    )
    lowrisk_safe = bool(
        float(row.get("primary_pitch_p95", np.nan)) <= SAFE_PITCH_P95_DEG
        and float(row.get("primary_roll_p95", np.nan)) <= SAFE_ROLL_P95_DEG
        and fallback_ratio <= 1e-9
    )
    flags = {
        "continuous_high_pitch": int(_max_run(pitch_high) >= 2),
        "continuous_high_roll": int(_max_run(roll_high) >= 2),
        "non_recovering_attitude": int(nonrecover_flag),
        "active_but_weak_response": int(bool(weak_bucket.any())),
        "no_recovery_intent": int(bool(low_intent.any())),
        "fallback_dominated": int(
            fallback_ratio >= FALLBACK_DOMINATED_RATIO or high_fallback_share >= 0.5
        ),
        "lowrisk_safe": int(lowrisk_safe),
    }
    if flags["lowrisk_safe"]:
        dominant = "lowrisk_safe"
    elif flags["fallback_dominated"]:
        dominant = "fallback_dominated"
    elif flags["active_but_weak_response"]:
        dominant = "active_but_weak_response"
    elif flags["no_recovery_intent"]:
        dominant = "no_recovery_intent"
    elif flags["non_recovering_attitude"]:
        dominant = "non_recovering_attitude"
    elif flags["continuous_high_pitch"] or flags["continuous_high_roll"]:
        dominant = "continuous_high_attitude"
    else:
        dominant = "no_sustained_high_attitude"
    flags.update(
        {
            "dominant_failure_mode": dominant,
            "high_pitch_bucket_count": int(sum(pitch_high)),
            "high_roll_bucket_count": int(sum(roll_high)),
            "high_attitude_bucket_count": int(high.sum()),
            "max_consecutive_high_pitch_buckets": _max_run(pitch_high),
            "max_consecutive_high_roll_buckets": _max_run(roll_high),
            "non_recovering_bucket_count": int(nonrecover_count),
            "active_weak_bucket_count": int(weak_bucket.sum()),
            "no_recovery_intent_bucket_count": int(low_intent.sum()),
            "high_bucket_fallback_share": float(high_fallback_share),
            "high_bucket_pressure_mean": float(
                buckets.loc[high, "pressure_norm"].mean() if bool(high.any()) else 0.0
            ),
            "high_bucket_pressure_supported_count": int(
                ((buckets["pressure_norm"] >= PRESSURE_RECOVERY_NORM) & high).sum()
            ),
        }
    )
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--casebook-dir",
        default="outputs/wind_prediction/medium_escalation_guard20_baseline",
        help="Existing casebook output directory to analyze.",
    )
    parser.add_argument(
        "--medium-comparison",
        default="outputs/wind_prediction/medium_escalation_guard20/medium_escalation_holdout_case_comparison.csv",
        help="Optional medium-escalation guard comparison CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/casebook_failure_modes",
    )
    args = parser.parse_args()

    casebook_dir = _resolve(args.casebook_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_csv(casebook_dir / "casebook_summary.csv")
    medium_path = _resolve(args.medium_comparison)
    medium = _read_csv(medium_path) if medium_path.exists() else pd.DataFrame()

    rows: list[dict] = []
    for _, case in summary.iterrows():
        case_id = str(case["case_id"])
        ts_path = _case_file(casebook_dir, "timeseries", case_id, "_timeseries.csv")
        log_path = _case_file(casebook_dir, "planner_logs", case_id, "_planner_log.csv")
        ts = _read_csv(ts_path) if ts_path is not None else pd.DataFrame()
        log = _read_csv(log_path) if log_path is not None else pd.DataFrame()
        buckets = _bucket_metrics(ts, log)
        flags = _mode_flags(buckets, case)
        med_row = (
            medium.loc[medium["case_id"].astype(str) == case_id].iloc[0]
            if not medium.empty
            and "case_id" in medium.columns
            and bool((medium["case_id"].astype(str) == case_id).any())
            else pd.Series(dtype=object)
        )
        rows.append(
            {
                "case_id": case_id,
                "label": str(case.get("label", "")),
                "case_group": _group_label(case_id, str(case.get("label", ""))),
                "primary_pump_work_m3": float(case.get("primary_pump_work_m3", np.nan)),
                "primary_pitch_p95_deg": float(case.get("primary_pitch_p95", np.nan)),
                "primary_roll_p95_deg": float(case.get("primary_roll_p95", np.nan)),
                "primary_safety_fallback_ratio": float(
                    case.get("primary_safety_fallback_ratio", 0.0)
                ),
                **flags,
                "medium_guard_triggered": int(med_row.get("triggered", 0) or 0),
                "medium_guard_bucket_count": int(
                    med_row.get("medium_escalation_bucket_count", 0) or 0
                ),
                "medium_guard_delta_pump_work_m3": float(
                    med_row.get("delta_pump_work_m3", 0.0) or 0.0
                ),
                "medium_guard_delta_pitch_p95_deg": float(
                    med_row.get("delta_pitch_p95_deg", 0.0) or 0.0
                ),
                "medium_guard_delta_fallback_ratio": float(
                    med_row.get("delta_fallback_ratio", 0.0) or 0.0
                ),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "casebook_failure_mode_table.csv", index=False)

    sustained = table.loc[
        (table["continuous_high_pitch"] == 1)
        | (table["continuous_high_roll"] == 1)
        | (table["non_recovering_attitude"] == 1)
    ]
    recovery_candidates = table.loc[
        (table["lowrisk_safe"] == 0)
        & (table["fallback_dominated"] == 0)
        & (
            (table["continuous_high_pitch"] == 1)
            | (table["continuous_high_roll"] == 1)
            | (table["non_recovering_attitude"] == 1)
        )
        & (
            (table["active_but_weak_response"] == 1)
            | (table["no_recovery_intent"] == 1)
            | (table["non_recovering_attitude"] == 1)
        )
    ]
    boundary_recovery_cases = table.loc[
        (table["lowrisk_safe"] == 0)
        & (table["fallback_dominated"] == 0)
        & (table["continuous_high_pitch"] == 0)
        & (table["continuous_high_roll"] == 0)
        & (table["non_recovering_attitude"] == 0)
        & (table["no_recovery_intent"] == 1)
    ]
    lowrisk_samples = table.loc[
        (table["lowrisk_safe"] == 1)
        | (table["case_group"] == "lowrisk")
    ]
    dominant_counts = table["dominant_failure_mode"].value_counts().to_dict()
    flag_counts = {
        name: int(table[name].sum())
        for name in [
            "continuous_high_pitch",
            "continuous_high_roll",
            "non_recovering_attitude",
            "active_but_weak_response",
            "no_recovery_intent",
            "fallback_dominated",
            "lowrisk_safe",
        ]
    }
    med_trigger = table.loc[table["medium_guard_triggered"] == 1]
    med_side_effects = med_trigger.loc[med_trigger["medium_guard_delta_pump_work_m3"] > 1.0]
    med_useful = med_trigger.loc[
        (med_trigger["medium_guard_delta_pitch_p95_deg"] < -1e-9)
        | (med_trigger["medium_guard_delta_fallback_ratio"] < -1e-9)
    ]

    lines = [
        "# Casebook Failure Mode Summary",
        "",
        "1. 当前长期高姿态问题不是只集中在 fr09。"
        f"guard20 baseline 中有 {len(sustained)} 个 sustained/high-attitude case: "
        f"{', '.join(sustained['case_id'].tolist()) or 'none'}。",
        "",
        "2. 主要问题不是单一来源，而是三类混合："
        f"fallback dominated={int(table['fallback_dominated'].sum())}，"
        f"active_but_weak_response={int(table['active_but_weak_response'].sum())}，"
        f"no_recovery_intent={int(table['no_recovery_intent'].sum())}。"
        "因此既不能只责怪 planner 没进入恢复，也不能只靠提高动作幅值解决。",
        "",
        "3. medium_escalation 的触发失败说明需要更上层的 recovery_mode。"
        f"guard20 中 medium_escalation 触发 {len(med_trigger)} 个 case，"
        f"只有 {len(med_useful)} 个按 pitch/fallback 指标有收益，"
        f"且 {len(med_side_effects)} 个出现泵量副作用；"
        "它更适合作为 recovery_mode 内部的动作幅值选择，而不是主逻辑。",
        "",
        "4. 恢复模式设计代表样本："
        f"{', '.join(recovery_candidates['case_id'].tolist()) or 'none'}。"
        "边界/非持续守门样本："
        f"{', '.join(boundary_recovery_cases['case_id'].tolist()) or 'none'}。",
        "",
        "5. 低风险保护样本："
        f"{', '.join(lowrisk_samples['case_id'].tolist()) or 'none'}。",
    ]
    (out_dir / "casebook_failure_mode_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(out_dir / "casebook_failure_mode_summary.md")


if __name__ == "__main__":
    main()
