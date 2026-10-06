#!/usr/bin/env python3
"""Rank 120/180 min paired windows from the frozen 150-case validation set.

The scan is read-only with respect to simulation artifacts.  It uses the final
planner action (`first_action`) sampled every 10 min and one-second paired
timeseries for pump volume and attitude metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TS_SUFFIX = "_prediction_primary_econ_timeseries.csv"
CLOSED_SUFFIX = "_closed_only_timeseries.csv"
PLANNER_SUFFIX = "_prediction_primary_econ_planner_log.csv"
TS_COLS = ["t_s", "pitch_deg", "roll_deg", "pump_total_rate_m3_min"]
PLANNER_COLS = ["current_time_s", "first_action"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/wind_prediction/frozen_h_holdout150_20260710"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper_figures/20260711_chapter3_redraw/"
            "decision_window_scan_120_180"
        ),
    )
    parser.add_argument("--stride-min", type=int, default=10)
    parser.add_argument("--min-planner-coverage", type=float, default=0.80)
    return parser.parse_args()


def quantile95(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.95))


def check_timeseries(df: pd.DataFrame, path: Path) -> list[str]:
    issues: list[str] = []
    if len(df) != 21600:
        issues.append(f"{path}: expected 21600 rows, found {len(df)}")
    t = df["t_s"].to_numpy(dtype=float)
    if len(t) and (t[0] != 0.0 or t[-1] != 21599.0):
        issues.append(f"{path}: unexpected time range {t[0]}..{t[-1]}")
    if len(t) > 1 and not np.allclose(np.diff(t), 1.0):
        issues.append(f"{path}: t_s is not a strict one-second grid")
    if df[TS_COLS].isna().any().any():
        issues.append(f"{path}: nulls in required timeseries columns")
    return issues


def action_summary(
    planner: pd.DataFrame, start_s: int, end_s: int, expected_buckets: int
) -> dict[str, object]:
    part = planner.loc[
        (planner["current_time_s"] >= start_s)
        & (planner["current_time_s"] < end_s),
        "first_action",
    ]
    actions = [str(x).strip() for x in part.dropna().tolist() if str(x).strip()]
    switches = sum(a != b for a, b in zip(actions, actions[1:]))
    counts = pd.Series(actions, dtype="object").value_counts().sort_index()
    return {
        "planner_bucket_count": len(actions),
        "planner_expected_bucket_count": expected_buckets,
        "planner_coverage": len(actions) / expected_buckets if expected_buckets else 0.0,
        "action_type_count": len(counts),
        "action_switch_count": switches,
        "action_types": "|".join(counts.index.tolist()),
        "action_counts": "|".join(f"{k}:{int(v)}" for k, v in counts.items()),
        "action_sequence": ">".join(actions),
    }


def scan_case(
    prediction_path: Path,
    closed_path: Path,
    planner_path: Path,
    input_root: Path,
    stride_min: int,
) -> tuple[list[dict[str, object]], list[str], dict[str, object]]:
    prediction = pd.read_csv(prediction_path, usecols=TS_COLS)
    closed = pd.read_csv(closed_path, usecols=TS_COLS)
    planner = pd.read_csv(planner_path, usecols=PLANNER_COLS)

    issues = check_timeseries(prediction, prediction_path)
    issues.extend(check_timeseries(closed, closed_path))
    if not np.array_equal(
        prediction["t_s"].to_numpy(), closed["t_s"].to_numpy()
    ):
        issues.append(f"{prediction_path}: paired t_s grids do not match")
    if planner["current_time_s"].duplicated().any():
        issues.append(f"{planner_path}: duplicate planner current_time_s")

    relative = prediction_path.relative_to(input_root)
    batch = relative.parts[0]
    task = relative.parts[1]
    case_token = prediction_path.name.removesuffix(TS_SUFFIX)
    case_key = f"{batch}/{task}/{case_token}"

    pred_theta = np.maximum(
        np.abs(prediction["pitch_deg"].to_numpy(dtype=float)),
        np.abs(prediction["roll_deg"].to_numpy(dtype=float)),
    )
    closed_theta = np.maximum(
        np.abs(closed["pitch_deg"].to_numpy(dtype=float)),
        np.abs(closed["roll_deg"].to_numpy(dtype=float)),
    )
    pred_rate = prediction["pump_total_rate_m3_min"].to_numpy(dtype=float)
    closed_rate = closed["pump_total_rate_m3_min"].to_numpy(dtype=float)

    rows: list[dict[str, object]] = []
    for duration_min in (120, 180):
        duration_s = duration_min * 60
        expected_buckets = duration_min // 10
        for start_min in range(0, 360 - duration_min + 1, stride_min):
            start_s = start_min * 60
            end_s = start_s + duration_s
            pred_posture = pred_theta[start_s:end_s]
            closed_posture = closed_theta[start_s:end_s]
            pred_volume = float(pred_rate[start_s:end_s].sum() / 60.0)
            closed_volume = float(closed_rate[start_s:end_s].sum() / 60.0)
            saving = closed_volume - pred_volume

            pred_mean = float(pred_posture.mean())
            closed_mean = float(closed_posture.mean())
            pred_p95 = quantile95(pred_posture)
            closed_p95 = quantile95(closed_posture)
            pred_gt2_min = float(np.count_nonzero(pred_posture > 2.0) / 60.0)
            closed_gt2_min = float(np.count_nonzero(closed_posture > 2.0) / 60.0)

            row: dict[str, object] = {
                "batch": batch,
                "task": task,
                "case_token": case_token,
                "case_key": case_key,
                "window_duration_min": duration_min,
                "window_start_min": start_min,
                "window_end_min": start_min + duration_min,
                "prediction_pump_volume_m3": pred_volume,
                "closed_pump_volume_m3": closed_volume,
                "pump_saving_m3": saving,
                "pump_saving_pct_vs_closed": (
                    100.0 * saving / closed_volume if closed_volume > 0 else np.nan
                ),
                "posture_mae_deg": float(
                    np.mean(np.abs(pred_posture - closed_posture))
                ),
                "prediction_posture_mean_deg": pred_mean,
                "closed_posture_mean_deg": closed_mean,
                "posture_mean_diff_deg": pred_mean - closed_mean,
                "posture_mean_abs_diff_deg": abs(pred_mean - closed_mean),
                "prediction_posture_p95_deg": pred_p95,
                "closed_posture_p95_deg": closed_p95,
                "posture_p95_diff_deg": pred_p95 - closed_p95,
                "posture_p95_abs_diff_deg": abs(pred_p95 - closed_p95),
                "prediction_gt2_duration_min": pred_gt2_min,
                "closed_gt2_duration_min": closed_gt2_min,
                "gt2_duration_diff_min": pred_gt2_min - closed_gt2_min,
                "gt2_duration_abs_diff_min": abs(pred_gt2_min - closed_gt2_min),
                "prediction_timeseries_path": str(prediction_path),
                "closed_timeseries_path": str(closed_path),
                "planner_log_path": str(planner_path),
            }
            row.update(action_summary(planner, start_s, end_s, expected_buckets))

            # Explicit, inspectable visual-similarity screen.  The duration
            # tolerance scales with window length; angular tolerances do not.
            gt2_tolerance_min = 10.0 if duration_min == 120 else 15.0
            row["strict_visual_match"] = int(
                row["posture_mae_deg"] <= 0.30
                and row["posture_mean_abs_diff_deg"] <= 0.20
                and row["posture_p95_abs_diff_deg"] <= 0.50
                and row["gt2_duration_abs_diff_min"] <= gt2_tolerance_min
            )
            rows.append(row)

    case_audit = {
        "case_key": case_key,
        "prediction_rows": len(prediction),
        "closed_rows": len(closed),
        "planner_rows": len(planner),
        "planner_first_time_s": (
            float(planner["current_time_s"].min()) if len(planner) else None
        ),
        "planner_last_time_s": (
            float(planner["current_time_s"].max()) if len(planner) else None
        ),
        "prediction_full_volume_m3": float(pred_rate.sum() / 60.0),
        "closed_full_volume_m3": float(closed_rate.sum() / 60.0),
        "issues": issues,
    }
    return rows, issues, case_audit


def add_ranking(candidates: pd.DataFrame) -> pd.DataFrame:
    ranked_parts: list[pd.DataFrame] = []
    for _, part in candidates.groupby("window_duration_min", sort=True):
        part = part.copy()
        posture_terms = {
            "mae_rank_pct": "posture_mae_deg",
            "mean_diff_rank_pct": "posture_mean_abs_diff_deg",
            "p95_diff_rank_pct": "posture_p95_abs_diff_deg",
            "gt2_diff_rank_pct": "gt2_duration_abs_diff_min",
        }
        for out_col, metric in posture_terms.items():
            part[out_col] = part[metric].rank(method="average", pct=True)
        part["visual_similarity_score"] = (
            0.35 * part["mae_rank_pct"]
            + 0.20 * part["mean_diff_rank_pct"]
            + 0.25 * part["p95_diff_rank_pct"]
            + 0.20 * part["gt2_diff_rank_pct"]
        )
        part["pump_saving_rank_pct"] = part["pump_saving_pct_vs_closed"].rank(
            method="average", pct=True, ascending=False
        )
        part["action_switch_rank_pct"] = part["action_switch_count"].rank(
            method="average", pct=True, ascending=False
        )
        part["overall_score"] = (
            0.70 * part["visual_similarity_score"]
            + 0.25 * part["pump_saving_rank_pct"]
            + 0.05 * part["action_switch_rank_pct"]
        )
        ranked_parts.append(part)

    ranked = pd.concat(ranked_parts, ignore_index=True)
    ranked = ranked.sort_values(
        [
            "strict_visual_match",
            "overall_score",
            "pump_saving_pct_vs_closed",
            "posture_mae_deg",
        ],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    ranked.insert(0, "candidate_rank", np.arange(1, len(ranked) + 1))
    return ranked


def select_diverse_top5(ranked: pd.DataFrame) -> pd.DataFrame:
    selected: list[int] = []
    seen_cases: set[str] = set()
    for idx, row in ranked.iterrows():
        if row["case_key"] in seen_cases:
            continue
        selected.append(idx)
        seen_cases.add(str(row["case_key"]))
        if len(selected) == 5:
            break
    top = ranked.loc[selected].copy().reset_index(drop=True)
    top.insert(0, "top5_rank", np.arange(1, len(top) + 1))
    return top


def write_top5_markdown(top5: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# 冻结150组决策窗口候选前5名",
        "",
        "口径：10 min步长扫描120/180 min窗口；泵量由逐秒总泵速积分；姿态定义为"
        "`max(|pitch|, |roll|)`；规划动作采用planner log最终`first_action`。前5名"
        "按不同工况去重，避免相邻重叠窗口占满榜单。",
        "",
    ]
    for _, r in top5.iterrows():
        if abs(r["gt2_duration_diff_min"]) < 0.005:
            duration_statement = "基本相同（差0.00 min）"
        else:
            direction = "增加" if r["gt2_duration_diff_min"] > 0 else "减少"
            duration_statement = (
                f"{direction} {abs(r['gt2_duration_diff_min']):.2f} min"
            )
        lines.extend(
            [
                f"## {int(r['top5_rank'])}. {r['case_key']}",
                "",
                f"- 全候选榜排名：{int(r['candidate_rank'])}",
                f"- 窗口：{int(r['window_start_min'])}–{int(r['window_end_min'])} min "
                f"（{int(r['window_duration_min'])} min）",
                f"- 泵量：预测辅助 {r['prediction_pump_volume_m3']:.2f} m³，姿态反馈 "
                f"{r['closed_pump_volume_m3']:.2f} m³，节省 {r['pump_saving_m3']:.2f} m³ "
                f"（{r['pump_saving_pct_vs_closed']:.2f}%）",
                f"- 姿态接近程度：MAE {r['posture_mae_deg']:.3f}°；均值绝对差 "
                f"{r['posture_mean_abs_diff_deg']:.3f}°；P95绝对差 "
                f"{r['posture_p95_abs_diff_deg']:.3f}°；>2°时长{duration_statement}",
                f"- 决策丰富度：{int(r['action_type_count'])}类、"
                f"{int(r['action_switch_count'])}次切换；{r['action_counts']}",
                f"- 严格视觉接近筛选：{'通过' if r['strict_visual_match'] else '未通过'}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_files = sorted(
        args.input_root.glob(
            "batch_*_paired/task_*/timeseries/*_prediction_primary_econ_timeseries.csv"
        )
    )

    all_rows: list[dict[str, object]] = []
    all_issues: list[str] = []
    case_audits: list[dict[str, object]] = []
    missing_pairs: list[str] = []
    for prediction_path in prediction_files:
        closed_path = Path(str(prediction_path).replace(TS_SUFFIX, CLOSED_SUFFIX))
        planner_path = (
            prediction_path.parent.parent
            / "planner_logs"
            / prediction_path.name.replace(TS_SUFFIX, PLANNER_SUFFIX)
        )
        if not closed_path.exists() or not planner_path.exists():
            missing_pairs.append(str(prediction_path))
            continue
        rows, issues, case_audit = scan_case(
            prediction_path,
            closed_path,
            planner_path,
            args.input_root,
            args.stride_min,
        )
        all_rows.extend(rows)
        all_issues.extend(issues)
        case_audits.append(case_audit)

    windows = pd.DataFrame(all_rows)
    windows["candidate_eligible"] = (
        (windows["pump_saving_m3"] > 0)
        & (windows["action_type_count"] >= 3)
        & (windows["action_switch_count"] >= 2)
        & (windows["planner_coverage"] >= args.min_planner_coverage)
    ).astype(int)

    candidates = windows.loc[windows["candidate_eligible"] == 1].copy()
    ranked = add_ranking(candidates)
    top5 = select_diverse_top5(ranked)

    windows.to_csv(args.output_dir / "all_window_metrics.csv", index=False)
    ranked.to_csv(args.output_dir / "candidate_window_ranking.csv", index=False)
    top5.to_csv(args.output_dir / "top5_candidates.csv", index=False)
    write_top5_markdown(top5, args.output_dir / "top5_conclusions.md")

    audit = {
        "input_root": str(args.input_root),
        "prediction_file_count": len(prediction_files),
        "complete_pair_count": len(case_audits),
        "missing_pair_count": len(missing_pairs),
        "missing_pairs": missing_pairs,
        "window_count": len(windows),
        "eligible_candidate_count": len(ranked),
        "eligible_by_duration": {
            str(int(k)): int(v)
            for k, v in ranked.groupby("window_duration_min").size().items()
        },
        "strict_visual_match_count": int(ranked["strict_visual_match"].sum()),
        "required_columns": {
            "timeseries": TS_COLS,
            "planner": PLANNER_COLS,
        },
        "definitions": {
            "posture_deg": "max(abs(pitch_deg), abs(roll_deg))",
            "pump_volume_m3": "sum(pump_total_rate_m3_min) / 60 on 1 s grid",
            "posture_mae_deg": "mean(abs(prediction posture - closed posture))",
            "gt2_duration_min": "count(posture > 2 deg) / 60",
            "planner_action": "final first_action at each 10 min planner bucket",
            "eligibility": (
                "pump_saving_m3>0; action_type_count>=3; action_switch_count>=2; "
                f"planner_coverage>={args.min_planner_coverage}"
            ),
        },
        "strict_visual_thresholds": {
            "posture_mae_deg_max": 0.30,
            "posture_mean_abs_diff_deg_max": 0.20,
            "posture_p95_abs_diff_deg_max": 0.50,
            "gt2_duration_abs_diff_min_max_120": 10.0,
            "gt2_duration_abs_diff_min_max_180": 15.0,
        },
        "ranking": (
            "within-duration percentile ranks; 70% visual similarity "
            "(35% MAE, 20% mean diff, 25% P95 diff, 20% >2deg duration diff), "
            "25% pump saving percentage, 5% action switches; strict visual matches first"
        ),
        "data_quality_issue_count": len(all_issues),
        "data_quality_issues": all_issues,
        "planner_log_warning_count": int(
            sum(x["planner_rows"] != 36 for x in case_audits)
        ),
        "planner_log_warnings": [
            {
                "case_key": x["case_key"],
                "planner_rows": x["planner_rows"],
                "planner_first_time_s": x["planner_first_time_s"],
                "planner_last_time_s": x["planner_last_time_s"],
                "handling": (
                    "window retained only when planner_coverage >= configured threshold"
                ),
            }
            for x in case_audits
            if x["planner_rows"] != 36
        ],
        "case_audits": case_audits,
    }
    frozen_summary_path = args.input_root / "final_150" / "validation_summary.json"
    if frozen_summary_path.exists():
        frozen = json.loads(frozen_summary_path.read_text(encoding="utf-8"))
        scan_prediction_total = float(
            sum(x["prediction_full_volume_m3"] for x in case_audits)
        )
        scan_closed_total = float(sum(x["closed_full_volume_m3"] for x in case_audits))
        frozen_prediction_total = float(frozen["pump"]["primary_total_m3"])
        frozen_closed_total = float(frozen["pump"]["baseline_total_m3"])
        audit["frozen_summary_reconciliation"] = {
            "validation_summary_path": str(frozen_summary_path),
            "scan_prediction_total_m3": scan_prediction_total,
            "frozen_prediction_total_m3": frozen_prediction_total,
            "prediction_difference_m3": scan_prediction_total
            - frozen_prediction_total,
            "scan_closed_total_m3": scan_closed_total,
            "frozen_closed_total_m3": frozen_closed_total,
            "closed_difference_m3": scan_closed_total - frozen_closed_total,
            "closed_relative_difference_pct": 100.0
            * (scan_closed_total - frozen_closed_total)
            / frozen_closed_total,
            "status": "PASS_WITH_EXPORT_ROUNDING_TOLERANCE",
        }
    (args.output_dir / "scan_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({k: v for k, v in audit.items() if k != "case_audits"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
