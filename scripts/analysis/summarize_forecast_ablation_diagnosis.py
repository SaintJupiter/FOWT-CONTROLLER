#!/usr/bin/env python3
"""Summarize why learned and persistence forecasts give similar control results.

This script is read-only with respect to experiments: it consumes existing
casebook summaries and planner logs, then writes a compact diagnosis package.
It is meant for paper/research logging, not for tuning controller parameters.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


RUNS = {
    "base": {
        "oracle": "forecast_ablation_oracle_vs_smooth_closed_4case_1h",
        "learned": "forecast_ablation_learned_vs_smooth_closed_4case_1h",
        "persistence": "forecast_ablation_persistence_vs_smooth_closed_4case_1h",
    },
    "pump_suppression": {
        "oracle": "forecast_ablation_oracle_suppression_4case_1h",
        "learned": "forecast_ablation_learned_suppression_4case_1h",
        "persistence": "forecast_ablation_persistence_suppression_4case_1h",
    },
    "target_change_reset": {
        "oracle": "forecast_ablation_oracle_target_change_4case_1h",
        "learned": "forecast_ablation_learned_target_change_4case_1h",
        "persistence": "forecast_ablation_persistence_target_change_4case_1h",
    },
    "active_bucket_reset": {
        "oracle": "forecast_ablation_oracle_active_bucket_4case_1h",
        "learned": "forecast_ablation_learned_active_bucket_4case_1h",
        "persistence": "forecast_ablation_persistence_active_bucket_4case_1h",
    },
    "event_risk_pressure_boost": {
        "oracle": "forecast_ablation_oracle_event_risk_boost_4case_1h",
        "learned": "forecast_ablation_learned_event_risk_boost_4case_1h",
        "persistence": "forecast_ablation_persistence_event_risk_boost_4case_1h",
    },
    "suppression_event_risk_guard": {
        "oracle": "forecast_ablation_oracle_supp_event_guard_4case_1h",
        "learned": "forecast_ablation_learned_supp_event_guard_4case_1h",
        "persistence": "forecast_ablation_persistence_supp_event_guard_4case_1h",
    },
}


PLANNER_KEY_COLS = [
    "history_end",
    "first_action",
    "best_sequence",
    "forecast_source",
    "prediction_primary_target_refreshed",
    "prediction_primary_target_reused",
    "prediction_primary_delta_abs_mean_kg",
    "pump_suppression_active",
    "pump_suppression_reason",
    "pressure_block0_norm",
    "pressure_block1_norm",
    "pressure_block2_norm",
    "pressure_block02_dot",
    "event_risk_prob_0_20m",
    "event_risk_prob_20_40m",
    "event_risk_prob_40_60m",
    "event_risk_scale_0_20m",
    "event_risk_scale_20_40m",
    "event_risk_scale_40_60m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        default="outputs/wind_prediction",
        help="Root containing forecast ablation output directories.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/forecast_ablation_diagnosis_20260506",
        help="Directory for diagnosis tables and report.",
    )
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _read_summary(outputs_root: Path, folder: str) -> pd.DataFrame:
    path = outputs_root / folder / "casebook_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _planner_log_scale_mean(outputs_root: Path, folder: str) -> float:
    vals: list[float] = []
    for path in _planner_log_paths(outputs_root, folder):
        df = pd.read_csv(path)
        cols = [
            col
            for col in (
                "event_risk_scale_0_20m",
                "event_risk_scale_20_40m",
                "event_risk_scale_40_60m",
            )
            if col in df.columns
        ]
        if cols:
            vals.extend(df[cols].to_numpy(dtype=float).reshape(-1).tolist())
    return float(np.mean(vals)) if vals else np.nan


def _event_risk_scale_value(outputs_root: Path, folder: str, df: pd.DataFrame) -> float:
    planner_mean = _planner_log_scale_mean(outputs_root, folder)
    if np.isfinite(planner_mean):
        return planner_mean
    return float(df.get("preview_event_risk_scale_mean", pd.Series(1.0, index=df.index)).mean())


def _case_number(case_id: str) -> str:
    text = str(case_id)
    return text.split("_", 1)[0] if "_" in text else text


def build_run_summary(outputs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment, source_map in RUNS.items():
        for source, folder in source_map.items():
            df = _read_summary(outputs_root, folder)
            event_risk_scale = _event_risk_scale_value(outputs_root, folder, df)
            rows.append(
                {
                    "experiment": experiment,
                    "source": source,
                    "cases": int(len(df)),
                    "mean_closed_pump_m3": float(df["closed_pump_work_m3"].mean()),
                    "mean_primary_pump_m3": float(df["primary_pump_work_m3"].mean()),
                    "mean_d_pump_work_pct": float(df["d_pump_work_pct"].mean()),
                    "mean_d_pitch_p95_deg": float(df["d_pitch_p95"].mean()),
                    "mean_d_roll_p95_deg": float(df["d_roll_p95"].mean()),
                    "mean_closed_latch": float(df["closed_latch_switches"].mean()),
                    "mean_primary_latch": float(df["primary_latch_switches"].mean()),
                    "mean_suppression_ratio": float(
                        df.get("preview_pump_suppression_ratio", pd.Series(0.0, index=df.index)).mean()
                    ),
                    "mean_event_risk_scale": float(event_risk_scale),
                    "case06_primary_pump_m3": float(
                        df.loc[df["case_id"].map(_case_number).eq("06"), "primary_pump_work_m3"].iloc[0]
                    )
                    if df["case_id"].map(_case_number).eq("06").any()
                    else np.nan,
                    "case06_d_pump_work_pct": float(
                        df.loc[df["case_id"].map(_case_number).eq("06"), "d_pump_work_pct"].iloc[0]
                    )
                    if df["case_id"].map(_case_number).eq("06").any()
                    else np.nan,
                    "case06_d_pitch_p95_deg": float(
                        df.loc[df["case_id"].map(_case_number).eq("06"), "d_pitch_p95"].iloc[0]
                    )
                    if df["case_id"].map(_case_number).eq("06").any()
                    else np.nan,
                    "case09_primary_pump_m3": float(
                        df.loc[df["case_id"].map(_case_number).eq("09"), "primary_pump_work_m3"].iloc[0]
                    )
                    if df["case_id"].map(_case_number).eq("09").any()
                    else np.nan,
                    "case09_d_pitch_p95_deg": float(
                        df.loc[df["case_id"].map(_case_number).eq("09"), "d_pitch_p95"].iloc[0]
                    )
                    if df["case_id"].map(_case_number).eq("09").any()
                    else np.nan,
                    "folder": folder,
                }
            )
    return pd.DataFrame(rows)


def build_pair_deltas(run_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment, group in run_summary.groupby("experiment", sort=False):
        by_source = group.set_index("source")
        if {"learned", "persistence"}.issubset(by_source.index):
            learned = by_source.loc["learned"]
            persist = by_source.loc["persistence"]
            rows.append(
                {
                    "experiment": experiment,
                    "pair": "learned_minus_persistence",
                    "mean_primary_pump_m3": float(
                        learned["mean_primary_pump_m3"] - persist["mean_primary_pump_m3"]
                    ),
                    "mean_d_pump_work_pct_point": float(
                        learned["mean_d_pump_work_pct"] - persist["mean_d_pump_work_pct"]
                    ),
                    "mean_d_pitch_p95_deg": float(
                        learned["mean_d_pitch_p95_deg"] - persist["mean_d_pitch_p95_deg"]
                    ),
                    "mean_d_roll_p95_deg": float(
                        learned["mean_d_roll_p95_deg"] - persist["mean_d_roll_p95_deg"]
                    ),
                    "mean_primary_latch": float(
                        learned["mean_primary_latch"] - persist["mean_primary_latch"]
                    ),
                    "case06_primary_pump_m3": float(
                        learned["case06_primary_pump_m3"] - persist["case06_primary_pump_m3"]
                    ),
                    "case06_d_pitch_p95_deg": float(
                        learned["case06_d_pitch_p95_deg"] - persist["case06_d_pitch_p95_deg"]
                    ),
                    "case09_primary_pump_m3": float(
                        learned["case09_primary_pump_m3"] - persist["case09_primary_pump_m3"]
                    ),
                    "case09_d_pitch_p95_deg": float(
                        learned["case09_d_pitch_p95_deg"] - persist["case09_d_pitch_p95_deg"]
                    ),
                }
            )
        if {"oracle", "persistence"}.issubset(by_source.index):
            oracle = by_source.loc["oracle"]
            persist = by_source.loc["persistence"]
            rows.append(
                {
                    "experiment": experiment,
                    "pair": "oracle_minus_persistence",
                    "mean_primary_pump_m3": float(
                        oracle["mean_primary_pump_m3"] - persist["mean_primary_pump_m3"]
                    ),
                    "mean_d_pump_work_pct_point": float(
                        oracle["mean_d_pump_work_pct"] - persist["mean_d_pump_work_pct"]
                    ),
                    "mean_d_pitch_p95_deg": float(
                        oracle["mean_d_pitch_p95_deg"] - persist["mean_d_pitch_p95_deg"]
                    ),
                    "mean_d_roll_p95_deg": float(
                        oracle["mean_d_roll_p95_deg"] - persist["mean_d_roll_p95_deg"]
                    ),
                    "mean_primary_latch": float(
                        oracle["mean_primary_latch"] - persist["mean_primary_latch"]
                    ),
                    "case06_primary_pump_m3": float(
                        oracle["case06_primary_pump_m3"] - persist["case06_primary_pump_m3"]
                    ),
                    "case06_d_pitch_p95_deg": float(
                        oracle["case06_d_pitch_p95_deg"] - persist["case06_d_pitch_p95_deg"]
                    ),
                    "case09_primary_pump_m3": float(
                        oracle["case09_primary_pump_m3"] - persist["case09_primary_pump_m3"]
                    ),
                    "case09_d_pitch_p95_deg": float(
                        oracle["case09_d_pitch_p95_deg"] - persist["case09_d_pitch_p95_deg"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _planner_log_paths(outputs_root: Path, folder: str) -> list[Path]:
    return sorted((outputs_root / folder / "planner_logs").glob("*planner_log.csv"))


def _read_planner_logs(outputs_root: Path, experiment: str, source: str, folder: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _planner_log_paths(outputs_root, folder):
        df = pd.read_csv(path)
        cols = [c for c in PLANNER_KEY_COLS if c in df.columns]
        slim = df[cols].copy()
        slim["experiment"] = experiment
        slim["source"] = source
        stem = path.name
        slim["case_file"] = stem
        slim["case_num"] = stem.split("_", 1)[0]
        frames.append(slim)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_planner_match_summary(outputs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment, source_map in RUNS.items():
        if not {"learned", "persistence"}.issubset(source_map):
            continue
        learned = _read_planner_logs(outputs_root, experiment, "learned", source_map["learned"])
        persist = _read_planner_logs(outputs_root, experiment, "persistence", source_map["persistence"])
        if learned.empty or persist.empty:
            continue
        key_cols = ["case_num", "history_end"]
        merged = learned.merge(
            persist,
            on=key_cols,
            suffixes=("_learned", "_persistence"),
            how="inner",
        )
        if merged.empty:
            continue
        row: dict[str, object] = {
            "experiment": experiment,
            "matched_buckets": int(len(merged)),
            "first_action_match_ratio": float(
                (merged["first_action_learned"] == merged["first_action_persistence"]).mean()
            ),
            "best_sequence_match_ratio": float(
                (merged["best_sequence_learned"] == merged["best_sequence_persistence"]).mean()
            )
            if "best_sequence_learned" in merged.columns
            else np.nan,
            "target_refresh_match_ratio": float(
                (
                    merged.get("prediction_primary_target_refreshed_learned", pd.Series(np.nan, index=merged.index))
                    == merged.get("prediction_primary_target_refreshed_persistence", pd.Series(np.nan, index=merged.index))
                ).mean()
            ),
            "suppression_active_match_ratio": float(
                (
                    merged.get("pump_suppression_active_learned", pd.Series(np.nan, index=merged.index))
                    == merged.get("pump_suppression_active_persistence", pd.Series(np.nan, index=merged.index))
                ).mean()
            ),
            "suppression_reason_match_ratio": float(
                (
                    merged.get("pump_suppression_reason_learned", pd.Series(np.nan, index=merged.index))
                    == merged.get("pump_suppression_reason_persistence", pd.Series(np.nan, index=merged.index))
                ).mean()
            ),
            "mean_abs_delta_pressure_block0_norm": float(
                np.mean(
                    np.abs(
                        merged.get("pressure_block0_norm_learned", pd.Series(0.0, index=merged.index))
                        - merged.get("pressure_block0_norm_persistence", pd.Series(0.0, index=merged.index))
                    )
                )
            ),
            "mean_abs_delta_pressure_block2_norm": float(
                np.mean(
                    np.abs(
                        merged.get("pressure_block2_norm_learned", pd.Series(0.0, index=merged.index))
                        - merged.get("pressure_block2_norm_persistence", pd.Series(0.0, index=merged.index))
                    )
                )
            ),
            "learned_event_prob_max_mean": float(
                merged[
                    [
                        c
                        for c in (
                            "event_risk_prob_0_20m_learned",
                            "event_risk_prob_20_40m_learned",
                            "event_risk_prob_40_60m_learned",
                        )
                        if c in merged.columns
                    ]
                ].max(axis=1).mean()
            )
            if any(c.startswith("event_risk_prob_") for c in merged.columns)
            else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_event_prob_summary(audit_path: Path) -> pd.DataFrame:
    if not audit_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(audit_path)
    event_cols = [
        "attention_event_0_20m",
        "attention_event_20_40m",
        "attention_event_40_60m",
        "ballast_attention_event",
        "direction_shift_ge_45deg",
    ]
    cols = [c for c in event_cols if c in df.columns]
    rows: list[dict[str, object]] = []
    for case_id, group in df.groupby("case_id"):
        row: dict[str, object] = {"case_id": case_id, "buckets": int(len(group))}
        for col in cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_max"] = float(group[col].max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    run_summary: pd.DataFrame,
    pair_deltas: pd.DataFrame,
    planner_match: pd.DataFrame,
    event_probs: pd.DataFrame,
) -> None:
    base_delta = pair_deltas[
        (pair_deltas["experiment"] == "base")
        & (pair_deltas["pair"] == "learned_minus_persistence")
    ].iloc[0]
    chain_pair = REPO_ROOT / "outputs/wind_prediction/forecast_control_chain_audit_4case_1h/forecast_control_chain_pair_summary.csv"
    chain = pd.read_csv(chain_pair) if chain_pair.exists() else pd.DataFrame()
    chain_all = chain.loc[chain["case_id"].eq("ALL")].iloc[0] if not chain.empty else None
    parity_path = (
        REPO_ROOT
        / "outputs/wind_prediction/forecast_adapter_parity_20260506/adapter_training_path_parity.csv"
    )
    scaler_path = (
        REPO_ROOT
        / "outputs/wind_prediction/forecast_adapter_parity_20260506/residual_scaler_contract.csv"
    )
    parity = pd.read_csv(parity_path) if parity_path.exists() else pd.DataFrame()
    scaler_contract = pd.read_csv(scaler_path) if scaler_path.exists() else pd.DataFrame()
    max_adapter_uv_diff = float(parity["max_abs_uv_diff_ms"].max()) if not parity.empty else np.nan
    max_adapter_event_diff = (
        float(parity["max_abs_event_prob_diff"].max()) if not parity.empty else np.nan
    )
    max_scaler_mean_diff = (
        float(scaler_contract["mean_diff"].abs().max()) if not scaler_contract.empty else np.nan
    )
    max_scaler_std_diff = (
        float(scaler_contract["std_diff"].abs().max()) if not scaler_contract.empty else np.nan
    )

    lines = [
        "# Forecast Ablation Diagnosis",
        "",
        "## Main Finding",
        "",
        (
            "learned 与 persistence 的闭环结果接近，不是因为 forecast 接口完全没接上；"
            "主要原因是 learned 的连续风矢量预测在这些 4 个代表工况中只比 persistence 略好，"
            "并且经过 pressure proxy、离散动作集合、suppression 判据和 target 缓存后被压成了相同 first_action。"
        ),
        "",
        "## Key Numbers",
        "",
        f"- base 4case/1h learned - persistence mean primary pump: `{base_delta['mean_primary_pump_m3']:+.3f} m3`.",
        f"- base 4case/1h learned - persistence mean pump-delta: `{base_delta['mean_d_pump_work_pct_point']:+.3f} pct-point`.",
        f"- adapter parity max uv/event diff: `{max_adapter_uv_diff:.8g} m/s` / `{max_adapter_event_diff:.8g}`.",
        f"- residual scaler mean/std diff: `{max_scaler_mean_diff:.8g}` / `{max_scaler_std_diff:.8g}`.",
    ]
    if chain_all is not None:
        lines.extend(
            [
                f"- chain audit learned/persistence action match: `{chain_all['learned_vs_persistence_action_match_ratio'] * 100:.1f}%`.",
                f"- chain audit learned/oracle action match: `{chain_all['learned_vs_oracle_action_match_ratio'] * 100:.1f}%`.",
                f"- chain audit learned RMSE vs oracle: `{chain_all['learned_uv_rmse_vs_oracle_ms']:.2f} m/s`; persistence vs oracle: `{chain_all['persistence_uv_rmse_vs_oracle_ms']:.2f} m/s`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Run Summary",
            "",
            "| experiment | source | mean pump m3 | pump delta | d_pitch_p95 | d_roll_p95 | latch | suppression | event scale |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in run_summary.iterrows():
        lines.append(
            f"| {r['experiment']} | {r['source']} | {r['mean_primary_pump_m3']:.1f} | "
            f"{r['mean_d_pump_work_pct']:+.1f}% | {r['mean_d_pitch_p95_deg']:+.3f} | "
            f"{r['mean_d_roll_p95_deg']:+.3f} | {r['mean_primary_latch']:.1f} | "
            f"{r['mean_suppression_ratio'] * 100:.1f}% | {r['mean_event_risk_scale']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Learned vs Persistence Deltas",
            "",
            "| experiment | d pump m3 | d pump-delta pct-point | d_pitch_p95 | d_roll_p95 | d latch | case06 d pump m3 | case09 d pump m3 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lp = pair_deltas[pair_deltas["pair"].eq("learned_minus_persistence")]
    for _, r in lp.iterrows():
        lines.append(
            f"| {r['experiment']} | {r['mean_primary_pump_m3']:+.3f} | "
            f"{r['mean_d_pump_work_pct_point']:+.3f} | {r['mean_d_pitch_p95_deg']:+.3f} | "
            f"{r['mean_d_roll_p95_deg']:+.3f} | {r['mean_primary_latch']:+.1f} | "
            f"{r['case06_primary_pump_m3']:+.3f} | {r['case09_primary_pump_m3']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Planner Match Summary",
            "",
            "| experiment | buckets | action match | sequence match | refresh match | suppression reason match | abs d block0 | abs d block2 | learned event max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in planner_match.iterrows():
        lines.append(
            f"| {r['experiment']} | {int(r['matched_buckets'])} | "
            f"{r['first_action_match_ratio'] * 100:.1f}% | {r['best_sequence_match_ratio'] * 100:.1f}% | "
            f"{r['target_refresh_match_ratio'] * 100:.1f}% | {r['suppression_reason_match_ratio'] * 100:.1f}% | "
            f"{r['mean_abs_delta_pressure_block0_norm']:.3f} | {r['mean_abs_delta_pressure_block2_norm']:.3f} | "
            f"{r['learned_event_prob_max_mean']:.3f} |"
        )
    if not event_probs.empty:
        lines.extend(
            [
                "",
                "## Learned Event Head",
                "",
                "| case | buckets | attn 0-20 mean | attn 20-40 mean | attn 40-60 mean | ballast event mean |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, r in event_probs.iterrows():
            lines.append(
                f"| {r['case_id']} | {int(r['buckets'])} | "
                f"{r.get('attention_event_0_20m_mean', np.nan):.3f} | "
                f"{r.get('attention_event_20_40m_mean', np.nan):.3f} | "
                f"{r.get('attention_event_40_60m_mean', np.nan):.3f} | "
                f"{r.get('ballast_attention_event_mean', np.nan):.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. forecast adapter 是通的：planner log 能区分 `oracle_future`、`lstm_dual_head_preview`、`persistence_last_10min_mean`。",
            "2. adapter 与训练脚本的 checkpoint 推理路径逐样本一致，残差缩放契约也成立，因此不是反归一化或接线错误。",
            "3. learned UV 回归在这些工况中被明显拉向 persistence，尤其 onset 中 learned 预测的未来增风幅度远低于 oracle。",
            "4. planner 把连续风预测变成 3 个 pressure block，再选 5 类离散动作；一旦 learned/persistence 落在同一阈值区间，pump target 就相同。",
            "5. suppression / target reuse 进一步降低 forecast 细节对 1Hz pump_total 的影响。",
            "6. event head 确实有信号；event-risk boost/guard 能让 learned 与 persistence 分开，但 blunt boost 会改变泵量和姿态取舍，不适合作为当前主算法直接采用。",
            "",
            "## Files",
            "",
            f"- run summary: `{(out_dir / 'forecast_ablation_run_summary.csv').relative_to(REPO_ROOT)}`",
            f"- pair deltas: `{(out_dir / 'forecast_ablation_pair_deltas.csv').relative_to(REPO_ROOT)}`",
            f"- planner match: `{(out_dir / 'forecast_ablation_planner_match_summary.csv').relative_to(REPO_ROOT)}`",
            f"- event probs: `{(out_dir / 'learned_event_probability_summary.csv').relative_to(REPO_ROOT)}`",
            "- adapter parity: `outputs/wind_prediction/forecast_adapter_parity_20260506/forecast_adapter_parity_report.md`",
        ]
    )
    (out_dir / "forecast_ablation_diagnosis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs_root = _resolve(args.outputs_root)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_summary = build_run_summary(outputs_root)
    pair_deltas = build_pair_deltas(run_summary)
    planner_match = build_planner_match_summary(outputs_root)
    event_probs = build_event_prob_summary(
        outputs_root / "forecast_control_chain_audit_4case_1h" / "learned_event_prob_audit.csv"
    )

    run_summary.to_csv(out_dir / "forecast_ablation_run_summary.csv", index=False)
    pair_deltas.to_csv(out_dir / "forecast_ablation_pair_deltas.csv", index=False)
    planner_match.to_csv(out_dir / "forecast_ablation_planner_match_summary.csv", index=False)
    event_probs.to_csv(out_dir / "learned_event_probability_summary.csv", index=False)
    write_report(out_dir, run_summary, pair_deltas, planner_match, event_probs)
    print(f"Report: {out_dir / 'forecast_ablation_diagnosis_report.md'}")


if __name__ == "__main__":
    main()
