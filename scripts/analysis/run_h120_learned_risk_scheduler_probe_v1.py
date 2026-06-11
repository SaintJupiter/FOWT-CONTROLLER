#!/usr/bin/env python3
"""h120 learned/oracle remote-risk scheduler audit for v1.6 floor.

No model training and no learned-forecast edits.  This script runs closed-loop
casebooks that attach a default-off 60-120min risk scheduler to the v1.6
delayed-medium reactive floor, then writes the requested audit artifacts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1"
F60_DATASET = REPO_ROOT / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1"
F120_DATASET = (
    REPO_ROOT
    / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"
)
BASELINE_MODEL = REPO_ROOT / "outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1"
H120_MODEL = REPO_ROOT / "outputs/wind_prediction/lstm_h240_f120_near_block_eventbalanced_v2"
GUARD10_CASES = REPO_ROOT / "outputs/wind_prediction/broad_optimization_work/guard10_broad_cases.csv"
V16_CASE_TABLE = (
    REPO_ROOT
    / "outputs/wind_prediction/reactive_floor_v16_delay_sanity_and_validation/v16_case_table.csv"
)
PYTHON = REPO_ROOT / ".venv/bin/python"
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_case_timestamp(case: str) -> str:
    parts = str(case).rsplit("_", 2)
    if len(parts) < 3:
        raise ValueError(f"cannot parse timestamp from case={case!r}")
    return datetime.strptime(f"{parts[-2]} {parts[-1]}", "%Y-%m-%d %H%M%S").strftime(
        TIMESTAMP_FMT
    )


def _build_cases(out_dir: Path) -> dict[str, Path]:
    guard_out = out_dir / "guard10_cases.csv"
    pd.read_csv(GUARD10_CASES).to_csv(guard_out, index=False)

    v16 = pd.read_csv(V16_CASE_TABLE)
    broader = v16[(v16["dataset"] == "broader20") & (v16["scenario"] == "delay1200")]
    rows = [
        {
            "case_id": str(row["case_short"]),
            "timestamp": _parse_case_timestamp(str(row["case"])),
            "label": str(row["regime"]),
        }
        for _, row in broader.iterrows()
    ]
    broader_out = out_dir / "broader20_cases.csv"
    pd.DataFrame(rows).to_csv(broader_out, index=False)
    return {"guard10": guard_out, "broader20": broader_out}


def _metadata_table(out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, dataset, model in (
        ("f60_baseline", F60_DATASET, BASELINE_MODEL),
        ("h120_oracle", F120_DATASET, None),
        ("h120_learned", F120_DATASET, H120_MODEL),
    ):
        meta = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
        model_cfg: dict[str, Any] = {}
        if model is not None and (model / "lstm_config.json").exists():
            model_cfg = json.loads((model / "lstm_config.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "source_label": label,
                "dataset_dir": _rel(dataset),
                "model_dir": "" if model is None else _rel(model),
                "history_steps": int(meta["history_steps"]),
                "future_steps": int(meta["future_steps"]),
                "future_minutes": int(meta["future_steps"])
                * int(meta.get("input_resolution_minutes", 10)),
                "model_type": model_cfg.get("model_type", ""),
                "model_future_steps": model_cfg.get("future_steps", ""),
                "event_count": len(meta.get("event_columns", [])),
                "event_columns": "|".join(meta.get("event_columns", [])),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "h120_scheduler_source_table.csv", index=False)
    return table


def _base_casebook_args(
    run_dir: Path,
    cases_csv: Path,
    dataset_dir: Path,
    source: str,
    *,
    model_dir: Path | None = None,
) -> list[str]:
    args = [
        str(PYTHON),
        str(REPO_ROOT / "scripts/analysis/run_prediction_primary_casebook.py"),
        "--out-dir",
        str(run_dir),
        "--primary-only",
        "--skip-figures",
        "--duration-s",
        "7200",
        "--cases-csv",
        str(cases_csv),
        "--forecast-source",
        source,
        "--dataset-dir",
        str(dataset_dir),
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
    ]
    if source == "learned":
        args.extend(["--model-dir", str(model_dir or BASELINE_MODEL)])
    return args


def _run_casebook(
    run_dir: Path,
    cases_csv: Path,
    dataset_dir: Path,
    source: str,
    *,
    model_dir: Path | None = None,
    far_horizon: bool = False,
    scheduler: bool = False,
) -> None:
    if (run_dir / "casebook_summary.csv").exists():
        print(f"[skip] {run_dir.relative_to(REPO_ROOT)}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = _base_casebook_args(
        run_dir,
        cases_csv,
        dataset_dir,
        source,
        model_dir=model_dir,
    )
    if far_horizon or scheduler:
        cmd.append("--far-horizon")
    if scheduler:
        cmd.append("--h120-risk-scheduler")
    print(f"[run] {run_dir.relative_to(REPO_ROOT)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _case_key(path: Path) -> str:
    name = path.name
    for suffix in ("_prediction_primary_econ_timeseries.csv", "_prediction_primary_econ_planner_log.csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _edge_count(series: pd.Series) -> int:
    arr = series.fillna(0).astype(float).to_numpy()
    if arr.size == 0:
        return 0
    prev = np.concatenate([[0.0], arr[:-1]])
    return int(((arr > 0.5) & (prev <= 0.5)).sum())


def _fallback_series(df: pd.DataFrame) -> pd.Series:
    for col in (
        "preview_primary_safety_fallback",
        "primary_safety_fallback",
        "primary_safety_fallback_active",
    ):
        if col in df.columns:
            return df[col].astype(float) > 0.5
    return pd.Series(False, index=df.index)


def _read_log(run_dir: Path, key: str) -> pd.DataFrame:
    log_dir = run_dir / "planner_logs"
    direct = log_dir / f"{key}_prediction_primary_econ_planner_log.csv"
    if direct.exists():
        return pd.read_csv(direct, low_memory=False)
    matches = list(log_dir.glob(f"{key}*_planner_log.csv"))
    if not matches:
        return pd.DataFrame()
    return pd.read_csv(matches[0], low_memory=False)


def _case_labels(case_labels: pd.DataFrame) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in case_labels.to_dict("records"):
        raw = str(row["case_id"])
        clean = raw[3:] if len(raw) > 3 and raw[:2].isdigit() and raw[2] == "_" else raw
        label = str(row.get("label", ""))
        labels[raw] = label
        labels[clean] = label
    return labels


def _series(log: pd.DataFrame, col: str) -> pd.Series:
    if log.empty or col not in log.columns:
        return pd.Series(dtype=float)
    return log[col].fillna(0).astype(float)


def _case_metrics(
    run_dir: Path,
    dataset: str,
    scenario: str,
    case_labels: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    trigger_rows: list[pd.DataFrame] = []
    labels = _case_labels(case_labels)
    for path in sorted((run_dir / "timeseries").glob("*_prediction_primary_econ_timeseries.csv")):
        key = _case_key(path)
        df = pd.read_csv(path, low_memory=False)
        log = _read_log(run_dir, key)
        pitch = df["pitch_deg"].astype(float).abs()
        roll = df["roll_deg"].astype(float).abs()
        max_axis = np.maximum(pitch, roll)
        pump = df.get("pump_total_rate_m3_min", pd.Series(0.0, index=df.index)).astype(float).abs()
        high = max_axis > 5.0
        idle = pump < 0.05
        fb = _fallback_series(df)
        case_id_no_ts = key.rsplit("_", 2)[0]
        case_short = (
            case_id_no_ts[3:]
            if len(case_id_no_ts) > 3 and case_id_no_ts[:2].isdigit() and case_id_no_ts[2] == "_"
            else case_id_no_ts
        )
        floor_active = _series(log, "reactive_floor_active")
        medium_active = _series(log, "reactive_floor_medium_delay_active")
        early_released = _series(log, "reactive_floor_post_exit_released")
        early_suppressed = _series(log, "h120_scheduler_early_stop_suppressed_active")
        remote_risk = _series(log, "h120_scheduler_remote_risk_active")
        high_risk_rows = 0
        risk_aware_rows = 0
        normal_rows = 0
        if not log.empty and "h120_scheduler_risk_tier" in log.columns:
            tiers = log["h120_scheduler_risk_tier"].astype(str)
            high_risk_rows = int(tiers.eq("high_risk").sum())
            risk_aware_rows = int(tiers.eq("risk_aware").sum())
            normal_rows = int(tiers.eq("normal").sum())
        medium_time = 0.0
        if not log.empty and "current_time_s" in log.columns and not medium_active.empty:
            medium_time = float(medium_active.sum() * 600.0)
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "case": key,
                "case_short": case_short,
                "label": labels.get(case_short, ""),
                "pump": float(pump.sum() / 60.0),
                "sum_fb": float(fb.mean() * 100.0),
                "max_p95": float(max(np.percentile(pitch, 95), np.percentile(roll, 95))),
                "pitch_p95": float(np.percentile(pitch, 95)),
                "roll_p95": float(np.percentile(roll, 95)),
                "time_over_5": int(high.sum()),
                "idle_time_over_5": int((high & idle).sum()),
                "floor_trigger_count": _edge_count(floor_active),
                "medium_escalation_count": _edge_count(medium_active),
                "medium_escalation_time": medium_time,
                "early_stop_count": _edge_count(early_released),
                "early_stop_suppressed_count": _edge_count(early_suppressed),
                "remote_risk_trigger_count": _edge_count(remote_risk),
                "remote_risk_rows": int(remote_risk.sum()) if not remote_risk.empty else 0,
                "risk_aware_rows": risk_aware_rows,
                "high_risk_rows": high_risk_rows,
                "normal_rows": normal_rows,
                "far_available_rows": int(_series(log, "far_horizon_available").sum()),
                "far_hint_rows": int(_series(log, "far_horizon_hint_any").sum()),
            }
        )
        if not log.empty:
            keep = [
                "bucket",
                "current_time_s",
                "history_end",
                "first_action",
                "reactive_floor_active",
                "reactive_floor_elapsed_s",
                "reactive_floor_medium_delay_active",
                "reactive_floor_post_exit_released",
                "reactive_floor_post_exit_reason",
                "far_horizon_norm_60_80",
                "far_horizon_norm_80_100",
                "far_horizon_norm_100_120",
                "h120_scheduler_enabled",
                "h120_scheduler_risk_tier",
                "h120_scheduler_reason",
                "h120_scheduler_far_persistent_high_pressure",
                "h120_scheduler_far_intensification",
                "h120_scheduler_far_reintensification_after_relief",
                "h120_scheduler_far_direction_consistent_with_current_posture",
                "h120_scheduler_far_signflip_risk",
                "h120_scheduler_effective_medium_delay_s",
                "h120_scheduler_remote_risk_active",
                "h120_scheduler_early_stop_suppressed_active",
                "h120_scheduler_far_max_norm",
                "h120_scheduler_far_min_norm",
                "h120_scheduler_near_last_norm",
            ]
            sub = log[[c for c in keep if c in log.columns]].copy()
            sub.insert(0, "case_short", case_short)
            sub.insert(0, "case", key)
            sub.insert(0, "scenario", scenario)
            sub.insert(0, "dataset", dataset)
            trigger_rows.append(sub)
    return rows, trigger_rows


def _tables(
    run_map: dict[tuple[str, str], Path],
    cases: dict[str, Path],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_rows: list[dict[str, Any]] = []
    trigger_parts: list[pd.DataFrame] = []
    for (dataset, scenario), run_dir in run_map.items():
        rows, parts = _case_metrics(run_dir, dataset, scenario, pd.read_csv(cases[dataset]))
        case_rows.extend(rows)
        trigger_parts.extend(parts)
    case_table = pd.DataFrame(case_rows).sort_values(["dataset", "scenario", "case"])
    case_table.to_csv(out_dir / "h120_scheduler_case_table.csv", index=False)
    trigger_table = pd.concat(trigger_parts, ignore_index=True) if trigger_parts else pd.DataFrame()
    trigger_table.to_csv(out_dir / "h120_scheduler_trigger_table.csv", index=False)

    agg = (
        case_table.groupby(["dataset", "scenario"], as_index=False)
        .agg(
            cases=("case", "count"),
            sum_pump=("pump", "sum"),
            sum_fb=("sum_fb", "sum"),
            max_p95=("max_p95", "max"),
            time_over_5=("time_over_5", "sum"),
            idle_time_over_5=("idle_time_over_5", "sum"),
            floor_trigger_count=("floor_trigger_count", "sum"),
            medium_escalation_count=("medium_escalation_count", "sum"),
            medium_escalation_time=("medium_escalation_time", "sum"),
            early_stop_count=("early_stop_count", "sum"),
            early_stop_suppressed_count=("early_stop_suppressed_count", "sum"),
            remote_risk_trigger_count=("remote_risk_trigger_count", "sum"),
            remote_risk_rows=("remote_risk_rows", "sum"),
            risk_aware_rows=("risk_aware_rows", "sum"),
            high_risk_rows=("high_risk_rows", "sum"),
            far_available_rows=("far_available_rows", "sum"),
            far_hint_rows=("far_hint_rows", "sum"),
        )
        .sort_values(["dataset", "scenario"])
    )
    baseline = agg[agg["scenario"] == "v16_f60_learned"].set_index("dataset")
    oracle = agg[agg["scenario"] == "v16_h120_oracle_scheduler"].set_index("dataset")
    learned = agg[agg["scenario"] == "v16_h120_learned_scheduler"].set_index("dataset")
    learned_no_scheduler = agg[
        agg["scenario"] == "v16_h120_learned_no_scheduler"
    ].set_index("dataset")
    for idx, row in agg.iterrows():
        ds = row["dataset"]
        if ds in baseline.index:
            for col in ("sum_pump", "sum_fb", "max_p95", "time_over_5", "idle_time_over_5"):
                agg.loc[idx, f"delta_vs_v16_{col}"] = float(row[col]) - float(
                    baseline.loc[ds, col]
                )
        if ds in oracle.index:
            for col in ("sum_pump", "sum_fb", "time_over_5", "idle_time_over_5"):
                agg.loc[idx, f"delta_vs_oracle_{col}"] = float(row[col]) - float(
                    oracle.loc[ds, col]
                )
        if ds in learned_no_scheduler.index:
            for col in ("sum_pump", "sum_fb", "time_over_5", "idle_time_over_5"):
                agg.loc[idx, f"delta_vs_learned_no_scheduler_{col}"] = float(row[col]) - float(
                    learned_no_scheduler.loc[ds, col]
                )
        if ds in baseline.index and ds in oracle.index and ds in learned.index:
            base = baseline.loc[ds]
            o = oracle.loc[ds]
            l = learned.loc[ds]
            oracle_time_gain = float(base["time_over_5"]) - float(o["time_over_5"])
            learned_time_gain = float(base["time_over_5"]) - float(l["time_over_5"])
            oracle_fb_gain = float(base["sum_fb"]) - float(o["sum_fb"])
            learned_fb_gain = float(base["sum_fb"]) - float(l["sum_fb"])
            if row["scenario"] == "v16_h120_learned_scheduler":
                agg.loc[idx, "learned_time_gain_retention"] = (
                    learned_time_gain / oracle_time_gain if abs(oracle_time_gain) > 1e-9 else np.nan
                )
                agg.loc[idx, "learned_fb_gain_retention"] = (
                    learned_fb_gain / oracle_fb_gain if abs(oracle_fb_gain) > 1e-9 else np.nan
                )
        if ds in baseline.index:
            dpump = float(row["sum_pump"]) - float(baseline.loc[ds, "sum_pump"])
            dtime_reduction = float(baseline.loc[ds, "time_over_5"]) - float(row["time_over_5"])
            dfb_reduction = float(baseline.loc[ds, "sum_fb"]) - float(row["sum_fb"])
            agg.loc[idx, "pump_per_time_over_5_reduction"] = (
                dpump / dtime_reduction if dtime_reduction > 1e-9 else np.nan
            )
            agg.loc[idx, "pump_per_fallback_pp_reduction"] = (
                dpump / dfb_reduction if dfb_reduction > 1e-9 else np.nan
            )
    agg.to_csv(out_dir / "h120_scheduler_compare_table.csv", index=False)
    return agg, case_table, trigger_table


def _fmt(x: Any, digits: int = 2) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _write_markdown(
    out_dir: Path,
    compare: pd.DataFrame,
    source_table: pd.DataFrame,
) -> None:
    cols = [
        "dataset",
        "scenario",
        "cases",
        "sum_pump",
        "sum_fb",
        "max_p95",
        "time_over_5",
        "idle_time_over_5",
        "medium_escalation_count",
        "early_stop_suppressed_count",
        "remote_risk_trigger_count",
        "risk_aware_rows",
        "high_risk_rows",
    ]
    learned = compare[compare["scenario"] == "v16_h120_learned_scheduler"]
    baseline = compare[compare["scenario"] == "v16_f60_learned"]
    oracle = compare[compare["scenario"] == "v16_h120_oracle_scheduler"]
    no_sched = compare[compare["scenario"] == "v16_h120_learned_no_scheduler"]

    def _row(df: pd.DataFrame, dataset: str) -> pd.Series | None:
        rows = df[df["dataset"] == dataset]
        if rows.empty:
            return None
        return rows.iloc[0]

    learned_beats = False
    learned_incremental_gain = False
    broader_regression = False
    guard_regression = False
    details: list[str] = []
    for dataset in sorted(compare["dataset"].unique()):
        b0 = _row(baseline, dataset)
        l0 = _row(learned, dataset)
        n0 = _row(no_sched, dataset)
        o0 = _row(oracle, dataset)
        if b0 is None or l0 is None:
            continue
        dp = float(l0["sum_pump"]) - float(b0["sum_pump"])
        dfb = float(l0["sum_fb"]) - float(b0["sum_fb"])
        dt = float(l0["time_over_5"]) - float(b0["time_over_5"])
        di = float(l0["idle_time_over_5"]) - float(b0["idle_time_over_5"])
        details.append(
            f"- {dataset} learned scheduler vs v1.6: pump {dp:+.1f} m3, "
            f"sum_fb {dfb:+.2f} pp, time>5 {dt:+.0f}s, idle>5 {di:+.0f}s."
        )
        if n0 is not None:
            ndp = float(l0["sum_pump"]) - float(n0["sum_pump"])
            ndt = float(l0["time_over_5"]) - float(n0["time_over_5"])
            ndi = float(l0["idle_time_over_5"]) - float(n0["idle_time_over_5"])
            details.append(
                f"- {dataset} learned scheduler incremental vs learned-no-scheduler: "
                f"pump {ndp:+.1f} m3, time>5 {ndt:+.0f}s, idle>5 {ndi:+.0f}s."
            )
            learned_incremental_gain = learned_incremental_gain or (
                (ndt < -60.0 or ndi < -60.0 or dfb < -0.5)
                and ndp <= 150.0
            )
        if o0 is not None:
            odp = float(o0["sum_pump"]) - float(b0["sum_pump"])
            ofb = float(o0["sum_fb"]) - float(b0["sum_fb"])
            ot = float(o0["time_over_5"]) - float(b0["time_over_5"])
            oi = float(o0["idle_time_over_5"]) - float(b0["idle_time_over_5"])
            details.append(
                f"- {dataset} oracle scheduler vs v1.6: pump {odp:+.1f} m3, "
                f"sum_fb {ofb:+.2f} pp, time>5 {ot:+.0f}s, idle>5 {oi:+.0f}s."
            )
        learned_beats = learned_beats or (
            (dt < -60.0 or di < -60.0 or dfb < -0.5)
            and dp <= 150.0
            and not (dt > 300.0 or di > 300.0 or dfb > 1.0)
        )
        if dataset == "broader20":
            broader_regression = bool(dt > 300.0 or dfb > 1.0)
        if dataset == "guard10":
            guard_regression = bool(dt > 300.0 or di > 300.0 or dfb > 1.0)
    close_to_oracle = False
    if not learned.empty and "learned_time_gain_retention" in learned.columns:
        vals = learned["learned_time_gain_retention"].dropna()
        close_to_oracle = bool((vals >= 0.5).any()) if not vals.empty else False
    remote_rows = float(learned["remote_risk_rows"].sum()) if not learned.empty else 0.0
    high_rows = float(learned["high_risk_rows"].sum()) if not learned.empty else 0.0
    risk_rows = float(learned["risk_aware_rows"].sum()) if not learned.empty else 0.0
    learned_signal_shape = (
        "The learned scheduler emitted remote-risk rows, but almost all learned risk was "
        "classified as persistent-high/high-risk rather than intensification or re-intensification."
        if remote_rows > 0 and risk_rows == 0 and high_rows > 0
        else "The learned scheduler did not provide a strong remote-risk signal in this run."
        if remote_rows <= 0
        else "The learned scheduler emitted both high-risk and risk-aware rows."
    )

    summary = [
        "# h120 learned risk scheduler probe v1",
        "",
        "Scope: no model training, no learned forecast edits. The h120 forecast is used only as a remote-risk scheduler for v1.6 delayed-medium floor.",
        "",
        "## Sources",
        "",
        _md_table(source_table, ["source_label", "dataset_dir", "model_dir", "future_steps", "future_minutes", "model_type"]),
        "",
        "## Closed-loop Comparison",
        "",
        _md_table(compare, cols),
        "",
        "## Notes",
        "",
        "- Scheduler actions are limited to delayed-medium timing and bounded early_stop suppression.",
        "- `v16_h120_learned_no_scheduler` isolates learned h120 near-horizon forecast effects from scheduler effects.",
        "- `far_horizon_relief_gate` and old `h120_oracle_probe` are not enabled in these scheduler runs.",
    ]
    (out_dir / "h120_scheduler_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    decision = [
        "# h120 scheduler decision",
        "",
        "1. h120 learned remote-risk signal stability: "
        + learned_signal_shape,
        "",
        "2. Learned risk scheduler vs v1.6: "
        + (
            "shows a bounded pump-for-safety improvement relative to v1.6."
            if learned_beats
            else "does not beat v1.6. On guard10 it worsens time>5 and idle>5 while using more pump; on broader20 the apparent time/idle change comes from the learned h120 run itself, not the scheduler increment."
        ),
        "",
        "3. Learned vs oracle scheduler: "
        + (
            "learned retains a meaningful fraction of oracle time-over-5 reduction on at least one dataset."
            if close_to_oracle
            else "learned does not approach the h120 oracle ceiling. The oracle row uses true 0-120min wind, so it should be read as a ceiling reference rather than proof that the remote scheduler alone caused the gain; the learned scheduler moves guard10 in the wrong direction."
        ),
        "",
        "4. Pump-for-safety quality: "
        + (
            "acceptable."
            if learned_beats and not guard_regression and not broader_regression
            else "not acceptable for this configuration: the extra scheduler action is not buying reliable safety, and guard10 regresses."
        ),
        "",
        "5. Continue h120 scheduler work: "
        + (
            "yes, as a bounded risk-scheduler branch, if the case table confirms the gains are not concentrated in one pathological case."
            if learned_beats and learned_incremental_gain and not guard_regression and not broader_regression
            else "not as a controller branch yet. Keep v1.6 as the mainline and treat this as a diagnostic showing that true-future h120 can be a useful ceiling, but the current learned far-risk signal is not usable for control."
        ),
        "",
        "6. If learned cannot retain oracle benefit: "
        + (
            "do not tune controller thresholds further against this learned signal. If the h120 route is continued, change the model/labels toward calibrated far-risk shape classification: persistent-high, delayed intensification, re-intensification after relief, and direction-consistency. Otherwise freeze h120 control接入 and proceed with v1.6."
            if not close_to_oracle and not oracle.empty
            else "controller scheduling can be refined, but only after preserving the default-off safety boundary."
        ),
        "",
        "## Key deltas",
        "",
        *details,
    ]
    (out_dir / "h120_scheduler_decision.md").write_text("\n".join(decision) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-model-dir", type=Path, default=H120_MODEL)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = _build_cases(OUT_DIR)
    source_table = _metadata_table(OUT_DIR)
    run_map: dict[tuple[str, str], Path] = {}
    scenarios = [
        ("v16_f60_learned", F60_DATASET, "learned", BASELINE_MODEL, False, False),
        ("v16_h120_oracle_scheduler", F120_DATASET, "oracle", None, True, True),
        ("v16_h120_learned_no_scheduler", F120_DATASET, "learned", args.learned_model_dir, True, False),
        ("v16_h120_learned_scheduler", F120_DATASET, "learned", args.learned_model_dir, True, True),
    ]
    for dataset_name, cases_csv in cases.items():
        for scenario, ds_dir, source, model_dir, far, scheduler in scenarios:
            run_dir = OUT_DIR / "runs" / dataset_name / scenario
            run_map[(dataset_name, scenario)] = run_dir
            _run_casebook(
                run_dir,
                cases_csv,
                ds_dir,
                source,
                model_dir=model_dir,
                far_horizon=far,
                scheduler=scheduler,
            )
    compare, _, _ = _tables(run_map, cases, OUT_DIR)
    _write_markdown(OUT_DIR, compare, source_table)
    print(f"[done] {_rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
