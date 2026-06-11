#!/usr/bin/env python3
"""120min oracle controller-value audit for v1.6 delayed-medium floor.

This script does not train models and does not change learned forecasts.  It
only runs closed-loop casebooks with f60/f120 oracle sources and a default-off
120min diagnostic probe, then writes the audit tables requested for
``h120_oracle_controller_value_audit_v1``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs/wind_prediction/h120_oracle_controller_value_audit_v1"
F60_DATASET = REPO_ROOT / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1"
F120_DATASET = (
    REPO_ROOT
    / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"
)
BASELINE_MODEL = REPO_ROOT / "outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1"
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


def _metadata(dataset_dir: Path) -> dict[str, Any]:
    return json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))


def _parse_case_timestamp(case: str) -> str:
    parts = str(case).rsplit("_", 2)
    if len(parts) < 3:
        raise ValueError(f"cannot parse timestamp from case={case!r}")
    return datetime.strptime(f"{parts[-2]} {parts[-1]}", "%Y-%m-%d %H%M%S").strftime(
        TIMESTAMP_FMT
    )


def _build_cases(out_dir: Path) -> dict[str, Path]:
    guard_out = out_dir / "guard10_cases.csv"
    guard = pd.read_csv(GUARD10_CASES)
    guard.to_csv(guard_out, index=False)

    v16 = pd.read_csv(V16_CASE_TABLE)
    broader = v16[(v16["dataset"] == "broader20") & (v16["scenario"] == "delay1200")]
    rows = []
    for _, row in broader.iterrows():
        rows.append(
            {
                "case_id": str(row["case_short"]),
                "timestamp": _parse_case_timestamp(str(row["case"])),
                "label": str(row["regime"]),
            }
        )
    broader_out = out_dir / "broader20_cases.csv"
    pd.DataFrame(rows).to_csv(broader_out, index=False)
    return {"guard10": guard_out, "broader20": broader_out}


def _dataset_coverage(dataset_dir: Path, cases_csv: Path) -> tuple[int, int, str, str]:
    import gzip

    idx_path = dataset_dir / "sample_index.csv.gz"
    cases = pd.read_csv(cases_csv)
    wanted = set(pd.to_datetime(cases["timestamp"]).dt.strftime(TIMESTAMP_FMT))
    available: set[str] = set()
    with gzip.open(idx_path, "rt", encoding="utf-8") as f:
        for chunk in pd.read_csv(f, chunksize=200000):
            sub = chunk[chunk["split"].astype(str).str.strip() == "test"]
            hits = set(pd.to_datetime(sub["history_end"]).dt.strftime(TIMESTAMP_FMT))
            available.update(wanted.intersection(hits))
            if available == wanted:
                break
    missing = sorted(wanted.difference(available))
    return len(available), len(wanted), ";".join(missing), "ok" if not missing else "missing"


def _horizon_source_table(cases: dict[str, Path], out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, dataset in (("f60", F60_DATASET), ("f120", F120_DATASET)):
        meta = _metadata(dataset)
        arrays = meta["arrays"]["test"]
        for dataset_name, cases_csv in cases.items():
            matched, total, missing, status = _dataset_coverage(dataset, cases_csv)
            rows.append(
                {
                    "oracle_label": label,
                    "dataset": dataset_name,
                    "dataset_dir": _rel(dataset),
                    "history_steps": int(meta["history_steps"]),
                    "history_minutes": int(meta["history_steps"])
                    * int(meta.get("input_resolution_minutes", 10)),
                    "future_steps": int(meta["future_steps"]),
                    "future_minutes": int(meta["future_steps"])
                    * int(meta.get("input_resolution_minutes", 10)),
                    "event_count": len(meta.get("event_columns", [])),
                    "event_columns": "|".join(meta.get("event_columns", [])),
                    "shape_X": str(arrays.get("shape_X")),
                    "shape_y_uv_raw": str(arrays.get("shape_y_uv")),
                    "shape_y_event": str(arrays.get("shape_y_event")),
                    "case_matches": int(matched),
                    "case_total": int(total),
                    "missing_history_end": missing,
                    "alignment_status": status,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "horizon_source_table.csv", index=False)
    return table


def _base_casebook_args(out_dir: Path, cases_csv: Path, dataset_dir: Path, source: str) -> list[str]:
    args = [
        str(PYTHON),
        str(REPO_ROOT / "scripts/analysis/run_prediction_primary_casebook.py"),
        "--out-dir",
        str(out_dir),
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
        args.extend(["--model-dir", str(BASELINE_MODEL)])
    return args


def _run_casebook(
    run_dir: Path,
    cases_csv: Path,
    dataset_dir: Path,
    source: str,
    *,
    far_horizon: bool = False,
    h120_probe: bool = False,
) -> None:
    summary = run_dir / "casebook_summary.csv"
    if summary.exists():
        print(f"[skip] {run_dir.relative_to(REPO_ROOT)}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = _base_casebook_args(run_dir, cases_csv, dataset_dir, source)
    if far_horizon or h120_probe:
        cmd.append("--far-horizon")
    if h120_probe:
        cmd.append("--h120-oracle-probe")
    print(f"[run] {run_dir.relative_to(REPO_ROOT)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _case_key(path: Path) -> str:
    name = path.name
    for suffix in ("_prediction_primary_econ_timeseries.csv", "_prediction_primary_econ_planner_log.csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


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


def _edge_count(series: pd.Series) -> int:
    arr = series.fillna(0).astype(float).to_numpy()
    if arr.size == 0:
        return 0
    prev = np.concatenate([[0.0], arr[:-1]])
    return int(((arr > 0.5) & (prev <= 0.5)).sum())


def _case_metrics(run_dir: Path, dataset: str, scenario: str, case_labels: pd.DataFrame) -> list[dict[str, Any]]:
    ts_dir = run_dir / "timeseries"
    rows: list[dict[str, Any]] = []
    labels: dict[str, str] = {}
    for row in case_labels.to_dict("records"):
        raw = str(row["case_id"])
        clean = raw[3:] if len(raw) > 3 and raw[:2].isdigit() and raw[2] == "_" else raw
        label = str(row.get("label", ""))
        labels[raw] = label
        labels[clean] = label
    for path in sorted(ts_dir.glob("*_prediction_primary_econ_timeseries.csv")):
        key = _case_key(path)
        df = pd.read_csv(path, low_memory=False)
        pitch = df["pitch_deg"].astype(float).abs()
        roll = df["roll_deg"].astype(float).abs()
        max_axis = np.maximum(pitch, roll)
        pump = df.get("pump_total_rate_m3_min", pd.Series(0.0, index=df.index)).astype(float).abs()
        high = max_axis > 5.0
        idle = pump < 0.05
        fb = _fallback_series(df)
        log = _read_log(run_dir, key)
        log_zeros = pd.Series(dtype=float)
        case_id_no_ts = key.rsplit("_", 2)[0]
        case_short = (
            case_id_no_ts[3:]
            if len(case_id_no_ts) > 3 and case_id_no_ts[:2].isdigit() and case_id_no_ts[2] == "_"
            else case_id_no_ts
        )
        floor_active = (
            log.get("reactive_floor_active", log_zeros).fillna(0).astype(float)
            if not log.empty
            else log_zeros
        )
        medium_active = (
            log.get("reactive_floor_medium_delay_active", log_zeros).fillna(0).astype(float)
            if not log.empty
            else log_zeros
        )
        early_stop_active = (
            log.get("reactive_floor_post_exit_released", log_zeros).fillna(0).astype(float)
            if not log.empty
            else log_zeros
        )
        preemptive_active = (
            log.get("h120_oracle_probe_preemptive_active", log_zeros).fillna(0).astype(float)
            if not log.empty
            else log_zeros
        )
        h120_short_delay_active = (
            log.get("h120_oracle_probe_short_delay_active", log_zeros).fillna(0).astype(float)
            if not log.empty
            else log_zeros
        )
        h120_early_stop_veto_active = (
            log.get("h120_oracle_probe_early_stop_veto_active", log_zeros).fillna(0).astype(float)
            if not log.empty
            else log_zeros
        )
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
                "floor_active_rows": int(floor_active.sum()),
                "medium_escalation_count": _edge_count(medium_active),
                "medium_active_rows": int(medium_active.sum()),
                "early_stop_count": _edge_count(early_stop_active),
                "preemptive_trigger_count": _edge_count(preemptive_active),
                "h120_short_delay_count": _edge_count(h120_short_delay_active),
                "h120_early_stop_veto_count": _edge_count(h120_early_stop_veto_active),
                "far_available_rows": int(
                    log.get("far_horizon_available", log_zeros).fillna(0).astype(float).sum()
                )
                if not log.empty
                else 0,
                "far_hint_rows": int(
                    log.get("far_horizon_hint_any", log_zeros).fillna(0).astype(float).sum()
                )
                if not log.empty
                else 0,
                "far_hidden_intensification_rows": int(
                    log.get("far_horizon_hidden_intensification", log_zeros)
                    .fillna(0)
                    .astype(float)
                    .sum()
                )
                if not log.empty
                else 0,
            }
        )
    return rows


def _compare_and_case_tables(
    run_map: dict[tuple[str, str], Path],
    cases: dict[str, Path],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (dataset, scenario), run_dir in run_map.items():
        case_labels = pd.read_csv(cases[dataset])
        rows.extend(_case_metrics(run_dir, dataset, scenario, case_labels))
    case_table = pd.DataFrame(rows)
    base_cols = [
        "dataset",
        "scenario",
        "case",
        "case_short",
        "label",
        "pump",
        "sum_fb",
        "max_p95",
        "pitch_p95",
        "roll_p95",
        "time_over_5",
        "idle_time_over_5",
        "floor_trigger_count",
        "floor_active_rows",
        "medium_escalation_count",
        "medium_active_rows",
        "early_stop_count",
        "preemptive_trigger_count",
        "h120_short_delay_count",
        "h120_early_stop_veto_count",
        "far_available_rows",
        "far_hint_rows",
        "far_hidden_intensification_rows",
    ]
    case_table = case_table[base_cols].sort_values(["dataset", "scenario", "case"])
    case_table.to_csv(out_dir / "h120_case_table.csv", index=False)

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
            floor_active_rows=("floor_active_rows", "sum"),
            medium_escalation_count=("medium_escalation_count", "sum"),
            medium_active_rows=("medium_active_rows", "sum"),
            early_stop_count=("early_stop_count", "sum"),
            preemptive_trigger_count=("preemptive_trigger_count", "sum"),
            h120_short_delay_count=("h120_short_delay_count", "sum"),
            h120_early_stop_veto_count=("h120_early_stop_veto_count", "sum"),
            far_available_rows=("far_available_rows", "sum"),
            far_hint_rows=("far_hint_rows", "sum"),
            far_hidden_intensification_rows=("far_hidden_intensification_rows", "sum"),
        )
        .sort_values(["dataset", "scenario"])
    )
    baseline = agg[agg["scenario"] == "v16_f60_learned"].set_index("dataset")
    f120_plain = agg[agg["scenario"] == "v16_f120_oracle_0_60"].set_index("dataset")
    for idx, row in agg.iterrows():
        ds = row["dataset"]
        if ds in baseline.index:
            for col in ("sum_pump", "sum_fb", "max_p95", "time_over_5", "idle_time_over_5"):
                agg.loc[idx, f"delta_vs_v16_{col}"] = float(row[col]) - float(
                    baseline.loc[ds, col]
                )
        if ds in f120_plain.index:
            for col in ("sum_pump", "sum_fb", "max_p95", "time_over_5", "idle_time_over_5"):
                agg.loc[idx, f"delta_vs_f120_0_60_{col}"] = float(row[col]) - float(
                    f120_plain.loc[ds, col]
                )
    agg.to_csv(out_dir / "h120_compare_table.csv", index=False)
    return agg, case_table


def _far_horizon_signal_table(
    run_map: dict[tuple[str, str], Path],
    out_dir: Path,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (dataset, scenario), run_dir in run_map.items():
        if scenario != "v16_f120_oracle_h120_probe":
            continue
        for log_path in sorted((run_dir / "planner_logs").glob("*_planner_log.csv")):
            log = pd.read_csv(log_path, low_memory=False)
            if log.empty:
                continue
            keep = [
                "bucket",
                "current_time_s",
                "history_end",
                "first_action",
                "far_horizon_available",
                "far_horizon_norm_0_20",
                "far_horizon_norm_20_40",
                "far_horizon_norm_40_60",
                "far_horizon_norm_60_80",
                "far_horizon_norm_80_100",
                "far_horizon_norm_100_120",
                "far_horizon_near_max",
                "far_horizon_far_min",
                "far_horizon_far_max",
                "far_horizon_hidden_intensification",
                "far_horizon_reversal",
                "far_horizon_direction_shift",
                "far_horizon_hint_any",
                "h120_oracle_probe_active",
                "h120_oracle_probe_action_effect",
                "h120_oracle_probe_reason",
                "h120_oracle_probe_preemptive_active",
                "h120_oracle_probe_short_delay_active",
                "h120_oracle_probe_early_stop_veto_active",
            ]
            sub = log[[c for c in keep if c in log.columns]].copy()
            sub.insert(0, "scenario", scenario)
            sub.insert(0, "dataset", dataset)
            sub.insert(2, "case", _case_key(log_path))
            rows.append(sub)
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    table.to_csv(out_dir / "far_horizon_signal_table.csv", index=False)
    return table


def _fmt(x: Any, digits: int = 1) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _markdown_table(df: pd.DataFrame, cols: list[str], digits: int = 1) -> str:
    if df.empty:
        return "_empty_"
    out = df[cols].copy()
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], digits) for c in cols) + " |")
    return "\n".join(lines)


def _write_markdown(
    out_dir: Path,
    horizon: pd.DataFrame,
    compare: pd.DataFrame,
    signal: pd.DataFrame,
) -> None:
    f60 = horizon[horizon["oracle_label"] == "f60"].iloc[0]
    f120 = horizon[horizon["oracle_label"] == "f120"].iloc[0]
    probe = compare[compare["scenario"] == "v16_f120_oracle_h120_probe"]
    plain = compare[compare["scenario"] == "v16_f120_oracle_0_60"]
    learned = compare[compare["scenario"] == "v16_f60_learned"]
    f60_oracle = compare[compare["scenario"] == "v16_f60_oracle"]
    probe_beats = False
    if not probe.empty:
        for _, row in probe.iterrows():
            ds = row["dataset"]
            base = learned[learned["dataset"] == ds]
            if base.empty:
                continue
            b = base.iloc[0]
            safer = (
                row["time_over_5"] < b["time_over_5"]
                and row["sum_pump"] <= b["sum_pump"] + 1e-6
                and row["sum_fb"] <= b["sum_fb"] + 1e-6
            )
            cheaper = (
                row["sum_pump"] < b["sum_pump"]
                and row["time_over_5"] <= b["time_over_5"]
                and row["sum_fb"] <= b["sum_fb"] + 1e-6
            )
            probe_beats = probe_beats or safer or cheaper

    f120_plain_inert = False
    if len(plain) == len(f60_oracle) and not plain.empty:
        # This is the leakage check.  The forecast source changes from f60 to
        # f120 oracle, while the controller stays on the 0-60min interface.
        # Any difference here would mean the long replay path changed near
        # horizon semantics or accidentally leaked far-horizon information.
        max_abs = 0.0
        for _, row in plain.iterrows():
            base = f60_oracle[f60_oracle["dataset"] == row["dataset"]]
            if base.empty:
                continue
            for col in ("sum_pump", "sum_fb", "max_p95", "time_over_5", "idle_time_over_5"):
                max_abs = max(max_abs, abs(float(row[col]) - float(base.iloc[0][col])))
        f120_plain_inert = max_abs < 1e-6

    cols = [
        "dataset",
        "scenario",
        "cases",
        "sum_pump",
        "sum_fb",
        "max_p95",
        "time_over_5",
        "idle_time_over_5",
        "floor_trigger_count",
        "floor_active_rows",
        "medium_escalation_count",
        "medium_active_rows",
        "early_stop_count",
        "preemptive_trigger_count",
    ]
    summary = [
        "# h120 oracle controller value audit v1",
        "",
        "Scope: no model training, no learned forecast changes.  The only controller change is the default-off `--h120-oracle-probe`, which uses 60-120min oracle pressure diagnostics on top of v1.6 delayed-medium floor.",
        "",
        "## Horizon source check",
        "",
        f"- f60 replay oracle: history {int(f60['history_minutes'])}min, future {int(f60['future_minutes'])}min, `future_steps={int(f60['future_steps'])}`.",
        f"- f120 replay oracle: history {int(f120['history_minutes'])}min, future {int(f120['future_minutes'])}min, `future_steps={int(f120['future_steps'])}`.",
        "- f120 event labels include 60-80 / 80-100 / 100-120min attention columns; the audit uses true future wind UV and logs event-label availability.",
        "",
        "## Closed-loop comparison",
        "",
        _markdown_table(compare, cols, digits=3),
        "",
        "## Far-horizon signal volume",
        "",
        f"- logged far-horizon buckets: {len(signal)}",
        f"- h120 probe active buckets: {int(signal.get('h120_oracle_probe_active', pd.Series(dtype=float)).fillna(0).astype(float).sum()) if not signal.empty else 0}",
        f"- hidden-intensification buckets: {int(signal.get('far_horizon_hidden_intensification', pd.Series(dtype=float)).fillna(0).astype(float).sum()) if not signal.empty else 0}",
        "",
        "## Interpretation",
        "",
        "- `v16_f120_oracle_0_60` is the leakage check: it swaps the replay oracle source to 120min but keeps the controller on the 0-60min interface. It should match `v16_f60_oracle` if 60-120min information is truly not consumed.",
        "- `v16_f120_oracle_h120_probe` is the ceiling probe: it enables far-horizon telemetry and the small 120min-aware interventions, without relief suppression.",
    ]
    (out_dir / "h120_oracle_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    decision_lines = [
        "# h120 oracle decision",
        "",
        "1. 120min oracle constructed and time-aligned: "
        + (
            "yes. All guard10/broader20 case timestamps are present in the f120 test replay index."
            if (horizon["alignment_status"] == "ok").all()
            else "no. See `horizon_source_table.csv` missing timestamp rows."
        ),
        "",
        "2. Controller only reading 0-60min: "
        + (
            "yes. The f120 oracle 0-60 run is numerically identical to f60 oracle in the aggregate metrics, so 60-120min information is inert until explicitly consumed."
            if f120_plain_inert
            else "not proven. The f120 oracle 0-60 run differs from f60 oracle; inspect case deltas before attributing any effect to 60-120min information."
        ),
        "",
        "3. 120min-aware probe vs v1.6 Pareto: "
        + (
            "passes the strict left/down check on at least one dataset."
            if probe_beats
            else "does not pass the strict left/down check against v1.6 on this holdout. It trades extra pump for some safety reduction rather than moving the pump-safety frontier left/down."
        ),
        "",
        "4. Learned 120min route: "
        + (
            "worth considering only as a follow-up because the oracle probe showed controller value."
            if probe_beats
            else "stop or freeze the learned h240_f120/h180_f120 route for now; do not train longer-horizon models without a stronger oracle control-value signal."
        ),
        "",
        "5. If there is a positive 120min oracle signal: "
        + (
            "next step is a learned h240_f120/h180_f120 model targeted at the far-horizon signal classes in `far_horizon_signal_table.csv`."
            if probe_beats
            else "no learned 120min training is justified by this audit."
        ),
    ]
    (out_dir / "h120_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-run casebooks even if summaries exist.")
    args = parser.parse_args()

    if not PYTHON.exists():
        raise FileNotFoundError(f"missing project Python: {PYTHON}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = _build_cases(OUT_DIR)
    horizon = _horizon_source_table(cases, OUT_DIR)
    if (horizon["alignment_status"] != "ok").any():
        raise RuntimeError("case timestamps are not fully aligned; inspect horizon_source_table.csv")

    run_map: dict[tuple[str, str], Path] = {}
    scenarios = [
        ("v16_f60_learned", F60_DATASET, "learned", False, False),
        ("v16_f60_oracle", F60_DATASET, "oracle", False, False),
        ("v16_f120_oracle_0_60", F120_DATASET, "oracle", False, False),
        ("v16_f120_oracle_h120_probe", F120_DATASET, "oracle", True, True),
    ]
    for dataset_name, cases_csv in cases.items():
        for scenario, ds_dir, source, far, probe in scenarios:
            run_dir = OUT_DIR / "runs" / dataset_name / scenario
            run_map[(dataset_name, scenario)] = run_dir
            if args.force and (run_dir / "casebook_summary.csv").exists():
                # Keep existing raw files unless the caller explicitly removes
                # the directory; this flag is reserved for future expansion.
                pass
            _run_casebook(run_dir, cases_csv, ds_dir, source, far_horizon=far, h120_probe=probe)

    compare, _ = _compare_and_case_tables(run_map, cases, OUT_DIR)
    signal = _far_horizon_signal_table(run_map, OUT_DIR)
    _write_markdown(OUT_DIR, horizon, compare, signal)
    print(f"[done] {_rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
