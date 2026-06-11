#!/usr/bin/env python3
"""Run a small PSC No.4 frame-dispatcher smoke validation.

This uses existing controller profiles only. It does not introduce a new
closed-loop controller; instead it splits the 12 planned smoke cases by frame
and runs the profile that the next dispatcher would select for that frame.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "analysis" / "run_prediction_primary_casebook.py"
BASE = REPO / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
PLAN = BASE / "psc_4hao_warning_frame_extension_20260603"
FRAME_PLAN = BASE / "psc_4hao_frame_optimization_plan_20260603"
DEFAULT_CASEBOOK = PLAN / "next_smoke_casebook.csv"
OUT = BASE / "psc_4hao_frame_dispatcher_smoke_v1"


DISPATCH = {
    "neutral_mhs_broader": {
        "label": "dispatcher_p2_generalized_gate",
        "profile": "psc_4hao_generalized_gate_v1",
        "forecast": "learned",
        "env": {"FOWT_4HAO_ACTIVE_POSTURE_REFRESH": "1"},
        "role": "P2/P3 main economy enable",
    },
    "lowrisk_stable_redundant_candidate": {
        "label": "dispatcher_lowrisk_generalized_gate",
        "profile": "psc_4hao_generalized_gate_v1",
        "forecast": "learned",
        "env": {"FOWT_4HAO_ACTIVE_POSTURE_REFRESH": "1"},
        "role": "small clean economy support",
    },
    "transient_peak_future_decay": {
        "label": "dispatcher_p1_relief_decay",
        "profile": "relief_decay_dispatcher_v2",
        "forecast": "learned",
        "env": {},
        "role": "P1 conditional relief-decay specialist",
    },
    "direction_reversal_boundary": {
        "label": "dispatcher_w1_layered_v8",
        "profile": "direction_reversal_layered_v8",
        "forecast": "learned",
        "env": {},
        "role": "W1 low-posture saving / high-pressure warning split",
    },
    "reintensification_boundary": {
        "label": "dispatcher_w2_baseline_watch",
        "profile": "dc_preserving_deadband_forecast_adaptive_v1",
        "forecast": "current_only",
        "env": {},
        "role": "W2 warning/veto fail-closed",
    },
    "sustained_high_safety_event": {
        "label": "dispatcher_w3_w7_baseline_watch",
        "profile": "dc_preserving_deadband_forecast_adaptive_v1",
        "forecast": "current_only",
        "env": {},
        "role": "W3/W7 baseline/watch fail-closed",
    },
    "quiet_low_opportunity": {
        "label": "dispatcher_quiet_baseline",
        "profile": "dc_preserving_deadband_forecast_adaptive_v1",
        "forecast": "current_only",
        "env": {},
        "role": "quiet no-op false-trigger control",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casebook", type=Path, default=DEFAULT_CASEBOOK)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--mode", choices=("dry-run", "run", "summarize"), default="run")
    parser.add_argument("--duration-s", type=float, default=21600.0)
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1",
    )
    parser.add_argument("--replay-split", choices=("test", "validation", "train"), default="test")
    parser.add_argument("--model-dir", default="outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _strip_numbered(case_id: str) -> str:
    return re.sub(r"^\d+_", "", str(case_id))


def _load_casebook(path: Path) -> pd.DataFrame:
    if not path.is_absolute():
        path = REPO / path
    df = pd.read_csv(path)
    required = {"case_id", "timestamp", "label", "stratum"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"casebook missing columns: {sorted(missing)}")
    unknown = sorted(set(df["stratum"]) - set(DISPATCH))
    if unknown:
        raise ValueError(f"no dispatch rule for strata: {unknown}")
    return df


def _materialize_groups(casebook: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    group_dir = out_dir / "casebooks"
    group_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for stratum, group in casebook.groupby("stratum", sort=False):
        role = DISPATCH[str(stratum)]["role"]
        out = group[["case_id", "timestamp", "label"]].copy()
        out["label"] = (
            out["label"].astype(str)
            + " | dispatcher_stratum="
            + str(stratum)
            + " | dispatcher_role="
            + str(role)
            + " | dispatcher_source_case_id="
            + out["case_id"].astype(str)
        )
        path = group_dir / f"{_safe_name(str(stratum))}_cases.csv"
        out.to_csv(path, index=False)
        paths[str(stratum)] = path
    return paths


def _command(args: argparse.Namespace, stratum: str, cases_csv: Path) -> tuple[list[str], dict[str, str], Path]:
    spec = DISPATCH[stratum]
    out_dir = args.out_dir / "runs" / _safe_name(stratum)
    cmd = [
        sys.executable,
        str(RUNNER),
        "--out-dir",
        str(out_dir),
        "--cases-csv",
        str(cases_csv),
        "--duration-s",
        str(args.duration_s),
        "--dataset-dir",
        str(args.dataset_dir),
        "--replay-split",
        str(args.replay_split),
        "--model-dir",
        str(args.model_dir),
        "--primary-label",
        str(spec["label"]),
        "--primary-only",
        "--skip-figures",
        "--primary-control-profile",
        str(spec["profile"]),
        "--forecast-source",
        str(spec["forecast"]),
    ]
    return cmd, dict(spec["env"]), out_dir


def _run_groups(args: argparse.Namespace, group_paths: dict[str, Path]) -> None:
    for stratum, cases_csv in group_paths.items():
        cmd, env_patch, out_dir = _command(args, stratum, cases_csv)
        summary = out_dir / "casebook_summary.csv"
        prefix = " ".join(f"{k}={v}" for k, v in env_patch.items())
        print(("+ " + prefix + " " if prefix else "+ ") + " ".join(cmd))
        if args.mode != "run":
            continue
        if args.skip_existing and summary.exists():
            print(f"skip existing {stratum}: {summary}")
            continue
        env = os.environ.copy()
        env.update(env_patch)
        subprocess.run(cmd, cwd=REPO, env=env, check=True)


def _extract_label_value(label: str, key: str) -> str:
    marker = f"{key}="
    text = str(label)
    if marker not in text:
        return ""
    return text.split(marker, 1)[1].split("|", 1)[0].strip()


def _find_timeseries(run_dir: Path, case_id: str, label: str) -> Path:
    matches = sorted((run_dir / "timeseries").glob(f"{case_id}_*_{label}_timeseries.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{run_dir}: {case_id}/{label} matches={len(matches)}")
    return matches[0]


def _run_metrics(run_dir: Path, label: str, stratum: str) -> pd.DataFrame:
    summary_path = run_dir / "casebook_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    rows: list[dict[str, Any]] = []
    for rec in summary.to_dict("records"):
        case_id = str(rec["case_id"])
        ts = pd.read_csv(_find_timeseries(run_dir, case_id, label), low_memory=False)
        pitch = pd.to_numeric(ts.get("pitch_deg", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
        roll = pd.to_numeric(ts.get("roll_deg", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
        axis = np.maximum(np.abs(pitch), np.abs(roll))
        fallback = pd.to_numeric(
            ts.get("preview_primary_safety_fallback", pd.Series(0.0, index=ts.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(float)
        raw_label = str(rec.get("label", ""))
        source_case_id = _extract_label_value(raw_label, "dispatcher_source_case_id")
        rows.append(
            {
                "source_case_id": source_case_id or _strip_numbered(case_id),
                "run_case_id": case_id,
                "stratum": stratum,
                "selected_profile": DISPATCH[stratum]["profile"],
                "selected_role": DISPATCH[stratum]["role"],
                "pump_m3": float(rec["primary_pump_work_m3"]),
                "time_gt3_s": float(np.sum(axis > 3.0)),
                "time_gt4_s": float(np.sum(axis > 4.0)),
                "time_gt5_s": float(np.sum(axis > 5.0)),
                "time_gt7_s": float(np.sum(axis > 7.0)),
                "fallback_s": float(np.sum(fallback > 0.0)),
                "p95_axis_deg": float(np.quantile(axis, 0.95)),
                "max_axis_deg": float(np.max(axis)),
            }
        )
    return pd.DataFrame(rows)


def _load_reference(casebook: pd.DataFrame) -> pd.DataFrame:
    path = FRAME_PLAN / "offline_policy_case_choices.csv"
    df = pd.read_csv(path)
    df["source_case_id"] = df["case_id"].map(_strip_numbered)
    keep = set(casebook["case_id"].astype(str))
    df = df[df["source_case_id"].isin(keep)].copy()
    df = df[df["policy"].isin({"baseline", "current_generalized_gate"})].copy()
    df["reference"] = df["policy"].map(
        {
            "baseline": "baseline",
            "current_generalized_gate": "current_generalized_gate",
        }
    )
    return df


def _safe_veto_reference(casebook: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    ref = reference.set_index(["source_case_id", "reference"], drop=False)
    economy_frames = {"neutral_mhs_broader", "lowrisk_stable_redundant_candidate"}
    for rec in casebook.to_dict("records"):
        source_case_id = str(rec["case_id"])
        stratum = str(rec["stratum"])
        choice = "current_generalized_gate" if stratum in economy_frames else "baseline"
        row = ref.loc[(source_case_id, choice)].copy()
        row["policy"] = "safe_veto_reference"
        row["selected_profile"] = (
            "psc_4hao_generalized_gate_v1"
            if choice == "current_generalized_gate"
            else "dc_preserving_deadband_forecast_adaptive_v1"
        )
        row["selected_role"] = (
            "P2/lowrisk economy enable" if choice == "current_generalized_gate" else "warning/watch fail-closed"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame, name: str, baseline_total: float) -> dict[str, Any]:
    pump = float(df["pump_m3"].sum())
    return {
        "policy": name,
        "cases": int(df["source_case_id"].nunique()),
        "pump_m3": pump,
        "saved_m3": baseline_total - pump,
        "saving_pct": 100.0 * (baseline_total - pump) / max(baseline_total, 1e-9),
        "time_gt3_s": float(df["time_gt3_s"].sum()),
        "time_gt4_s": float(df["time_gt4_s"].sum()),
        "time_gt5_s": float(df["time_gt5_s"].sum()),
        "time_gt7_s": float(df["time_gt7_s"].sum()),
        "fallback_s": float(df["fallback_s"].sum()),
        "worst_p95_axis_deg": float(df["p95_axis_deg"].max()),
        "worst_max_axis_deg": float(df["max_axis_deg"].max()),
    }


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for rec in df[cols].to_dict("records"):
        values: list[str] = []
        for col in cols:
            value = rec[col]
            if isinstance(value, float):
                if col.endswith("_pct"):
                    values.append(f"{value:.2f}%")
                elif col.endswith("_deg"):
                    values.append(f"{value:.2f}")
                elif col.endswith("_m3"):
                    values.append(f"{value:.1f}")
                elif col.endswith("_s"):
                    values.append(f"{value:.0f}")
                else:
                    values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize(args: argparse.Namespace, casebook: pd.DataFrame, group_paths: dict[str, Path]) -> None:
    frames: list[pd.DataFrame] = []
    for stratum in group_paths:
        spec = DISPATCH[stratum]
        run_dir = args.out_dir / "runs" / _safe_name(stratum)
        frames.append(_run_metrics(run_dir, str(spec["label"]), stratum))
    dispatcher = pd.concat(frames, ignore_index=True)
    dispatcher.to_csv(args.out_dir / "dispatcher_case_metrics.csv", index=False)

    reference = _load_reference(casebook)
    baseline = reference[reference["reference"].eq("baseline")].copy()
    current = reference[reference["reference"].eq("current_generalized_gate")].copy()
    safe_veto = _safe_veto_reference(casebook, reference)
    baseline_total = float(baseline["pump_m3"].sum())

    summary = pd.DataFrame(
        [
            _aggregate(baseline, "baseline", baseline_total),
            _aggregate(current, "current_generalized_gate", baseline_total),
            _aggregate(safe_veto, "safe_veto_reference", baseline_total),
            _aggregate(dispatcher, "frame_dispatcher_smoke", baseline_total),
        ]
    )
    summary.to_csv(args.out_dir / "dispatcher_summary.csv", index=False)

    case_compare = dispatcher.merge(
        baseline[["source_case_id", "pump_m3", "time_gt5_s", "time_gt7_s", "fallback_s"]],
        on="source_case_id",
        how="left",
        suffixes=("", "_baseline"),
    ).merge(
        current[["source_case_id", "pump_m3", "time_gt5_s", "time_gt7_s", "fallback_s"]],
        on="source_case_id",
        how="left",
        suffixes=("", "_current_gate"),
    )
    case_compare["saved_vs_baseline_m3"] = case_compare["pump_m3_baseline"] - case_compare["pump_m3"]
    case_compare["saved_vs_current_gate_m3"] = case_compare["pump_m3_current_gate"] - case_compare["pump_m3"]
    case_compare["d_gt5_vs_current_gate_s"] = case_compare["time_gt5_s"] - case_compare["time_gt5_s_current_gate"]
    case_compare["d_gt7_vs_current_gate_s"] = case_compare["time_gt7_s"] - case_compare["time_gt7_s_current_gate"]
    case_compare["d_fallback_vs_current_gate_s"] = case_compare["fallback_s"] - case_compare["fallback_s_current_gate"]
    case_compare.to_csv(args.out_dir / "dispatcher_case_compare.csv", index=False)

    by_stratum = (
        case_compare.groupby("stratum", dropna=False)
        .agg(
            cases=("source_case_id", "count"),
            selected_profile=("selected_profile", "first"),
            saved_vs_baseline_m3=("saved_vs_baseline_m3", "sum"),
            saved_vs_current_gate_m3=("saved_vs_current_gate_m3", "sum"),
            time_gt5_s=("time_gt5_s", "sum"),
            time_gt7_s=("time_gt7_s", "sum"),
            fallback_s=("fallback_s", "sum"),
            d_gt5_vs_current_gate_s=("d_gt5_vs_current_gate_s", "sum"),
            d_gt7_vs_current_gate_s=("d_gt7_vs_current_gate_s", "sum"),
            d_fallback_vs_current_gate_s=("d_fallback_vs_current_gate_s", "sum"),
        )
        .reset_index()
    )
    by_stratum.to_csv(args.out_dir / "dispatcher_by_stratum.csv", index=False)
    safe_by_stratum = (
        safe_veto.groupby("stratum", dropna=False)
        .agg(
            cases=("source_case_id", "count"),
            selected_profile=("selected_profile", "first"),
            pump_m3=("pump_m3", "sum"),
            time_gt5_s=("time_gt5_s", "sum"),
            time_gt7_s=("time_gt7_s", "sum"),
            fallback_s=("fallback_s", "sum"),
        )
        .reset_index()
    )
    base_by = baseline.groupby("stratum", dropna=False).agg(baseline_pump_m3=("pump_m3", "sum")).reset_index()
    cur_by = (
        current.groupby("stratum", dropna=False)
        .agg(
            current_pump_m3=("pump_m3", "sum"),
            current_gt5_s=("time_gt5_s", "sum"),
            current_gt7_s=("time_gt7_s", "sum"),
            current_fallback_s=("fallback_s", "sum"),
        )
        .reset_index()
    )
    safe_by_stratum = safe_by_stratum.merge(base_by, on="stratum", how="left").merge(cur_by, on="stratum", how="left")
    safe_by_stratum["saved_vs_baseline_m3"] = safe_by_stratum["baseline_pump_m3"] - safe_by_stratum["pump_m3"]
    safe_by_stratum["saved_vs_current_gate_m3"] = safe_by_stratum["current_pump_m3"] - safe_by_stratum["pump_m3"]
    safe_by_stratum["d_gt5_vs_current_gate_s"] = safe_by_stratum["time_gt5_s"] - safe_by_stratum["current_gt5_s"]
    safe_by_stratum["d_gt7_vs_current_gate_s"] = safe_by_stratum["time_gt7_s"] - safe_by_stratum["current_gt7_s"]
    safe_by_stratum["d_fallback_vs_current_gate_s"] = safe_by_stratum["fallback_s"] - safe_by_stratum["current_fallback_s"]
    safe_by_stratum.to_csv(args.out_dir / "safe_veto_by_stratum.csv", index=False)

    readout = [
        "# PSC No.4 Frame Dispatcher Smoke v1",
        "",
        "This smoke run selects existing profiles by frame on the 12 planned cases.",
        "",
        "## Summary",
        "",
        _md_table(
            summary,
            [
                "policy",
                "cases",
                "pump_m3",
                "saved_m3",
                "saving_pct",
                "time_gt5_s",
                "time_gt7_s",
                "fallback_s",
                "worst_p95_axis_deg",
                "worst_max_axis_deg",
            ],
        ),
        "",
        "## By Frame",
        "",
        _md_table(
            by_stratum,
            [
                "stratum",
                "cases",
                "selected_profile",
                "saved_vs_baseline_m3",
                "saved_vs_current_gate_m3",
                "time_gt5_s",
                "time_gt7_s",
                "fallback_s",
                "d_gt5_vs_current_gate_s",
                "d_gt7_vs_current_gate_s",
                "d_fallback_vs_current_gate_s",
            ],
        ),
        "",
        "## Safe Veto Reference",
        "",
        "This reference keeps generalized_gate only on P2/lowrisk frames and fails P1/W1/W2/W3/W7 closed to baseline/watch.",
        "",
        _md_table(
            safe_by_stratum,
            [
                "stratum",
                "cases",
                "selected_profile",
                "saved_vs_baseline_m3",
                "saved_vs_current_gate_m3",
                "time_gt5_s",
                "time_gt7_s",
                "fallback_s",
                "d_gt5_vs_current_gate_s",
                "d_gt7_vs_current_gate_s",
                "d_fallback_vs_current_gate_s",
            ],
        ),
    ]
    (args.out_dir / "dispatcher_smoke_readout.md").write_text("\n".join(readout) + "\n", encoding="utf-8")
    print(args.out_dir / "dispatcher_smoke_readout.md")
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    casebook = _load_casebook(args.casebook)
    group_paths = _materialize_groups(casebook, args.out_dir)
    _run_groups(args, group_paths)
    if args.mode in {"run", "summarize"}:
        summarize(args, casebook, group_paths)


if __name__ == "__main__":
    main()
