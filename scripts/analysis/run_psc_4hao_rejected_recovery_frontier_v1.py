#!/usr/bin/env python3
"""Run a frontier recovery check for high-potential rejected PSC No.4 cases.

The regime-addback-v2 policy intentionally rejected several cases that saved
pump under the current generalized gate but carried fallback or safety debt.
This runner replays those frontier cases with existing controller profiles to
see whether any can be safely added back without designing a new controller.
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
V2 = BASE / "psc_4hao_regime_addback_rules_v2_20260603"
CASEBOOK_DIR = BASE / "psc_4hao_broader_6h_test_only_main_v1" / "casebooks"
OUT = BASE / "psc_4hao_rejected_recovery_frontier_v1_20260603"

FRONTIER_IDS = [
    "10_negative_pool_10",
    "01_positive_pool_01",
    "06_positive_pool_06",
    "11_positive_pool_11",
]

PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "psc_4hao_generalized_gate_v1": {
        "label": "frontier_generalized_gate",
        "profile": "psc_4hao_generalized_gate_v1",
        "strata": {"direction_reversal_boundary", "transient_peak_future_decay"},
        "forecast": "learned",
        "env": {"FOWT_4HAO_ACTIVE_POSTURE_REFRESH": "1"},
    },
    "psc_4hao_guarded_v3": {
        "label": "frontier_guarded_v3",
        "profile": "psc_4hao_guarded_v3",
        "strata": {"direction_reversal_boundary", "transient_peak_future_decay"},
        "forecast": "learned",
        "env": {"FOWT_4HAO_ACTIVE_POSTURE_REFRESH": "1"},
    },
    "psc_4hao_exec_guard_v1": {
        "label": "frontier_exec_guard_v1",
        "profile": "psc_4hao_exec_guard_v1",
        "strata": {"direction_reversal_boundary", "transient_peak_future_decay"},
        "forecast": "learned",
        "env": {"FOWT_4HAO_ACTIVE_POSTURE_REFRESH": "1"},
    },
    "relief_decay_dispatcher_v2": {
        "label": "frontier_relief_decay_v2",
        "profile": "relief_decay_dispatcher_v2",
        "strata": {"transient_peak_future_decay"},
        "forecast": "learned",
        "env": {},
    },
    "direction_reversal_layered_v8": {
        "label": "frontier_direction_v8",
        "profile": "direction_reversal_layered_v8",
        "strata": {"direction_reversal_boundary"},
        "forecast": "learned",
        "env": {},
    },
    "direction_reversal_layered_v8_holdgate_v1": {
        "label": "frontier_direction_v8_holdgate",
        "profile": "direction_reversal_layered_v8_holdgate_v1",
        "strata": {"direction_reversal_boundary"},
        "forecast": "learned",
        "env": {},
    },
    "direction_reversal_layered_v8_holdgate_off_v1": {
        "label": "frontier_direction_v8_holdgate_off",
        "profile": "direction_reversal_layered_v8_holdgate_off_v1",
        "strata": {"direction_reversal_boundary"},
        "forecast": "learned",
        "env": {},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def _strip_number(case_id: str) -> str:
    parts = str(case_id).split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else str(case_id)


def _load_outcomes() -> pd.DataFrame:
    df = pd.read_csv(V2 / "regime_addback_v2_case_outcomes.csv")
    selected = df[df["case_id"].isin(FRONTIER_IDS)].copy()
    missing = sorted(set(FRONTIER_IDS) - set(selected["case_id"].astype(str)))
    if missing:
        raise ValueError(f"missing frontier cases in v2 outcomes: {missing}")
    return selected


def _load_source_casebooks() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in CASEBOOK_DIR.glob("*_pool_cases.csv")]
    if not frames:
        raise FileNotFoundError(CASEBOOK_DIR)
    return pd.concat(frames, ignore_index=True).drop_duplicates("case_id", keep="first")


def _frontier_source_cases(outcomes: pd.DataFrame) -> pd.DataFrame:
    source = _load_source_casebooks()
    selected = outcomes.copy()
    selected["source_case_id"] = selected["case_id"].map(_strip_number)
    merged = selected.merge(source, left_on="source_case_id", right_on="case_id", how="left", suffixes=("_synthetic", ""))
    if merged["timestamp"].isna().any():
        missing = merged.loc[merged["timestamp"].isna(), "case_id_synthetic"].tolist()
        raise ValueError(f"missing source casebook rows: {missing}")
    return pd.DataFrame(
        {
            "synthetic_case_id": merged["case_id_synthetic"],
            "case_id": merged["case_id"],
            "timestamp": merged["timestamp"],
            "label": merged["label"],
            "stratum": merged["stratum"],
        }
    )


def _materialize_casebooks(outcomes: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    source_cases = _frontier_source_cases(outcomes)
    casebook_dir = out_dir / "casebooks"
    casebook_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, spec in PROFILE_SPECS.items():
        group = source_cases[source_cases["stratum"].isin(spec["strata"])].copy()
        if group.empty:
            continue
        label = str(spec["label"])
        group["label"] = (
            group["label"].astype(str)
            + " | frontier_profile="
            + name
            + " | synthetic_case_id="
            + group["synthetic_case_id"].astype(str)
            + " | frontier_source_case_id="
            + group["case_id"].astype(str)
        )
        out = group[["case_id", "timestamp", "label"]]
        path = casebook_dir / f"{_safe_name(name)}_frontier_cases.csv"
        out.to_csv(path, index=False)
        paths[name] = path
        group.to_csv(casebook_dir / f"{_safe_name(name)}_frontier_casebook_with_meta.csv", index=False)
    return paths


def _command(args: argparse.Namespace, name: str, cases_csv: Path) -> tuple[list[str], dict[str, str], Path]:
    spec = PROFILE_SPECS[name]
    out_dir = args.out_dir / "runs" / _safe_name(name)
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


def _run_profiles(args: argparse.Namespace, casebooks: dict[str, Path]) -> None:
    for name, cases_csv in casebooks.items():
        cmd, env_patch, out_dir = _command(args, name, cases_csv)
        summary = out_dir / "casebook_summary.csv"
        prefix = " ".join(f"{key}={value}" for key, value in env_patch.items())
        print(("+ " + prefix + " " if prefix else "+ ") + " ".join(cmd))
        if args.mode != "run":
            continue
        if args.skip_existing and summary.exists():
            print(f"skip existing {name}: {summary}")
            continue
        env = os.environ.copy()
        env.update(env_patch)
        subprocess.run(cmd, cwd=REPO, env=env, check=True)


def _label_value(label: str, key: str) -> str:
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


def _profile_metrics(run_dir: Path, name: str) -> pd.DataFrame:
    spec = PROFILE_SPECS[name]
    label = str(spec["label"])
    summary = pd.read_csv(run_dir / "casebook_summary.csv")
    rows: list[dict[str, Any]] = []
    for rec in summary.to_dict("records"):
        run_case_id = str(rec["case_id"])
        raw_label = str(rec.get("label", ""))
        synthetic_case_id = _label_value(raw_label, "synthetic_case_id")
        stratum = _label_value(raw_label, "broader6h_stratum")
        ts = pd.read_csv(_find_timeseries(run_dir, run_case_id, label), low_memory=False)
        pitch = pd.to_numeric(ts.get("pitch_deg", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
        roll = pd.to_numeric(ts.get("roll_deg", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
        axis = np.maximum(np.abs(pitch), np.abs(roll))
        fallback = pd.to_numeric(
            ts.get("preview_primary_safety_fallback", pd.Series(0.0, index=ts.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(float)
        rows.append(
            {
                "profile_key": name,
                "selected_profile": str(spec["profile"]),
                "synthetic_case_id": synthetic_case_id,
                "run_case_id": run_case_id,
                "stratum": stratum,
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


def _compare(metrics: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    ref_cols = [
        "case_id",
        "stratum",
        "baseline_pump_m3",
        "baseline_time_gt5_s",
        "baseline_time_gt7_s",
        "baseline_fallback_s",
        "baseline_p95_axis_deg",
        "pump_m3",
        "time_gt5_s",
        "time_gt7_s",
        "fallback_s",
        "p95_axis_deg",
        "saved_m3",
    ]
    ref = outcomes[ref_cols].rename(
        columns={
            "case_id": "synthetic_case_id",
            "pump_m3": "current_gate_pump_m3",
            "time_gt5_s": "current_gate_time_gt5_s",
            "time_gt7_s": "current_gate_time_gt7_s",
            "fallback_s": "current_gate_fallback_s",
            "p95_axis_deg": "current_gate_p95_axis_deg",
            "saved_m3": "current_gate_saved_m3",
        }
    )
    compare = metrics.merge(ref, on=["synthetic_case_id", "stratum"], how="left")
    compare["saved_vs_baseline_m3"] = compare["baseline_pump_m3"] - compare["pump_m3"]
    compare["saved_vs_current_gate_m3"] = compare["current_gate_pump_m3"] - compare["pump_m3"]
    compare["d_gt5_vs_baseline_s"] = compare["time_gt5_s"] - compare["baseline_time_gt5_s"]
    compare["d_gt7_vs_baseline_s"] = compare["time_gt7_s"] - compare["baseline_time_gt7_s"]
    compare["d_fallback_vs_baseline_s"] = compare["fallback_s"] - compare["baseline_fallback_s"]
    compare["d_gt5_vs_current_gate_s"] = compare["time_gt5_s"] - compare["current_gate_time_gt5_s"]
    compare["d_gt7_vs_current_gate_s"] = compare["time_gt7_s"] - compare["current_gate_time_gt7_s"]
    compare["d_fallback_vs_current_gate_s"] = compare["fallback_s"] - compare["current_gate_fallback_s"]
    compare["safe_addback_candidate"] = (
        (compare["saved_vs_baseline_m3"] > 0.0)
        & (compare["fallback_s"] <= 0.0)
        & (compare["d_gt7_vs_baseline_s"] <= 0.0)
    )
    return compare.sort_values(["safe_addback_candidate", "saved_vs_baseline_m3"], ascending=[False, False])


def _aggregate(df: pd.DataFrame, name: str, baseline_total: float) -> dict[str, object]:
    pump = float(df["pump_m3"].sum())
    return {
        "policy": name,
        "cases": int(df["synthetic_case_id"].nunique()),
        "pump_m3": pump,
        "saved_m3": baseline_total - pump,
        "saving_pct": 100.0 * (baseline_total - pump) / max(baseline_total, 1e-9),
        "time_gt5_s": float(df["time_gt5_s"].sum()),
        "time_gt7_s": float(df["time_gt7_s"].sum()),
        "fallback_s": float(df["fallback_s"].sum()),
        "worst_p95_axis_deg": float(df["p95_axis_deg"].max()),
    }


def _frontier_summary(compare: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    baseline_total = float(outcomes["baseline_pump_m3"].sum())
    current = pd.DataFrame(
        {
            "synthetic_case_id": outcomes["case_id"],
            "pump_m3": outcomes["pump_m3"],
            "time_gt5_s": outcomes["time_gt5_s"],
            "time_gt7_s": outcomes["time_gt7_s"],
            "fallback_s": outcomes["fallback_s"],
            "p95_axis_deg": outcomes["p95_axis_deg"],
        }
    )
    baseline = pd.DataFrame(
        {
            "synthetic_case_id": outcomes["case_id"],
            "pump_m3": outcomes["baseline_pump_m3"],
            "time_gt5_s": outcomes["baseline_time_gt5_s"],
            "time_gt7_s": outcomes["baseline_time_gt7_s"],
            "fallback_s": outcomes["baseline_fallback_s"],
            "p95_axis_deg": outcomes["baseline_p95_axis_deg"],
        }
    )
    rows = [_aggregate(baseline, "baseline_frontier_4", baseline_total)]
    rows.append(_aggregate(current, "current_generalized_gate_frontier_4", baseline_total))
    for name, group in compare.groupby("profile_key", sort=False):
        profile_baseline_total = float(group["baseline_pump_m3"].sum())
        rows.append(_aggregate(group, name, profile_baseline_total))
    safe = compare[compare["safe_addback_candidate"]].copy()
    if not safe.empty:
        chosen = safe.sort_values("saved_vs_baseline_m3", ascending=False).drop_duplicates("synthetic_case_id")
        mixed_rows: list[pd.Series] = []
        chosen_by_case = chosen.set_index("synthetic_case_id")
        for rec in baseline.to_dict("records"):
            case_id = str(rec["synthetic_case_id"])
            mixed_rows.append(chosen_by_case.loc[case_id] if case_id in chosen_by_case.index else pd.Series(rec))
        rows.append(_aggregate(pd.DataFrame(mixed_rows), "best_safe_recovery_mix", baseline_total))
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for rec in df[cols].to_dict("records"):
        values: list[str] = []
        for col in cols:
            value = rec[col]
            if isinstance(value, float):
                if col.endswith("_pct"):
                    values.append(f"{value:.2f}%")
                elif col.endswith("_m3"):
                    values.append(f"{value:.1f}")
                elif col.endswith("_s"):
                    values.append(f"{value:.0f}")
                elif col.endswith("_deg"):
                    values.append(f"{value:.2f}")
                else:
                    values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize(args: argparse.Namespace, outcomes: pd.DataFrame, casebooks: dict[str, Path]) -> None:
    metrics = []
    for name in casebooks:
        run_dir = args.out_dir / "runs" / _safe_name(name)
        metrics.append(_profile_metrics(run_dir, name))
    all_metrics = pd.concat(metrics, ignore_index=True)
    compare = _compare(all_metrics, outcomes)
    safe = compare[compare["safe_addback_candidate"]].copy()
    summary = _frontier_summary(compare, outcomes)

    all_metrics.to_csv(args.out_dir / "frontier_run_metrics.csv", index=False)
    compare.to_csv(args.out_dir / "frontier_case_compare.csv", index=False)
    safe.to_csv(args.out_dir / "frontier_safe_candidates.csv", index=False)
    summary.to_csv(args.out_dir / "frontier_summary.csv", index=False)

    top = compare.sort_values("saved_vs_baseline_m3", ascending=False).head(12)
    readout = [
        "# PSC No.4 Rejected Recovery Frontier v1 - 2026-06-03",
        "",
        "This check replays four rejected but high-potential cases with existing profiles.",
        "",
        "Acceptance rule: saved_vs_baseline_m3 > 0, fallback_s = 0, and d_gt7_vs_baseline_s <= 0.",
        "",
        "## Frontier Summary",
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
            ],
        ),
        "",
        "## Safe Addback Candidates",
        "",
        _md_table(
            safe,
            [
                "synthetic_case_id",
                "stratum",
                "selected_profile",
                "saved_vs_baseline_m3",
                "d_gt5_vs_baseline_s",
                "d_gt7_vs_baseline_s",
                "fallback_s",
                "p95_axis_deg",
            ],
        )
        if not safe.empty
        else "No candidate passed the safe addback rule.",
        "",
        "## Highest Saving Attempts",
        "",
        _md_table(
            top,
            [
                "synthetic_case_id",
                "selected_profile",
                "saved_vs_baseline_m3",
                "d_gt5_vs_baseline_s",
                "d_gt7_vs_baseline_s",
                "fallback_s",
                "p95_axis_deg",
                "safe_addback_candidate",
            ],
        ),
    ]
    (args.out_dir / "frontier_readout.md").write_text("\n".join(readout) + "\n", encoding="utf-8")
    print(args.out_dir / "frontier_readout.md")
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outcomes = _load_outcomes()
    casebooks = _materialize_casebooks(outcomes, args.out_dir)
    _run_profiles(args, casebooks)
    if args.mode in {"run", "summarize"}:
        summarize(args, outcomes, casebooks)


if __name__ == "__main__":
    main()
