#!/usr/bin/env python3
"""Replay the PSC No.4 regime-addback-v2 selected cases.

The v2 policy is a safe-veto dispatcher:

- selected cases run the existing `psc_4hao_generalized_gate_v1` profile;
- rejected cases stay on the frozen baseline.

This runner re-executes the selected casebook so the result is not only a
spreadsheet recombination of previous outputs.  It still is not a provider-level
runtime classifier; the selected casebook is supplied by the offline v2 rule.
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
DEFAULT_CASEBOOK = V2 / "regime_addback_v2_casebook.csv"
OUT = BASE / "psc_4hao_regime_addback_v2_replay_20260603"


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


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def _load_casebook(path: Path) -> pd.DataFrame:
    df = pd.read_csv(_resolve(path))
    required = {"case_id", "timestamp", "label", "stratum", "synthetic_case_id", "regime_v2_reason"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"casebook missing columns: {sorted(missing)}")
    return df


def _materialize_selected(casebook: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = casebook[["case_id", "timestamp", "label"]].copy()
    out["label"] = (
        out["label"].astype(str)
        + " | replay_dispatcher=regime_addback_v2"
        + " | selected_profile=psc_4hao_generalized_gate_v1"
    )
    path = out_dir / "regime_addback_v2_selected_cases.csv"
    out.to_csv(path, index=False)
    return path


def _command(args: argparse.Namespace, selected_csv: Path) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--out-dir",
        str(args.out_dir / "selected_generalized_gate"),
        "--cases-csv",
        str(selected_csv),
        "--duration-s",
        str(args.duration_s),
        "--dataset-dir",
        str(args.dataset_dir),
        "--replay-split",
        str(args.replay_split),
        "--model-dir",
        str(args.model_dir),
        "--primary-label",
        "regime_addback_v2_selected_gate",
        "--primary-only",
        "--skip-figures",
        "--primary-control-profile",
        "psc_4hao_generalized_gate_v1",
        "--forecast-source",
        "learned",
    ]


def _run_selected(args: argparse.Namespace, selected_csv: Path) -> None:
    run_dir = args.out_dir / "selected_generalized_gate"
    summary = run_dir / "casebook_summary.csv"
    cmd = _command(args, selected_csv)
    print("+ FOWT_4HAO_ACTIVE_POSTURE_REFRESH=1 " + " ".join(cmd))
    if args.mode != "run":
        return
    if args.skip_existing and summary.exists():
        print(f"skip existing selected replay: {summary}")
        return
    env = os.environ.copy()
    env["FOWT_4HAO_ACTIVE_POSTURE_REFRESH"] = "1"
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


def _selected_metrics(run_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(run_dir / "casebook_summary.csv")
    rows: list[dict[str, Any]] = []
    for rec in summary.to_dict("records"):
        run_case_id = str(rec["case_id"])
        raw_label = str(rec.get("label", ""))
        synthetic_case_id = _label_value(raw_label, "synthetic_case_id")
        stratum = _label_value(raw_label, "broader6h_stratum") or _label_value(raw_label, "stratum")
        reason = _label_value(raw_label, "regime_v2_reason")
        ts = pd.read_csv(_find_timeseries(run_dir, run_case_id, "regime_addback_v2_selected_gate"), low_memory=False)
        pitch = pd.to_numeric(ts.get("pitch_deg", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
        roll = pd.to_numeric(ts.get("roll_deg", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
        axis = np.maximum(np.abs(pitch), np.abs(roll))
        fallback = pd.to_numeric(
            ts.get("preview_primary_safety_fallback", pd.Series(0.0, index=ts.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(float)
        rows.append(
            {
                "synthetic_case_id": synthetic_case_id,
                "run_case_id": run_case_id,
                "stratum": stratum,
                "regime_v2_reason": reason,
                "selected_arm": "selected_replay_generalized_gate",
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


def _offline_policy_rows() -> pd.DataFrame:
    cases = pd.read_csv(V2 / "regime_addback_v2_case_outcomes.csv")
    return cases


def _baseline_rows(cases: pd.DataFrame, rejected_ids: set[str]) -> pd.DataFrame:
    rows = cases[cases["case_id"].isin(rejected_ids)].copy()
    return pd.DataFrame(
        {
            "synthetic_case_id": rows["case_id"],
            "run_case_id": rows["case_id"],
            "stratum": rows["stratum"],
            "regime_v2_reason": "fail_closed",
            "selected_arm": "baseline_fail_closed",
            "pump_m3": rows["baseline_pump_m3"],
            "time_gt3_s": rows["baseline_time_gt3_s"],
            "time_gt4_s": rows["baseline_time_gt4_s"],
            "time_gt5_s": rows["baseline_time_gt5_s"],
            "time_gt7_s": rows["baseline_time_gt7_s"],
            "fallback_s": rows["baseline_fallback_s"],
            "p95_axis_deg": rows["baseline_p95_axis_deg"],
            "max_axis_deg": rows["baseline_max_axis_deg"],
        }
    )


def _aggregate(df: pd.DataFrame, name: str, baseline_total: float) -> dict[str, object]:
    pump = float(df["pump_m3"].sum())
    return {
        "policy": name,
        "cases": int(df["synthetic_case_id"].nunique()),
        "gate_cases": int(df["selected_arm"].eq("selected_replay_generalized_gate").sum()),
        "pump_m3": pump,
        "saved_m3": baseline_total - pump,
        "saving_pct": 100.0 * (baseline_total - pump) / max(baseline_total, 1e-9),
        "time_gt5_s": float(df["time_gt5_s"].sum()),
        "time_gt7_s": float(df["time_gt7_s"].sum()),
        "fallback_s": float(df["fallback_s"].sum()),
        "worst_p95_axis_deg": float(df["p95_axis_deg"].max()),
        "worst_max_axis_deg": float(df["max_axis_deg"].max()),
    }


def _reference_summary() -> pd.DataFrame:
    return pd.read_csv(V2 / "regime_addback_v2_summary.csv")


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


def summarize(args: argparse.Namespace, casebook: pd.DataFrame) -> None:
    cases = _offline_policy_rows()
    selected = _selected_metrics(args.out_dir / "selected_generalized_gate")
    selected_ids = set(selected["synthetic_case_id"].astype(str))
    all_ids = set(cases["case_id"].astype(str))
    rejected = _baseline_rows(cases, all_ids - selected_ids)
    replay_policy = pd.concat([selected, rejected], ignore_index=True)
    replay_policy.to_csv(args.out_dir / "regime_addback_v2_replay_policy_cases.csv", index=False)
    selected.to_csv(args.out_dir / "regime_addback_v2_selected_replay_metrics.csv", index=False)

    baseline_total = float(cases["baseline_pump_m3"].sum())
    replay_summary = pd.DataFrame([_aggregate(replay_policy, "regime_addback_v2_replay", baseline_total)])
    summary = pd.concat([_reference_summary(), replay_summary], ignore_index=True)
    summary.to_csv(args.out_dir / "regime_addback_v2_replay_summary.csv", index=False)

    by_stratum = (
        replay_policy.groupby("stratum", dropna=False)
        .agg(
            cases=("synthetic_case_id", "count"),
            gate_cases=("selected_arm", lambda s: int((s == "selected_replay_generalized_gate").sum())),
            pump_m3=("pump_m3", "sum"),
            time_gt5_s=("time_gt5_s", "sum"),
            time_gt7_s=("time_gt7_s", "sum"),
            fallback_s=("fallback_s", "sum"),
            worst_p95_axis_deg=("p95_axis_deg", "max"),
        )
        .reset_index()
    )
    base_by = cases.groupby("stratum", dropna=False).agg(baseline_pump_m3=("baseline_pump_m3", "sum")).reset_index()
    by_stratum = by_stratum.merge(base_by, on="stratum", how="left")
    by_stratum["saved_m3"] = by_stratum["baseline_pump_m3"] - by_stratum["pump_m3"]
    by_stratum.to_csv(args.out_dir / "regime_addback_v2_replay_by_stratum.csv", index=False)

    readout = [
        "# PSC No.4 Regime Addback v2 Replay - 2026-06-03",
        "",
        "This run replays the 48 v2-selected cases with `psc_4hao_generalized_gate_v1` and composes rejected cases with the frozen baseline.",
        "",
        "It is a stronger check than the offline table, but still not a provider-level runtime classifier.",
        "",
        "## Summary",
        "",
        _md_table(
            summary,
            [
                "policy",
                "cases",
                "gate_cases",
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
        "## Replay By Regime",
        "",
        _md_table(
            by_stratum,
            [
                "stratum",
                "cases",
                "gate_cases",
                "saved_m3",
                "time_gt5_s",
                "time_gt7_s",
                "fallback_s",
                "worst_p95_axis_deg",
            ],
        ),
    ]
    (args.out_dir / "regime_addback_v2_replay_readout.md").write_text("\n".join(readout) + "\n", encoding="utf-8")
    print(args.out_dir / "regime_addback_v2_replay_readout.md")
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    casebook = _load_casebook(args.casebook)
    selected_csv = _materialize_selected(casebook, args.out_dir)
    _run_selected(args, selected_csv)
    if args.mode in {"run", "summarize"}:
        summarize(args, casebook)


if __name__ == "__main__":
    main()
