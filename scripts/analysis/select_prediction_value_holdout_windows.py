#!/usr/bin/env python3
"""Select holdout windows using only physical wind/pressure features.

The selector deliberately does NOT use learned-vs-persistence outcomes. It
creates CSV case lists for follow-up closed-loop runs.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.wind_prediction.ballast_planner import PlannerConfig, compute_pressure_blocks, norm_term
from src.wind_prediction.replay_dataset import Fino1ReplayDataset


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


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_planner_config() -> PlannerConfig:
    sign_path = REPO_ROOT / "outputs" / "wind_prediction" / "planner_a1_dryrun" / "diagnostics" / "a01_pressure_vec_sign_convention.json"
    if sign_path.exists():
        sign_cfg = json.loads(sign_path.read_text(encoding="utf-8"))
        return PlannerConfig(pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]))
    return PlannerConfig()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1e-9 or bn <= 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (an * bn))


def _speed_ms(uv: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.asarray(uv, dtype=float) ** 2, axis=1))


def _features_for_sample(sample, cfg: PlannerConfig) -> dict:
    blocks = compute_pressure_blocks(sample.y_uv_raw, [1.0, 0.85, 0.70], cfg)
    raw_vecs = [np.asarray(b["pressure_vec_raw"], dtype=float) for b in blocks]
    raw_norms = [norm_term(v, cfg) for v in raw_vecs]
    speeds = _speed_ms(sample.y_uv_raw)
    return {
        "timestamp": sample.history_end.strftime(TIMESTAMP_FMT),
        "current_speed_ms": float(speeds[:2].mean()),
        "mid_speed_ms": float(speeds[2:4].mean()),
        "future_speed_ms": float(speeds[4:6].mean()),
        "relief_ms": float(speeds[:2].mean() - speeds[4:6].mean()),
        "raw_norm0": float(raw_norms[0]),
        "raw_norm1": float(raw_norms[1]),
        "raw_norm2": float(raw_norms[2]),
        "norm_relief02": float(raw_norms[0] - raw_norms[2]),
        "cos02": _cosine(raw_vecs[0], raw_vecs[2]),
        "cos01": _cosine(raw_vecs[0], raw_vecs[1]),
        "cos12": _cosine(raw_vecs[1], raw_vecs[2]),
    }


def _has_followup_samples(dataset: Fino1ReplayDataset, ts: datetime, buckets: int) -> bool:
    for idx in range(int(buckets)):
        if dataset.sample_for_history_end(ts + timedelta(seconds=idx * dataset.update_interval_s)) is None:
            return False
    return True


def _far_from_calibration(ts: datetime, guard_min: float) -> bool:
    guard = timedelta(minutes=float(guard_min))
    for raw in CALIBRATION_TIMESTAMPS:
        if abs(ts - datetime.strptime(raw, TIMESTAMP_FMT)) <= guard:
            return False
    return True


def _thin_by_time(rows: pd.DataFrame, min_separation_min: float, limit: int) -> pd.DataFrame:
    selected = []
    used: list[datetime] = []
    sep = timedelta(minutes=float(min_separation_min))
    for _, row in rows.iterrows():
        ts = datetime.strptime(str(row["timestamp"]), TIMESTAMP_FMT)
        if all(abs(ts - prev) >= sep for prev in used):
            selected.append(row)
            used.append(ts)
        if len(selected) >= int(limit):
            break
    if not selected:
        return rows.head(0).copy()
    return pd.DataFrame(selected)


def build_candidate_table(
    dataset: Fino1ReplayDataset,
    cfg: PlannerConfig,
    duration_buckets: int,
    calibration_guard_min: float,
) -> pd.DataFrame:
    rows = []
    for sample in dataset.samples:
        ts = sample.history_end
        if not _has_followup_samples(dataset, ts, duration_buckets):
            continue
        if not _far_from_calibration(ts, calibration_guard_min):
            continue
        rows.append(_features_for_sample(sample, cfg))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # These labels are physical scenario tags, not outcome-based labels.
    df["case07_like_signflip"] = (
        (df["raw_norm0"] >= 1.20)
        & (df["raw_norm2"] >= 0.70)
        & (df["cos02"] <= -0.30)
        & (df["current_speed_ms"] >= 10.0)
    )
    df["future_relief"] = (
        (df["raw_norm0"] >= 1.20)
        & ((df["raw_norm2"] <= 0.85) | (df["relief_ms"] >= 2.0))
        & (df["current_speed_ms"] >= 10.0)
    )
    df["high_persistent_no_flip"] = (
        (df["raw_norm0"] >= 1.20)
        & (df["raw_norm2"] >= 1.00)
        & (df["cos02"] >= 0.50)
    )
    return df


def write_cases_csv(rows: pd.DataFrame, out_path: Path, prefix: str, label: str) -> None:
    out_rows = []
    for idx, row in enumerate(rows.to_dict("records"), start=1):
        out_rows.append(
            {
                "case_id": f"{prefix}_{idx:02d}",
                "timestamp": row["timestamp"],
                "label": label,
                "raw_norm0": row["raw_norm0"],
                "raw_norm2": row["raw_norm2"],
                "cos02": row["cos02"],
                "relief_ms": row["relief_ms"],
            }
        )
    pd.DataFrame(out_rows).to_csv(out_path, index=False)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1",
    )
    parser.add_argument("--out-dir", default="outputs/wind_prediction/prediction_value_holdout_selection_v1")
    parser.add_argument("--duration-s", type=float, default=7200.0)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--random-limit", type=int, default=12)
    parser.add_argument("--min-separation-min", type=float, default=180.0)
    parser.add_argument("--calibration-guard-min", type=float, default=180.0)
    parser.add_argument("--random-seed", type=int, default=20260507)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = Fino1ReplayDataset(dataset_dir=_resolve(args.dataset_dir), split="test")
    cfg = _load_planner_config()
    duration_buckets = int(np.ceil(float(args.duration_s) / dataset.update_interval_s))
    candidates = build_candidate_table(
        dataset=dataset,
        cfg=cfg,
        duration_buckets=duration_buckets,
        calibration_guard_min=float(args.calibration_guard_min),
    )
    candidates.to_csv(out_dir / "all_candidate_features.csv", index=False)

    signflip = candidates[candidates["case07_like_signflip"]].copy()
    signflip = signflip.sort_values(
        ["cos02", "raw_norm0", "raw_norm2"],
        ascending=[True, False, False],
    )
    signflip_sel = _thin_by_time(
        signflip,
        min_separation_min=float(args.min_separation_min),
        limit=int(args.limit),
    )
    write_cases_csv(
        signflip_sel,
        out_dir / "case07_like_holdout_cases.csv",
        prefix="holdout_signflip",
        label="UV-only sustained signflip holdout",
    )

    relief = candidates[candidates["future_relief"]].copy()
    relief = relief.sort_values(
        ["relief_ms", "norm_relief02", "raw_norm0"],
        ascending=[False, False, False],
    )
    relief_sel = _thin_by_time(
        relief,
        min_separation_min=float(args.min_separation_min),
        limit=int(args.limit),
    )
    write_cases_csv(
        relief_sel,
        out_dir / "future_relief_holdout_cases.csv",
        prefix="holdout_relief",
        label="UV-only future relief holdout",
    )

    random_pool = candidates.sample(
        frac=1.0,
        random_state=int(args.random_seed),
    )
    random_sel = _thin_by_time(
        random_pool,
        min_separation_min=float(args.min_separation_min),
        limit=int(args.random_limit),
    )
    write_cases_csv(
        random_sel,
        out_dir / "random_holdout_cases.csv",
        prefix="holdout_random",
        label="UV-only random test holdout",
    )

    summary = pd.DataFrame(
        [
            {"set": "all_eligible", "count": int(len(candidates))},
            {"set": "case07_like_signflip_all", "count": int(len(signflip))},
            {"set": "case07_like_signflip_selected", "count": int(len(signflip_sel))},
            {"set": "future_relief_all", "count": int(len(relief))},
            {"set": "future_relief_selected", "count": int(len(relief_sel))},
            {"set": "random_selected", "count": int(len(random_sel))},
        ]
    )
    summary.to_csv(out_dir / "selection_summary.csv", index=False)
    report = [
        "# Prediction Value Holdout Selection v1",
        "",
        "Selection uses only true UV-derived pressure features and timestamps.",
        "It does not use learned, persistence, oracle control outcomes, or pump metrics.",
        "",
        "## Rules",
        "",
        "- case07-like signflip: `raw_norm0 >= 1.20`, `raw_norm2 >= 0.70`, `cos02 <= -0.30`, `current_speed_ms >= 10.0`.",
        "- future relief: `raw_norm0 >= 1.20`, (`raw_norm2 <= 0.85` or `relief_ms >= 2.0`), `current_speed_ms >= 10.0`.",
        "- exclude windows within calibration guard around the 10 manual casebook timestamps.",
        "- thin selected windows by minimum timestamp separation before writing case CSVs.",
        "",
        "## Counts",
        "",
        markdown_table(summary),
        "",
        "## Outputs",
        "",
        f"- `{out_dir / 'case07_like_holdout_cases.csv'}`",
        f"- `{out_dir / 'future_relief_holdout_cases.csv'}`",
        f"- `{out_dir / 'random_holdout_cases.csv'}`",
        f"- `{out_dir / 'all_candidate_features.csv'}`",
    ]
    (out_dir / "selection_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Report: {out_dir / 'selection_report.md'}")


if __name__ == "__main__":
    main()
