#!/usr/bin/env python3
"""Lightweight mechanism-case selection for prediction-primary validation.

This scans FINO1 replay windows with the planner only. It does not run the
platform/controller simulation, so it is intended as a gate before spending
time on multi-hour casebook runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from wind_prediction.ballast_planner import (  # noqa: E402
    DEFAULT_PLANT_INFO,
    PlannerConfig,
    compute_pressure_blocks,
    select_best_economic,
    evaluate_sequence,
    _ALL_SEQUENCES,
)
from wind_prediction.replay_dataset import Fino1ReplayDataset  # noqa: E402


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
VALIDATION_GROUPS = [
    "high_sustained",
    "onset",
    "decay",
    "signflip",
    "lowrisk_hold",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/prediction_primary_mechanism_selection",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--stride", type=int, default=3, help="Scan every Nth 10-min replay sample.")
    parser.add_argument("--duration-min", type=float, default=120.0)
    parser.add_argument("--top-per-group", type=int, default=30)
    parser.add_argument(
        "--planner-candidates-per-group",
        type=int,
        default=80,
        help="Run the expensive planner only on this many coarse candidates per group.",
    )
    parser.add_argument("--validation-per-group", type=int, default=1)
    parser.add_argument("--show-top", type=int, default=8)
    return parser.parse_args()


def circular_delta_deg(a: float, b: float) -> float:
    return float(((float(b) - float(a) + 180.0) % 360.0) - 180.0)


def uv_speed_dir(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(uv, dtype=float)
    speed = np.linalg.norm(uv, axis=1)
    direction = (np.rad2deg(np.arctan2(uv[:, 1], uv[:, 0])) + 360.0) % 360.0
    return speed, direction


def planner_best(sample, cfg: PlannerConfig, discounts: list[float]) -> tuple[dict, list[dict]]:
    blocks = compute_pressure_blocks(np.asarray(sample.y_uv_raw, dtype=float), discounts, cfg)
    evaluated = [
        evaluate_sequence(seq, blocks, DEFAULT_PLANT_INFO, cfg)
        for seq in _ALL_SEQUENCES
    ]
    ranked, _ = select_best_economic(evaluated, cfg)
    if ranked:
        best = ranked[0]
    else:
        best = {
            "sequence": ("hold", "hold", "hold"),
            "hard_reject_reason": "no_feasible",
            "costs": {"max_envelope_norm": float("nan")},
            "best_scalar_cost": float("nan"),
        }
    return best, blocks


def classify_row(row: dict) -> list[tuple[str, float]]:
    groups: list[tuple[str, float]] = []
    b0 = float(row["block0_norm"])
    b2 = float(row["block2_norm"])
    max_b = float(row["max_block_norm"])
    min_b = float(row["min_block_norm"])
    dir_change = abs(float(row["dir_change_deg"]))
    action = str(row["first_action"])
    active_ratio = float(row["active_action_ratio"])

    if min_b > 1.0 and active_ratio >= 0.50:
        groups.append(("high_sustained", max_b + 0.15 * active_ratio))
    if b0 < 0.65 and b2 > 1.0:
        groups.append(("onset", (b2 - b0) + 0.04 * float(row["speed_growth"])))
    if b0 > 1.0 and b2 < 0.75:
        groups.append(("decay", (b0 - b2) + 0.04 * max(0.0, -float(row["speed_growth"]))))
    if dir_change >= 100.0 and max_b > 0.85:
        groups.append(("signflip", dir_change / 180.0 + max_b))
    if max_b < 0.65 and action == "hold":
        groups.append(("lowrisk_hold", 1.0 - max_b))
    return groups


def classify_coarse_row(row: dict) -> list[tuple[str, float]]:
    groups: list[tuple[str, float]] = []
    b0 = float(row["block0_norm"])
    b2 = float(row["block2_norm"])
    max_b = float(row["max_block_norm"])
    min_b = float(row["min_block_norm"])
    dir_change = abs(float(row["dir_change_deg"]))
    if min_b > 1.0:
        groups.append(("high_sustained", max_b))
    if b0 < 0.65 and b2 > 1.0:
        groups.append(("onset", (b2 - b0) + 0.04 * float(row["speed_growth"])))
    if b0 > 1.0 and b2 < 0.75:
        groups.append(("decay", (b0 - b2) + 0.04 * max(0.0, -float(row["speed_growth"]))))
    if dir_change >= 100.0 and max_b > 0.85:
        groups.append(("signflip", dir_change / 180.0 + max_b))
    if max_b < 0.65:
        groups.append(("lowrisk_hold", 1.0 - max_b))
    return groups


def base_row_from_sample(sample, blocks: list[dict]) -> dict:
    future = np.asarray(sample.y_uv_raw, dtype=float)
    speed, direction = uv_speed_dir(future)
    pressure_norms = [float(b["pressure_norm"]) for b in blocks]
    dir_now = float(direction[1] if direction.size > 1 else direction[0])
    dir_future = float(_circular_mean(direction[4:6] if direction.size >= 6 else direction[-2:]))
    speed_now = float(speed[1] if speed.size > 1 else speed[0])
    speed_future = float(np.mean(speed[4:6] if speed.size >= 6 else speed[-2:]))
    return {
        "timestamp": sample.history_end.strftime(TIMESTAMP_FMT),
        "block0_norm": pressure_norms[0],
        "block1_norm": pressure_norms[1],
        "block2_norm": pressure_norms[-1],
        "max_block_norm": float(np.max(pressure_norms)),
        "min_block_norm": float(np.min(pressure_norms)),
        "growth_norm": float(pressure_norms[-1] - pressure_norms[0]),
        "speed_now_ms": speed_now,
        "speed_future_ms": speed_future,
        "speed_max_ms": float(np.max(speed[:6])),
        "speed_growth": float(speed_future - speed_now),
        "wind_dir_now_deg": dir_now,
        "wind_dir_future_deg": dir_future,
        "dir_change_deg": abs(circular_delta_deg(dir_now, dir_future)),
    }


def min_gap_ok(ts: datetime, selected: list[dict], min_gap: timedelta) -> bool:
    return all(abs(ts - datetime.strptime(str(r["timestamp"]), TIMESTAMP_FMT)) >= min_gap for r in selected)


def select_group_cases(df: pd.DataFrame, group: str, count: int, min_gap: timedelta) -> pd.DataFrame:
    sub = df[df["selection_group"] == group].sort_values("selection_score", ascending=False)
    selected: list[dict] = []
    for row in sub.to_dict("records"):
        ts = datetime.strptime(str(row["timestamp"]), TIMESTAMP_FMT)
        if min_gap_ok(ts, selected, min_gap):
            selected.append(row)
        if len(selected) >= count:
            break
    return pd.DataFrame(selected)


def row_label(group: str, row: pd.Series) -> str:
    action = str(row["first_action"])
    ws0 = float(row["speed_now_ms"])
    ws2 = float(row["speed_future_ms"])
    b0 = float(row["block0_norm"])
    b2 = float(row["block2_norm"])
    if group == "high_sustained":
        return f"High sustained pressure, planner {action}, WS {ws0:.1f}->{ws2:.1f} m/s"
    if group == "onset":
        return f"Transient onset, future pressure grows {b0:.2f}->{b2:.2f}"
    if group == "decay":
        return f"Pressure decay, planner should avoid chasing old target {b0:.2f}->{b2:.2f}"
    if group == "signflip":
        return f"Direction reversal, delta WD {float(row['dir_change_deg']):.0f} deg"
    if group == "lowrisk_hold":
        return f"Low-risk hold check, max pressure {float(row['max_block_norm']):.2f}"
    return str(group)


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_planner_out = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    sign_cfg = json.loads((base_planner_out / "diagnostics" / "a01_pressure_vec_sign_convention.json").read_text())
    discount_cfg = json.loads((base_planner_out / "diagnostics" / "a1_block_discount_config.json").read_text())
    cfg = PlannerConfig(pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]))
    discounts = discount_cfg["default_discount_blocks"]

    replay = Fino1ReplayDataset(
        dataset_dir=repo_root / "data" / "processed" / "wind_ml_10min" / "ballast_decision_fino1_meteo_aux_v1",
        split=str(args.split),
    )

    coarse_rows: list[dict] = []
    samples = replay.samples[:: max(1, int(args.stride))]
    sample_by_ts = {sample.history_end.strftime(TIMESTAMP_FMT): sample for sample in samples}
    for idx, sample in enumerate(samples):
        if idx % 10000 == 0 and idx > 0:
            print(f"coarse scanned {idx}/{len(samples)} samples", flush=True)
        future = np.asarray(sample.y_uv_raw, dtype=float)
        if future.shape[0] < 6:
            continue
        try:
            blocks = compute_pressure_blocks(future, discounts, cfg)
        except Exception:
            continue
        row = base_row_from_sample(sample, blocks)
        for group, score in classify_coarse_row(row):
            coarse_rows.append({**row, "selection_group": group, "coarse_score": float(score)})

    coarse = pd.DataFrame(coarse_rows)
    coarse.to_csv(out_dir / "mechanism_coarse_scan.csv", index=False)

    rows: list[dict] = []
    planner_candidates = max(1, int(args.planner_candidates_per_group))
    for group in VALIDATION_GROUPS:
        sub = (
            coarse[coarse["selection_group"] == group]
            .sort_values("coarse_score", ascending=False)
            .head(planner_candidates)
        )
        for row in sub.to_dict("records"):
            sample = sample_by_ts.get(str(row["timestamp"]))
            if sample is None:
                continue
            try:
                best, _ = planner_best(sample, cfg, discounts)
            except Exception:
                continue
            first_action = str(best["sequence"][0])
            active_actions = sum(1 for a in best["sequence"] if str(a) != "hold")
            enriched = {
                **row,
                "first_action": first_action,
                "best_sequence": ">".join(best["sequence"]),
                "active_action_ratio": float(active_actions / max(len(best["sequence"]), 1)),
                "max_envelope_norm": float(best.get("costs", {}).get("max_envelope_norm", float("nan"))),
                "hard_reject_reason": str(best.get("hard_reject_reason", "")),
            }
            for refined_group, score in classify_row(enriched):
                if refined_group == group:
                    rows.append({**enriched, "selection_score": float(score)})

    full = pd.DataFrame(rows)
    full.to_csv(out_dir / "mechanism_candidate_scan.csv", index=False)
    top_parts = []
    for group in VALIDATION_GROUPS:
        top_parts.append(
            full[full["selection_group"] == group]
            .sort_values("selection_score", ascending=False)
            .head(max(1, int(args.top_per_group)))
        )
    top = pd.concat(top_parts, ignore_index=True) if top_parts else pd.DataFrame()
    top.to_csv(out_dir / "mechanism_top_candidates.csv", index=False)

    validation_parts = [
        select_group_cases(full, group, int(args.validation_per_group), timedelta(hours=24))
        for group in VALIDATION_GROUPS
    ]
    validation = pd.concat(validation_parts, ignore_index=True) if validation_parts else pd.DataFrame()
    case_rows = []
    for idx, row in enumerate(validation.to_dict("records"), start=1):
        group = str(row["selection_group"])
        case_rows.append({
            "case_id": f"{group}_{idx:02d}",
            "timestamp": row["timestamp"],
            "selection_group": group,
            "label": row_label(group, pd.Series(row)),
            "first_action": row["first_action"],
            "best_sequence": row["best_sequence"],
            "block0_norm": row["block0_norm"],
            "block2_norm": row["block2_norm"],
            "speed_now_ms": row["speed_now_ms"],
            "speed_future_ms": row["speed_future_ms"],
            "wind_dir_now_deg": row["wind_dir_now_deg"],
            "wind_dir_future_deg": row["wind_dir_future_deg"],
            "dir_change_deg": row["dir_change_deg"],
            "selection_score": row["selection_score"],
        })
    cases = pd.DataFrame(case_rows)
    cases.to_csv(out_dir / "mechanism_validation_cases.csv", index=False)

    lines = [
        "# Prediction-primary mechanism case selection",
        "",
        f"- elapsed: `{time.perf_counter() - t0:.1f}s`",
        f"- replay split: `{args.split}`",
        f"- samples scanned: `{len(samples)}` using stride `{int(args.stride)}`",
        f"- coarse candidate rows: `{len(coarse)}`",
        f"- planner-refined candidate rows: `{len(full)}`",
        f"- planner candidates per group: `{planner_candidates}`",
        f"- proposed mechanism validation cases: `{len(cases)}`",
        "",
        "## Proposed mechanism validation cases",
        "",
        "| group | timestamp | action | b0->b2 | WS now->future | WD now->future | label |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for _, r in cases.iterrows():
        lines.append(
            f"| {r['selection_group']} | {r['timestamp']} | {r['first_action']} | "
            f"{float(r['block0_norm']):.2f}->{float(r['block2_norm']):.2f} | "
            f"{float(r['speed_now_ms']):.1f}->{float(r['speed_future_ms']):.1f} | "
            f"{float(r['wind_dir_now_deg']):.0f}->{float(r['wind_dir_future_deg']):.0f} | "
            f"{r['label']} |"
        )
    lines += ["", "## Top candidates by group"]
    for group in VALIDATION_GROUPS:
        lines += ["", f"### {group}", "", "| timestamp | action | score | b0 | b2 | WS now | WS future | dWD |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        sub = top[top["selection_group"] == group].head(max(1, int(args.show_top)))
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['timestamp']} | {r['first_action']} | {float(r['selection_score']):.2f} | "
                f"{float(r['block0_norm']):.2f} | {float(r['block2_norm']):.2f} | "
                f"{float(r['speed_now_ms']):.1f} | {float(r['speed_future_ms']):.1f} | "
                f"{float(r['dir_change_deg']):.0f} |"
            )
    report = out_dir / "mechanism_selection_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


def _circular_mean(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    rad = np.deg2rad(vals)
    return float((np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) + 360.0) % 360.0)


if __name__ == "__main__":
    main()
