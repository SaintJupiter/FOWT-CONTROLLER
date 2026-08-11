#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


THRESHOLDS = ("1p5", "2", "3", "4", "5", "7p5", "10")


def numeric_sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        raise ValueError(f"missing required column: {column}")
    return float(pd.to_numeric(frame[column], errors="raise").sum())


def bootstrap_pump_reduction(
    baseline: np.ndarray,
    primary: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_cases = len(baseline)
    values = np.empty(iterations, dtype=float)
    for idx in range(iterations):
        sample = rng.integers(0, n_cases, n_cases)
        denominator = float(np.sum(baseline[sample]))
        values[idx] = 100.0 * float(
            np.sum(baseline[sample] - primary[sample])
        ) / denominator
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--duration-s", type=float, default=21600.0)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    summary_paths = sorted(
        path
        for batch_dir in args.batch_dir
        for path in batch_dir.glob("task_*/casebook_summary.csv")
    )
    protocol_paths = sorted(
        path
        for batch_dir in args.batch_dir
        for path in batch_dir.glob("task_*/run_protocol.json")
    )
    if not summary_paths:
        raise ValueError(f"no task summaries found below {args.batch_dir}")
    if len(summary_paths) != len(protocol_paths):
        raise ValueError("task summary/protocol count mismatch")

    frames = []
    for path in summary_paths:
        frame = pd.read_csv(path)
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    {"source_task": [f"{path.parent.parent.name}/{path.parent.name}"] * len(frame)}
                ),
            ],
            axis=1,
        )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != int(args.expected_cases):
        raise ValueError(
            f"expected {args.expected_cases} cases, found {len(combined)}"
        )
    if combined["case_id"].duplicated().any():
        duplicates = combined.loc[combined["case_id"].duplicated(), "case_id"].tolist()
        raise ValueError(f"duplicate case ids: {duplicates}")

    protocols = [json.loads(path.read_text(encoding="utf-8")) for path in protocol_paths]
    for protocol in protocols:
        duration = float(protocol["inputs"]["duration_s"])
        if duration != float(args.duration_s):
            raise ValueError(f"unexpected duration_s: {duration}")
        if protocol["identity"]["forecast_source_effective"] != "lstm_dual_head_preview":
            raise ValueError("task did not use the frozen learned forecast source")
        if int(protocol["result"]["issue_count"]) != 0:
            raise ValueError("task protocol reports validation issues")

    baseline = pd.to_numeric(combined["closed_pump_work_m3"], errors="raise").to_numpy(float)
    primary = pd.to_numeric(combined["primary_pump_work_m3"], errors="raise").to_numpy(float)
    per_case_reduction = 100.0 * (baseline - primary) / baseline
    aggregate_reduction = 100.0 * float(np.sum(baseline - primary)) / float(np.sum(baseline))
    ci_low, ci_high = bootstrap_pump_reduction(
        baseline,
        primary,
        seed=int(args.seed),
        iterations=int(args.bootstrap_iterations),
    )

    duration_total = float(args.duration_s) * len(combined)
    posture = {}
    for threshold in THRESHOLDS:
        baseline_time = numeric_sum(combined, f"closed_time_over_{threshold}deg_s")
        primary_time = numeric_sum(combined, f"primary_time_over_{threshold}deg_s")
        baseline_pct = 100.0 * baseline_time / duration_total
        primary_pct = 100.0 * primary_time / duration_total
        posture[threshold] = {
            "baseline_time_s": baseline_time,
            "primary_time_s": primary_time,
            "baseline_time_pct": baseline_pct,
            "primary_time_pct": primary_pct,
            "delta_percentage_points": primary_pct - baseline_pct,
        }

    compact = pd.DataFrame(
        {
            "case_id": combined["case_id"],
            "timestamp": combined["timestamp"],
            "source_task": combined["source_task"],
            "baseline_pump_m3": baseline,
            "primary_pump_m3": primary,
            "pump_reduction_pct": per_case_reduction,
            "baseline_switches": pd.to_numeric(
                combined["closed_latch_switches"], errors="raise"
            ),
            "primary_switches": pd.to_numeric(
                combined["primary_latch_switches"], errors="raise"
            ),
        }
    ).sort_values("timestamp")

    out_dir = args.out_dir if args.out_dir is not None else args.batch_dir[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    compact.to_csv(out_dir / "paired_results.csv", index=False)
    combined.to_csv(out_dir / "combined_casebook_summary.csv", index=False)

    baseline_switches = numeric_sum(combined, "closed_latch_switches")
    primary_switches = numeric_sum(combined, "primary_latch_switches")
    result = {
        "schema_version": "frozen_h_validation_summary.v1",
        "case_count": int(len(combined)),
        "duration_s_per_case": float(args.duration_s),
        "total_duration_h": duration_total / 3600.0,
        "task_count": int(len(summary_paths)),
        "issue_count": int(sum(int(p["result"]["issue_count"]) for p in protocols)),
        "pump": {
            "baseline_total_m3": float(np.sum(baseline)),
            "primary_total_m3": float(np.sum(primary)),
            "aggregate_reduction_pct": aggregate_reduction,
            "bootstrap_95pct_ci": [ci_low, ci_high],
            "improved_case_count": int(np.sum(primary < baseline)),
            "improved_case_fraction": float(np.mean(primary < baseline)),
            "per_case_reduction_median_pct": float(np.median(per_case_reduction)),
            "per_case_reduction_p25_pct": float(np.percentile(per_case_reduction, 25)),
            "per_case_reduction_p75_pct": float(np.percentile(per_case_reduction, 75)),
        },
        "switches": {
            "baseline_total": baseline_switches,
            "primary_total": primary_switches,
            "reduction_pct": 100.0 * (baseline_switches - primary_switches) / baseline_switches,
        },
        "posture": posture,
        "source_summaries": [str(path) for path in summary_paths],
        "source_protocols": [str(path) for path in protocol_paths],
    }
    output = out_dir / "validation_summary.json"
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
