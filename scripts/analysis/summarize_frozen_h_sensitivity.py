#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


THRESHOLDS = ("1p5", "2", "3", "4", "5", "7p5", "10")


def raw_case_id(case_id: str) -> str:
    parts = str(case_id).split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else str(case_id)


def intervals_overlap(
    start_a: pd.Timestamp,
    end_a: pd.Timestamp,
    start_b: pd.Timestamp,
    end_b: pd.Timestamp,
) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def connected_components(cases: pd.DataFrame, start_col: str, end_col: str) -> list[list[int]]:
    ordered = cases.sort_values(start_col).copy()
    components: list[list[int]] = []
    current: list[int] = []
    current_end: pd.Timestamp | None = None
    for row in ordered.itertuples():
        start = getattr(row, start_col)
        end = getattr(row, end_col)
        if current_end is None or start >= current_end:
            if current:
                components.append(current)
            current = [row.Index]
            current_end = end
        else:
            current.append(row.Index)
            current_end = max(current_end, end)
    if current:
        components.append(current)
    return components


def greedy_nonoverlap(cases: pd.DataFrame, start_col: str, end_col: str) -> list[int]:
    keep: list[int] = []
    previous_end: pd.Timestamp | None = None
    for row in cases.sort_values(start_col).itertuples():
        start = getattr(row, start_col)
        end = getattr(row, end_col)
        if previous_end is None or start >= previous_end:
            keep.append(row.Index)
            previous_end = end
    return keep


def metric_summary(frame: pd.DataFrame, duration_s: float) -> dict:
    baseline = pd.to_numeric(frame["closed_pump_work_m3"], errors="raise").to_numpy(float)
    primary = pd.to_numeric(frame["primary_pump_work_m3"], errors="raise").to_numpy(float)
    reduction = 100.0 * float(np.sum(baseline - primary)) / float(np.sum(baseline))
    total_duration = float(duration_s) * len(frame)
    posture = {}
    for threshold in THRESHOLDS:
        baseline_time = float(frame[f"closed_time_over_{threshold}deg_s"].sum())
        primary_time = float(frame[f"primary_time_over_{threshold}deg_s"].sum())
        posture[threshold] = {
            "baseline_pct": 100.0 * baseline_time / total_duration,
            "primary_pct": 100.0 * primary_time / total_duration,
            "delta_percentage_points": 100.0 * (primary_time - baseline_time) / total_duration,
        }
    return {
        "case_count": int(len(frame)),
        "nominal_duration_h": float(len(frame) * duration_s / 3600.0),
        "baseline_pump_m3": float(np.sum(baseline)),
        "primary_pump_m3": float(np.sum(primary)),
        "pump_reduction_pct": reduction,
        "improved_case_count": int(np.sum(primary < baseline)),
        "improved_case_fraction": float(np.mean(primary < baseline)),
        "posture": posture,
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    components: list[list[int]],
    *,
    seed: int,
    iterations: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    baseline = pd.to_numeric(frame["closed_pump_work_m3"], errors="raise")
    primary = pd.to_numeric(frame["primary_pump_work_m3"], errors="raise")
    cluster_totals = np.array(
        [
            [float(baseline.loc[indices].sum()), float(primary.loc[indices].sum())]
            for indices in components
        ],
        dtype=float,
    )
    values = np.empty(iterations, dtype=float)
    n_clusters = len(components)
    for idx in range(iterations):
        sampled = rng.integers(0, n_clusters, n_clusters)
        base = float(cluster_totals[sampled, 0].sum())
        pred = float(cluster_totals[sampled, 1].sum())
        values[idx] = 100.0 * (base - pred) / base
    return np.percentile(values, [2.5, 97.5]).astype(float).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--holdout-csv", type=Path, required=True)
    parser.add_argument("--development-csv", type=Path, required=True)
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=21600.0)
    parser.add_argument("--history-min", type=float, default=110.0)
    parser.add_argument("--forecast-min", type=float, default=60.0)
    parser.add_argument("--resolution-min", type=float, default=10.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv).copy()
    summary["raw_case_id"] = summary["case_id"].map(raw_case_id)
    summary["timestamp"] = pd.to_datetime(summary["timestamp"], errors="raise")
    holdout = pd.read_csv(args.holdout_csv)
    holdout["timestamp"] = pd.to_datetime(holdout["timestamp"], errors="raise")
    development = pd.read_csv(args.development_csv)
    development["timestamp"] = pd.to_datetime(development["timestamp"], errors="raise")
    if set(summary["raw_case_id"]) != set(holdout["case_id"]):
        raise ValueError("summary and holdout case sets differ")

    cases = summary.merge(
        holdout[["case_id", "timestamp"]].rename(
            columns={"case_id": "raw_case_id", "timestamp": "holdout_timestamp"}
        ),
        on="raw_case_id",
        how="left",
        validate="one_to_one",
    )
    if not (cases["timestamp"] == cases["holdout_timestamp"]).all():
        raise ValueError("summary timestamps do not match frozen holdout manifest")

    cases = cases.copy()
    duration = pd.Timedelta(seconds=float(args.duration_s))
    history = pd.Timedelta(minutes=float(args.history_min))
    forecast_tail = pd.Timedelta(
        minutes=float(args.forecast_min) - float(args.resolution_min)
    )
    cases = cases.assign(
        control_start=cases["timestamp"],
        control_end=cases["timestamp"] + duration,
        footprint_start=cases["timestamp"] - history,
        footprint_end=cases["timestamp"] + duration + forecast_tail,
    )

    development["footprint_start"] = development["timestamp"] - history
    development["footprint_end"] = development["timestamp"] + duration + forecast_tail
    development_overlap_ids = []
    for row in cases.itertuples():
        if any(
            intervals_overlap(
                row.footprint_start,
                row.footprint_end,
                dev.footprint_start,
                dev.footprint_end,
            )
            for dev in development.itertuples()
        ):
            development_overlap_ids.append(row.raw_case_id)

    sample_index = pd.read_csv(args.sample_index, usecols=["split", "history_end"])
    test_timestamps = set(
        pd.to_datetime(
            sample_index.loc[sample_index["split"] == "test", "history_end"],
            errors="raise",
        )
    )
    missing_buckets = Counter()
    bucket_count = int(round(float(args.duration_s) / (float(args.resolution_min) * 60.0)))
    for row in cases.itertuples():
        for bucket in range(bucket_count):
            timestamp = row.timestamp + pd.Timedelta(
                minutes=float(args.resolution_min) * bucket
            )
            if timestamp not in test_timestamps:
                missing_buckets[row.raw_case_id] += 1

    control_components = connected_components(cases, "control_start", "control_end")
    footprint_clean = cases.loc[
        ~cases["raw_case_id"].isin(development_overlap_ids)
    ].copy()
    strict_indices = greedy_nonoverlap(
        footprint_clean, "footprint_start", "footprint_end"
    )
    complete_forecast = cases.loc[
        ~cases["raw_case_id"].isin(missing_buckets.keys())
    ].copy()
    strict_complete = cases.loc[
        cases.index.isin(strict_indices)
        & ~cases["raw_case_id"].isin(missing_buckets.keys())
    ].copy()

    subsets = {
        "all_150": cases,
        "complete_forecast_only": complete_forecast,
        "strict_nonoverlap_footprint": cases.loc[strict_indices].copy(),
        "strict_nonoverlap_and_complete_forecast": strict_complete,
    }
    result = {
        "schema_version": "frozen_h_sensitivity_summary.v1",
        "selection_rules": {
            "development_overlap_exclusion": (
                "Exclude holdout cases whose [start-110min, end+50min] model/control "
                "footprint overlaps a development-case footprint."
            ),
            "holdout_nonoverlap_rule": (
                "Chronological greedy retention on the same footprint, independent of outcomes."
            ),
            "complete_forecast_rule": (
                "Exclude cases with any missing 10-minute test-split LSTM sample."
            ),
        },
        "development_overlap_case_ids": sorted(development_overlap_ids),
        "missing_forecast_buckets_by_case": dict(sorted(missing_buckets.items())),
        "control_interval_cluster_count": len(control_components),
        "control_interval_cluster_sizes": sorted(
            [len(component) for component in control_components], reverse=True
        ),
        "cluster_bootstrap_95pct_ci": cluster_bootstrap(
            cases,
            control_components,
            seed=int(args.seed),
            iterations=int(args.bootstrap_iterations),
        ),
        "subsets": {
            name: {
                **metric_summary(frame, float(args.duration_s)),
                "case_ids": frame["raw_case_id"].tolist(),
            }
            for name, frame in subsets.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
