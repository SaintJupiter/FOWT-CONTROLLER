#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CORE_HASHES = {
    "src/wind_prediction/ballast_planner_provider.py": (
        "f6fe5d74c567cbfa39d1ea97c402d602fb4cf42a2d0acf4daa5cc4b3e3aaa594"
    ),
    "scripts/analysis/run_prediction_primary_casebook.py": (
        "5455b38f469c7c96e55dad5dd417c330fde1c13b109d0bc8e6867d95bd5a584c"
    ),
    "archive/legacy_fowt_control/controllers_extras.py": (
        "3bf06f111aff63319c0b489123b0ce8e28b7ea152bfaae563420e5928821da14"
    ),
    "archive/legacy_fowt_control/run_validation.py": (
        "155b4bf99259318eda96e64cffab7dff1a748909f6bc65af2e40afc7bf1cd9a8"
    ),
}

PAIR_COLUMNS = ("t_s", "wind_speed", "wind_dir_deg", "thrust_n")
INITIAL_COLUMNS = (
    "t_s",
    "wind_speed",
    "wind_dir_deg",
    "thrust_n",
    "pitch_deg",
    "roll_deg",
    "heave_m",
    "tank1_kg",
    "tank2_kg",
    "tank3_kg",
    "ballast_total_kg",
)
PRIMARY_TELEMETRY = (
    "forecast_safe_deadband_enabled",
    "forecast_safe_deadband_active",
    "forecast_safe_deadband_has_future",
    "forecast_safe_deadband_trust_ok",
    "forecast_safe_deadband_event_probability",
    "forecast_safe_deadband_event_probability_available",
    "preview_event_risk_prob_0_20m",
    "preview_event_risk_prob_20_40m",
    "preview_event_risk_prob_40_60m",
    "deadband_target_release_active",
    "forecast_safe_deadband_provider_target_synced",
    "forecast_safe_deadband_provider_replan_requested",
    "preview_primary_enabled",
    "preview_primary_active",
    "preview_primary_applied",
    "preview_primary_candidate_applied",
)
BASELINE_DISABLED_COLUMNS = (
    "forecast_safe_deadband_enabled",
    "forecast_safe_deadband_active",
    "forecast_safe_deadband_event_probability_available",
    "preview_primary_enabled",
    "preview_primary_active",
    "preview_primary_applied",
    "preview_primary_candidate_applied",
    "deadband_target_release_active",
)
POSTURE_THRESHOLDS = {
    "1p5": 1.5,
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
    "7p5": 7.5,
    "10": 10.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_timestamp(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="raise").dt.strftime("%Y-%m-%d %H:%M:%S")


def scalar_equal(left: Any, right: Any, atol: float = 1e-9) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, (float, int, np.number)) and isinstance(
        right, (float, int, np.number)
    ):
        return bool(np.isclose(float(left), float(right), atol=atol, rtol=0.0))
    return bool(left == right)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, details: Any, severity: str = "critical") -> None:
        self.checks.append(
            {
                "name": name,
                "status": "pass" if passed else "fail",
                "severity": severity,
                "details": details,
            }
        )

    def warning(self, name: str, details: Any) -> None:
        self.checks.append(
            {
                "name": name,
                "status": "warning",
                "severity": "caveat",
                "details": details,
            }
        )

    def result(self, evidence: dict[str, Any]) -> dict[str, Any]:
        failures = [item for item in self.checks if item["status"] == "fail"]
        warnings = [item for item in self.checks if item["status"] == "warning"]
        return {
            "schema_version": "frozen_h_run_integrity_audit.v1",
            "overall_status": "fail" if failures else ("pass_with_caveats" if warnings else "pass"),
            "failure_count": len(failures),
            "warning_count": len(warnings),
            "checks": self.checks,
            "evidence": evidence,
        }


def interval_overlap_report(cases: pd.DataFrame, duration_h: float) -> dict[str, Any]:
    starts = pd.to_datetime(cases["timestamp"], errors="raise")
    intervals = sorted(
        [
            (
                str(row.case_id),
                pd.Timestamp(row.timestamp),
                pd.Timestamp(row.timestamp) + pd.Timedelta(hours=duration_h),
            )
            for row in cases.assign(timestamp=starts).itertuples(index=False)
        ],
        key=lambda item: item[1],
    )
    overlaps: list[dict[str, Any]] = []
    for idx, (case_a, start_a, end_a) in enumerate(intervals):
        for case_b, start_b, end_b in intervals[idx + 1 :]:
            if start_b >= end_a:
                break
            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            if overlap_start < overlap_end:
                overlaps.append(
                    {
                        "case_a": case_a,
                        "case_b": case_b,
                        "start_a": str(start_a),
                        "start_b": str(start_b),
                        "overlap_h": (overlap_end - overlap_start).total_seconds() / 3600.0,
                    }
                )

    merged: list[list[pd.Timestamp]] = []
    for _, start, end in intervals:
        if not merged or start >= merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    union_h = sum((end - start).total_seconds() for start, end in merged) / 3600.0
    nominal_h = len(intervals) * duration_h
    return {
        "nominal_h": nominal_h,
        "nonoverlapping_union_h": union_h,
        "duplicated_exposure_h": nominal_h - union_h,
        "overlap_pair_count": len(overlaps),
        "overlaps": overlaps,
    }


def event_stats(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        result[column] = {
            "count": int(values.size),
            "finite_count": int(np.isfinite(values).sum()),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/wind_prediction/frozen_h_holdout150_20260710"),
    )
    parser.add_argument("--paired-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--duplicate-primary-dir", type=Path)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--duration-s", type=int, default=21600)
    parser.add_argument(
        "--source-cases",
        type=Path,
        default=Path(
            "outputs/wind_prediction/posture_debt_full170_20260613/original170_cases.csv"
        ),
    )
    parser.add_argument(
        "--development-cases",
        type=Path,
        default=Path(
            "outputs/wind_prediction/fullsync_stratified20_audit_20260710/cases/stratified20_master.csv"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = Audit()
    evidence: dict[str, Any] = {}
    cases_dir = args.root / "cases"
    manifest_path = cases_dir / "frozen_h_holdout_manifest.json"
    holdout_path = cases_dir / "frozen_h_holdout150.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = pd.read_csv(args.source_cases)
    development = pd.read_csv(args.development_cases)
    holdout = pd.read_csv(holdout_path)
    for frame in (source, development, holdout):
        frame["timestamp"] = normalize_timestamp(frame["timestamp"])

    source_ts = set(source["timestamp"])
    development_ts = set(development["timestamp"])
    holdout_ts = set(holdout["timestamp"])
    audit.add(
        "holdout_is_exact_source_minus_development",
        holdout_ts == source_ts - development_ts,
        {
            "source": len(source_ts),
            "development": len(development_ts),
            "holdout": len(holdout_ts),
            "expected_holdout": len(source_ts - development_ts),
            "development_holdout_overlap": len(development_ts & holdout_ts),
        },
    )
    audit.add(
        "case_start_timestamps_unique",
        len(source_ts) == len(source)
        and len(development_ts) == len(development)
        and len(holdout_ts) == len(holdout),
        {
            "source_duplicate_starts": int(source["timestamp"].duplicated().sum()),
            "development_duplicate_starts": int(development["timestamp"].duplicated().sum()),
            "holdout_duplicate_starts": int(holdout["timestamp"].duplicated().sum()),
        },
    )
    audit.add(
        "holdout_case_ids_unique",
        not holdout["case_id"].duplicated().any(),
        {"duplicate_case_ids": holdout.loc[holdout["case_id"].duplicated(), "case_id"].tolist()},
    )
    label_splits = holdout["label"].str.extract(r"split=([^ |]+)", expand=False)
    declared_splits = label_splits.dropna()
    audit.add(
        "holdout_label_split_declarations_do_not_conflict",
        bool((declared_splits == "test").all()),
        {
            "declared": declared_splits.value_counts().to_dict(),
            "undeclared_count": int(label_splits.isna().sum()),
        },
    )

    manifest_mismatches = []
    for name, expected_hash in manifest["files"].items():
        path = cases_dir / name
        observed_hash = sha256(path) if path.exists() else None
        if observed_hash != expected_hash:
            manifest_mismatches.append(
                {"path": str(path), "expected": expected_hash, "observed": observed_hash}
            )
    audit.add(
        "case_manifest_hashes_match",
        not manifest_mismatches,
        {"file_count": len(manifest["files"]), "mismatches": manifest_mismatches},
    )

    core_mismatches = []
    core_observed: dict[str, str | None] = {}
    for raw_path, expected_hash in CORE_HASHES.items():
        path = Path(raw_path)
        observed_hash = sha256(path) if path.exists() else None
        core_observed[raw_path] = observed_hash
        if observed_hash != expected_hash:
            core_mismatches.append(
                {"path": raw_path, "expected": expected_hash, "observed": observed_hash}
            )
    audit.add(
        "frozen_core_hashes_match",
        not core_mismatches,
        {"observed": core_observed, "mismatches": core_mismatches},
    )

    overlap_report = interval_overlap_report(holdout, args.duration_s / 3600.0)
    if overlap_report["overlap_pair_count"]:
        audit.warning("holdout_intervals_partially_overlap", overlap_report)
    else:
        audit.add("holdout_intervals_do_not_overlap", True, overlap_report)
    evidence["holdout_intervals"] = overlap_report

    sample_index = pd.read_csv(args.dataset_dir / "sample_index.csv.gz", usecols=["split", "history_end"])
    test_history_end = set(
        normalize_timestamp(sample_index.loc[sample_index["split"] == "test", "history_end"])
    )
    missing_replay_timestamps = []
    for row in holdout.itertuples(index=False):
        start = pd.Timestamp(row.timestamp)
        for bucket in range(int(args.duration_s // 600)):
            timestamp = (start + timedelta(minutes=10 * bucket)).strftime("%Y-%m-%d %H:%M:%S")
            if timestamp not in test_history_end:
                missing_replay_timestamps.append(
                    {"case_id": str(row.case_id), "bucket": bucket, "timestamp": timestamp}
                )
    if missing_replay_timestamps:
        audit.warning(
            "some_holdout_buckets_have_no_lstm_sample_and_must_fail_closed",
            {
                "required_bucket_count": int(len(holdout) * (args.duration_s // 600)),
                "missing_count": len(missing_replay_timestamps),
                "examples": missing_replay_timestamps[:10],
            },
        )
    else:
        audit.add(
            "all_holdout_replay_buckets_exist_in_test_split",
            True,
            {
                "required_bucket_count": int(len(holdout) * (args.duration_s // 600)),
                "missing_count": 0,
                "examples": [],
            },
        )

    summary_paths = sorted(
        path for paired_dir in args.paired_dir for path in paired_dir.glob("task_*/casebook_summary.csv")
    )
    protocol_paths = sorted(
        path for paired_dir in args.paired_dir for path in paired_dir.glob("task_*/run_protocol.json")
    )
    audit.add(
        "task_summary_protocol_counts_match",
        len(summary_paths) == len(protocol_paths),
        {"summaries": len(summary_paths), "protocols": len(protocol_paths)},
    )

    frames = []
    protocol_issues = []
    task_case_mismatches = []
    timeseries_count = 0
    planner_log_count = 0
    for summary_path in summary_paths:
        task_dir = summary_path.parent
        frame = pd.read_csv(summary_path)
        frame["timestamp"] = normalize_timestamp(frame["timestamp"])
        frame = pd.concat(
            [
                frame,
                pd.DataFrame({"source_task": [str(task_dir)] * len(frame)}),
            ],
            axis=1,
        )
        frames.append(frame)
        protocol_path = task_dir / "run_protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        identity = protocol["identity"]
        inputs = protocol["inputs"]
        result = protocol["result"]
        expected = {
            "primary_only": False,
            "forecast_source_effective": "lstm_dual_head_preview",
            "replay_split": "test",
            "duration_s": float(args.duration_s),
            "completed_cases": len(frame),
            "issue_count": 0,
        }
        observed = {
            "primary_only": identity.get("primary_only"),
            "forecast_source_effective": identity.get("forecast_source_effective"),
            "replay_split": identity.get("replay_split"),
            "duration_s": float(inputs.get("duration_s")),
            "completed_cases": int(result.get("completed_cases")),
            "issue_count": int(result.get("issue_count")),
        }
        differences = {
            key: {"expected": expected[key], "observed": observed[key]}
            for key in expected
            if not scalar_equal(expected[key], observed[key])
        }
        if differences:
            protocol_issues.append({"task": str(task_dir), "differences": differences})

        case_source = Path(identity["cases_source"])
        input_cases = pd.read_csv(case_source)
        input_cases["timestamp"] = normalize_timestamp(input_cases["timestamp"])
        expected_keys = set(zip(input_cases["case_id"], input_cases["timestamp"]))
        observed_keys = set(zip(frame["case_id"].str.replace(r"^\d+_", "", regex=True), frame["timestamp"]))
        if expected_keys != observed_keys:
            task_case_mismatches.append(
                {
                    "task": str(task_dir),
                    "missing": sorted(expected_keys - observed_keys),
                    "extra": sorted(observed_keys - expected_keys),
                }
            )
        timeseries_count += len(list((task_dir / "timeseries").glob("*.csv")))
        planner_log_count += len(list((task_dir / "planner_logs").glob("*.csv")))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    audit.add(
        "paired_protocols_are_frozen_and_complete",
        not protocol_issues,
        {"task_count": len(protocol_paths), "issues": protocol_issues},
    )
    audit.add(
        "task_outputs_match_declared_input_cases",
        not task_case_mismatches,
        {"issues": task_case_mismatches},
    )
    audit.add(
        "paired_case_count_matches_request",
        len(combined) == args.expected_cases and not combined["case_id"].duplicated().any(),
        {
            "observed": len(combined),
            "expected": args.expected_cases,
            "duplicate_case_ids": combined.loc[
                combined["case_id"].duplicated(), "case_id"
            ].tolist(),
        },
    )
    audit.add(
        "expected_raw_artifact_counts",
        timeseries_count == 2 * args.expected_cases and planner_log_count == args.expected_cases,
        {
            "timeseries": timeseries_count,
            "expected_timeseries": 2 * args.expected_cases,
            "planner_logs": planner_log_count,
            "expected_planner_logs": args.expected_cases,
        },
    )

    pair_mismatches = []
    initial_mismatches = []
    metric_mismatches = []
    primary_event_frames = []
    baseline_nonzero = Counter()
    planner_sources = Counter()
    planner_actions = Counter()
    planner_sequences = Counter()
    planner_future_values = Counter()
    planner_event_values: list[np.ndarray] = []
    for paired_dir in args.paired_dir:
        for primary_path in sorted(
            paired_dir.glob("task_*/timeseries/*_prediction_primary_econ_timeseries.csv")
        ):
            closed_path = primary_path.with_name(
                primary_path.name.replace("prediction_primary_econ", "closed_only")
            )
            task_dir = primary_path.parent.parent
            prefix = primary_path.name.removesuffix("_prediction_primary_econ_timeseries.csv")
            summary = pd.read_csv(task_dir / "casebook_summary.csv")
            summary_row = summary.loc[
                summary["case_id"].astype(str).map(
                    lambda case_id: prefix.startswith(case_id + "_")
                )
            ]
            if len(summary_row) != 1:
                metric_mismatches.append(
                    {"case_id": prefix, "issue": "summary row not uniquely resolved"}
                )
                continue
            summary_row = summary_row.iloc[0]

            needed = sorted(
                set(PAIR_COLUMNS)
                | set(INITIAL_COLUMNS)
                | set(PRIMARY_TELEMETRY)
                | set(BASELINE_DISABLED_COLUMNS)
                | {
                    "pump_rate1_m3min",
                    "pump_rate2_m3min",
                    "pump_rate3_m3min",
                    "pitch_deg",
                    "roll_deg",
                }
            )
            primary = pd.read_csv(primary_path, usecols=needed)
            closed = pd.read_csv(closed_path, usecols=needed)
            if len(primary) != args.duration_s or len(closed) != args.duration_s:
                pair_mismatches.append(
                    {
                        "case_id": prefix,
                        "issue": "unexpected row count",
                        "primary_rows": len(primary),
                        "closed_rows": len(closed),
                    }
                )
                continue
            differing_pair_columns = [
                column
                for column in PAIR_COLUMNS
                if not np.array_equal(
                    primary[column].to_numpy(), closed[column].to_numpy(), equal_nan=True
                )
            ]
            if differing_pair_columns:
                pair_mismatches.append(
                    {"case_id": prefix, "differing_columns": differing_pair_columns}
                )
            differing_initial_columns = [
                column
                for column in INITIAL_COLUMNS
                if not scalar_equal(primary.iloc[0][column], closed.iloc[0][column])
            ]
            if differing_initial_columns:
                initial_mismatches.append(
                    {"case_id": prefix, "differing_columns": differing_initial_columns}
                )

            primary_rate = primary[
                ["pump_rate1_m3min", "pump_rate2_m3min", "pump_rate3_m3min"]
            ].abs().sum(axis=1).to_numpy(float)
            closed_rate = closed[
                ["pump_rate1_m3min", "pump_rate2_m3min", "pump_rate3_m3min"]
            ].abs().sum(axis=1).to_numpy(float)
            primary_pump = float(
                np.trapezoid(primary_rate, primary["t_s"].to_numpy(float)) / 60.0
            )
            closed_pump = float(
                np.trapezoid(closed_rate, closed["t_s"].to_numpy(float)) / 60.0
            )
            if not np.isclose(primary_pump, summary_row["primary_pump_work_m3"], atol=1e-7):
                metric_mismatches.append(
                    {
                        "case_id": prefix,
                        "metric": "primary_pump_work_m3",
                        "recomputed": primary_pump,
                        "summary": float(summary_row["primary_pump_work_m3"]),
                    }
                )
            if not np.isclose(closed_pump, summary_row["closed_pump_work_m3"], atol=1e-7):
                metric_mismatches.append(
                    {
                        "case_id": prefix,
                        "metric": "closed_pump_work_m3",
                        "recomputed": closed_pump,
                        "summary": float(summary_row["closed_pump_work_m3"]),
                    }
                )
            for label, frame in (("primary", primary), ("closed", closed)):
                posture = np.maximum(
                    np.abs(pd.to_numeric(frame["pitch_deg"], errors="raise").to_numpy(float)),
                    np.abs(pd.to_numeric(frame["roll_deg"], errors="raise").to_numpy(float)),
                )
                for suffix, threshold in POSTURE_THRESHOLDS.items():
                    recomputed = float(np.sum(posture > threshold))
                    column = f"{label}_time_over_{suffix}deg_s"
                    if not np.isclose(recomputed, summary_row[column], atol=1e-9):
                        metric_mismatches.append(
                            {
                                "case_id": prefix,
                                "metric": column,
                                "recomputed": recomputed,
                                "summary": float(summary_row[column]),
                            }
                        )

            primary_event_frames.append(primary.loc[:, PRIMARY_TELEMETRY])
            for column in BASELINE_DISABLED_COLUMNS:
                baseline_nonzero[column] += int(
                    np.count_nonzero(pd.to_numeric(closed[column], errors="raise").to_numpy(float))
                )

            planner_path = task_dir / "planner_logs" / (
                prefix + "_prediction_primary_econ_planner_log.csv"
            )
            planner = pd.read_csv(
                planner_path,
                usecols=[
                    "forecast_source",
                    "forecast_has_future",
                    "first_action",
                    "best_sequence",
                    "event_risk_raw_prob_0_20m",
                    "event_risk_raw_prob_20_40m",
                    "event_risk_raw_prob_40_60m",
                ],
            )
            planner_sources.update(planner["forecast_source"].astype(str))
            planner_future_values.update(planner["forecast_has_future"].astype(str))
            planner_actions.update(planner["first_action"].astype(str))
            planner_sequences.update(planner["best_sequence"].astype(str))
            planner_event_values.append(
                planner[
                    [
                        "event_risk_raw_prob_0_20m",
                        "event_risk_raw_prob_20_40m",
                        "event_risk_raw_prob_40_60m",
                    ]
                ].to_numpy(float)
            )

    audit.add(
        "paired_external_inputs_are_rowwise_identical",
        not pair_mismatches,
        {"pair_count": args.expected_cases, "issues": pair_mismatches[:20]},
    )
    audit.add(
        "paired_initial_states_are_identical",
        not initial_mismatches,
        {"pair_count": args.expected_cases, "issues": initial_mismatches[:20]},
    )
    audit.add(
        "summary_metrics_recompute_from_raw_timeseries",
        not metric_mismatches,
        {"issue_count": len(metric_mismatches), "issues": metric_mismatches[:20]},
    )
    audit.add(
        "baseline_prediction_path_is_disabled",
        sum(baseline_nonzero.values()) == 0,
        dict(baseline_nonzero),
    )

    primary_events = pd.concat(primary_event_frames, ignore_index=True)
    telemetry_stats = event_stats(primary_events, PRIMARY_TELEMETRY)
    event_columns = (
        "preview_event_risk_prob_0_20m",
        "preview_event_risk_prob_20_40m",
        "preview_event_risk_prob_40_60m",
    )
    availability = pd.to_numeric(
        primary_events["forecast_safe_deadband_event_probability_available"],
        errors="raise",
    ).to_numpy(float)
    has_future = pd.to_numeric(
        primary_events["forecast_safe_deadband_has_future"], errors="raise"
    ).to_numpy(float)
    unavailable = availability < 0.5
    unavailable_fail_closed = True
    if np.any(unavailable):
        unavailable_fail_closed = bool(
            np.all(
                pd.to_numeric(
                    primary_events.loc[unavailable, "forecast_safe_deadband_active"],
                    errors="raise",
                ).to_numpy(float)
                == 0.0
            )
            and np.all(
                pd.to_numeric(
                    primary_events.loc[unavailable, "deadband_target_release_active"],
                    errors="raise",
                ).to_numpy(float)
                == 0.0
            )
            and all(
                np.all(
                    pd.to_numeric(
                        primary_events.loc[unavailable, column], errors="raise"
                    ).to_numpy(float)
                    == 0.0
                )
                for column in event_columns
            )
        )
    telemetry_ok = (
        np.array_equal(availability, has_future)
        and telemetry_stats["forecast_safe_deadband_enabled"]["min"] == 1.0
        and telemetry_stats["preview_primary_enabled"]["min"] == 1.0
        and telemetry_stats["preview_primary_applied"]["max"] == 1.0
        and unavailable_fail_closed
        and all(
            telemetry_stats[column]["finite_count"] == telemetry_stats[column]["count"]
            and 0.0 <= telemetry_stats[column]["min"]
            and telemetry_stats[column]["max"] <= 1.0
            and telemetry_stats[column]["std"] > 0.0
            for column in event_columns
        )
    )
    audit.add(
        "prediction_event_telemetry_is_available_finite_and_time_varying",
        telemetry_ok,
        {
            "telemetry": telemetry_stats,
            "unavailable_row_count": int(np.sum(unavailable)),
            "unavailable_rows_fail_closed_for_deadband_release": unavailable_fail_closed,
        },
    )
    if telemetry_stats["forecast_safe_deadband_provider_target_synced"]["max"] == 0.0:
        audit.warning(
            "provider_target_sync_path_not_used_by_frozen_h_profile",
            {
                "provider_target_synced_rows": 0,
                "provider_replan_requested_rows": int(
                    telemetry_stats[
                        "forecast_safe_deadband_provider_replan_requested"
                    ]["mean"]
                    * telemetry_stats[
                        "forecast_safe_deadband_provider_replan_requested"
                    ]["count"]
                ),
                "active_release_field": "deadband_target_release_active",
                "interpretation": (
                    "The frozen profile intentionally releases the command-layer target "
                    "without synchronizing the planner's retained absolute target."
                ),
            },
        )
    audit.add(
        "planner_logs_confirm_lstm_and_candidate_sequences",
        set(planner_sources) == {"lstm_dual_head_preview"}
        and set(planner_future_values) == {"1"}
        and len(planner_actions) >= 2
        and len(planner_sequences) >= 2,
        {
            "forecast_sources": dict(planner_sources),
            "forecast_has_future": dict(planner_future_values),
            "first_actions": dict(planner_actions),
            "unique_sequence_count": len(planner_sequences),
            "top_sequences": planner_sequences.most_common(10),
        },
    )
    planner_event_array = np.concatenate(planner_event_values, axis=0)
    audit.add(
        "planner_event_probabilities_are_finite_and_time_varying",
        bool(
            np.isfinite(planner_event_array).all()
            and np.all((planner_event_array >= 0.0) & (planner_event_array <= 1.0))
            and np.all(np.std(planner_event_array, axis=0) > 0.0)
        ),
        {
            "rows": int(planner_event_array.shape[0]),
            "min_by_head": np.min(planner_event_array, axis=0).tolist(),
            "max_by_head": np.max(planner_event_array, axis=0).tolist(),
            "mean_by_head": np.mean(planner_event_array, axis=0).tolist(),
            "std_by_head": np.std(planner_event_array, axis=0).tolist(),
        },
    )

    duplicate_report: dict[str, Any] | None = None
    if args.duplicate_primary_dir is not None:
        original_paths = sorted(
            args.duplicate_primary_dir.glob(
                "task_*/timeseries/*_prediction_primary_econ_timeseries.csv"
            )
        )
        paired_paths = sorted(
            path
            for paired_dir in args.paired_dir
            for path in paired_dir.glob(
                "task_*/timeseries/*_prediction_primary_econ_timeseries.csv"
            )
        )
        original_hashes = {path.name: sha256(path) for path in original_paths}
        paired_hashes = {path.name: sha256(path) for path in paired_paths}
        common = sorted(set(original_hashes) & set(paired_hashes))
        mismatches = [name for name in common if original_hashes[name] != paired_hashes[name]]
        duplicate_report = {
            "original_count": len(original_hashes),
            "matching_paired_count": len(common),
            "byte_identical_count": len(common) - len(mismatches),
            "mismatches": mismatches,
        }
        audit.add(
            "duplicate_primary_run_is_byte_deterministic",
            bool(common) and not mismatches and len(common) == len(original_hashes),
            duplicate_report,
        )
    evidence["duplicate_primary_reproduction"] = duplicate_report

    model_hashes = {
        str(path.relative_to(args.model_dir)): sha256(path)
        for path in sorted(args.model_dir.glob("*"))
        if path.is_file()
    }
    evidence["model_artifact_hashes"] = model_hashes
    evidence["core_hashes"] = core_observed
    evidence["telemetry_stats"] = telemetry_stats
    evidence["planner"] = {
        "forecast_sources": dict(planner_sources),
        "forecast_has_future": dict(planner_future_values),
        "first_actions": dict(planner_actions),
        "unique_sequence_count": len(planner_sequences),
    }
    evidence["audited_case_count"] = int(len(combined))
    evidence["audited_duration_h"] = float(len(combined) * args.duration_s / 3600.0)

    result = audit.result(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result["failure_count"] else 0)


if __name__ == "__main__":
    main()
