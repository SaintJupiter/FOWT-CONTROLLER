#!/usr/bin/env python3
"""Run a few six-hour cases through the compact controller and real plant."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import platform
import subprocess
import sys
from dataclasses import replace
from importlib import metadata as importlib_metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
LEGACY_DIR = REPO_ROOT / "archive" / "legacy_fowt_control"
for module_dir in (SRC_DIR, LEGACY_DIR):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import numpy as np

from core_model import FloatingPlatform
from wind_env import wind_speed_to_thrust_n
from wind_prediction.controller_chain_runner import run_controller_plant_chain
from wind_prediction.controller_configuration import load_controller_config
from wind_prediction.controller_plant_adapter import CompactControllerPlantAdapter
from wind_prediction.controller_replay_adapter import ReplayForecastEvidenceSource
from wind_prediction.forecast_adapter import ForecastModelAdapter
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.replay_dataset import Fino1ReplayDataset


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/controller_chain_smoke_v2.json",
        help="Chain-smoke JSON configuration relative to the repository root.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/wind_prediction/controller_v2_chain_smoke",
        help="Output directory relative to the repository root.",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Optional leading case count for integration debugging.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Run only the named configured case; repeat to select several cases.",
    )
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_smoke_config(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "controller_chain_smoke.v1":
        raise ValueError("unsupported chain-smoke schema_version")
    cases = document.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 4:
        raise ValueError("chain-smoke configuration must contain one to four cases")
    if float(document.get("duration_s", 0.0)) != 21600.0:
        raise ValueError("all chain-smoke cases must run for exactly six hours")
    if not isinstance(document.get("platform_profile"), str) or not document[
        "platform_profile"
    ].strip():
        raise ValueError("chain-smoke configuration must name platform_profile")
    probe_case_id = document.get("forecast_content_probe_case_id")
    configured_case_ids = {str(case.get("case_id", "")) for case in cases}
    if not isinstance(probe_case_id, str) or probe_case_id not in configured_case_ids:
        raise ValueError(
            "chain-smoke configuration must name a configured "
            "forecast_content_probe_case_id"
        )
    return document


def pump_config(controller_config) -> dict[str, Any]:
    execution = controller_config.execution
    return {
        "pump_stop_err_kg": execution.stop_error_kg,
        "pump_restart_err_kg": execution.restart_error_kg,
        "pump_min_on_s": execution.min_on_s,
        "pump_min_off_s": execution.min_off_s,
        "pump_hold_before_stop_s": execution.near_target_hold_s,
        "pump_ramp_up_m3_min_per_s": execution.ramp_up_m3_min_per_s,
        "pump_ramp_down_m3_min_per_s": execution.ramp_down_m3_min_per_s,
        "pump_rate_schedule_m3_min": execution.pump_rate_schedule_m3_min,
    }


def initialize_plant(
    stiffness_file: Path,
    controller_config,
    platform_profile: str,
) -> FloatingPlatform:
    plant = FloatingPlatform(
        str(stiffness_file),
        pump_cfg=pump_config(controller_config),
        platform_profile=platform_profile,
        platform_profile_purpose="framework_smoke",
        allow_linear_mooring_fallback=False,
    )
    if plant.load_reference_mode == "legacy_mixed":
        k33 = float(plant.K_hydro[2])
        waterplane_area = k33 / (plant.rho * plant.g)
        if waterplane_area <= 0.0:
            raise RuntimeError("plant hydrostatic heave stiffness is invalid")
        plant.state[2] = -float(np.sum(plant.current_ballast_mass)) / (
            plant.rho * waterplane_area
        )
    return plant


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def result_row(case: dict[str, Any], variant: str, result) -> dict[str, Any]:
    return {
        "case_id": str(case["case_id"]),
        "category": str(case["category"]),
        "timestamp": str(case["timestamp"]),
        "variant": variant,
        **result.summary.to_dict(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def artifact_hashes(paths: list[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted({item.resolve() for item in paths}, key=str):
        if not path.is_file():
            raise FileNotFoundError(path)
        records[relative_path(path)] = {
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return records


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def wind_trace_sha256(trace: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("ws", "wd"):
        values = np.ascontiguousarray(np.asarray(trace[key], dtype=np.float64))
        digest.update(key.encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def forecast_content_probe(
    *,
    case_id: str,
    state: Any,
    initial_tank_masses_kg: Any,
    wind_observation: Mapping[str, Any],
    forecast: ForecastEvidence,
    controller_config: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold measured state fixed while changing only the future wind vectors."""

    masses = np.asarray(initial_tank_masses_kg, dtype=float).reshape(-1)
    if masses.size != 3 or not np.all(np.isfinite(masses)):
        raise ValueError("content probe requires three finite initial tank masses")
    variants = (
        ("weakened_future", 0.5),
        ("model_future", 1.0),
        ("strengthened_future", 1.5),
    )
    rows: list[dict[str, Any]] = []
    for label, scale in variants:
        modified = replace(
            forecast,
            uv_ms=np.asarray(forecast.uv_ms, dtype=float) * float(scale),
        )
        adapter = CompactControllerPlantAdapter(
            initial_tank_masses_kg=masses,
            forecast_source=lambda _time, _wind, evidence=modified: evidence,
            config=controller_config,
        )
        output = adapter.compute(
            state=np.asarray(state, dtype=float).copy(),
            wind_obs=dict(wind_observation),
            plant_info_prev={
                "tank_masses": masses.copy(),
                "target_ballast_mass": masses.copy(),
                "pump_rate_cmd_m3_min": np.zeros(3, dtype=float),
                "pump_latched": np.zeros(3, dtype=bool),
            },
            current_time=0.0,
        )
        decision_record = adapter.records[0]
        rows.append(
            {
                "case_id": case_id,
                "probe_variant": label,
                "future_vector_scale": float(scale),
                "selected_action": output["preview_primary_action"],
                "target_operation": output["preview_target_operation"],
                "target_masses_kg": output["preview_primary_target_kg"],
                "forecast_payload_sha256": decision_record[
                    "forecast_payload_sha256"
                ],
                "decision_state_sha256": decision_record[
                    "decision_state_sha256"
                ],
                "candidate_ranking_sha256": decision_record[
                    "candidate_ranking_sha256"
                ],
                "event_probs_sha256": json_sha256(
                    {
                        str(key): float(value)
                        for key, value in sorted(modified.event_probs.items())
                    }
                ),
            }
        )

    state_hashes = {row["decision_state_sha256"] for row in rows}
    payload_hashes = {row["forecast_payload_sha256"] for row in rows}
    ranking_hashes = {row["candidate_ranking_sha256"] for row in rows}
    event_hashes = {row["event_probs_sha256"] for row in rows}
    targets = [np.asarray(row["target_masses_kg"], dtype=float) for row in rows]
    max_target_difference_kg = max(
        float(np.max(np.abs(left - right)))
        for index, left in enumerate(targets)
        for right in targets[index + 1 :]
    )
    threshold_kg = float(controller_config.execution.restart_error_kg)
    checks = [
        {
            "case_id": case_id,
            "variant": "forecast_content_probe",
            "check": "probe_holds_decision_state_fixed",
            "passed": len(state_hashes) == 1,
        },
        {
            "case_id": case_id,
            "variant": "forecast_content_probe",
            "check": "probe_changes_only_future_payload_content",
            "passed": len(payload_hashes) == len(variants) and len(event_hashes) == 1,
        },
        {
            "case_id": case_id,
            "variant": "forecast_content_probe",
            "check": "future_content_changes_candidate_ranking",
            "passed": len(ranking_hashes) == len(variants),
        },
        {
            "case_id": case_id,
            "variant": "forecast_content_probe",
            "check": "future_content_changes_executable_target",
            "passed": max_target_difference_kg > threshold_kg,
            "max_target_difference_kg": max_target_difference_kg,
            "threshold_kg": threshold_kg,
        },
    ]
    return rows, checks


def git_identity() -> dict[str, Any]:
    def _run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        head = _run("rev-parse", "HEAD")
        status = _run("status", "--porcelain=v1", "--untracked-files=normal")
    except (OSError, subprocess.CalledProcessError):
        return {"git_available": False}
    return {
        "git_available": True,
        "head": head,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def imported_repo_python_files() -> list[Path]:
    source_roots = {"src", "archive", "scripts"}
    paths: set[Path] = set()
    for module in tuple(sys.modules.values()):
        try:
            source = inspect.getsourcefile(module)
        except (OSError, TypeError):
            source = None
        if not source:
            continue
        path = Path(source).resolve()
        try:
            relative = path.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if (
            relative.parts
            and relative.parts[0] in source_roots
            and path.is_file()
            and path.suffix == ".py"
        ):
            paths.add(path)
    return sorted(paths, key=str)


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "torch", "pandas", "scipy", "openpyxl"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def decision_comparisons(
    run_results: Mapping[tuple[str, str], Any],
    *,
    target_difference_threshold_kg: float,
) -> list[dict[str, Any]]:
    if not math.isfinite(target_difference_threshold_kg) or target_difference_threshold_kg <= 0.0:
        raise ValueError("target_difference_threshold_kg must be finite and positive")
    comparisons: list[dict[str, Any]] = []
    case_ids = sorted({case_id for case_id, _ in run_results})
    for case_id in case_ids:
        predicted = run_results.get((case_id, "prediction_assisted"))
        feedback = run_results.get((case_id, "feedback_only"))
        if predicted is None or feedback is None:
            continue
        predicted_by_time = {float(row["time_s"]): row for row in predicted.decisions}
        feedback_by_time = {float(row["time_s"]): row for row in feedback.decisions}
        common_times = sorted(set(predicted_by_time) & set(feedback_by_time))
        action_difference_count = 0
        target_difference_count = 0
        max_target_difference_kg = 0.0
        for time_s in common_times:
            predicted_row = predicted_by_time[time_s]
            feedback_row = feedback_by_time[time_s]
            action_difference_count += int(
                predicted_row["selected_action"] != feedback_row["selected_action"]
            )
            predicted_target = np.asarray(predicted_row["target_masses_kg"], dtype=float)
            feedback_target = np.asarray(feedback_row["target_masses_kg"], dtype=float)
            target_difference = float(np.max(np.abs(predicted_target - feedback_target)))
            target_difference_count += int(
                target_difference > target_difference_threshold_kg
            )
            max_target_difference_kg = max(max_target_difference_kg, target_difference)
        comparisons.append(
            {
                "case_id": case_id,
                "common_decision_count": len(common_times),
                "action_difference_count": action_difference_count,
                "target_difference_threshold_kg": target_difference_threshold_kg,
                "target_difference_count": target_difference_count,
                "max_target_difference_kg": max_target_difference_kg,
            }
        )
    return comparisons


def sanity_checks(
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    comparison_by_case = {row["case_id"]: row for row in comparisons}
    for row in rows:
        by_case.setdefault(row["case_id"], {})[row["variant"]] = row
        checks.extend(
            [
                {
                    "case_id": row["case_id"],
                    "variant": row["variant"],
                    "check": "completed_and_finite",
                    "passed": bool(row["completed"] and row["finite_state"]),
                },
                {
                    "case_id": row["case_id"],
                    "variant": row["variant"],
                    "check": "tank_capacity",
                    "passed": bool(row["capacity_respected"]),
                },
                {
                    "case_id": row["case_id"],
                    "variant": row["variant"],
                    "check": "exact_target_transfer",
                    "passed": bool(row["max_command_transfer_error_kg"] <= 1e-6),
                },
                {
                    "case_id": row["case_id"],
                    "variant": row["variant"],
                    "check": "no_obvious_attitude_blowup",
                    "passed": bool(
                        max(row["max_abs_pitch_deg"], row["max_abs_roll_deg"]) < 45.0
                    ),
                },
                {
                    "case_id": row["case_id"],
                    "variant": row["variant"],
                    "check": "platform_mass_telemetry_available",
                    "passed": bool(row["platform_mass_telemetry_available"]),
                },
                {
                    "case_id": row["case_id"],
                    "variant": row["variant"],
                    "check": "platform_mass_balance",
                    "passed": bool(
                        row["max_platform_mass_balance_error_kg"] is not None
                        and row["max_platform_mass_balance_error_kg"] <= 1e-6
                    ),
                },
            ]
        )
        if float(row["pump_volume_m3"]) > 1e-6:
            checks.extend(
                [
                    {
                        "case_id": row["case_id"],
                        "variant": row["variant"],
                        "check": "active_pumping_has_target_revision",
                        "passed": bool(row["target_revision_count"] > 0),
                    },
                    {
                        "case_id": row["case_id"],
                        "variant": row["variant"],
                        "check": "active_pumping_changes_tank_mass",
                        "passed": bool(row["max_abs_tank_mass_change_kg"] > 1e-6),
                    },
                    {
                        "case_id": row["case_id"],
                        "variant": row["variant"],
                        "check": "active_pumping_changes_mass_properties",
                        "passed": bool(
                            row["max_abs_platform_mass_change_kg"] > 1e-6
                            or row["max_platform_center_of_mass_shift_m"] > 1e-12
                            or row["max_abs_platform_inertia_change_kg_m2"] > 1e-3
                        ),
                    },
                ]
            )
    for case_id, variants in by_case.items():
        predicted = variants.get("prediction_assisted")
        feedback = variants.get("feedback_only")
        if predicted is None or feedback is None:
            continue
        reference = max(float(feedback["pump_volume_m3"]), 1e-9)
        checks.extend(
            [
                {
                    "case_id": case_id,
                    "variant": "prediction_assisted",
                    "check": "forecast_reached_decisions",
                    "passed": bool(predicted["forecast_decision_count"] > 0),
                },
                {
                    "case_id": case_id,
                    "variant": "feedback_only",
                    "check": "forecast_absent_in_feedback_only",
                    "passed": bool(feedback["forecast_decision_count"] == 0),
                },
                {
                    "case_id": case_id,
                    "variant": "prediction_assisted",
                    "check": "no_order_of_magnitude_pump_regression",
                    "passed": bool(float(predicted["pump_volume_m3"]) <= 3.0 * reference),
                },
            ]
        )
        category = str(predicted.get("category", ""))
        comparison = comparison_by_case.get(case_id)
        if category != "low_disturbance" and comparison is not None:
            checks.append(
                {
                    "case_id": case_id,
                    "variant": "prediction_assisted",
                    "check": "forecast_changes_executable_target",
                    "passed": bool(comparison["target_difference_count"] > 0),
                }
            )
    return checks


def main() -> int:
    args = parse_args()
    smoke_path = repo_path(args.config).resolve()
    smoke = load_smoke_config(smoke_path)
    cases = list(smoke["cases"])
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if str(case["case_id"]) in requested]
        found = {str(case["case_id"]) for case in cases}
        missing = requested - found
        if missing:
            raise ValueError(f"unknown configured case ids: {sorted(missing)}")
    if args.case_limit is not None:
        if not 1 <= args.case_limit <= len(cases):
            raise ValueError("case-limit must select at least one configured case")
        cases = cases[: args.case_limit]

    controller_path = repo_path(smoke["controller_config"]).resolve()
    loaded_controller = load_controller_config(controller_path)
    dataset_dir = repo_path(smoke["dataset_dir"]).resolve()
    model_dir = repo_path(smoke["model_dir"]).resolve()
    stiffness_file = repo_path(smoke["stiffness_file"]).resolve()
    platform_profile = str(smoke["platform_profile"])
    for required in (dataset_dir, model_dir, stiffness_file):
        if not required.exists():
            raise FileNotFoundError(required)

    duration_s = float(smoke["duration_s"])
    dt_s = float(smoke["dt_s"])
    sample_interval_s = float(smoke["sample_interval_s"])
    replay = Fino1ReplayDataset(dataset_dir, split="test")
    model = ForecastModelAdapter(model_dir, dataset_dir, device="cpu")
    output_dir = repo_path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory must be empty for a traceable run: {output_dir}"
        )

    rows: list[dict[str, Any]] = []
    run_results: dict[tuple[str, str], Any] = {}
    resolved_platform: dict[str, Any] | None = None
    case_inputs: list[dict[str, Any]] = []
    content_probe_rows: list[dict[str, Any]] = []
    content_probe_checks: list[dict[str, Any]] = []
    probe_case_id = str(smoke["forecast_content_probe_case_id"])
    for case in cases:
        start = datetime.strptime(str(case["timestamp"]), TIMESTAMP_FORMAT)
        replay_rows = int(round(duration_s / replay.update_interval_s))
        trace = replay.build_wind_trace(start, replay_rows, dt_s)
        case_input = {
            "case_id": str(case["case_id"]),
            "timestamp": str(case["timestamp"]),
            "replay_row_count": replay_rows,
            "wind_trace_sha256": wind_trace_sha256(trace),
        }
        case_inputs.append(case_input)
        forecast_source = ReplayForecastEvidenceSource(
            replay_dataset=replay,
            start_timestamp=start,
            forecast_adapter=model,
        )
        if str(case["case_id"]) == probe_case_id:
            probe_plant = initialize_plant(
                stiffness_file,
                loaded_controller.config,
                platform_profile,
            )
            probe_identity = probe_plant.resolved_platform_identity()
            if resolved_platform is None:
                resolved_platform = probe_identity
            elif probe_identity != resolved_platform:
                raise RuntimeError(
                    "resolved platform identity changed before forecast content probe"
                )
            probe_wind_observation = {
                "ws": float(np.asarray(trace["ws"], dtype=float).reshape(-1)[0]),
                "wd_deg": float(
                    np.asarray(trace["wd"], dtype=float).reshape(-1)[0]
                ),
            }
            probe_forecast = forecast_source(0.0, probe_wind_observation)
            if probe_forecast is None:
                raise RuntimeError(
                    f"configured forecast content probe has no forecast: {probe_case_id}"
                )
            probe_rows, probe_checks = forecast_content_probe(
                case_id=probe_case_id,
                state=probe_plant.state,
                initial_tank_masses_kg=probe_plant.current_ballast_mass,
                wind_observation=probe_wind_observation,
                forecast=probe_forecast,
                controller_config=loaded_controller.config,
            )
            content_probe_rows.extend(probe_rows)
            content_probe_checks.extend(probe_checks)
        for variant, source in (
            ("feedback_only", None),
            ("prediction_assisted", forecast_source),
        ):
            plant = initialize_plant(
                stiffness_file,
                loaded_controller.config,
                platform_profile,
            )
            run_platform_identity = plant.resolved_platform_identity()
            if resolved_platform is None:
                resolved_platform = run_platform_identity
            elif run_platform_identity != resolved_platform:
                raise RuntimeError(
                    "resolved platform identity changed between smoke runs"
                )
            adapter = CompactControllerPlantAdapter(
                initial_tank_masses_kg=plant.current_ballast_mass,
                forecast_source=source,
                config=loaded_controller.config,
            )
            result = run_controller_plant_chain(
                plant=plant,
                controller=adapter,
                wind_trace=trace,
                thrust_model=wind_speed_to_thrust_n,
                duration_s=duration_s,
                dt_s=dt_s,
                sample_interval_s=sample_interval_s,
            )
            rows.append(result_row(case, variant, result))
            run_results[(str(case["case_id"]), variant)] = result
            if variant == "prediction_assisted":
                forecast_inputs = [
                    {
                        "time_s": float(record["time_s"]),
                        "forecast_origin_time": record["forecast_origin_time"],
                        "input_window_sha256": record[
                            "forecast_input_window_sha256"
                        ],
                        "forecast_payload_sha256": record[
                            "forecast_payload_sha256"
                        ],
                        "decision_state_sha256": record[
                            "decision_state_sha256"
                        ],
                        "candidate_ranking_sha256": record[
                            "candidate_ranking_sha256"
                        ],
                    }
                    for record in result.decisions
                    if record.get("forecast_available")
                ]
                invalid_input_digests = [
                    row["input_window_sha256"]
                    for row in forecast_inputs
                    if not isinstance(row["input_window_sha256"], str)
                    or len(row["input_window_sha256"]) != 64
                ]
                if invalid_input_digests:
                    raise RuntimeError(
                        "prediction decisions lack traceable forecast input windows"
                    )
                case_input["forecast_inputs"] = forecast_inputs
                case_input["forecast_input_bundle_sha256"] = json_sha256(
                    forecast_inputs
                )
            stem = f"{case['case_id']}__{variant}"
            write_jsonl(output_dir / f"{stem}__timeseries.jsonl", result.sampled_timeseries)
            write_jsonl(output_dir / f"{stem}__decisions.jsonl", result.decisions)
            print(
                f"{case['case_id']} {variant}: "
                f"pump={result.summary.pump_volume_m3:.2f} m3, "
                f"pitch={result.summary.max_abs_pitch_deg:.2f} deg, "
                f"roll={result.summary.max_abs_roll_deg:.2f} deg, "
                f"decisions={result.summary.decision_count}, "
                f"forecast={result.summary.forecast_decision_count}",
                flush=True,
            )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = decision_comparisons(
        run_results,
        target_difference_threshold_kg=(
            loaded_controller.config.execution.restart_error_kg
        ),
    )
    comparison_path = output_dir / "decision_comparisons.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    checks = sanity_checks(rows, comparisons)
    if probe_case_id in {str(case["case_id"]) for case in cases}:
        checks.extend(content_probe_checks)
    probe_path = output_dir / "forecast_content_probes.jsonl"
    write_jsonl(probe_path, content_probe_rows)
    sanity_path = output_dir / "sanity_checks.jsonl"
    write_jsonl(sanity_path, checks)

    model_artifacts = list(model_dir.glob("*"))
    dataset_runtime_artifacts = [
        dataset_dir / "metadata.json",
        dataset_dir / "scaler_train.json",
        dataset_dir / "sample_index.csv.gz",
        dataset_dir / "X_test.npy",
        replay.canonical_path,
    ]
    runtime_code_artifacts = [
        REPO_ROOT / "archive/legacy_fowt_control/core_model.py",
        REPO_ROOT / "archive/legacy_fowt_control/defaults.py",
        REPO_ROOT / "archive/legacy_fowt_control/wind_env.py",
        REPO_ROOT / "src/wind_prediction/action_plan.py",
        REPO_ROOT / "src/wind_prediction/ballast_allocation.py",
        REPO_ROOT / "src/wind_prediction/ballast_mass_properties.py",
        REPO_ROOT / "src/wind_prediction/controller_chain_runner.py",
        REPO_ROOT / "src/wind_prediction/controller_configuration.py",
        REPO_ROOT / "src/wind_prediction/controller_core.py",
        REPO_ROOT / "src/wind_prediction/controller_plant_adapter.py",
        REPO_ROOT / "src/wind_prediction/controller_replay_adapter.py",
        REPO_ROOT / "src/wind_prediction/controller_runtime.py",
        REPO_ROOT / "src/wind_prediction/dataset_manifest.py",
        REPO_ROOT / "src/wind_prediction/execution_rollout.py",
        REPO_ROOT / "src/wind_prediction/forecast_contract.py",
        REPO_ROOT / "src/wind_prediction/forecast_evidence.py",
        REPO_ROOT / "src/wind_prediction/forecast_action_policy.py",
        REPO_ROOT / "src/wind_prediction/forecast_adapter.py",
        REPO_ROOT / "src/wind_prediction/replay_dataset.py",
        REPO_ROOT / "scripts/validation/run_controller_v2_chain_smoke.py",
    ]
    model_hashes = artifact_hashes(model_artifacts)
    dataset_hashes = artifact_hashes(dataset_runtime_artifacts)
    runtime_code_artifacts = sorted(
        set(runtime_code_artifacts) | set(imported_repo_python_files()),
        key=str,
    )
    runtime_code_hashes = artifact_hashes(runtime_code_artifacts)
    resolved_platform_sha256 = json_sha256(resolved_platform)
    formal_suite_complete = bool(
        len(cases) == len(smoke["cases"])
        and args.case_id is None
        and args.case_limit is None
    )
    manifest = {
        "schema_version": "controller_chain_smoke_result.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "end_to_end_framework_check_not_performance_evidence",
        "smoke_config": str(smoke_path.relative_to(REPO_ROOT)),
        "smoke_config_sha256": sha256_file(smoke_path),
        "controller_config": str(controller_path.relative_to(REPO_ROOT)),
        "controller_config_sha256": loaded_controller.sha256,
        "controller_config_semantic_sha256": loaded_controller.sha256,
        "controller_config_file_sha256": sha256_file(controller_path),
        "dataset_dir": str(dataset_dir.relative_to(REPO_ROOT)),
        "model_dir": str(model_dir.relative_to(REPO_ROOT)),
        "stiffness_file": str(stiffness_file.relative_to(REPO_ROOT)),
        "platform_profile": platform_profile,
        "resolved_platform": resolved_platform,
        "resolved_platform_sha256": resolved_platform_sha256,
        "duration_s": duration_s,
        "dt_s": dt_s,
        "sample_interval_s": sample_interval_s,
        "dataset_split": "test",
        "forecast_device": "cpu",
        "case_count": len(cases),
        "selected_cases": cases,
        "case_inputs": case_inputs,
        "forecast_content_probe_case_id": probe_case_id,
        "forecast_content_probe_file": probe_path.name,
        "forecast_content_probe_sha256": sha256_file(probe_path),
        "forecast_input_bundle_sha256": json_sha256(
            [
                {
                    "case_id": row["case_id"],
                    "forecast_input_bundle_sha256": row.get(
                        "forecast_input_bundle_sha256"
                    ),
                }
                for row in case_inputs
            ]
        ),
        "command_selection": {
            "case_id": args.case_id,
            "case_limit": args.case_limit,
        },
        "variants": ["feedback_only", "prediction_assisted"],
        "model_artifacts": model_hashes,
        "model_bundle_sha256": json_sha256(model_hashes),
        "dataset_runtime_artifacts": dataset_hashes,
        "dataset_runtime_bundle_sha256": json_sha256(dataset_hashes),
        "runtime_code_artifacts": runtime_code_hashes,
        "runtime_source_bundle_sha256": json_sha256(runtime_code_hashes),
        "stiffness_file_sha256": sha256_file(stiffness_file),
        "decision_comparison_file": comparison_path.name,
        "decision_comparison_sha256": sha256_file(comparison_path),
        "sanity_check_file": sanity_path.name,
        "sanity_check_sha256": sha256_file(sanity_path),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        },
        "git": git_identity(),
        "formal_suite_complete": formal_suite_complete,
        "all_checks_passed": bool(all(row["passed"] for row in checks)),
        "failed_checks": [row for row in checks if not row["passed"]],
    }
    current_outputs = [
        path for path in output_dir.iterdir() if path.name != "manifest.json"
    ]
    manifest["output_artifacts"] = artifact_hashes(current_outputs)
    manifest["output_bundle_sha256"] = json_sha256(manifest["output_artifacts"])
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {summary_path}")
    print(f"all checks passed: {manifest['all_checks_passed']}")
    return 0 if manifest["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
