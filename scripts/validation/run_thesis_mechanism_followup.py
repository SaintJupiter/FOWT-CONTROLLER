"""Bounded interventions on saved diagnosis snapshots, without controller tuning."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import fields, replace
import gzip
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

import run_thesis_prediction_mechanism_diagnosis as diagnosis
import preview_mpc_experiment_runtime as runtime_module
from wind_prediction.execution_rollout import ExecutionRolloutRequest
from wind_prediction.physical_execution_platform_path import advance_physical_execution_platform_path
from real_lstm_preview_fixture import REFERENCE_TANK_MASSES_KG, TANK_CAPACITY_KG
from wind_prediction.source_bound_rotor_preview import RotorPreviewOperatingDomainError


ROOT = diagnosis.ROOT
EVIDENCE = diagnosis.OUTPUT
OUTPUT = EVIDENCE / "followup"
CASES = ("strengthening_subwindow", "rise_then_fall", "turning_subwindow",
         "turning_during_turn", "steady")


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def execution_variant(name):
    original = runtime_module.execution_config
    original_diagnosis = diagnosis.execution_config

    def config(*, block_duration_s):
        base = original(block_duration_s=block_duration_s)
        if name == "one_second":
            return replace(base, internal_step_s=1.0)
        if name == "zero_deadband":
            return replace(base, stop_error_kg=0.0, restart_error_kg=0.0)
        return base

    runtime_module.execution_config = config
    diagnosis.execution_config = config
    try:
        yield config
    finally:
        runtime_module.execution_config = original
        diagnosis.execution_config = original_diagnosis


def limited_information(record, retained_leads):
    uv = np.array(record.forecast.uv_ms, copy=True)
    uv[retained_leads:] = record.current_enu_downwind_wind_mps[:2]
    metadata = dict(record.forecast.metadata)
    metadata.update(diagnostic_hybrid=True, retained_lstm_leads=retained_leads,
                    remaining_leads_source="current_observation_persistence")
    return replace(record, forecast=replace(record.forecast, uv_ms=uv,
        source=f"diagnostic_hybrid_lstm_{retained_leads}_persistence_tail",
        metadata=metadata))


def followup_mode(fork):
    return fork.get("subsequent_common_information_rule",
                    fork["alignment"].get("subsequent_common_information_rule", "lstm"))


def metrics(cycles):
    duration = volume = squared = 0.0
    peaks = np.zeros(2)
    starts = stops = switches = 0
    for cycle in cycles:
        net = np.zeros(3)
        amount = local_squared = 0.0
        for step in cycle["actual_path"]["substeps"]:
            dt = step["duration_s"]
            flow = np.asarray(step["actual_flow_m3_min"])
            if np.max(np.abs(flow)) > 1.0 + 1e-8:
                raise RuntimeError("actual flow exceeded frozen hardware limit")
            delta = flow * dt / 60.0
            net += delta * 1025.0
            amount += float(np.sum(np.abs(delta)))
            angles = np.rad2deg(np.asarray(step["start_position"])[[3, 4]])
            endpoint = np.rad2deg(np.asarray(step["end_position"])[[3, 4]])
            peaks = np.maximum(peaks, np.maximum(np.abs(angles), np.abs(endpoint)))
            local_squared += float(np.max(np.abs(angles)) ** 2 * dt)
            duration += dt
            starts += sum(step["start_counts"])
            stops += sum(step["stop_counts"])
            switches += sum(step["direction_switch_counts"])
        actual = np.asarray(cycle["complete_end_state"]["execution"]["masses_kg"]) - np.asarray(cycle["complete_start_state"]["execution"]["masses_kg"])
        if not np.allclose(net, actual, rtol=0, atol=1e-7):
            raise RuntimeError("signed flow does not reproduce tank mass change")
        if abs(amount - cycle["execution"]["transferred_volume_m3"]) > 1e-7:
            raise RuntimeError("absolute flow integral mismatch")
        volume += amount
        squared += local_squared
    return {"cycle_count": len(cycles), "duration_minutes": duration / 60.0,
            "volume_m3": volume, "dominant_tilt_rms_deg": (squared / duration) ** .5 if duration else None,
            "peak_abs_roll_pitch_deg": peaks.tolist(), "starts": starts,
            "stops": stops, "switches": switches,
            "final_state": cycles[-1]["complete_end_state"] if cycles else None}


def run_branch(*, resources, runtime, modes, records, fork, override, count):
    p, e = diagnosis.restore(fork["complete_snapshot"])
    index = fork["fork_cycle_index"]
    cycles = []
    termination = None
    for i in range(index, index + count):
        try:
            cycle, p, e = diagnosis.execute_record(
                resources=resources, runtime=runtime, modes=modes, records=records,
                index=i, mode="lstm" if i == index else followup_mode(fork),
                platform=p, execution=e, override=override if i == index else None,
            )
        except (ValueError, diagnosis.experiment.PreviewMPCNoSafeCandidateError) as exc:
            termination = {"index": i, "error": str(exc)}
            break
        if not cycle["execution"]["realised_execution_within_working_model_scope"]:
            termination = {"index": i, "error": "realised working-model scope exceeded"}
            break
        cycles.append(cycle)
        print(f"  valid block {i}", flush=True)
    return {"cycles": cycles, "requested_count": count, "termination": termination,
            "completed": len(cycles) == count, "metrics": metrics(cycles)}


def fixed_request(*, resources, runtime, records, fork, saved_cycle, config):
    p, e = diagnosis.restore(fork["complete_snapshot"])
    index = fork["fork_cycle_index"]
    loads = diagnosis.assemble_recorded_interval_rotor_load_substeps(
        resources=resources, current_record=records[index], following_record=records[index + 1],
        platform_state=p, substep_count=int(np.ceil(config.block_duration_s / config.internal_step_s)))
    committed = saved_cycle["execution"]["committed_target_masses_kg"]
    operation = saved_cycle["execution"]["selected_target_lifecycle"]
    request = ExecutionRolloutRequest.release_to_current() if operation == "release_to_actual" else ExecutionRolloutRequest.track(committed)
    path = advance_physical_execution_platform_path(
        platform_state=p, execution_state=e, execution_request=request,
        execution_config=config, runtime_assembly=runtime,
        reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
        tank_capacities_kg=np.full(3, TANK_CAPACITY_KG),
        tank_coordinates_m=diagnosis.TANK_COORDINATES_M,
        rotor_load=diagnosis.average_rotor_loads(loads), rotor_load_substeps=loads,
        wave_load=np.zeros(6), other_load=np.zeros(6), duration_s=600.0)
    cycle = {"complete_start_state": fork["complete_snapshot"],
             "complete_end_state": diagnosis.state_record(path.final_platform_state, path.final_execution_state),
             "actual_path": diagnosis.path_record(SimpleNamespace(execution_path=path)),
             "execution": {"transferred_volume_m3": path.transferred_volume_m3},
             "fixed_committed_target_kg": committed, "fixed_operation": operation}
    return {"cycles": [cycle], "metrics": metrics([cycle]),
            "no_optimizer_invoked": True}


def candidate_release(*, resources, runtime, modes, records, fork):
    candidate = next(item for item in fork["new_forecast_first_cycle"]["execution"]["physical_precheck_candidates"]
                     if item["source"] == "release_to_actual")
    if not candidate["eligible"]:
        raise RuntimeError("diagnostic release was not a lawful eligible candidate")
    cycle = {"execution": {"selected_target_lifecycle": "release_to_actual",
                           "committed_target_masses_kg": fork["complete_snapshot"]["execution"]["masses_kg"]}}
    result = fixed_request(resources=resources, runtime=runtime, records=records,
                          fork=fork, saved_cycle=cycle, config=diagnosis.execution_config(block_duration_s=600))
    cycles = result["cycles"]
    p, e = diagnosis.restore(cycles[-1]["complete_end_state"])
    index = fork["fork_cycle_index"]
    count = fork["common_valid_cycle_count"]
    termination = None
    for i in range(index + 1, index + count):
        try:
            next_cycle, p, e = diagnosis.execute_record(
                resources=resources, runtime=runtime, modes=modes, records=records,
                index=i, mode=followup_mode(fork), platform=p, execution=e)
        except (ValueError, diagnosis.experiment.PreviewMPCNoSafeCandidateError) as exc:
            termination = {"index": i, "error": str(exc)}
            break
        if not next_cycle["execution"]["realised_execution_within_working_model_scope"]:
            termination = {"index": i, "error": "realised working-model scope exceeded"}
            break
        cycles.append(next_cycle)
        print(f"  release follow-up block {i}", flush=True)
    return {"cycles": cycles, "requested_count": count, "termination": termination,
            "completed": len(cycles) == count, "metrics": metrics(cycles),
            "candidate_nominal_cost": candidate["combined_physical_prefix_tail_objective"],
            "selected_nominal_cost": next(item["combined_physical_prefix_tail_objective"]
                for item in fork["new_forecast_first_cycle"]["execution"]["physical_precheck_candidates"]
                if item["source"] == fork["new_forecast_first_cycle"]["execution"]["selected_target_lifecycle"])}


def domain_fallback(*, resources, runtime, modes):
    records = diagnosis.load_saved_records("turning")
    baseline = read(EVIDENCE / "turning_persistence.json.gz")
    p, e = diagnosis.restore(baseline["rich_cycles"][0]["complete_start_state"])
    cycles, admission = [], []
    excluded_cycles = []
    termination = None
    for index in range(15):
        mode = "lstm"
        decision = {"index": index, "origin": records[index].forecast.origin_time,
                    "requested_mode": "lstm", "admitted_mode": mode}
        stage = "planner_admission"
        try:
            try:
                diagnosis.assemble_planner_rotor_preview_from_record(
                    resources=resources, source_record=records[index], platform_state=p,
                    planner_forecast_mode="lstm", horizon_blocks=6)
            except RotorPreviewOperatingDomainError as exc:
                if "current" in exc.unsupported_labels:
                    raise
                mode = "persistence"
                decision.update(admitted_mode=mode, unsupported_labels=list(exc.unsupported_labels),
                                operating_domain=exc.operating_domain, reason=str(exc))
            stage = "actual_environment_and_execution"
            cycle, p, e = diagnosis.execute_record(resources=resources, runtime=runtime, modes=modes,
                records=records, index=index, mode=mode, platform=p, execution=e)
        except (RotorPreviewOperatingDomainError, diagnosis.experiment.PreviewMPCNoSafeCandidateError) as exc:
            termination = {"index": index, "stage": stage, "error": str(exc), "no_further_fallback": True}
            admission.append(decision)
            break
        if not cycle["execution"]["realised_execution_within_working_model_scope"]:
            termination = {"index": index, "error": "realised working-model scope exceeded"}
            excluded_cycles.append(cycle)
            admission.append(decision)
            break
        cycles.append(cycle)
        admission.append(decision)
        print(f"fallback turning {index + 1}/15 admitted={mode}", flush=True)
    diagnosis.save(OUTPUT / "turning_lstm_domain_fallback.json.gz", {
        "strategy": "LSTM with explicit future-only domain admission to Persistence",
        "requested_count": 15, "completed": len(cycles) == 15,
        "termination": termination, "admission": admission, "cycles": cycles,
        "diagnostic_cycles_excluded_from_metrics": excluded_cycles,
        "metrics": metrics(cycles) if cycles else None,
        "baseline_shared_initial_state": baseline["rich_cycles"][0]["complete_start_state"],
        "interpretation": "domain-handling diagnostic, not repaired rotor physics or a new control algorithm"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("execution", "information", "candidate", "fallback"), required=True)
    parser.add_argument("--case", choices=CASES, action="append")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    print("load frozen resources", flush=True)
    resources = diagnosis.prepare_real_lstm_preview_resources(device="cpu")
    runtime = diagnosis.research_runtime_assembly()
    modes = diagnosis.ThreeTankDifferentialModes(
        tank_coordinates_m=diagnosis.TANK_COORDINATES_M, gravity_m_s2=runtime.gravity_m_s2)
    identity = diagnosis.experiment.implementation_identity()
    if args.stage == "fallback":
        domain_fallback(resources=resources, runtime=runtime, modes=modes)
        diagnosis.save(OUTPUT / "fallback_identity.json", {"implementation": identity,
            "source_change": "typed ValueError-compatible domain exception only; no numerical model/control change"})
        return
    cases = args.case or (["turning_subwindow", "steady"] if args.stage in ("information", "candidate") else
                         ["strengthening_subwindow", "steady"])
    for case in cases:
        fork = read(EVIDENCE / f"{case}_fork.json.gz")
        name = case.replace("_subwindow", "").replace("_during_turn", "")
        records = diagnosis.load_saved_records(name)
        if case.endswith("_subwindow"):
            start = 3 if name == "strengthening" else 5
            records = records[start:]
        current = records[fork["fork_cycle_index"]]
        aligned, _ = diagnosis.forecast_update(records, fork["fork_cycle_index"])
        if args.stage == "candidate":
            print(f"candidate {case}: one eligible release then same rolling LSTM", flush=True)
            result = candidate_release(resources=resources, runtime=runtime, modes=modes, records=records, fork=fork)
            diagnosis.save(OUTPUT / f"{case}_release_once.json.gz", {
                "case": case, "stage": args.stage, "variant": "release_once",
                "snapshot_sha256": fork["snapshot_sha256"], "origin": current.forecast.origin_time,
                "same_available_information": True, "first_candidate_eligible": True,
                "interpretation": "forced existing candidate diagnostic; not an optimized or proposed deployed policy",
                "subsequent_common_information_rule": followup_mode(fork), "result": result})
            continue
        variants = (("baseline", current), ("zero_deadband", current)) if args.stage == "execution" else (
            ("retained_0", limited_information(current, 0)),
            ("retained_2", limited_information(current, 2)))
        for variant, override in variants:
            started = time.monotonic()
            count = fork["common_valid_cycle_count"] if args.stage == "information" and case in ("turning_subwindow", "steady") else 1
            config_name = variant if args.stage == "execution" else "baseline"
            print(f"{args.stage} {case} {variant} count={count}", flush=True)
            with execution_variant(config_name) as config:
                record = {"case": case, "stage": args.stage, "variant": variant,
                    "snapshot_sha256": fork["snapshot_sha256"],
                    "origin": current.forecast.origin_time,
                    "optimizer_horizon_blocks": 6,
                    "information_intervention": "one origin only; same observed wind, event probabilities, reliability and later information rule",
                    "input_wind_uv_mps": override.forecast.uv_ms.tolist(),
                    "retained_lstm_leads": int(variant[-1]) if variant.startswith("retained_") else 6,
                    "hybrid_forecast_is_not_original_lstm": variant.startswith("retained_"),
                    "subsequent_common_information_rule": followup_mode(fork),
                    "execution_config": {f.name: diagnosis.plain(getattr(config(block_duration_s=600), f.name)) for f in fields(config(block_duration_s=600))},
                    "interpretation": "fixed-request software sensitivity, not hardware identification or new controller" if args.stage == "execution" else "future wind information extent; fixed objective and optimization horizon"}
                if args.stage == "execution":
                    record["new_information"] = fixed_request(resources=resources, runtime=runtime, records=records,
                        fork=fork, saved_cycle=fork["new_forecast_first_cycle"], config=config(block_duration_s=600))
                    record["aligned_old_information"] = fixed_request(resources=resources, runtime=runtime, records=records,
                        fork=fork, saved_cycle=fork["aligned_previous_first_cycle"], config=config(block_duration_s=600))
                    if variant == "baseline":
                        for arm, key in (("new_information", "new_forecast_first_cycle"), ("aligned_old_information", "aligned_previous_first_cycle")):
                            if diagnosis.digest(record[arm]["cycles"][0]["actual_path"]) != diagnosis.digest(fork[key]["actual_path"]):
                                raise RuntimeError("fixed-request baseline did not exactly replay saved physical path")
                        record["fixed_request_physical_replay_exact"] = True
                else:
                    record["new_information"] = run_branch(resources=resources, runtime=runtime, modes=modes,
                        records=records, fork=fork, override=override, count=count)
            record["elapsed_s"] = time.monotonic() - started
            diagnosis.save(OUTPUT / f"{case}_{variant}.json.gz", record)
            print(f"saved {case} {variant} elapsed={record['elapsed_s']:.1f}s", flush=True)
    if diagnosis.experiment.implementation_identity() != identity:
        raise RuntimeError("active production source changed during follow-up")
    diagnosis.save(OUTPUT / f"{args.stage}_identity.json", {
        "implementation": identity, "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": 0, "cases": cases, "model_weights_changed": False,
        "design_parameters_changed": False})
    print(f"{args.stage} complete", flush=True)


if __name__ == "__main__":
    main()
