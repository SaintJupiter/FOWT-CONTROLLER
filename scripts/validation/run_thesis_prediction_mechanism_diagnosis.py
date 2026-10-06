#!/usr/bin/env python3
"""Bounded mechanism evidence using the existing continuous MPC experiment.

Four wind-only selected windows, three information sources, one aligned
forecast-update fork per window. No controller weights or plant rules change.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
from datetime import datetime, timedelta
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).parent)]

import numpy as np
import torch
import preview_mpc_continuous_experiment as experiment
from preview_mpc_experiment_runtime import (
    average_rotor_loads, execution_config, research_runtime_assembly,
    run_preview_mpc_cycle,
)
from real_lstm_preview_fixture import (
    RealLstmWindRecord, TANK_COORDINATES_M,
    assemble_planner_rotor_preview_from_record,
    assemble_recorded_interval_rotor_load_substeps,
    prepare_real_lstm_preview_resources,
)
from fowt_platform import IncrementalState, ThreeTankDifferentialModes
from wind_prediction.execution_rollout import ExecutionRolloutState
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.preview_mpc_design import research_preview_mpc_design_v4
from wind_prediction.replay_dataset import TIMESTAMP_FMT

OUTPUT = ROOT / "artifacts/thesis_mechanism_diagnosis_20261005"
MAIN_CYCLES = 12
TAIL_CYCLES = 3
TOTAL_CYCLES = MAIN_CYCLES + TAIL_CYCLES
MODES = ("persistence", "lstm", "recorded_oracle")


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def save(path, value):
    payload = json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    else:
        path.write_text(payload + "\n", encoding="utf-8")


def state_record(platform, execution):
    return {
        "platform": {f.name: plain(getattr(platform, f.name)) for f in fields(platform)},
        "execution": {f.name: plain(getattr(execution, f.name)) for f in fields(execution)},
    }


def restore(record):
    return IncrementalState(**record["platform"]), ExecutionRolloutState(**record["execution"])


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def path_record(result):
    path = result.execution_path
    trace = []
    elapsed = 0.0
    for step in path.substeps:
        execution = step.execution_step
        trace.append({
            "offset_s": elapsed,
            "duration_s": step.duration_s,
            "start_position": step.start_platform_state.position.tolist(),
            "start_velocity": step.start_platform_state.velocity.tolist(),
            "end_position": step.next_platform_state.position.tolist(),
            "end_velocity": step.next_platform_state.velocity.tolist(),
            "start_masses_kg": step.start_execution_state.masses_kg.tolist(),
            "end_masses_kg": step.next_execution_state.masses_kg.tolist(),
            "mass_delta_kg": execution.mass_delta_kg.tolist(),
            "actual_flow_m3_min": execution.state.pump_rates_m3_min.tolist(),
            "command_rate_m3_min": execution.state.pump_command_rates_m3_min.tolist(),
            "pump_latched": execution.state.pump_latched.tolist(),
            "pump_volume_m3": execution.pump_volume_m3.tolist(),
            "start_counts": execution.start_counts.tolist(),
            "stop_counts": execution.stop_counts.tolist(),
            "direction_switch_counts": execution.direction_switch_counts.tolist(),
        })
        elapsed += step.duration_s
    dm = np.asarray([r["mass_delta_kg"] for r in trace])
    incoming = np.sum(np.maximum(dm, 0.0), axis=0) / 1025.0
    outgoing = np.sum(np.maximum(-dm, 0.0), axis=0) / 1025.0
    if abs(float(np.sum(incoming + outgoing)) - path.transferred_volume_m3) > 1e-8:
        raise RuntimeError("absolute flow integral does not match reported pump volume")
    return {
        "substeps": trace, "inflow_m3_by_tank": incoming.tolist(),
        "outflow_m3_by_tank": outgoing.tolist(),
        "absolute_flow_integral_verified": True,
    }


def observed_cycle(**kwargs):
    start = state_record(kwargs["platform_state"], kwargs["execution_state"])
    captures = []
    cycle, platform, execution = run_preview_mpc_cycle(
        **kwargs, cycle_result_observer=lambda result: captures.append(path_record(result)),
    )
    cycle["complete_start_state"] = start
    cycle["complete_end_state"] = state_record(platform, execution)
    cycle["actual_path"] = captures[0]
    return cycle, platform, execution


def execute_record(*, resources, runtime, modes, records, index, mode, platform, execution, override=None):
    current = records[index] if override is None else override
    config = execution_config(block_duration_s=current.forecast.sample_period_s)
    realised = assemble_recorded_interval_rotor_load_substeps(
        resources=resources, current_record=records[index], following_record=records[index + 1],
        platform_state=platform,
        substep_count=int(np.ceil(config.block_duration_s / config.internal_step_s)),
    )
    oracle = tuple(records[index + 1:index + 7]) if mode == "recorded_oracle" else None
    oracle_payload = None if oracle is None else tuple({
        "recorded_observation_time": r.forecast.origin_time,
        "enu_downwind_wind_mps": r.current_enu_downwind_wind_mps.tolist(),
    } for r in oracle)
    preview = assemble_planner_rotor_preview_from_record(
        resources=resources, source_record=current, platform_state=platform,
        planner_forecast_mode=mode, horizon_blocks=6, oracle_records=oracle,
    )
    cycle, platform, execution = observed_cycle(
        source_preview=preview, runtime=runtime, modes=modes,
        platform_state=platform, execution_state=execution,
        planner_forecast_mode=mode, planner_horizon_blocks=6,
        planner_oracle_future_wind_records=oracle_payload,
        plant_first_block_rotor_load=average_rotor_loads(realised),
        plant_first_block_rotor_load_substeps=realised,
        plant_load_source="recorded_replay_linear_enu_substep_reconstruction_converted_under_each_variant_cycle_initial_platform_state",
    )
    cycle["cycle_index"] = index
    return cycle, platform, execution


def summary(cycles):
    compact = []
    for index, cycle in enumerate(cycles):
        item = dict(cycle, cycle_index=index)
        compact.append(experiment._compact_cycle(item))
    return experiment._aggregate(compact) if compact else None


def forecast_update(records, index):
    old = records[index - 1].forecast
    new = records[index].forecast
    # The previous lead 2..6 and current lead 1..5 refer to identical times.
    # The unshared sixth endpoint is the new forecast in both fork arms.
    aligned = np.vstack([old.uv_ms[1:6], new.uv_ms[5:6]])
    origin = datetime.strptime(str(new.origin_time), TIMESTAMP_FMT)
    old_origin = datetime.strptime(str(old.origin_time), TIMESTAMP_FMT)
    new_times = [origin + timedelta(seconds=(i + 1) * new.sample_period_s) for i in range(6)]
    old_times = [old_origin + timedelta(seconds=(i + 1) * old.sample_period_s) for i in range(6)]
    assert old_times[1:] == new_times[:5]
    delta = np.asarray(new.uv_ms[:5]) - np.asarray(old.uv_ms[1:6])
    return RealLstmWindRecord(
        forecast=replace(new, uv_ms=aligned, metadata={
            **dict(new.metadata), "diagnostic_intervention": "retain_aligned_previous_common_five_endpoints",
            "previous_forecast_origin": old.origin_time,
        }), current_enu_downwind_wind_mps=records[index].current_enu_downwind_wind_mps,
    ), {
        "previous_origin": old.origin_time, "current_origin": new.origin_time,
        "common_future_times": [v.strftime(TIMESTAMP_FMT) for v in new_times[:5]],
        "shared_tail_time": new_times[5].strftime(TIMESTAMP_FMT),
        "tail_rule": "current new sixth endpoint used identically in both arms",
        "mean_common_uv_update_mps": float(np.mean(np.linalg.norm(delta, axis=1))),
        "max_common_uv_update_mps": float(np.max(np.linalg.norm(delta, axis=1))),
        "old_aligned_uv_mps": aligned.tolist(), "new_uv_mps": new.uv_ms.tolist(),
        "intervention_scope": "only one origin's future uv; same current observation, reliability, state, configuration and subsequent common information rule",
    }


def run_fork(*, resources, runtime, modes, records, cycles, case_name,
             total_cycle_count=TOTAL_CYCLES, main_cycle_count=MAIN_CYCLES,
             followup_mode="lstm", fixed_index=None):
    eligible = range(1, min(main_cycle_count, len(cycles)))
    index = fixed_index if fixed_index is not None else max(eligible, key=lambda i: forecast_update(records, i)[1]["mean_common_uv_update_mps"])
    intervention, alignment = forecast_update(records, index)
    alignment["subsequent_common_information_rule"] = followup_mode
    snapshot = cycles[index]["complete_start_state"]
    baseline_platform, baseline_execution = restore(snapshot)
    replay, p, e = execute_record(
        resources=resources, runtime=runtime, modes=modes, records=records,
        index=index, mode="lstm", platform=baseline_platform, execution=baseline_execution,
    )
    # Require exact equality of the entire physical and planning evidence.
    matched = digest(replay) == digest(cycles[index])
    if not matched:
        raise RuntimeError(f"same-input full-state replay differs for {case_name} cycle {index}")
    branch = []
    diagnostic_cycles = []
    p, e = restore(snapshot)
    termination = None
    for offset in range(index, total_cycle_count):
        try:
            cycle, p, e = execute_record(
                resources=resources, runtime=runtime, modes=modes, records=records,
                index=offset, mode="lstm" if offset == index else followup_mode,
                platform=p, execution=e,
                override=intervention if offset == index else None,
            )
        except (ValueError, experiment.PreviewMPCNoSafeCandidateError) as exc:
            termination = {"cycle_index": offset, "error": str(exc)}
            break
        if not cycle["execution"]["realised_execution_within_working_model_scope"]:
            diagnostic_cycles.append(cycle)
            termination = {"cycle_index": offset, "error": "realised working-model scope exceeded"}
            break
        branch.append(cycle)
        print(f"fork {case_name} {offset + 1}/{total_cycle_count}", flush=True)
    common_count = min(len(branch), len(cycles) - index)
    original = cycles[index:index + common_count]
    comparable = branch[:common_count]
    return {
        "case": case_name, "fork_cycle_index": index, "alignment": alignment,
        "subsequent_common_information_rule": followup_mode,
        "complete_snapshot": snapshot, "snapshot_sha256": digest(snapshot),
        "random_states_at_fork": {"python": plain(random.getstate()),
            "numpy": plain(np.random.get_state()), "torch_cpu": torch.get_rng_state().tolist()},
        "randomness_role": "no random sampling in active MPC or execution; cached eval-mode forecasts shared",
        "same_input_full_evidence_replay_exact": matched,
        "termination": termination,
        "comparison_eligible": common_count == total_cycle_count - index and termination is None,
        "common_valid_cycle_count": common_count,
        "new_forecast_summary": summary(original), "aligned_previous_summary": summary(comparable),
        "new_forecast_first_cycle": original[0] if original else None,
        "aligned_previous_first_cycle": comparable[0] if comparable else None,
        "new_forecast_final_state": original[-1]["complete_end_state"] if original else None,
        "aligned_previous_final_state": comparable[-1]["complete_end_state"] if comparable else None,
        "diagnostic_cycles_excluded_from_aggregate": diagnostic_cycles,
        "aligned_previous_cycles": branch,
    }


def load_saved_records(name):
    payload = json.loads((OUTPUT / f"{name}_sources.json").read_text())
    return [RealLstmWindRecord(forecast=ForecastEvidence(**r["forecast"]),
        current_enu_downwind_wind_mps=r["current_observation"]["enu_downwind_wind_mps"])
        for r in payload["records"]]


def recover_subwindows(*, resources, runtime, modes):
    """Stay inside the four original records; preserve their domain failures."""
    for name, start in (("strengthening", 3), ("turning", 5)):
        records = load_saved_records(name)[start:]
        total = TOTAL_CYCLES - start
        main_count = MAIN_CYCLES - start
        initial = None
        if name == "turning":
            with gzip.open(OUTPUT / "turning_persistence.json.gz", "rt") as handle:
                parent = json.load(handle)
            initial = parent["rich_cycles"][start]["complete_start_state"]
        collected = {}
        for mode in MODES:
            rich = []
            def capture(**kwargs):
                cycle, p, e = observed_cycle(**kwargs)
                rich.append(cycle)
                print(f"subwindow {name} {mode} {len(rich)}/{total}", flush=True)
                return cycle, p, e
            original_cycle = experiment.run_preview_mpc_cycle
            original_initial = experiment._precondition_window_initial_state
            experiment.run_preview_mpc_cycle = capture
            if initial is not None:
                def common_initial(**kwargs):
                    p, e = restore(initial)
                    return p, e, {"kind": "complete_state_from_shared_current_observation_persistence_prefix",
                        "original_case_start_cycle": start, "snapshot_sha256": digest(initial)}
                experiment._precondition_window_initial_state = common_initial
            try:
                result = experiment.run_variant(name=mode, planner_forecast_mode=mode,
                    planner_horizon_blocks=6, source_records=records, resources=resources,
                    runtime=runtime, modes=modes, cycle_count=total)
            finally:
                experiment.run_preview_mpc_cycle = original_cycle
                experiment._precondition_window_initial_state = original_initial
            result["original_case_start_cycle"] = start
            result["main_cycle_count"] = main_count
            result["main_window_complete"] = len(result["cycles"]) >= main_count
            result["main_window_aggregate"] = summary(rich[:min(main_count, len(result["cycles"]))])
            result["rich_cycles"] = rich
            result["selection_basis"] = "source-domain support inside the original fixed wind record; no pump/posture outcome ranking"
            collected[mode] = result
            save(OUTPUT / f"{name}_subwindow_{mode}.json.gz", result)
        lstm = collected["lstm"]["rich_cycles"][:len(collected["lstm"]["cycles"])]
        if len(lstm) >= 2:
            fork = run_fork(resources=resources, runtime=runtime, modes=modes,
                records=records, cycles=lstm, case_name=f"{name}_subwindow",
                total_cycle_count=total, main_cycle_count=main_count)
            save(OUTPUT / f"{name}_subwindow_fork.json.gz", fork)

    # A second turn-time fork uses the lawful Persistence prefix at cycle 3.
    # Both arms see LSTM only at that origin, then the same Persistence rule.
    records = load_saved_records("turning")
    with gzip.open(OUTPUT / "turning_persistence.json.gz", "rt") as handle:
        parent = json.load(handle)
    start = 3
    p, e = restore(parent["rich_cycles"][start]["complete_start_state"])
    baseline = list(parent["rich_cycles"][:start])
    for index in range(start, TOTAL_CYCLES):
        cycle, p, e = execute_record(resources=resources, runtime=runtime, modes=modes,
            records=records, index=index, mode="lstm" if index == start else "persistence",
            platform=p, execution=e)
        baseline.append(cycle)
    fork = run_fork(resources=resources, runtime=runtime, modes=modes,
        records=records, cycles=baseline, case_name="turning_during_turn",
        fixed_index=start, followup_mode="persistence")
    save(OUTPUT / "turning_during_turn_fork.json.gz", fork)
    save(OUTPUT / "turning_during_turn_new_forecast_baseline.json.gz", {"cycles": baseline})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=["strengthening", "rise_then_fall", "turning", "steady"])
    parser.add_argument("--recover-subwindows", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    print("loading frozen real-LSTM resources", flush=True)
    resources = prepare_real_lstm_preview_resources(device="cpu")
    runtime = research_runtime_assembly()
    modes = ThreeTankDifferentialModes(tank_coordinates_m=TANK_COORDINATES_M, gravity_m_s2=runtime.gravity_m_s2)
    identity = {"implementation": experiment.implementation_identity(),
        "design": research_preview_mpc_design_v4().as_dict(),
        "execution_config": {f.name: plain(getattr(execution_config(block_duration_s=600), f.name))
            for f in fields(execution_config(block_duration_s=600))},
        "runtime_provenance": runtime.provenance,
        "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
        "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "main_cycles": MAIN_CYCLES, "tail_cycles": TAIL_CYCLES,
        "tail_scope": "same recorded 30min extension, each variant retains its information rule; no forced equalization of terminal masses",
    }
    save(OUTPUT / ("subwindow_identity.json" if args.recover_subwindows else "frozen_identity.json"), identity)
    if args.recover_subwindows:
        recover_subwindows(resources=resources, runtime=runtime, modes=modes)
        if experiment.implementation_identity() != identity["implementation"]:
            raise RuntimeError("active source identity changed during diagnosis")
        print("subwindow diagnosis finished; source identity unchanged", flush=True)
        return
    selection = json.loads((OUTPUT / "wind_selection.json").read_text())
    for regime in selection["regimes"]:
        name = regime["regime"]
        if args.case and name not in args.case:
            continue
        candidate = regime["candidates"][0]
        origin = datetime.strptime(candidate["origin"], TIMESTAMP_FMT)
        print(f"prepare {name} {origin}", flush=True)
        records = experiment.prepare_source_records(origin=origin, device="cpu",
            cycle_count=TOTAL_CYCLES, lookahead_count=6, resources=resources)
        preflight = experiment.preflight_source_operating_domain(source_records=records,
            resources=resources, origin=origin, cycle_count=TOTAL_CYCLES)
        save(OUTPUT / f"{name}_sources.json", {
            "selection": candidate, "preflight": preflight,
            "records": [r.source_record_payload() for r in records],
        })
        collected = {}
        for mode in MODES:
            rich = []
            def capture(**kwargs):
                cycle, p, e = observed_cycle(**kwargs)
                rich.append(cycle)
                print(f"{name} {mode} {len(rich)}/{TOTAL_CYCLES}", flush=True)
                return cycle, p, e
            original = experiment.run_preview_mpc_cycle
            experiment.run_preview_mpc_cycle = capture
            started = time.monotonic()
            try:
                result = experiment.run_variant(name=mode, planner_forecast_mode=mode,
                    planner_horizon_blocks=6, source_records=records, resources=resources,
                    runtime=runtime, modes=modes, cycle_count=TOTAL_CYCLES)
            finally:
                experiment.run_preview_mpc_cycle = original
            result["runtime_seconds"] = time.monotonic() - started
            valid_rich = rich[:len(result["cycles"])]
            result["main_window_aggregate"] = summary(valid_rich[:MAIN_CYCLES])
            result["main_window_complete"] = len(valid_rich) >= MAIN_CYCLES
            result["rich_cycles"] = rich
            collected[mode] = result
            save(OUTPUT / f"{name}_{mode}.json.gz", result)
        lstm = collected["lstm"]["rich_cycles"][:len(collected["lstm"]["cycles"])]
        if len(lstm) >= 2:
            fork = run_fork(resources=resources, runtime=runtime, modes=modes,
                records=records, cycles=lstm, case_name=name)
            save(OUTPUT / f"{name}_fork.json.gz", fork)
        print(f"completed case {name}", flush=True)
    if experiment.implementation_identity() != identity["implementation"]:
        raise RuntimeError("active source identity changed during diagnosis")
    print("diagnosis run finished; source identity unchanged", flush=True)


if __name__ == "__main__":
    main()
