import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from core_model import FloatingPlatform
from controllers import MIMOController
from controllers_extras import (
    ClosedLoopPolicy,
    CommandRateLimiter,
    HeaveBiasBalancer,
    SetpointShaper,
    TrimGovernor,
)
from defaults import (
    DEFAULT_CONTROLLER_CFG,
    DEFAULT_HEAVE_CFG,
    DEFAULT_MAIN_EXPERIMENT_PROTOCOL,
    DEFAULT_TARGET_SHAPE_CFG,
    TRIM_CFG_TEST_E,
    TRIM_CFG_TEST_F,
    clone_cfg,
)
from wind_env import MarkovWindGenerator, WindEnvMarkov, WindTracePlayer, wind_speed_to_thrust_n

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def compute_heave_equilibrium_from_ballast(plant):
    """Compute static heave equilibrium from current ballast and hydrostatic K33."""
    k33 = float(plant.K_hydro[2]) if len(plant.K_hydro) > 2 else 0.0
    if k33 <= 0.0 or plant.rho <= 1e-9 or plant.g <= 1e-9:
        return {"ok": False, "z_eq_m": float("nan"), "a_waterplane_m2": float("nan")}
    a_wp = k33 / (plant.rho * plant.g)
    if a_wp <= 0.0:
        return {"ok": False, "z_eq_m": float("nan"), "a_waterplane_m2": float("nan")}
    z_eq = -float(np.sum(plant.current_ballast_mass)) / (plant.rho * a_wp)
    return {"ok": True, "z_eq_m": float(z_eq), "a_waterplane_m2": float(a_wp)}


def discover_stiffness_file():
    candidates = [
        p
        for p in Path("data").glob("*.xlsx")
        if "frequency-matrix" not in p.name and "Thrust force" not in p.name
    ]
    return str(candidates[0]) if candidates else ""


def discover_wind_matrix_file():
    candidates = sorted(
        p for p in Path("data").glob("*frequency-matrix.xlsx") if not p.name.startswith("~$")
    )
    if not candidates:
        candidates = sorted(
            p for p in Path("data").glob("*frequency-matrix.csv") if not p.name.startswith("~$")
        )
    return str(candidates[0]) if candidates else ""


def create_wind_source(
    wind_gen,
    dt,
    wind_trace=None,
    update_interval_s=3600.0,
    mean_lpf_tau_s=120.0,
    target_ws=11.5,
    target_wd=0.0,
    seed=RANDOM_SEED,
):
    if wind_trace is not None:
        return WindTracePlayer(wind_trace), int(wind_trace.get("seed", seed))
    if wind_gen is None:
        return None, np.nan
    env = WindEnvMarkov(
        wind_gen=wind_gen,
        dt=dt,
        update_interval_s=update_interval_s,
        lpf_tau_s=mean_lpf_tau_s,
        target_ws=target_ws,
        target_wd=target_wd,
        seed=seed,
    )
    return env, int(seed)


def build_constant_wind_trace(n_steps, ws_mps, wd_deg, dt, seed=RANDOM_SEED, wind_gen=None):
    ws = np.full(int(n_steps), float(ws_mps), dtype=float)
    wd = np.full(int(n_steps), float(wd_deg), dtype=float)
    thrust_n = np.full(int(n_steps), float(wind_speed_to_thrust_n(ws_mps)), dtype=float)
    trace = {
        "ws": ws,
        "wd": wd,
        "thrust_n": thrust_n,
        "seed": int(seed),
        "n_steps": int(n_steps),
        "dt": float(dt),
        "update_interval_s": 0.0,
        "mean_lpf_tau_s": 0.0,
    }
    # For fixed-trace experiments, attach the nearest discrete Markov state when
    # the matrix is available. This keeps the physical wind constant while making
    # G2 table lookup auditable and actually usable on F-only constant-wind runs.
    if wind_gen is not None:
        try:
            sid = int(wind_gen.get_state_id(float(ws_mps), float(wd_deg)))
            ws_state, wd_state = wind_gen.curr_states[sid]
            trace["state_id"] = np.full(int(n_steps), sid, dtype=int)
            trace["ws_state"] = np.full(int(n_steps), float(ws_state), dtype=float)
            trace["wd_state"] = np.full(int(n_steps), float(wd_state), dtype=float)
        except Exception:
            pass
    return trace


def resolve_controller_cfg(controller_cfg=None):
    cfg = clone_cfg(DEFAULT_CONTROLLER_CFG)
    if controller_cfg:
        cfg.update(controller_cfg)
    return cfg


def apply_experiment_protocol(
    experiment_protocol,
    controller_cfg=None,
    heave_cfg=None,
    target_shape_cfg=None,
):
    controller_cfg_eff = resolve_controller_cfg(controller_cfg)

    heave_cfg_eff = clone_cfg(DEFAULT_HEAVE_CFG)
    if heave_cfg:
        heave_cfg_eff.update(heave_cfg)

    target_shape_cfg_eff = clone_cfg(DEFAULT_TARGET_SHAPE_CFG)
    if target_shape_cfg:
        target_shape_cfg_eff.update(target_shape_cfg)

    protocol_name = "custom"
    token = "" if experiment_protocol is None else str(experiment_protocol).strip().lower()
    if token in ("", "none", "off", "custom"):
        pass
    elif token in ("main", "paper_main", "main_v1"):
        proto = clone_cfg(DEFAULT_MAIN_EXPERIMENT_PROTOCOL)
        controller_cfg_eff.update(proto.get("controller_overrides", {}))
        heave_cfg_eff.update(proto.get("heave_overrides", {}))
        # Enforce one shared shaping config for main comparisons.
        target_shape_cfg_eff = clone_cfg(DEFAULT_TARGET_SHAPE_CFG)
        target_shape_cfg_eff.update(proto.get("target_shape_cfg", {}))
        protocol_name = str(proto.get("name", "main_protocol"))
    else:
        raise ValueError(
            f"unsupported experiment_protocol={experiment_protocol}; "
            "expected one of: main/custom/none"
        )

    return controller_cfg_eff, heave_cfg_eff, target_shape_cfg_eff, protocol_name


def _build_protocol_meta(protocol_name, controller_cfg, heave_cfg, target_shape_cfg, trim_cfg):
    ff_pitch = float(controller_cfg.get("k_wind_comp_pitch", 0.0))
    ff_roll = float(controller_cfg.get("k_wind_comp_roll", 0.0))
    ff_disabled = int(abs(ff_pitch) <= 1e-12 and abs(ff_roll) <= 1e-12)

    heave_enabled_cfg = int(bool(heave_cfg.get("enabled", False)))
    shaper_enabled = int(bool(target_shape_cfg.get("enable_setpoint_shaping", True)))

    trim_map_version = ""
    target_mode = "g1"
    target_table_path = ""
    target_lookup_key = "state_id"
    target_table_strict = 0
    if isinstance(trim_cfg, dict):
        trim_map_version = str(trim_cfg.get("trim_map_version", "")).strip().lower()
        target_mode = str(trim_cfg.get("target_mode", "g1")).strip().lower() or "g1"
        target_table_path = str(trim_cfg.get("target_table_path", ""))
        target_lookup_key = str(trim_cfg.get("target_lookup_key", "state_id"))
        target_table_strict = int(bool(trim_cfg.get("target_table_strict", False)))

    return {
        "name": str(protocol_name),
        "trim_map_version": trim_map_version,
        "target_mode": target_mode,
        "target_table_path": target_table_path,
        "target_lookup_key": target_lookup_key,
        "target_table_strict": target_table_strict,
        "ff_pitch": ff_pitch,
        "ff_roll": ff_roll,
        "ff_disabled": ff_disabled,
        "heave_enabled_cfg": heave_enabled_cfg,
        "heave_disabled_cfg": int(not bool(heave_enabled_cfg)),
        "shaper_enabled": shaper_enabled,
        "shaper_alpha": float(target_shape_cfg.get("setpoint_alpha", np.nan)),
        "shaper_rate_deg_s": float(target_shape_cfg.get("setpoint_rate_deg_s", np.nan)),
        "primary_safety_fallback_enabled": int(
            bool(target_shape_cfg.get("primary_safety_fallback_enabled", True))
        ),
        "primary_safety_pitch_enter_deg": float(
            target_shape_cfg.get("primary_safety_pitch_enter_deg", np.nan)
        ),
        "primary_safety_roll_enter_deg": float(
            target_shape_cfg.get("primary_safety_roll_enter_deg", np.nan)
        ),
        "primary_safety_pitch_exit_deg": float(
            target_shape_cfg.get("primary_safety_pitch_exit_deg", np.nan)
        ),
        "primary_safety_roll_exit_deg": float(
            target_shape_cfg.get("primary_safety_roll_exit_deg", np.nan)
        ),
        "primary_safety_use_envelope": int(
            bool(target_shape_cfg.get("primary_safety_use_envelope", False))
        ),
        "primary_safety_envelope_enter_norm": float(
            target_shape_cfg.get("primary_safety_envelope_enter_norm", np.nan)
        ),
        "primary_safety_enter_hold_s": float(
            target_shape_cfg.get("primary_safety_enter_hold_s", np.nan)
        ),
        "primary_safety_emergency_pitch_enter_deg": float(
            target_shape_cfg.get("primary_safety_emergency_pitch_enter_deg", np.nan)
        ),
        "primary_safety_emergency_roll_enter_deg": float(
            target_shape_cfg.get("primary_safety_emergency_roll_enter_deg", np.nan)
        ),
        "primary_safety_exit_hold_s": float(
            target_shape_cfg.get("primary_safety_exit_hold_s", np.nan)
        ),
        "primary_safety_exit_required_windows": int(
            target_shape_cfg.get("primary_safety_exit_required_windows", 1)
        ),
    }


def build_default_controller(plant, dt, controller_cfg=None):
    cfg = resolve_controller_cfg(controller_cfg)

    baseline = plant.current_ballast_mass.copy()
    return MIMOController(
        kp_p=cfg["kp_p"], ki_p=cfg["ki_p"], kd_p=cfg["kd_p"],
        kp_r=cfg["kp_r"], ki_r=cfg["ki_r"], kd_r=cfg["kd_r"],
        kp_h=cfg["kp_h"], ki_h=cfg["ki_h"], kd_h=cfg["kd_h"],
        dt=dt,
        max_capacity=plant.tank_capacity,
        baselines=baseline,
        setpoints={"pitch": 0.0, "roll": 0.0, "heave": -8.1},
        deadbands={
            "pitch": cfg["deadband_pitch"],
            "roll": cfg["deadband_roll"],
            "heave": cfg["deadband_heave"],
        },
        deadband_exit_ratio=cfg["deadband_exit_ratio"],
        filter_tau=cfg["filter_tau"],
        update_interval=cfg["update_interval"],
        tank_pos=plant.tank_pos,
        k_wind_comp_pitch=cfg["k_wind_comp_pitch"],
        k_wind_comp_roll=cfg["k_wind_comp_roll"],
    )


def run_test_a(excel_path):
    print("\n=== Test A: Working Condition Equilibrium ===")
    plant = FloatingPlatform(excel_path)
    plant.set_irregular_wave(Hs=0, Tp=10, n_freqs=0)
    print(f"Initial Ballast: {plant.current_ballast_mass / 1000} tons")

    dt = 0.1
    for i in range(10000):
        plant.step(0.0, 0.0, dt, i * dt)

    final_heave = plant.state[2]
    final_pitch = np.degrees(plant.state[4])
    print(f"Result: Heave={final_heave:.3f}m, Pitch={final_pitch:.3f}deg")
    if abs(final_heave - (-8.1)) < 0.5 and abs(final_pitch) < 0.5:
        print(f"[PASS] Platform is balanced (Pitch={final_pitch:.3f} deg).")
    else:
        print(f"[WARN] Equilibrium offset. Pitch={final_pitch:.3f} deg")


def run_step_case(
    excel_path,
    delta_target_kg,
    n_steps,
    dt,
    case_name,
    wind_gen=None,
    wind_trace=None,
    wind_update_interval_s=3600.0,
    wind_mean_lpf_tau_s=120.0,
    wind_target_ws=11.5,
    wind_target_wd=0.0,
    platform_profile=None,
    platform_cfg=None,
):
    plant = FloatingPlatform(excel_path, platform_profile=platform_profile, platform_cfg=platform_cfg)
    plant.set_irregular_wave(0, 10, 0, 0)
    initial_m1 = plant.current_ballast_mass[0]
    target_mass = initial_m1 + delta_target_kg
    plant.set_ballast_target(target_mass, plant.current_ballast_mass[1], plant.current_ballast_mass[2])

    reached_target = False
    time_to_reach = np.nan
    tol = plant.pump_stop_err_kg
    prev_active = False
    prev_rate = 0.0
    switch_count = 0
    tv = 0.0
    sat_count = 0
    pitch_sq_sum = 0.0
    roll_sq_sum = 0.0
    pitch_abs_max = 0.0
    roll_abs_max = 0.0

    has_wind = (wind_gen is not None) or (wind_trace is not None)
    wind_source, wind_seed = create_wind_source(
        wind_gen=wind_gen,
        dt=dt,
        wind_trace=wind_trace,
        update_interval_s=wind_update_interval_s,
        mean_lpf_tau_s=wind_mean_lpf_tau_s,
        target_ws=wind_target_ws,
        target_wd=wind_target_wd,
        seed=RANDOM_SEED,
    )
    if wind_trace is not None and len(wind_trace["ws"]) < n_steps:
        raise ValueError(
            f"wind_trace too short for {case_name}: need {n_steps}, got {len(wind_trace['ws'])}"
        )
    if has_wind and (n_steps * dt < wind_update_interval_s) and (wind_trace is None):
        print(
            f"   [WARN] {case_name}: sim window {n_steps * dt:.1f}s < wind update interval "
            f"{wind_update_interval_s:.1f}s; Markov state will not transition."
        )

    for i in range(n_steps):
        t_sim = i * dt
        if not has_wind:
            thrust_n = 0.0
            wind_dir_deg = 0.0
        else:
            wind_obs = wind_source.step()
            wind_dir_deg = float(wind_obs["wd_deg"])
            thrust_n = float(wind_obs["thrust_n"])

        _, info = plant.step(thrust_n, wind_dir_deg, dt, t_sim)
        current_m1 = info["tank_masses"][0]
        err = abs(current_m1 - target_mass)

        if (not reached_target) and (err < tol):
            reached_target = True
            time_to_reach = t_sim

        active = bool(info["pump_active"][0])
        rate = float(info["pump_rate_cmd_m3_min"][0])
        sat = bool(info["pump_saturated"][0])
        switch_count += int(active != prev_active)
        tv += abs(rate - prev_rate)
        sat_count += int(sat)
        prev_active = active
        prev_rate = rate
        pitch_deg = float(info["pitch_deg"])
        roll_deg = float(info["roll_deg"])
        pitch_sq_sum += pitch_deg * pitch_deg
        roll_sq_sum += roll_deg * roll_deg
        pitch_abs_max = max(pitch_abs_max, abs(pitch_deg))
        roll_abs_max = max(roll_abs_max, abs(roll_deg))

        if i % 500 == 0:
            print(
                f"   {case_name}: T={t_sim:.1f}s, m1={current_m1:.1f}, "
                f"err={err:.1f}, rate={rate:.2f}"
            )

    final_err = abs(info["tank_masses"][0] - target_mass)
    sat_ratio = sat_count / n_steps
    pitch_rms_deg = float(np.sqrt(pitch_sq_sum / n_steps))
    roll_rms_deg = float(np.sqrt(roll_sq_sum / n_steps))
    passed = np.isfinite(time_to_reach) and final_err <= plant.pump_restart_err_kg

    print(
        f"   Result {case_name}: reach={time_to_reach:.1f}s, final_err={final_err:.1f}, "
        f"switches={switch_count}, tv={tv:.1f}, sat_ratio={sat_ratio:.3f}, "
        f"pitch_rms={pitch_rms_deg:.3f}, roll_rms={roll_rms_deg:.3f}"
    )
    print("[PASS] Step response converged." if passed else "[WARN] Not fully converged in window.")

    return {
        "case": case_name,
        "delta_ton": delta_target_kg / 1000.0,
        "n_steps": n_steps,
        "dt": dt,
        "reach_time_s": float(time_to_reach) if np.isfinite(time_to_reach) else np.nan,
        "final_err_kg": float(final_err),
        "switches": int(switch_count),
        "tv_m3min": float(tv),
        "sat_ratio": float(sat_ratio),
        "pitch_rms_deg": pitch_rms_deg,
        "roll_rms_deg": roll_rms_deg,
        "pitch_abs_max_deg": float(pitch_abs_max),
        "roll_abs_max_deg": float(roll_abs_max),
        "wind_on": int(has_wind),
        "wind_seed": wind_seed,
        "wind_update_interval_s": float(wind_update_interval_s),
        "wind_mean_lpf_tau_s": float(wind_mean_lpf_tau_s),
        "experiment_protocol": "custom",
        "protocol_trim_map_version": "",
        "protocol_target_mode_cfg": "g1",
        "protocol_target_table_path_cfg": "",
        "protocol_target_lookup_key_cfg": "state_id",
        "protocol_target_table_strict_cfg": 0,
        "target_mode": "g1",
        "target_table_path": "",
        "target_lookup_key": "state_id",
        "target_lookup_total": 0,
        "target_lookup_hit": 0,
        "target_lookup_fallback": 0,
        "fallback_reason_missing_state": 0,
        "fallback_reason_missing_table": 0,
        "fallback_reason_missing_columns": 0,
        "fallback_reason_invalid_state_id": 0,
        "protocol_ff_pitch": np.nan,
        "protocol_ff_roll": np.nan,
        "protocol_ff_disabled": 0,
        "protocol_heave_enabled_cfg": 0,
        "protocol_heave_disabled_cfg": 1,
        "protocol_shaper_enabled": 0,
        "protocol_shaper_alpha": np.nan,
        "protocol_shaper_rate_deg_s": np.nan,
        "pass": int(passed),
    }


def _sign_token(v, eps=1e-6):
    if v > eps:
        return "+"
    if v < -eps:
        return "-"
    return "0"


def run_sign_tests(excel_path, dt=0.1, horizon_s=100.0, delta_ton=50.0):
    print("\n=== Test S: Tank Sign Sanity (No Wind/No Wave/Open Loop) ===")
    n_steps = max(1, int(horizon_s / dt))

    plant_ref = FloatingPlatform(excel_path)
    plant_ref.set_irregular_wave(0, 10, 0, 0)
    base_mass = plant_ref.current_ballast_mass.copy()
    tank_pos = plant_ref.tank_pos.copy()
    front_idx = int(np.argmax(tank_pos[:, 0]))
    right_idx = int(np.argmin(tank_pos[:, 1]))
    delta_kg = float(delta_ton * 1000.0)

    def _run_case(case_name, tank_idx):
        plant = FloatingPlatform(excel_path)
        plant.set_irregular_wave(0, 10, 0, 0)
        target = base_mass.copy()
        target[tank_idx] += delta_kg
        plant.set_ballast_target(target[0], target[1], target[2])

        pitch0 = float(np.degrees(plant.state[4]))
        roll0 = float(np.degrees(plant.state[3]))
        info = None
        for i in range(n_steps):
            _, info = plant.step(0.0, 0.0, dt, i * dt)

        pitch_delta = float(info["pitch_deg"] - pitch0)
        roll_delta = float(info["roll_deg"] - roll0)
        pitch_sign = _sign_token(pitch_delta)
        roll_sign = _sign_token(roll_delta)
        moved_mass = float(info["tank_masses"][tank_idx] - base_mass[tank_idx])
        moved_enough = abs(moved_mass) > 0.5 * delta_kg
        response_enough = max(abs(pitch_delta), abs(roll_delta)) > 0.05
        # FLU/ENU right-handed convention: +pitch=bow-down, +roll=starboard-down.
        expected_pitch_sign = "+" if case_name == "S_front_tank_step_50t" else None
        expected_roll_sign = "+" if case_name == "S_right_tank_step_50t" else None
        sign_ok = True
        if expected_pitch_sign is not None:
            sign_ok = sign_ok and (pitch_sign == expected_pitch_sign)
        if expected_roll_sign is not None:
            sign_ok = sign_ok and (roll_sign == expected_roll_sign)
        passed = moved_enough and response_enough and sign_ok

        print(
            f"   {case_name}: tank={tank_idx+1}, dm={moved_mass/1000:.1f}t, "
            f"dPitch={pitch_delta:+.3f}deg, dRoll={roll_delta:+.3f}deg, sign=(P{pitch_sign},R{roll_sign}), "
            f"expect=(P{expected_pitch_sign or '*'},R{expected_roll_sign or '*'})"
        )
        print("[PASS] Sign response observed." if passed else "[FAIL] Weak/invalid sign response.")

        return {
            "case": case_name,
            "delta_ton": float(delta_ton),
            "n_steps": int(n_steps),
            "dt": float(dt),
            "reach_time_s": np.nan,
            "final_err_kg": np.nan,
            "switches": np.nan,
            "tv_m3min": np.nan,
            "sat_ratio": np.nan,
            "pitch_rms_deg": np.nan,
            "roll_rms_deg": np.nan,
            "pitch_abs_max_deg": abs(pitch_delta),
            "roll_abs_max_deg": abs(roll_delta),
            "sign_pitch_delta_deg": float(pitch_delta),
            "sign_roll_delta_deg": float(roll_delta),
            "sign_pitch_token": pitch_sign,
            "sign_roll_token": roll_sign,
            "expected_pitch_sign": expected_pitch_sign if expected_pitch_sign is not None else "",
            "expected_roll_sign": expected_roll_sign if expected_roll_sign is not None else "",
            "sign_ok": int(sign_ok),
            "sign_test_tank_idx": int(tank_idx + 1),
            "wind_on": 0,
            "pass": int(passed),
        }

    row_front = _run_case("S_front_tank_step_50t", front_idx)
    row_right = _run_case("S_right_tank_step_50t", right_idx)
    return [row_front, row_right]


def run_test_c(wind_matrix_path, min_states=600, expected_states=647):
    print("\n=== Test C: Wind Markov Matrix Integrity ===")
    if not wind_matrix_path:
        print("[FAIL] Wind frequency matrix not found in ./data")
        return None, False

    print(f"Matrix file: {wind_matrix_path}")
    wind = MarkovWindGenerator(str(wind_matrix_path))
    n_curr = len(wind.curr_states)
    n_next = len(wind.next_states)
    print(f"Result: curr_states={n_curr}, next_states={n_next}, probs_shape={wind.probs.shape}")

    ok_shape = wind.probs.shape == (n_curr, n_next)
    ok_size = n_curr >= min_states and n_next >= min_states
    if expected_states is not None:
        ok_size = ok_size and (n_curr == expected_states and n_next == expected_states)
    row_sums = wind.probs.sum(axis=1)
    ok_rowsum = np.allclose(row_sums, 1.0, atol=1e-6)
    ok_nan = not np.isnan(wind.probs).any()
    print(
        f"RowSum stats: min={row_sums.min():.6f}, max={row_sums.max():.6f}, mean={row_sums.mean():.6f}"
    )

    rng = np.random.default_rng(42)
    samples = min(5, n_curr)
    sample_ids = rng.choice(n_curr, size=samples, replace=False)
    next_set = set(wind.next_states)
    sample_ok = True
    for i, sid in enumerate(sample_ids, 1):
        cws, cwd = wind.curr_states[int(sid)]
        nws, nwd = wind.get_next_wind(cws, cwd)
        in_grid = (nws, nwd) in next_set
        sample_ok = sample_ok and in_grid
        print(f"   sample{i}: curr=({cws},{cwd}) -> next=({nws},{nwd}), in_grid={in_grid}")

    if n_curr == 1 and n_next == 1:
        print("[FAIL] Wind matrix fell back to 1x1 default. Check source file and parsing.")
    if ok_shape and ok_size and ok_rowsum and ok_nan and sample_ok and not (n_curr == 1 and n_next == 1):
        print("[PASS] Wind matrix loaded and sampled correctly.")
        ok_all = True
    else:
        print(
            f"[FAIL] Wind matrix check failed: shape={ok_shape}, size={ok_size}, "
            f"rowsum={ok_rowsum}, nan={ok_nan}, sampling={sample_ok}"
        )
        ok_all = False
    return wind, ok_all


def _as_vec3(values, fallback=None):
    base = fallback if values is None else values
    arr = np.asarray(base, dtype=float).reshape(-1)
    if arr.size >= 3:
        return arr[:3].astype(float)
    out = np.zeros(3, dtype=float)
    if arr.size > 0:
        out[: arr.size] = arr.astype(float)
    return out


def _append_timeseries_row(
    timeseries,
    t_sim,
    wind_obs,
    thrust_n,
    plant,
    dbg,
    info,
    obs,
    control_enabled,
):
    target_masses = obs["target_masses"]
    err_masses = obs["err_masses"]
    alloc_deltas = obs["alloc_deltas"]
    alloc_raw = obs["alloc_raw"]
    alloc_cmd = obs["alloc_cmd"]
    alloc_clip_any = obs["alloc_clip_any"]
    alloc_clip_mask = obs["alloc_clip_mask"]

    timeseries.append(
        {
            "t_s": float(t_sim),
            "wind_speed": float(wind_obs["ws"]),
            "wind_dir_deg": float(wind_obs["wd_deg"]),
            "wind_state_id": int(obs["wind_state_id"]) if obs["wind_state_id"] is not None else np.nan,
            "wind_ws_state": float(obs["wind_ws_state"]) if obs["wind_ws_state"] is not None else np.nan,
            "wind_wd_state": float(obs["wind_wd_state"]) if obs["wind_wd_state"] is not None else np.nan,
            "thrust_n": float(thrust_n),
            "pitch_deg": float(obs["pitch_deg"]),
            "roll_deg": float(obs["roll_deg"]),
            "heave_m": float(plant.state[2]),
            "hm_smoothed_heave_m": float(dbg.get("hm_smoothed_heave_m", plant.state[2])),
            "hm_heave_error_m": float(dbg.get("hm_heave_error_m", 0.0)),
            "hm_total_ballast_kg": float(
                dbg.get("hm_total_ballast_kg", np.sum(info["tank_masses"]))
            ),
            "ballast_total_kg": float(
                dbg.get("ballast_total_kg", np.sum(info["tank_masses"]))
            ),
            "heave_correction_per_tank_kg": float(
                dbg.get("heave_correction_per_tank_kg", 0.0)
            ),
            "tank1_kg": float(info["tank_masses"][0]),
            "tank2_kg": float(info["tank_masses"][1]),
            "tank3_kg": float(info["tank_masses"][2]),
            "target_tank1_kg": float(target_masses[0]),
            "target_tank2_kg": float(target_masses[1]),
            "target_tank3_kg": float(target_masses[2]),
            "err_tank1_kg": float(err_masses[0]),
            "err_tank2_kg": float(err_masses[1]),
            "err_tank3_kg": float(err_masses[2]),
            "pump_rate1_m3min": float(info["pump_rate_cmd_m3_min"][0]),
            "pump_rate2_m3min": float(info["pump_rate_cmd_m3_min"][1]),
            "pump_rate3_m3min": float(info["pump_rate_cmd_m3_min"][2]),
            "pump_total_rate_m3_min": float(np.sum(info["pump_rate_cmd_m3_min"])),
            "pump_target_rate1_m3min": float(_as_vec3(info.get("pump_rate_target_m3_min", [0.0, 0.0, 0.0]))[0]),
            "pump_target_rate2_m3min": float(_as_vec3(info.get("pump_rate_target_m3_min", [0.0, 0.0, 0.0]))[1]),
            "pump_target_rate3_m3min": float(_as_vec3(info.get("pump_rate_target_m3_min", [0.0, 0.0, 0.0]))[2]),
            "pump_released_rate1_m3min": float(_as_vec3(info.get("pump_rate_released_m3_min", [0.0, 0.0, 0.0]))[0]),
            "pump_released_rate2_m3min": float(_as_vec3(info.get("pump_rate_released_m3_min", [0.0, 0.0, 0.0]))[1]),
            "pump_released_rate3_m3min": float(_as_vec3(info.get("pump_rate_released_m3_min", [0.0, 0.0, 0.0]))[2]),
            "pump_stage_idx1": int(_as_vec3(info.get("pump_stage_idx", [0, 0, 0]))[0]),
            "pump_stage_idx2": int(_as_vec3(info.get("pump_stage_idx", [0, 0, 0]))[1]),
            "pump_stage_idx3": int(_as_vec3(info.get("pump_stage_idx", [0, 0, 0]))[2]),
            "pump_stage_switch_count": int(info.get("pump_stage_switch_count", 0)),
            "pump_total_backlog_kg": float(info.get("pump_total_backlog_kg", 0.0)),
            "pump_target_motion_kg_s": float(info.get("pump_target_motion_kg_s", 0.0)),
            "pump_global_quiet_s": float(info.get("pump_global_quiet_s", 0.0)),
            "pump_global_quiet_ready": int(info.get("pump_global_quiet_ready", 0)),
            "pump_quiet_stop_block_count": int(info.get("pump_quiet_stop_block_count", 0)),
            "pump_latch_switch_count": int(info.get("pump_latch_switch_count", 0)),
            "pump_latched1": int(bool(_as_vec3(info.get("pump_latched", [0, 0, 0]))[0])),
            "pump_latched2": int(bool(_as_vec3(info.get("pump_latched", [0, 0, 0]))[1])),
            "pump_latched3": int(bool(_as_vec3(info.get("pump_latched", [0, 0, 0]))[2])),
            "pump_near_target_mean_s": float(
                np.mean(_as_vec3(info.get("pump_near_target_s", np.zeros(3, dtype=float))))
            ),
            "pump_fullspeed_any": int(obs["pump_fullspeed_any"]),
            "pump_backlog_max_kg": float(obs["pump_backlog_max_kg"]),
            "alloc_delta_tank1_kg": float(alloc_deltas[0]),
            "alloc_delta_tank2_kg": float(alloc_deltas[1]),
            "alloc_delta_tank3_kg": float(alloc_deltas[2]),
            "alloc_raw_tank1_kg": float(alloc_raw[0]),
            "alloc_raw_tank2_kg": float(alloc_raw[1]),
            "alloc_raw_tank3_kg": float(alloc_raw[2]),
            "alloc_cmd_tank1_kg": float(alloc_cmd[0]),
            "alloc_cmd_tank2_kg": float(alloc_cmd[1]),
            "alloc_cmd_tank3_kg": float(alloc_cmd[2]),
            "alloc_clip_any": int(alloc_clip_any),
            "alloc_clip_tank1": int(alloc_clip_mask[0]),
            "alloc_clip_tank2": int(alloc_clip_mask[1]),
            "alloc_clip_tank3": int(alloc_clip_mask[2]),
            "pitch_sp_deg": float(obs["pitch_sp"]),
            "roll_sp_deg": float(obs["roll_sp"]),
            "pitch_sp_raw_deg": float(obs["pitch_sp_raw"]),
            "roll_sp_raw_deg": float(obs["roll_sp_raw"]),
            "current_pitch_trim_raw_deg": float(obs["current_pitch_trim_raw"]),
            "current_roll_trim_raw_deg": float(obs["current_roll_trim_raw"]),
            "preview_pitch_bias_deg": float(obs["preview_pitch_bias"]),
            "preview_roll_bias_deg": float(obs["preview_roll_bias"]),
            "combined_pitch_trim_raw_deg": float(obs["combined_pitch_trim_raw"]),
            "combined_roll_trim_raw_deg": float(obs["combined_roll_trim_raw"]),
            "current_pitch_trim_scaled_deg": float(obs["current_pitch_trim_scaled"]),
            "current_roll_trim_scaled_deg": float(obs["current_roll_trim_scaled"]),
            "preview_pitch_scaled_deg": float(obs["preview_pitch_scaled"]),
            "preview_roll_scaled_deg": float(obs["preview_roll_scaled"]),
            "preview_scale_mode": str(obs["preview_scale_mode"]),
            "preview_scale_eff": float(obs["preview_scale_eff"]),
            "preview_trim_active": int(obs["preview_trim_active"]),
            "preview_trim_source": str(obs["preview_trim_source"]),
            "trim_scale": float(obs["trim_scale"]),
            "trim_enabled": int(obs["trim_enabled"]),
            "trim_freeze": int(obs["trim_freeze"]),
            "trim_pressure_recent": float(obs["trim_pressure_recent"]),
            "trim_pressure_clip_recent": float(obs["trim_pressure_clip_recent"]),
            "trim_pressure_cmd_gap_recent": float(obs["trim_pressure_cmd_gap_recent"]),
            "trim_pressure_fullspeed_recent": float(obs["trim_pressure_fullspeed_recent"]),
            "trim_pressure_backlog_recent": float(obs["trim_pressure_backlog_recent"]),
            "trim_pressure_sample": float(obs["trim_pressure_sample"]),
            "trim_pressure_backlog_max_kg": float(obs["trim_pressure_backlog_max_kg"]),
            "trim_pressure_cmd_gap_kg": float(obs["trim_pressure_cmd_gap_kg"]),
            "trim_pressure_clip": int(obs["trim_pressure_clip"]),
            "trim_pressure_fullspeed": int(obs["trim_pressure_fullspeed"]),
            "steady_mean_abs_err_deg": float(dbg.get("steady_mean_abs_err_deg", 0.0)),
            "steady_std_err_deg": float(dbg.get("steady_std_err_deg", 0.0)),
            "trim_update_tick": int(dbg.get("trim_update_tick", 0)),
            "cmd_gap_kg": float(obs["cmd_gap"]),
            "sat_recent_ratio": float(obs["sat_recent_ratio"]),
            "cmd_gap_recent_kg": float(obs["cmd_gap_recent"]),
            "target_mode": str(obs["target_mode"]),
            "target_source": str(obs["target_source"]),
            "target_fallback_reason_last": str(obs["target_fallback_reason_last"]),
            "target_table_path": str(obs["target_table_path"]),
            "target_lookup_key": str(obs["target_lookup_key"]),
            "target_lookup_total": int(obs["target_lookup_total"]),
            "target_lookup_hit": int(obs["target_lookup_hit"]),
            "target_lookup_fallback": int(obs["target_lookup_fallback"]),
            "fallback_reason_missing_state": int(obs["fallback_reason_missing_state"]),
            "fallback_reason_missing_table": int(obs["fallback_reason_missing_table"]),
            "fallback_reason_missing_columns": int(obs["fallback_reason_missing_columns"]),
            "fallback_reason_invalid_state_id": int(obs["fallback_reason_invalid_state_id"]),
            "control_on": int(control_enabled),
            "ctrl_status": str(dbg.get("ctrl_status", "")),
            "ctrl_clipped": int(dbg.get("ctrl_clipped", 0)),
            "ctrl_base_clipped": int(dbg.get("ctrl_base_clipped", 0)),
            "ctrl_chain_clipped": int(
                dbg.get("ctrl_chain_clipped", dbg.get("ctrl_clipped", 0))
            ),
            "ctrl_sign_warn": int(dbg.get("ctrl_sign_warn", 0)),
            "ctrl_filter_alpha": float(dbg.get("ctrl_filter_alpha", 0.0)),
            "ctrl_effective_dt": float(dbg.get("ctrl_effective_dt", 0.0)),
            "ctrl_update_tick": int(dbg.get("ctrl_update_tick", 0)),
            "ctrl_raw_pitch_deg": float(
                dbg.get("ctrl_raw_pitch_deg", np.degrees(plant.state[4]))
            ),
            "ctrl_raw_roll_deg": float(
                dbg.get("ctrl_raw_roll_deg", np.degrees(plant.state[3]))
            ),
            "ctrl_raw_heave_m": float(dbg.get("ctrl_raw_heave_m", plant.state[2])),
            "ctrl_filtered_pitch_deg": float(
                dbg.get("ctrl_filtered_pitch_deg", np.degrees(plant.state[4]))
            ),
            "ctrl_filtered_roll_deg": float(
                dbg.get("ctrl_filtered_roll_deg", np.degrees(plant.state[3]))
            ),
            "ctrl_filtered_heave_m": float(
                dbg.get("ctrl_filtered_heave_m", plant.state[2])
            ),
            "ctrl_pitch_err_raw_deg": float(dbg.get("ctrl_pitch_err_raw_deg", 0.0)),
            "ctrl_roll_err_raw_deg": float(dbg.get("ctrl_roll_err_raw_deg", 0.0)),
            "ctrl_heave_err_raw_m": float(dbg.get("ctrl_heave_err_raw_m", 0.0)),
            "ctrl_pitch_err_deg": float(dbg.get("ctrl_pitch_err_deg", 0.0)),
            "ctrl_roll_err_deg": float(dbg.get("ctrl_roll_err_deg", 0.0)),
            "ctrl_heave_err_m": float(dbg.get("ctrl_heave_err_m", 0.0)),
            "ctrl_pitch_u_total": float(dbg.get("ctrl_pitch_u_total", 0.0)),
            "ctrl_roll_u_total": float(dbg.get("ctrl_roll_u_total", 0.0)),
            "ctrl_heave_u_total": float(dbg.get("ctrl_heave_u_total", 0.0)),
            "ctrl_pitch_in_deadband": int(dbg.get("ctrl_pitch_in_deadband", 0)),
            "ctrl_roll_in_deadband": int(dbg.get("ctrl_roll_in_deadband", 0)),
            "ctrl_heave_in_deadband": int(dbg.get("ctrl_heave_in_deadband", 0)),
            "ctrl_deadband_pitch_state": int(dbg.get("ctrl_deadband_pitch_state", 0)),
            "ctrl_deadband_roll_state": int(dbg.get("ctrl_deadband_roll_state", 0)),
            "ctrl_deadband_heave_state": int(dbg.get("ctrl_deadband_heave_state", 0)),
            "heave_bias_delta_mean_kg": float(
                dbg.get("heave_bias_delta_mean_kg", 0.0)
            ),
            "limiter_delta_mean_kg": float(dbg.get("limiter_delta_mean_kg", 0.0)),
            "post_chain_delta_mean_kg": float(
                dbg.get("post_chain_delta_mean_kg", 0.0)
            ),
            "post_chain_adjusted": int(dbg.get("post_chain_adjusted", 0)),
            "deadband_target_release_active": int(
                dbg.get("deadband_target_release_active", 0)
            ),
            "deadband_target_release_reason": str(
                dbg.get("deadband_target_release_reason", "")
            ),
            "deadband_target_release_delta_mean_kg": float(
                dbg.get("deadband_target_release_delta_mean_kg", 0.0)
            ),
            "deadband_target_release_pitch_ok": int(
                dbg.get("deadband_target_release_pitch_ok", 0)
            ),
            "deadband_target_release_roll_ok": int(
                dbg.get("deadband_target_release_roll_ok", 0)
            ),
            "deadband_target_release_exit_pitch_ok": int(
                dbg.get("deadband_target_release_exit_pitch_ok", 0)
            ),
            "deadband_target_release_exit_roll_ok": int(
                dbg.get("deadband_target_release_exit_roll_ok", 0)
            ),
            "deadband_target_release_latched": int(
                dbg.get("deadband_target_release_latched", 0)
            ),
            "deadband_target_release_reset_limiter": int(
                dbg.get("deadband_target_release_reset_limiter", 0)
            ),
            "preview_pump_suppression_active": int(
                dbg.get("preview_pump_suppression_active", 0)
            ),
            "preview_pump_restart_err_kg": float(
                dbg.get("preview_pump_restart_err_kg", 0.0)
            ),
            "preview_pump_suppression_reason": str(
                dbg.get("preview_pump_suppression_reason", "")
            ),
            "preview_primary_enabled": int(dbg.get("preview_primary_enabled", 0)),
            "preview_primary_active": int(dbg.get("preview_primary_active", 0)),
            "preview_primary_applied": int(dbg.get("preview_primary_applied", 0)),
            "preview_primary_candidate_applied": int(
                dbg.get("preview_primary_candidate_applied", 0)
            ),
            "preview_primary_target_t1_kg": float(dbg.get("preview_primary_target_t1_kg", 0.0)),
            "preview_primary_target_t2_kg": float(dbg.get("preview_primary_target_t2_kg", 0.0)),
            "preview_primary_target_t3_kg": float(dbg.get("preview_primary_target_t3_kg", 0.0)),
            "preview_primary_delta_t1_kg": float(dbg.get("preview_primary_delta_t1_kg", 0.0)),
            "preview_primary_delta_t2_kg": float(dbg.get("preview_primary_delta_t2_kg", 0.0)),
            "preview_primary_delta_t3_kg": float(dbg.get("preview_primary_delta_t3_kg", 0.0)),
            "preview_primary_delta_mean_kg": float(dbg.get("preview_primary_delta_mean_kg", 0.0)),
            "preview_primary_candidate_delta_mean_kg": float(
                dbg.get("preview_primary_candidate_delta_mean_kg", 0.0)
            ),
            "preview_primary_action": str(dbg.get("preview_primary_action", "")),
            "preview_primary_event_reset": int(dbg.get("preview_primary_event_reset", 0)),
            "preview_primary_target_refreshed": int(
                dbg.get("preview_primary_target_refreshed", 0)
            ),
            "preview_primary_target_reused": int(
                dbg.get("preview_primary_target_reused", 0)
            ),
            "preview_primary_target_age_s": float(
                dbg.get("preview_primary_target_age_s", 0.0)
            ),
            "preview_primary_safety_enabled": int(
                dbg.get("preview_primary_safety_enabled", 0)
            ),
            "preview_primary_safety_active": int(
                dbg.get("preview_primary_safety_active", 0)
            ),
            "preview_primary_safety_fallback": int(
                dbg.get("preview_primary_safety_fallback", 0)
            ),
            "preview_primary_safety_reason": str(
                dbg.get("preview_primary_safety_reason", "")
            ),
            "preview_primary_safety_pitch_abs_deg": float(
                dbg.get("preview_primary_safety_pitch_abs_deg", 0.0)
            ),
            "preview_primary_safety_roll_abs_deg": float(
                dbg.get("preview_primary_safety_roll_abs_deg", 0.0)
            ),
            "preview_primary_safety_env_norm": float(
                dbg.get("preview_primary_safety_env_norm", 0.0)
            ),
            "preview_primary_safety_use_envelope": int(
                dbg.get("preview_primary_safety_use_envelope", 0)
            ),
            "preview_primary_safety_envelope_enter_norm": float(
                dbg.get("preview_primary_safety_envelope_enter_norm", 0.0)
            ),
            "preview_primary_safety_normal_enter": int(
                dbg.get("preview_primary_safety_normal_enter", 0)
            ),
            "preview_primary_safety_emergency_enter": int(
                dbg.get("preview_primary_safety_emergency_enter", 0)
            ),
            "preview_primary_safety_enter_elapsed_s": float(
                dbg.get("preview_primary_safety_enter_elapsed_s", 0.0)
            ),
            "preview_primary_safety_enter_hold_s": float(
                dbg.get("preview_primary_safety_enter_hold_s", 0.0)
            ),
            "preview_primary_safety_pitch_enter_deg": float(
                dbg.get("preview_primary_safety_pitch_enter_deg", 0.0)
            ),
            "preview_primary_safety_roll_enter_deg": float(
                dbg.get("preview_primary_safety_roll_enter_deg", 0.0)
            ),
            "preview_primary_safety_emergency_pitch_enter_deg": float(
                dbg.get("preview_primary_safety_emergency_pitch_enter_deg", 0.0)
            ),
            "preview_primary_safety_emergency_roll_enter_deg": float(
                dbg.get("preview_primary_safety_emergency_roll_enter_deg", 0.0)
            ),
            "preview_primary_safety_pitch_exit_deg": float(
                dbg.get("preview_primary_safety_pitch_exit_deg", 0.0)
            ),
            "preview_primary_safety_roll_exit_deg": float(
                dbg.get("preview_primary_safety_roll_exit_deg", 0.0)
            ),
            "preview_primary_safety_exit_window_max_pitch_abs_deg": float(
                dbg.get("preview_primary_safety_exit_window_max_pitch_abs_deg", 0.0)
            ),
            "preview_primary_safety_exit_window_max_roll_abs_deg": float(
                dbg.get("preview_primary_safety_exit_window_max_roll_abs_deg", 0.0)
            ),
            "preview_primary_safety_exit_env_norm": float(
                dbg.get("preview_primary_safety_exit_env_norm", 0.0)
            ),
            "preview_primary_safety_exit_clean_windows": int(
                dbg.get("preview_primary_safety_exit_clean_windows", 0)
            ),
            "preview_primary_safety_exit_hold_s": float(
                dbg.get("preview_primary_safety_exit_hold_s", 0.0)
            ),
            "preview_primary_safety_exit_required_windows": int(
                dbg.get("preview_primary_safety_exit_required_windows", 0)
            ),
            "preview_pressure_block0_norm": float(
                dbg.get("preview_pressure_block0_norm", 0.0)
            ),
            "preview_pressure_block1_norm": float(
                dbg.get("preview_pressure_block1_norm", 0.0)
            ),
            "preview_pressure_block2_norm": float(
                dbg.get("preview_pressure_block2_norm", 0.0)
            ),
            "preview_pressure_block02_dot": float(
                dbg.get("preview_pressure_block02_dot", 0.0)
            ),
            "suppression_blocked_tanks": int(dbg.get("suppression_blocked_tanks", 0)),
            "suppression_blocked_mass_kg": float(
                dbg.get("suppression_blocked_mass_kg", 0.0)
            ),
            "suppression_delta_mean_kg": float(dbg.get("suppression_delta_mean_kg", 0.0)),
            "suppression_mask_t1": int(dbg.get("suppression_mask_t1", 0)),
            "suppression_mask_t2": int(dbg.get("suppression_mask_t2", 0)),
            "suppression_mask_t3": int(dbg.get("suppression_mask_t3", 0)),
        }
    )

def _build_case_components(
    excel_path,
    case_name,
    dt,
    n_steps,
    wind_gen,
    wind_trace,
    wind_update_interval_s,
    wind_mean_lpf_tau_s,
    wind_target_ws,
    wind_target_wd,
    trim_cfg,
    heave_cfg,
    controller_cfg,
    target_shape_cfg,
    pump_cfg=None,
    platform_profile=None,
    platform_cfg=None,
    control_enabled=True,
    start_from_heave_equilibrium=True,
    preview_trim_provider=None,
):
    plant = FloatingPlatform(
        excel_path,
        pump_cfg=pump_cfg,
        platform_profile=platform_profile,
        platform_cfg=platform_cfg,
    )
    plant.set_irregular_wave(0, 10, 0, 0)

    heave_init_info = {"ok": False, "z_eq_m": np.nan, "a_waterplane_m2": np.nan}
    if start_from_heave_equilibrium:
        heave_init_info = compute_heave_equilibrium_from_ballast(plant)
        if heave_init_info["ok"]:
            # Start from static heave equilibrium to avoid artificial initialization transient.
            plant.state[2] = float(heave_init_info["z_eq_m"])
            plant.state[8] = 0.0
        else:
            print(f"   [WARN] {case_name}: failed to compute heave equilibrium, keep z=0 init.")

    ctrl = build_default_controller(plant, dt, controller_cfg=controller_cfg)

    limits = {
        "pitch_deg": 8.0,
        "roll_deg": 8.0,
        "sat_ratio_any_max": 0.98,
        "switch_per_min_max": 30.0,
    }

    has_wind = (wind_gen is not None) or (wind_trace is not None)
    wind_source, wind_seed = create_wind_source(
        wind_gen=wind_gen,
        dt=dt,
        wind_trace=wind_trace,
        update_interval_s=wind_update_interval_s,
        mean_lpf_tau_s=wind_mean_lpf_tau_s,
        target_ws=wind_target_ws,
        target_wd=wind_target_wd,
        seed=RANDOM_SEED,
    )
    if wind_trace is not None and len(wind_trace["ws"]) < n_steps:
        raise ValueError(
            f"wind_trace too short for {case_name}: need {n_steps}, got {len(wind_trace['ws'])}"
        )
    if has_wind and n_steps * dt < wind_update_interval_s and (wind_trace is None):
        print(
            f"   [WARN] {case_name}: sim window {n_steps * dt:.1f}s < wind update interval "
            f"{wind_update_interval_s:.1f}s; Markov state will not transition."
        )

    if target_shape_cfg is None:
        target_shape_cfg = clone_cfg(DEFAULT_TARGET_SHAPE_CFG)
    if trim_cfg is None:
        trim_cfg = {}
    if heave_cfg is None:
        heave_cfg = {}

    trim_governor = None
    if has_wind and len(trim_cfg) > 0:
        if "trim_map_version" not in trim_cfg:
            raise ValueError(
                "trim_cfg must explicitly include trim_map_version (v1 or v2); "
                "implicit trim-map defaults are disabled."
            )
        allowed_trim_keys = {
            "k_trim_deg_per_mps",
            "max_trim_deg",
            "sat_gate_on",
            "sat_gate_off",
            "cmd_gap_decay_start_kg",
            "cmd_gap_decay_full_kg",
            "sat_window_s",
            "cmd_gap_window_s",
            "trim_min_scale",
            "trim_decay_per_step",
            "trim_recover_per_step",
            "wind_lpf_tau_s",
            "steady_window_s",
            "steady_mean_on_deg",
            "steady_mean_off_deg",
            "steady_std_on_deg",
            "steady_std_off_deg",
            "trim_update_s",
            "pressure_backlog_start_kg",
            "pressure_backlog_full_kg",
            "trim_logic",
            "simple_decay_on_clip",
            "simple_recover_per_step",
            "simple_pressure_hold_th",
            "simple_pressure_decay_th",
            "simple_freeze_enabled",
            "simple_pressure_freeze_th",
            "simple_pressure_unfreeze_th",
            "trim_map_version",
            "ws_rated_mps",
            "ws_cutout_mps",
            "post_rated_mode",
            "post_rated_decay_ratio",
            "protect_trim_scale",
            "target_mode",
            "target_table_path",
            "target_lookup_key",
            "target_table_strict",
            "preview_scale_mode",
            "preview_scale",
        }
        trim_cfg_checked = {
            k: trim_cfg[k] for k in trim_cfg if k in allowed_trim_keys
        }
        trim_governor = TrimGovernor(dt=dt, **trim_cfg_checked)


    setpoint_shaper = SetpointShaper(
        dt=dt,
        alpha=target_shape_cfg.get("setpoint_alpha", 0.98),
        rate_deg_s=target_shape_cfg.get("setpoint_rate_deg_s", 0.02),
        enabled=target_shape_cfg.get("enable_setpoint_shaping", True),
        initial=(ctrl.setpoints["pitch"], ctrl.setpoints["roll"]),
    )

    heave_balancer = None
    if control_enabled and bool(heave_cfg.get("enabled", False)):
        k33 = float(plant.K_hydro[2]) if len(plant.K_hydro) > 2 else 0.0
        a_waterplane = 0.0
        if plant.rho > 1e-6 and plant.g > 1e-6 and k33 > 0.0:
            a_waterplane = k33 / (plant.rho * plant.g)
        if a_waterplane <= 0.0:
            print(
                f"   [WARN] {case_name}: invalid A_waterplane from K_hydro[2]={k33:.3e}, "
                "heave balancer disabled."
            )
        else:
            target_heave_cfg = heave_cfg.get("target_heave_m", None)
            if (
                target_heave_cfg is None
                or (isinstance(target_heave_cfg, str) and target_heave_cfg.lower() == "auto")
            ):
                target_heave_m = -float(np.sum(plant.current_ballast_mass)) / (
                    plant.rho * a_waterplane
                )
            else:
                target_heave_m = float(target_heave_cfg)
            heave_balancer = HeaveBiasBalancer(
                dt=dt,
                rho=plant.rho,
                g=plant.g,
                a_waterplane_m2=a_waterplane,
                target_heave_m=float(target_heave_m),
                kp=float(heave_cfg.get("kp", 0.005)),
                ki=float(heave_cfg.get("ki", 0.0)),
                max_correction_per_tank_kg=float(
                    heave_cfg.get("max_correction_per_tank_kg", 2.0)
                ),
                m_min_op_kg=float(heave_cfg.get("m_min_op_kg", 200000.0)),
                m_max_op_ratio=float(heave_cfg.get("m_max_op_ratio", 0.90)),
                history_window_s=float(heave_cfg.get("history_window_s", 30.0)),
                enabled=True,
            )

    rate_limit_enabled = bool(target_shape_cfg.get("enable_rate_limit", False))
    cmd_limiter = None
    if rate_limit_enabled:
        cmd_limiter = CommandRateLimiter(
            dt=dt,
            rho=plant.rho,
            max_capacity=plant.tank_capacity,
            rate_limit_m3_min=target_shape_cfg.get("rate_limit_m3_min", 12.0),
            enabled=True,
        )

    policy = ClosedLoopPolicy(
        controller=ctrl,
        trim_governor=trim_governor,
        setpoint_shaper=setpoint_shaper,
        command_rate_limiter=cmd_limiter,
        heave_balancer=heave_balancer,
        preview_trim_provider=preview_trim_provider,
        primary_safety_cfg=target_shape_cfg,
    )
    initial_ws = 0.0
    if wind_trace is not None and len(wind_trace["ws"]) > 0:
        initial_ws = float(wind_trace["ws"][0])
    policy.reset(initial_cmd=plant.current_ballast_mass.copy(), initial_ws=initial_ws)

    return {
        "plant": plant,
        "ctrl": ctrl,
        "limits": limits,
        "has_wind": has_wind,
        "wind_source": wind_source,
        "wind_seed": wind_seed,
        "target_shape_cfg": target_shape_cfg,
        "trim_governor": trim_governor,
        "heave_balancer": heave_balancer,
        "rate_limit_enabled": rate_limit_enabled,
        "policy": policy,
        "heave_init_info": heave_init_info,
    }


def _build_open_loop_debug(plant, ctrl):
    return {
        "pitch_sp_deg": float(ctrl.setpoints["pitch"]),
        "roll_sp_deg": float(ctrl.setpoints["roll"]),
        "pitch_sp_raw_deg": float(ctrl.setpoints["pitch"]),
        "roll_sp_raw_deg": float(ctrl.setpoints["roll"]),
        "current_pitch_trim_raw_deg": 0.0,
        "current_roll_trim_raw_deg": 0.0,
        "preview_pitch_bias_deg": 0.0,
        "preview_roll_bias_deg": 0.0,
        "combined_pitch_trim_raw_deg": 0.0,
        "combined_roll_trim_raw_deg": 0.0,
        "current_pitch_trim_scaled_deg": 0.0,
        "current_roll_trim_scaled_deg": 0.0,
        "preview_pitch_scaled_deg": 0.0,
        "preview_roll_scaled_deg": 0.0,
        "preview_scale_mode": "none",
        "preview_scale_eff": 0.0,
        "preview_trim_active": 0,
        "preview_trim_source": "open_loop",
        "trim_scale": 0.0,
        "sat_recent_ratio": 0.0,
        "trim_pressure_recent": 0.0,
        "trim_pressure_clip_recent": 0.0,
        "trim_pressure_cmd_gap_recent": 0.0,
        "trim_pressure_fullspeed_recent": 0.0,
        "trim_pressure_backlog_recent": 0.0,
        "trim_pressure_sample": 0.0,
        "trim_pressure_backlog_max_kg": 0.0,
        "trim_pressure_cmd_gap_kg": 0.0,
        "trim_pressure_clip": 0,
        "trim_pressure_fullspeed": 0,
        "cmd_gap_recent_kg": 0.0,
        "target_mode": "g1",
        "target_source": "open_loop",
        "target_fallback_reason_last": "",
        "target_table_path": "",
        "target_lookup_key": "state_id",
        "target_lookup_total": 0,
        "target_lookup_hit": 0,
        "target_lookup_fallback": 0,
        "fallback_reason_missing_state": 0,
        "fallback_reason_missing_table": 0,
        "fallback_reason_missing_columns": 0,
        "fallback_reason_invalid_state_id": 0,
        "cmd_gap_kg": 0.0,
        "ctrl_status": "open",
        "ctrl_clipped": 0,
        "ctrl_base_clipped": 0,
        "ctrl_chain_clipped": 0,
        "alloc_clip_any": 0,
        "ctrl_sign_warn": 0,
        "ctrl_filter_alpha": 0.0,
        "ctrl_effective_dt": 0.0,
        "ctrl_raw_pitch_deg": float(np.degrees(plant.state[4])),
        "ctrl_raw_roll_deg": float(np.degrees(plant.state[3])),
        "ctrl_raw_heave_m": float(plant.state[2]),
        "ctrl_filtered_pitch_deg": float(np.degrees(plant.state[4])),
        "ctrl_filtered_roll_deg": float(np.degrees(plant.state[3])),
        "ctrl_filtered_heave_m": float(plant.state[2]),
        "heave_bias_delta_mean_kg": 0.0,
        "limiter_delta_mean_kg": 0.0,
        "post_chain_delta_mean_kg": 0.0,
        "post_chain_adjusted": 0,
        "hm_smoothed_heave_m": float(plant.state[2]),
        "hm_heave_error_m": 0.0,
        "hm_total_ballast_kg": float(np.sum(plant.current_ballast_mass)),
        "ballast_total_kg": float(np.sum(plant.current_ballast_mass)),
        "heave_correction_per_tank_kg": 0.0,
        "heave_balancer_active": 0,
    }


def _extract_step_obs(plant, info, dbg, ctrl_diag, limits, wind_obs):
    pitch_sp = float(dbg.get("pitch_sp_deg", 0.0))
    roll_sp = float(dbg.get("roll_sp_deg", 0.0))
    pitch_sp_raw = float(dbg.get("pitch_sp_raw_deg", pitch_sp))
    roll_sp_raw = float(dbg.get("roll_sp_raw_deg", roll_sp))
    current_pitch_trim_raw = float(dbg.get("current_pitch_trim_raw_deg", 0.0))
    current_roll_trim_raw = float(dbg.get("current_roll_trim_raw_deg", 0.0))
    preview_pitch_bias = float(dbg.get("preview_pitch_bias_deg", 0.0))
    preview_roll_bias = float(dbg.get("preview_roll_bias_deg", 0.0))
    combined_pitch_trim_raw = float(dbg.get("combined_pitch_trim_raw_deg", pitch_sp_raw))
    combined_roll_trim_raw = float(dbg.get("combined_roll_trim_raw_deg", roll_sp_raw))
    current_pitch_trim_scaled = float(dbg.get("current_pitch_trim_scaled_deg", 0.0))
    current_roll_trim_scaled = float(dbg.get("current_roll_trim_scaled_deg", 0.0))
    preview_pitch_scaled = float(dbg.get("preview_pitch_scaled_deg", preview_pitch_bias))
    preview_roll_scaled = float(dbg.get("preview_roll_scaled_deg", preview_roll_bias))
    preview_scale_mode = str(dbg.get("preview_scale_mode", "none"))
    preview_scale_eff = float(dbg.get("preview_scale_eff", 0.0))
    preview_trim_active = int(dbg.get("preview_trim_active", 0))
    preview_trim_source = str(dbg.get("preview_trim_source", "none"))
    preview_pump_suppression_active = int(dbg.get("preview_pump_suppression_active", 0))
    preview_pump_restart_err_kg = float(dbg.get("preview_pump_restart_err_kg", 0.0))
    preview_pump_suppression_reason = str(dbg.get("preview_pump_suppression_reason", ""))
    preview_primary_enabled = int(dbg.get("preview_primary_enabled", 0))
    preview_primary_active = int(dbg.get("preview_primary_active", 0))
    preview_primary_applied = int(dbg.get("preview_primary_applied", 0))
    preview_primary_candidate_applied = int(dbg.get("preview_primary_candidate_applied", 0))
    preview_primary_target_t1_kg = float(dbg.get("preview_primary_target_t1_kg", 0.0))
    preview_primary_target_t2_kg = float(dbg.get("preview_primary_target_t2_kg", 0.0))
    preview_primary_target_t3_kg = float(dbg.get("preview_primary_target_t3_kg", 0.0))
    preview_primary_delta_t1_kg = float(dbg.get("preview_primary_delta_t1_kg", 0.0))
    preview_primary_delta_t2_kg = float(dbg.get("preview_primary_delta_t2_kg", 0.0))
    preview_primary_delta_t3_kg = float(dbg.get("preview_primary_delta_t3_kg", 0.0))
    preview_primary_delta_mean_kg = float(dbg.get("preview_primary_delta_mean_kg", 0.0))
    preview_primary_candidate_delta_mean_kg = float(
        dbg.get("preview_primary_candidate_delta_mean_kg", 0.0)
    )
    preview_primary_action = str(dbg.get("preview_primary_action", ""))
    preview_primary_event_reset = int(dbg.get("preview_primary_event_reset", 0))
    preview_primary_target_refreshed = int(
        dbg.get("preview_primary_target_refreshed", 0)
    )
    preview_primary_target_reused = int(dbg.get("preview_primary_target_reused", 0))
    preview_primary_target_age_s = float(dbg.get("preview_primary_target_age_s", 0.0))
    preview_primary_safety_enabled = int(dbg.get("preview_primary_safety_enabled", 0))
    preview_primary_safety_active = int(dbg.get("preview_primary_safety_active", 0))
    preview_primary_safety_fallback = int(dbg.get("preview_primary_safety_fallback", 0))
    preview_primary_safety_reason = str(dbg.get("preview_primary_safety_reason", ""))
    preview_primary_safety_pitch_abs_deg = float(
        dbg.get("preview_primary_safety_pitch_abs_deg", 0.0)
    )
    preview_primary_safety_roll_abs_deg = float(
        dbg.get("preview_primary_safety_roll_abs_deg", 0.0)
    )
    preview_primary_safety_env_norm = float(dbg.get("preview_primary_safety_env_norm", 0.0))
    preview_primary_safety_use_envelope = int(
        dbg.get("preview_primary_safety_use_envelope", 0)
    )
    preview_primary_safety_envelope_enter_norm = float(
        dbg.get("preview_primary_safety_envelope_enter_norm", 0.0)
    )
    preview_primary_safety_normal_enter = int(
        dbg.get("preview_primary_safety_normal_enter", 0)
    )
    preview_primary_safety_emergency_enter = int(
        dbg.get("preview_primary_safety_emergency_enter", 0)
    )
    preview_primary_safety_enter_elapsed_s = float(
        dbg.get("preview_primary_safety_enter_elapsed_s", 0.0)
    )
    preview_primary_safety_enter_hold_s = float(
        dbg.get("preview_primary_safety_enter_hold_s", 0.0)
    )
    preview_primary_safety_pitch_enter_deg = float(
        dbg.get("preview_primary_safety_pitch_enter_deg", 0.0)
    )
    preview_primary_safety_roll_enter_deg = float(
        dbg.get("preview_primary_safety_roll_enter_deg", 0.0)
    )
    preview_primary_safety_emergency_pitch_enter_deg = float(
        dbg.get("preview_primary_safety_emergency_pitch_enter_deg", 0.0)
    )
    preview_primary_safety_emergency_roll_enter_deg = float(
        dbg.get("preview_primary_safety_emergency_roll_enter_deg", 0.0)
    )
    preview_primary_safety_pitch_exit_deg = float(
        dbg.get("preview_primary_safety_pitch_exit_deg", 0.0)
    )
    preview_primary_safety_roll_exit_deg = float(
        dbg.get("preview_primary_safety_roll_exit_deg", 0.0)
    )
    preview_primary_safety_exit_window_max_pitch_abs_deg = float(
        dbg.get("preview_primary_safety_exit_window_max_pitch_abs_deg", 0.0)
    )
    preview_primary_safety_exit_window_max_roll_abs_deg = float(
        dbg.get("preview_primary_safety_exit_window_max_roll_abs_deg", 0.0)
    )
    preview_primary_safety_exit_env_norm = float(
        dbg.get("preview_primary_safety_exit_env_norm", 0.0)
    )
    preview_primary_safety_exit_clean_windows = int(
        dbg.get("preview_primary_safety_exit_clean_windows", 0)
    )
    preview_primary_safety_exit_hold_s = float(
        dbg.get("preview_primary_safety_exit_hold_s", 0.0)
    )
    preview_primary_safety_exit_required_windows = int(
        dbg.get("preview_primary_safety_exit_required_windows", 0)
    )
    preview_pressure_block0_norm = float(dbg.get("preview_pressure_block0_norm", 0.0))
    preview_pressure_block1_norm = float(dbg.get("preview_pressure_block1_norm", 0.0))
    preview_pressure_block2_norm = float(dbg.get("preview_pressure_block2_norm", 0.0))
    preview_pressure_block02_dot = float(dbg.get("preview_pressure_block02_dot", 0.0))
    suppression_blocked_tanks = int(dbg.get("suppression_blocked_tanks", 0))
    suppression_blocked_mass_kg = float(dbg.get("suppression_blocked_mass_kg", 0.0))
    suppression_delta_mean_kg = float(dbg.get("suppression_delta_mean_kg", 0.0))
    suppression_mask_t1 = int(dbg.get("suppression_mask_t1", 0))
    suppression_mask_t2 = int(dbg.get("suppression_mask_t2", 0))
    suppression_mask_t3 = int(dbg.get("suppression_mask_t3", 0))
    deadband_target_release_active = int(dbg.get("deadband_target_release_active", 0))
    deadband_target_release_reason = str(dbg.get("deadband_target_release_reason", ""))
    deadband_target_release_delta_mean_kg = float(
        dbg.get("deadband_target_release_delta_mean_kg", 0.0)
    )
    deadband_target_release_pitch_ok = int(dbg.get("deadband_target_release_pitch_ok", 0))
    deadband_target_release_roll_ok = int(dbg.get("deadband_target_release_roll_ok", 0))
    deadband_target_release_exit_pitch_ok = int(
        dbg.get("deadband_target_release_exit_pitch_ok", 0)
    )
    deadband_target_release_exit_roll_ok = int(
        dbg.get("deadband_target_release_exit_roll_ok", 0)
    )
    deadband_target_release_latched = int(
        dbg.get("deadband_target_release_latched", 0)
    )
    deadband_target_release_reset_limiter = int(
        dbg.get("deadband_target_release_reset_limiter", 0)
    )
    trim_scale = float(dbg.get("trim_scale", 0.0))
    trim_enabled = int(dbg.get("trim_enabled", 0))
    trim_freeze = int(dbg.get("trim_freeze", 0))
    sat_recent_ratio = float(dbg.get("sat_recent_ratio", 0.0))
    trim_pressure_recent = float(dbg.get("trim_pressure_recent", sat_recent_ratio))
    trim_pressure_clip_recent = float(dbg.get("trim_pressure_clip_recent", 0.0))
    trim_pressure_cmd_gap_recent = float(dbg.get("trim_pressure_cmd_gap_recent", 0.0))
    trim_pressure_fullspeed_recent = float(dbg.get("trim_pressure_fullspeed_recent", 0.0))
    trim_pressure_backlog_recent = float(dbg.get("trim_pressure_backlog_recent", 0.0))
    trim_pressure_sample = float(dbg.get("trim_pressure_sample", trim_pressure_recent))
    trim_pressure_backlog_max_kg = float(dbg.get("trim_pressure_backlog_max_kg", 0.0))
    trim_pressure_cmd_gap_kg = float(dbg.get("trim_pressure_cmd_gap_kg", 0.0))
    trim_pressure_clip = int(dbg.get("trim_pressure_clip", 0))
    trim_pressure_fullspeed = int(dbg.get("trim_pressure_fullspeed", 0))
    cmd_gap_recent = float(dbg.get("cmd_gap_recent_kg", 0.0))
    cmd_gap = float(dbg.get("cmd_gap_kg", 0.0))
    target_mode = str(dbg.get("target_mode", "g1"))
    target_source = str(dbg.get("target_source", "g1_formula"))
    target_fallback_reason_last = str(dbg.get("target_fallback_reason_last", ""))
    target_table_path = str(dbg.get("target_table_path", ""))
    target_lookup_key = str(dbg.get("target_lookup_key", "state_id"))
    target_lookup_total = int(dbg.get("target_lookup_total", 0))
    target_lookup_hit = int(dbg.get("target_lookup_hit", 0))
    target_lookup_fallback = int(dbg.get("target_lookup_fallback", 0))
    fallback_reason_missing_state = int(dbg.get("fallback_reason_missing_state", 0))
    fallback_reason_missing_table = int(dbg.get("fallback_reason_missing_table", 0))
    fallback_reason_missing_columns = int(dbg.get("fallback_reason_missing_columns", 0))
    fallback_reason_invalid_state_id = int(dbg.get("fallback_reason_invalid_state_id", 0))

    wind_state_raw = wind_obs.get("state_id", None) if isinstance(wind_obs, dict) else None
    try:
        if wind_state_raw is None:
            wind_state_id = None
        elif isinstance(wind_state_raw, float) and (not np.isfinite(wind_state_raw)):
            wind_state_id = None
        else:
            wind_state_id = int(wind_state_raw)
    except Exception:
        wind_state_id = None

    wind_ws_state = wind_obs.get("ws_state", None) if isinstance(wind_obs, dict) else None
    wind_wd_state = wind_obs.get("wd_state", None) if isinstance(wind_obs, dict) else None

    pitch_deg = float(info["pitch_deg"])
    roll_deg = float(info["roll_deg"])
    tank_masses = _as_vec3(info.get("tank_masses", plant.current_ballast_mass))
    target_masses = _as_vec3(
        info.get("target_ballast_mass", plant.target_ballast_mass),
        fallback=plant.target_ballast_mass,
    )
    err_masses = _as_vec3(
        info.get("ballast_err_kg", target_masses - tank_masses),
        fallback=(target_masses - tank_masses),
    )
    alloc_deltas = _as_vec3(ctrl_diag.get("alloc_m_deltas_kg", np.zeros(3, dtype=float)))
    alloc_raw = _as_vec3(ctrl_diag.get("alloc_m_raw_kg", np.zeros(3, dtype=float)))
    alloc_cmd = _as_vec3(ctrl_diag.get("alloc_m_cmds_kg", np.zeros(3, dtype=float)))
    alloc_clip_mask = np.abs(alloc_raw - alloc_cmd) > 1e-6
    alloc_clip_any = int(np.any(alloc_clip_mask))
    pump_fullspeed_any = int(info.get("pump_fullspeed_any", 0))
    pump_backlog_max_kg = float(info.get("pump_backlog_max_kg", np.max(np.abs(err_masses))))
    pump_total_backlog_kg = float(info.get("pump_total_backlog_kg", np.sum(np.abs(err_masses))))
    pump_target_motion_kg_s = float(info.get("pump_target_motion_kg_s", 0.0))
    pump_global_quiet_s = float(info.get("pump_global_quiet_s", 0.0))
    pump_quiet_stop_block_count = int(info.get("pump_quiet_stop_block_count", 0))
    pump_latch_switch_count = int(info.get("pump_latch_switch_count", 0))
    pump_near_target_mean_s = float(
        np.mean(_as_vec3(info.get("pump_near_target_s", np.zeros(3, dtype=float))))
    )
    sat_any = int(any(info["pump_saturated"]))
    act = [bool(v) for v in info["pump_active"]]
    total_rate = float(np.sum(info["pump_rate_cmd_m3_min"]))

    return {
        "pitch_sp": pitch_sp,
        "roll_sp": roll_sp,
        "pitch_sp_raw": pitch_sp_raw,
        "roll_sp_raw": roll_sp_raw,
        "current_pitch_trim_raw": current_pitch_trim_raw,
        "current_roll_trim_raw": current_roll_trim_raw,
        "preview_pitch_bias": preview_pitch_bias,
        "preview_roll_bias": preview_roll_bias,
        "combined_pitch_trim_raw": combined_pitch_trim_raw,
        "combined_roll_trim_raw": combined_roll_trim_raw,
        "current_pitch_trim_scaled": current_pitch_trim_scaled,
        "current_roll_trim_scaled": current_roll_trim_scaled,
        "preview_pitch_scaled": preview_pitch_scaled,
        "preview_roll_scaled": preview_roll_scaled,
        "preview_scale_mode": preview_scale_mode,
        "preview_scale_eff": preview_scale_eff,
        "preview_trim_active": preview_trim_active,
        "preview_trim_source": preview_trim_source,
        "preview_pump_suppression_active": preview_pump_suppression_active,
        "preview_pump_restart_err_kg": preview_pump_restart_err_kg,
        "preview_pump_suppression_reason": preview_pump_suppression_reason,
        "preview_primary_enabled": preview_primary_enabled,
        "preview_primary_active": preview_primary_active,
        "preview_primary_applied": preview_primary_applied,
        "preview_primary_candidate_applied": preview_primary_candidate_applied,
        "preview_primary_target_t1_kg": preview_primary_target_t1_kg,
        "preview_primary_target_t2_kg": preview_primary_target_t2_kg,
        "preview_primary_target_t3_kg": preview_primary_target_t3_kg,
        "preview_primary_delta_t1_kg": preview_primary_delta_t1_kg,
        "preview_primary_delta_t2_kg": preview_primary_delta_t2_kg,
        "preview_primary_delta_t3_kg": preview_primary_delta_t3_kg,
        "preview_primary_delta_mean_kg": preview_primary_delta_mean_kg,
        "preview_primary_candidate_delta_mean_kg": preview_primary_candidate_delta_mean_kg,
        "preview_primary_action": preview_primary_action,
        "preview_primary_event_reset": preview_primary_event_reset,
        "preview_primary_target_refreshed": preview_primary_target_refreshed,
        "preview_primary_target_reused": preview_primary_target_reused,
        "preview_primary_target_age_s": preview_primary_target_age_s,
        "preview_primary_safety_enabled": preview_primary_safety_enabled,
        "preview_primary_safety_active": preview_primary_safety_active,
        "preview_primary_safety_fallback": preview_primary_safety_fallback,
        "preview_primary_safety_reason": preview_primary_safety_reason,
        "preview_primary_safety_pitch_abs_deg": preview_primary_safety_pitch_abs_deg,
        "preview_primary_safety_roll_abs_deg": preview_primary_safety_roll_abs_deg,
        "preview_primary_safety_env_norm": preview_primary_safety_env_norm,
        "preview_primary_safety_use_envelope": preview_primary_safety_use_envelope,
        "preview_primary_safety_envelope_enter_norm": preview_primary_safety_envelope_enter_norm,
        "preview_primary_safety_normal_enter": preview_primary_safety_normal_enter,
        "preview_primary_safety_emergency_enter": preview_primary_safety_emergency_enter,
        "preview_primary_safety_enter_elapsed_s": preview_primary_safety_enter_elapsed_s,
        "preview_primary_safety_enter_hold_s": preview_primary_safety_enter_hold_s,
        "preview_primary_safety_pitch_enter_deg": preview_primary_safety_pitch_enter_deg,
        "preview_primary_safety_roll_enter_deg": preview_primary_safety_roll_enter_deg,
        "preview_primary_safety_emergency_pitch_enter_deg": preview_primary_safety_emergency_pitch_enter_deg,
        "preview_primary_safety_emergency_roll_enter_deg": preview_primary_safety_emergency_roll_enter_deg,
        "preview_primary_safety_pitch_exit_deg": preview_primary_safety_pitch_exit_deg,
        "preview_primary_safety_roll_exit_deg": preview_primary_safety_roll_exit_deg,
        "preview_primary_safety_exit_window_max_pitch_abs_deg": preview_primary_safety_exit_window_max_pitch_abs_deg,
        "preview_primary_safety_exit_window_max_roll_abs_deg": preview_primary_safety_exit_window_max_roll_abs_deg,
        "preview_primary_safety_exit_env_norm": preview_primary_safety_exit_env_norm,
        "preview_primary_safety_exit_clean_windows": preview_primary_safety_exit_clean_windows,
        "preview_primary_safety_exit_hold_s": preview_primary_safety_exit_hold_s,
        "preview_primary_safety_exit_required_windows": preview_primary_safety_exit_required_windows,
        "preview_pressure_block0_norm": preview_pressure_block0_norm,
        "preview_pressure_block1_norm": preview_pressure_block1_norm,
        "preview_pressure_block2_norm": preview_pressure_block2_norm,
        "preview_pressure_block02_dot": preview_pressure_block02_dot,
        "suppression_blocked_tanks": suppression_blocked_tanks,
        "suppression_blocked_mass_kg": suppression_blocked_mass_kg,
        "suppression_delta_mean_kg": suppression_delta_mean_kg,
        "suppression_mask_t1": suppression_mask_t1,
        "suppression_mask_t2": suppression_mask_t2,
        "suppression_mask_t3": suppression_mask_t3,
        "deadband_target_release_active": deadband_target_release_active,
        "deadband_target_release_reason": deadband_target_release_reason,
        "deadband_target_release_delta_mean_kg": deadband_target_release_delta_mean_kg,
        "deadband_target_release_pitch_ok": deadband_target_release_pitch_ok,
        "deadband_target_release_roll_ok": deadband_target_release_roll_ok,
        "deadband_target_release_exit_pitch_ok": deadband_target_release_exit_pitch_ok,
        "deadband_target_release_exit_roll_ok": deadband_target_release_exit_roll_ok,
        "deadband_target_release_latched": deadband_target_release_latched,
        "deadband_target_release_reset_limiter": deadband_target_release_reset_limiter,
        "trim_scale": trim_scale,
        "trim_enabled": trim_enabled,
        "trim_freeze": trim_freeze,
        "sat_recent_ratio": sat_recent_ratio,
        "trim_pressure_recent": trim_pressure_recent,
        "trim_pressure_clip_recent": trim_pressure_clip_recent,
        "trim_pressure_cmd_gap_recent": trim_pressure_cmd_gap_recent,
        "trim_pressure_fullspeed_recent": trim_pressure_fullspeed_recent,
        "trim_pressure_backlog_recent": trim_pressure_backlog_recent,
        "trim_pressure_sample": trim_pressure_sample,
        "trim_pressure_backlog_max_kg": trim_pressure_backlog_max_kg,
        "trim_pressure_cmd_gap_kg": trim_pressure_cmd_gap_kg,
        "trim_pressure_clip": trim_pressure_clip,
        "trim_pressure_fullspeed": trim_pressure_fullspeed,
        "cmd_gap_recent": cmd_gap_recent,
        "cmd_gap": cmd_gap,
        "target_mode": target_mode,
        "target_source": target_source,
        "target_fallback_reason_last": target_fallback_reason_last,
        "target_table_path": target_table_path,
        "target_lookup_key": target_lookup_key,
        "target_lookup_total": target_lookup_total,
        "target_lookup_hit": target_lookup_hit,
        "target_lookup_fallback": target_lookup_fallback,
        "fallback_reason_missing_state": fallback_reason_missing_state,
        "fallback_reason_missing_table": fallback_reason_missing_table,
        "fallback_reason_missing_columns": fallback_reason_missing_columns,
        "fallback_reason_invalid_state_id": fallback_reason_invalid_state_id,
        "wind_state_id": wind_state_id,
        "wind_ws_state": wind_ws_state,
        "wind_wd_state": wind_wd_state,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "pitch_exceed": int(abs(pitch_deg) > limits["pitch_deg"]),
        "roll_exceed": int(abs(roll_deg) > limits["roll_deg"]),
        "tank_masses": tank_masses,
        "target_masses": target_masses,
        "err_masses": err_masses,
        "alloc_deltas": alloc_deltas,
        "alloc_raw": alloc_raw,
        "alloc_cmd": alloc_cmd,
        "alloc_clip_mask": alloc_clip_mask,
        "alloc_clip_any": alloc_clip_any,
        "pump_fullspeed_any": pump_fullspeed_any,
        "pump_backlog_max_kg": pump_backlog_max_kg,
        "pump_total_backlog_kg": pump_total_backlog_kg,
        "pump_target_motion_kg_s": pump_target_motion_kg_s,
        "pump_global_quiet_s": pump_global_quiet_s,
        "pump_quiet_stop_block_count": pump_quiet_stop_block_count,
        "pump_latch_switch_count": pump_latch_switch_count,
        "pump_near_target_mean_s": pump_near_target_mean_s,
        "sat_any": sat_any,
        "act": act,
        "total_rate": total_rate,
    }


def _update_case_accumulators(acc, obs, dbg, t_sim):
    acc["sp_pitch_sum"] += obs["pitch_sp"]
    acc["sp_roll_sum"] += obs["roll_sp"]
    acc["trim_scale_sum"] += obs["trim_scale"]
    acc["trim_pressure_sum"] += obs["trim_pressure_recent"]
    acc["trim_gate_steps"] += int(obs["trim_scale"] < 0.999)
    acc["trim_enabled_steps"] += obs["trim_enabled"]
    acc["trim_freeze_steps"] += obs["trim_freeze"]
    acc["trim_pressure_clip_steps"] += obs["trim_pressure_clip"]
    acc["trim_pressure_fullspeed_steps"] += obs["trim_pressure_fullspeed"]
    acc["cmd_gap_sum"] += obs["cmd_gap"]
    acc["cmd_gap_values"].append(obs["cmd_gap"])
    acc["cmd_limited_steps"] += int(obs["cmd_gap"] > 1e-6)

    acc["hm_heave_error_abs_sum"] += abs(float(dbg.get("hm_heave_error_m", 0.0)))
    acc["hm_corr_abs_sum"] += abs(float(dbg.get("heave_correction_per_tank_kg", 0.0)))
    acc["hm_active_steps"] += int(dbg.get("heave_balancer_active", 0))

    acc["pitch_sq_sum"] += obs["pitch_deg"] * obs["pitch_deg"]
    acc["roll_sq_sum"] += obs["roll_deg"] * obs["roll_deg"]
    acc["pitch_abs_max"] = max(acc["pitch_abs_max"], abs(obs["pitch_deg"]))
    acc["roll_abs_max"] = max(acc["roll_abs_max"], abs(obs["roll_deg"]))
    acc["pitch_exceed_steps"] += obs["pitch_exceed"]
    acc["roll_exceed_steps"] += obs["roll_exceed"]

    sat_any = obs["sat_any"]
    acc["sat_any_steps"] += sat_any
    acc["sat_streak"] = acc["sat_streak"] + 1 if sat_any else 0
    acc["sat_streak_max"] = max(acc["sat_streak_max"], acc["sat_streak"])

    acc["ctrl_clipped_steps"] += int(dbg.get("ctrl_chain_clipped", dbg.get("ctrl_clipped", 0)) == 1)
    acc["ctrl_base_clipped_steps"] += int(dbg.get("ctrl_base_clipped", 0) == 1)
    acc["alloc_clip_any_steps"] += int(obs["alloc_clip_any"] == 1)
    acc["pump_fullspeed_any_steps"] += int(obs["pump_fullspeed_any"] == 1)
    acc["pump_backlog_max_values"].append(obs["pump_backlog_max_kg"])
    acc["pump_near_target_sum"] += float(obs.get("pump_near_target_mean_s", 0.0))
    acc["pump_quiet_hold_sum"] += float(obs.get("pump_global_quiet_s", 0.0))
    acc["pump_quiet_stop_block_count"] = int(
        obs.get("pump_quiet_stop_block_count", acc.get("pump_quiet_stop_block_count", 0))
    )
    acc["pump_latch_switch_count"] = int(
        obs.get("pump_latch_switch_count", acc.get("pump_latch_switch_count", 0))
    )
    acc["sign_warn_steps"] += int(dbg.get("ctrl_sign_warn", 0) == 1)
    acc["ctrl_update_steps"] += int(dbg.get("ctrl_status") == "update")

    acc["target_mode"] = str(obs.get("target_mode", acc.get("target_mode", "g1")))
    acc["target_table_path"] = str(obs.get("target_table_path", acc.get("target_table_path", "")))
    acc["target_lookup_key"] = str(obs.get("target_lookup_key", acc.get("target_lookup_key", "state_id")))
    acc["target_lookup_total"] = int(obs.get("target_lookup_total", acc.get("target_lookup_total", 0)))
    acc["target_lookup_hit"] = int(obs.get("target_lookup_hit", acc.get("target_lookup_hit", 0)))
    acc["target_lookup_fallback"] = int(obs.get("target_lookup_fallback", acc.get("target_lookup_fallback", 0)))
    acc["fallback_reason_missing_state"] = int(obs.get("fallback_reason_missing_state", acc.get("fallback_reason_missing_state", 0)))
    acc["fallback_reason_missing_table"] = int(obs.get("fallback_reason_missing_table", acc.get("fallback_reason_missing_table", 0)))
    acc["fallback_reason_missing_columns"] = int(obs.get("fallback_reason_missing_columns", acc.get("fallback_reason_missing_columns", 0)))
    acc["fallback_reason_invalid_state_id"] = int(obs.get("fallback_reason_invalid_state_id", acc.get("fallback_reason_invalid_state_id", 0)))
    safety_active = int(obs.get("preview_primary_safety_active", 0))
    safety_fallback = int(obs.get("preview_primary_safety_fallback", 0))
    acc["preview_primary_safety_active_steps"] += safety_active
    acc["preview_primary_safety_fallback_steps"] += safety_fallback
    if safety_active != int(acc.get("preview_primary_safety_prev_active", 0)):
        acc["preview_primary_safety_transition_count"] += 1
    acc["preview_primary_safety_prev_active"] = safety_active

    if t_sim >= acc["tail_start_s"]:
        acc["tail_steps"] += 1
        acc["tail_pitch_sum"] += obs["pitch_deg"]
        acc["tail_roll_sum"] += obs["roll_deg"]
        acc["tail_total_rate_sum"] += obs["total_rate"]
        acc["tail_trim_scale_sum"] += obs["trim_scale"]
        acc["tail_trim_pressure_sum"] += obs["trim_pressure_recent"]
        acc["tail_pump_target_motion_values"].append(float(obs.get("pump_target_motion_kg_s", 0.0)))
        acc["tail_pump_global_backlog_values"].append(float(obs.get("pump_total_backlog_kg", 0.0)))

    for k in range(3):
        acc["total_switches"] += int(obs["act"][k] != acc["prev_active"][k])
        acc["prev_active"][k] = obs["act"][k]


def _finalize_case_summary(
    case_name,
    n_steps,
    dt,
    acc,
    limits,
    target_shape_cfg,
    trim_governor,
    heave_balancer,
    heave_init_info,
    control_enabled,
    has_wind,
    wind_seed,
    wind_update_interval_s,
    wind_mean_lpf_tau_s,
    rate_limit_enabled,
    protocol_meta,
):
    pitch_rms = float(np.sqrt(acc["pitch_sq_sum"] / n_steps))
    roll_rms = float(np.sqrt(acc["roll_sq_sum"] / n_steps))
    pitch_exceed_ratio = acc["pitch_exceed_steps"] / n_steps
    roll_exceed_ratio = acc["roll_exceed_steps"] / n_steps
    sat_ratio_any = acc["sat_any_steps"] / n_steps
    ctrl_clip_ratio = acc["ctrl_clipped_steps"] / n_steps
    ctrl_base_clip_ratio = acc["ctrl_base_clipped_steps"] / n_steps
    alloc_clip_any_ratio = acc["alloc_clip_any_steps"] / n_steps
    switch_per_min = acc["total_switches"] / ((n_steps * dt) / 60.0)
    mean_pitch_sp = acc["sp_pitch_sum"] / n_steps
    mean_roll_sp = acc["sp_roll_sum"] / n_steps
    mean_trim_scale = acc["trim_scale_sum"] / n_steps
    mean_trim_pressure = acc["trim_pressure_sum"] / n_steps
    trim_gate_ratio = acc["trim_gate_steps"] / n_steps
    trim_enabled_ratio = acc["trim_enabled_steps"] / n_steps
    trim_freeze_ratio = acc["trim_freeze_steps"] / n_steps
    trim_pressure_clip_ratio = acc["trim_pressure_clip_steps"] / n_steps
    trim_pressure_fullspeed_ratio = acc["trim_pressure_fullspeed_steps"] / n_steps
    cmd_gap_mean = acc["cmd_gap_sum"] / n_steps
    cmd_limited_ratio = acc["cmd_limited_steps"] / n_steps
    preview_primary_safety_active_ratio = (
        acc.get("preview_primary_safety_active_steps", 0) / n_steps
    )
    preview_primary_safety_fallback_ratio = (
        acc.get("preview_primary_safety_fallback_steps", 0) / n_steps
    )
    p95_cmd_gap = float(np.percentile(np.array(acc["cmd_gap_values"]), 95))
    pump_fullspeed_any_ratio = acc["pump_fullspeed_any_steps"] / n_steps

    if len(acc["pump_backlog_max_values"]) > 0:
        pump_backlog_max_p95_kg = float(
            np.percentile(np.asarray(acc["pump_backlog_max_values"]), 95)
        )
    else:
        pump_backlog_max_p95_kg = 0.0
    if len(acc["tail_pump_target_motion_values"]) > 0:
        pump_target_motion_p95_kg_s = float(
            np.percentile(np.asarray(acc["tail_pump_target_motion_values"]), 95)
        )
    else:
        pump_target_motion_p95_kg_s = 0.0
    if len(acc["tail_pump_global_backlog_values"]) > 0:
        pump_global_backlog_p95_kg = float(
            np.percentile(np.asarray(acc["tail_pump_global_backlog_values"]), 95)
        )
    else:
        pump_global_backlog_p95_kg = 0.0

    if acc["tail_steps"] > 0:
        tail_mean_pitch_deg = float(acc["tail_pitch_sum"] / acc["tail_steps"])
        tail_mean_roll_deg = float(acc["tail_roll_sum"] / acc["tail_steps"])
        tail_mean_total_rate = float(acc["tail_total_rate_sum"] / acc["tail_steps"])
        tail_mean_trim_scale = float(acc["tail_trim_scale_sum"] / acc["tail_steps"])
        tail_mean_trim_pressure = float(acc["tail_trim_pressure_sum"] / acc["tail_steps"])
    else:
        tail_mean_pitch_deg = np.nan
        tail_mean_roll_deg = np.nan
        tail_mean_total_rate = np.nan
        tail_mean_trim_scale = np.nan
        tail_mean_trim_pressure = np.nan

    max_consecutive_sat_s = float(acc["sat_streak_max"] * dt)
    sign_warn_ratio = acc["sign_warn_steps"] / n_steps
    hm_mean_abs_heave_error_m = acc["hm_heave_error_abs_sum"] / n_steps
    hm_mean_abs_correction_per_tank_kg = acc["hm_corr_abs_sum"] / n_steps
    hm_active_ratio = acc["hm_active_steps"] / n_steps
    pump_near_target_mean_s = acc["pump_near_target_sum"] / n_steps
    pump_quiet_hold_mean_s = acc["pump_quiet_hold_sum"] / n_steps

    passed = (
        acc["pitch_abs_max"] <= limits["pitch_deg"]
        and acc["roll_abs_max"] <= limits["roll_deg"]
        and sat_ratio_any <= limits["sat_ratio_any_max"]
        and switch_per_min <= limits["switch_per_min_max"]
    )

    print(
        f"   Result {case_name}: pitch_rms={pitch_rms:.3f}, roll_rms={roll_rms:.3f}, "
        f"pitch_max={acc['pitch_abs_max']:.3f}, roll_max={acc['roll_abs_max']:.3f}, "
        f"sat_any={sat_ratio_any:.3f}, clip={ctrl_clip_ratio:.3f}, switch/min={switch_per_min:.2f}, "
        f"sp_mean=({mean_pitch_sp:.3f},{mean_roll_sp:.3f}), trim_scale={mean_trim_scale:.3f}, "
        f"cmd_gap={cmd_gap_mean:.1f}, p95_gap={p95_cmd_gap:.1f}, max_sat={max_consecutive_sat_s:.1f}s, "
        f"hm_err={hm_mean_abs_heave_error_m:.4f}m, hm_corr={hm_mean_abs_correction_per_tank_kg:.2f}kg, "
        f"sign_warn={sign_warn_ratio:.3f}"
    )
    print(
        f"   Diag {case_name}: trim_gate_ratio={trim_gate_ratio:.3f}, "
        f"trim_enabled_ratio={trim_enabled_ratio:.3f}, "
        f"trim_freeze_ratio={trim_freeze_ratio:.3f}, "
        f"trim_pressure={mean_trim_pressure:.3f}, "
        f"cmd_limited_ratio={cmd_limited_ratio:.3f}, "
        f"pump_target_motion_p95={pump_target_motion_p95_kg_s:.1f}kg/s, "
        f"pump_global_backlog_p95={pump_global_backlog_p95_kg:.1f}kg, "
        f"pump_quiet_blocks={int(acc['pump_quiet_stop_block_count'])}, "
        f"pump_latch_switches={int(acc['pump_latch_switch_count'])}, "
        f"primary_safety_fallback={preview_primary_safety_fallback_ratio:.3f}, "
        f"rate_limit_enabled={int(rate_limit_enabled)}, "
        f"authority={target_shape_cfg.get('actuator_authority', 'plant')}, "
        f"target_mode={acc.get('target_mode', 'g1')}, "
        f"lookup={int(acc.get('target_lookup_hit', 0))}/{int(acc.get('target_lookup_total', 0))}"
    )
    print(
        f"   Protocol {case_name}: profile={protocol_meta.get('name', 'custom')}, "
        f"trim_map={protocol_meta.get('trim_map_version', '') or 'n/a'}, "
        f"target_mode={acc.get('target_mode', 'g1')}, "
        f"target_key={acc.get('target_lookup_key', 'state_id')}, "
        f"ff=({protocol_meta.get('ff_pitch', np.nan):.1f},{protocol_meta.get('ff_roll', np.nan):.1f}), "
        f"ff_disabled={int(protocol_meta.get('ff_disabled', 0))}, "
        f"heave_enabled_cfg={int(protocol_meta.get('heave_enabled_cfg', 0))}, "
        f"shaper_enabled={int(protocol_meta.get('shaper_enabled', 0))}, "
        f"alpha={protocol_meta.get('shaper_alpha', np.nan):.3f}, "
        f"rate={protocol_meta.get('shaper_rate_deg_s', np.nan):.3f}"
    )
    print("[PASS] Closed-loop safety within limits." if passed else "[WARN] Closed-loop exceeded limits.")

    summary = {
        "case": case_name,
        "delta_ton": np.nan,
        "n_steps": n_steps,
        "dt": dt,
        "reach_time_s": np.nan,
        "final_err_kg": np.nan,
        "switches": int(acc["total_switches"]),
        "tv_m3min": np.nan,
        "sat_ratio": float(sat_ratio_any),
        "pitch_rms_deg": pitch_rms,
        "roll_rms_deg": roll_rms,
        "pitch_abs_max_deg": float(acc["pitch_abs_max"]),
        "roll_abs_max_deg": float(acc["roll_abs_max"]),
        "pitch_exceed_ratio": float(pitch_exceed_ratio),
        "roll_exceed_ratio": float(roll_exceed_ratio),
        "ctrl_clip_ratio": float(ctrl_clip_ratio),
        "ctrl_base_clip_ratio": float(ctrl_base_clip_ratio),
        "alloc_clip_any_ratio": float(alloc_clip_any_ratio),
        "sign_warn_ratio": float(sign_warn_ratio),
        "ctrl_update_steps": int(acc["ctrl_update_steps"]),
        "switch_per_min": float(switch_per_min),
        "trim_on": int(trim_governor is not None),
        "mean_pitch_sp_deg": float(mean_pitch_sp),
        "mean_roll_sp_deg": float(mean_roll_sp),
        "mean_trim_scale": float(mean_trim_scale),
        "mean_trim_pressure": float(mean_trim_pressure),
        "trim_gate_ratio": float(trim_gate_ratio),
        "trim_enabled_ratio": float(trim_enabled_ratio),
        "trim_freeze_ratio": float(trim_freeze_ratio),
        "trim_pressure_clip_ratio": float(trim_pressure_clip_ratio),
        "trim_pressure_fullspeed_ratio": float(trim_pressure_fullspeed_ratio),
        "cmd_gap_mean_kg": float(cmd_gap_mean),
        "cmd_gap_p95_kg": p95_cmd_gap,
        "pump_fullspeed_any_ratio": float(pump_fullspeed_any_ratio),
        "pump_backlog_max_p95_kg": float(pump_backlog_max_p95_kg),
        "pump_target_motion_p95_kg_s": float(pump_target_motion_p95_kg_s),
        "pump_global_backlog_p95_kg": float(pump_global_backlog_p95_kg),
        "pump_quiet_stop_block_count": int(acc["pump_quiet_stop_block_count"]),
        "pump_active_latch_switch_count": int(acc["pump_latch_switch_count"]),
        "pump_near_target_mean_s": float(pump_near_target_mean_s),
        "pump_quiet_hold_mean_s": float(pump_quiet_hold_mean_s),
        "tail_window_start_s": float(acc["tail_start_s"]),
        "tail_window_end_s": float(n_steps * dt),
        "tail_mean_pitch_deg": float(tail_mean_pitch_deg),
        "tail_mean_roll_deg": float(tail_mean_roll_deg),
        "tail_mean_total_rate": float(tail_mean_total_rate),
        "tail_mean_trim_scale": float(tail_mean_trim_scale),
        "tail_mean_trim_pressure": float(tail_mean_trim_pressure),
        "cmd_limited_ratio": float(cmd_limited_ratio),
        "preview_primary_safety_active_ratio": float(preview_primary_safety_active_ratio),
        "preview_primary_safety_fallback_ratio": float(preview_primary_safety_fallback_ratio),
        "preview_primary_safety_transition_count": int(
            acc.get("preview_primary_safety_transition_count", 0)
        ),
        "max_consecutive_sat_s": max_consecutive_sat_s,
        "hm_active_ratio": float(hm_active_ratio),
        "hm_mean_abs_heave_error_m": float(hm_mean_abs_heave_error_m),
        "hm_mean_abs_correction_per_tank_kg": float(hm_mean_abs_correction_per_tank_kg),
        "heave_init_applied": int(heave_init_info["ok"]),
        "heave_init_z_eq_m": float(heave_init_info["z_eq_m"]),
        "heave_init_a_wp_m2": float(heave_init_info["a_waterplane_m2"]),
        "rate_limit_m3_min": float(target_shape_cfg.get("rate_limit_m3_min", np.nan)),
        "rate_limit_enabled": int(rate_limit_enabled),
        "actuator_authority": str(target_shape_cfg.get("actuator_authority", "plant")),
        "heave_bias_on": int(heave_balancer is not None),
        "control_on": int(control_enabled),
        "wind_on": int(has_wind),
        "wind_seed": wind_seed,
        "wind_update_interval_s": float(wind_update_interval_s),
        "wind_mean_lpf_tau_s": float(wind_mean_lpf_tau_s),
        "experiment_protocol": str(protocol_meta.get("name", "custom")),
        "protocol_trim_map_version": str(protocol_meta.get("trim_map_version", "")),
        "protocol_target_mode_cfg": str(protocol_meta.get("target_mode", "g1")),
        "protocol_target_table_path_cfg": str(protocol_meta.get("target_table_path", "")),
        "protocol_target_lookup_key_cfg": str(protocol_meta.get("target_lookup_key", "state_id")),
        "protocol_target_table_strict_cfg": int(protocol_meta.get("target_table_strict", 0)),
        "target_mode": str(acc.get("target_mode", "g1")),
        "target_table_path": str(acc.get("target_table_path", "")),
        "target_lookup_key": str(acc.get("target_lookup_key", "state_id")),
        "target_lookup_total": int(acc.get("target_lookup_total", 0)),
        "target_lookup_hit": int(acc.get("target_lookup_hit", 0)),
        "target_lookup_fallback": int(acc.get("target_lookup_fallback", 0)),
        "fallback_reason_missing_state": int(acc.get("fallback_reason_missing_state", 0)),
        "fallback_reason_missing_table": int(acc.get("fallback_reason_missing_table", 0)),
        "fallback_reason_missing_columns": int(acc.get("fallback_reason_missing_columns", 0)),
        "fallback_reason_invalid_state_id": int(acc.get("fallback_reason_invalid_state_id", 0)),
        "protocol_ff_pitch": float(protocol_meta.get("ff_pitch", np.nan)),
        "protocol_ff_roll": float(protocol_meta.get("ff_roll", np.nan)),
        "protocol_ff_disabled": int(protocol_meta.get("ff_disabled", 0)),
        "protocol_heave_enabled_cfg": int(protocol_meta.get("heave_enabled_cfg", 0)),
        "protocol_heave_disabled_cfg": int(protocol_meta.get("heave_disabled_cfg", 0)),
        "protocol_shaper_enabled": int(protocol_meta.get("shaper_enabled", 0)),
        "protocol_shaper_alpha": float(protocol_meta.get("shaper_alpha", np.nan)),
        "protocol_shaper_rate_deg_s": float(protocol_meta.get("shaper_rate_deg_s", np.nan)),
        "protocol_primary_safety_fallback_enabled": int(
            protocol_meta.get("primary_safety_fallback_enabled", 0)
        ),
        "protocol_primary_safety_pitch_enter_deg": float(
            protocol_meta.get("primary_safety_pitch_enter_deg", np.nan)
        ),
        "protocol_primary_safety_roll_enter_deg": float(
            protocol_meta.get("primary_safety_roll_enter_deg", np.nan)
        ),
        "protocol_primary_safety_pitch_exit_deg": float(
            protocol_meta.get("primary_safety_pitch_exit_deg", np.nan)
        ),
        "protocol_primary_safety_roll_exit_deg": float(
            protocol_meta.get("primary_safety_roll_exit_deg", np.nan)
        ),
        "protocol_primary_safety_use_envelope": int(
            protocol_meta.get("primary_safety_use_envelope", 0)
        ),
        "protocol_primary_safety_envelope_enter_norm": float(
            protocol_meta.get("primary_safety_envelope_enter_norm", np.nan)
        ),
        "protocol_primary_safety_enter_hold_s": float(
            protocol_meta.get("primary_safety_enter_hold_s", np.nan)
        ),
        "protocol_primary_safety_emergency_pitch_enter_deg": float(
            protocol_meta.get("primary_safety_emergency_pitch_enter_deg", np.nan)
        ),
        "protocol_primary_safety_emergency_roll_enter_deg": float(
            protocol_meta.get("primary_safety_emergency_roll_enter_deg", np.nan)
        ),
        "protocol_primary_safety_exit_hold_s": float(
            protocol_meta.get("primary_safety_exit_hold_s", np.nan)
        ),
        "protocol_primary_safety_exit_required_windows": int(
            protocol_meta.get("primary_safety_exit_required_windows", 1)
        ),
        "pass": int(passed),
    }
    return summary


def run_closed_loop_case(
    excel_path,
    case_name,
    dt=0.1,
    n_steps=3000,
    wind_gen=None,
    wind_trace=None,
    wind_update_interval_s=3600.0,
    wind_mean_lpf_tau_s=120.0,
    wind_target_ws=11.5,
    wind_target_wd=0.0,
    trim_cfg=None,
    heave_cfg=None,
    controller_cfg=None,
    target_shape_cfg=None,
    target_shape_override_cfg=None,
    pump_cfg=None,
    platform_profile=None,
    platform_cfg=None,
    experiment_protocol=None,
    control_enabled=True,
    record_timeseries=False,
    start_from_heave_equilibrium=True,
    preview_trim_provider=None,
):
    (
        controller_cfg_eff,
        heave_cfg_eff,
        target_shape_cfg_eff,
        protocol_name,
    ) = apply_experiment_protocol(
        experiment_protocol=experiment_protocol,
        controller_cfg=controller_cfg,
        heave_cfg=heave_cfg,
        target_shape_cfg=target_shape_cfg,
    )
    if target_shape_override_cfg:
        target_shape_cfg_eff = clone_cfg(target_shape_cfg_eff)
        target_shape_cfg_eff.update(target_shape_override_cfg)
    protocol_meta = _build_protocol_meta(
        protocol_name=protocol_name,
        controller_cfg=controller_cfg_eff,
        heave_cfg=heave_cfg_eff,
        target_shape_cfg=target_shape_cfg_eff,
        trim_cfg=trim_cfg,
    )

    components = _build_case_components(
        excel_path=excel_path,
        case_name=case_name,
        dt=dt,
        n_steps=n_steps,
        wind_gen=wind_gen,
        wind_trace=wind_trace,
        wind_update_interval_s=wind_update_interval_s,
        wind_mean_lpf_tau_s=wind_mean_lpf_tau_s,
        wind_target_ws=wind_target_ws,
        wind_target_wd=wind_target_wd,
        trim_cfg=trim_cfg,
        heave_cfg=heave_cfg_eff,
        controller_cfg=controller_cfg_eff,
        target_shape_cfg=target_shape_cfg_eff,
        pump_cfg=pump_cfg,
        platform_profile=platform_profile,
        platform_cfg=platform_cfg,
        control_enabled=control_enabled,
        start_from_heave_equilibrium=start_from_heave_equilibrium,
        preview_trim_provider=preview_trim_provider,
    )

    plant = components["plant"]
    ctrl = components["ctrl"]
    limits = components["limits"]
    has_wind = components["has_wind"]
    wind_source = components["wind_source"]
    wind_seed = components["wind_seed"]
    target_shape_cfg = components["target_shape_cfg"]
    trim_governor = components["trim_governor"]
    heave_balancer = components["heave_balancer"]
    rate_limit_enabled = components["rate_limit_enabled"]
    policy = components["policy"]
    heave_init_info = components["heave_init_info"]

    acc = {
        "pitch_sq_sum": 0.0,
        "roll_sq_sum": 0.0,
        "pitch_abs_max": 0.0,
        "roll_abs_max": 0.0,
        "pitch_exceed_steps": 0,
        "roll_exceed_steps": 0,
        "sat_any_steps": 0,
        "ctrl_clipped_steps": 0,
        "ctrl_base_clipped_steps": 0,
        "alloc_clip_any_steps": 0,
        "ctrl_update_steps": 0,
        "total_switches": 0,
        "sp_pitch_sum": 0.0,
        "sp_roll_sum": 0.0,
        "trim_scale_sum": 0.0,
        "trim_pressure_sum": 0.0,
        "trim_gate_steps": 0,
        "trim_enabled_steps": 0,
        "trim_freeze_steps": 0,
        "trim_pressure_clip_steps": 0,
        "trim_pressure_fullspeed_steps": 0,
        "cmd_gap_sum": 0.0,
        "cmd_limited_steps": 0,
        "cmd_gap_values": [],
        "sign_warn_steps": 0,
        "hm_heave_error_abs_sum": 0.0,
        "hm_corr_abs_sum": 0.0,
        "hm_active_steps": 0,
        "prev_active": [False, False, False],
        "sat_streak": 0,
        "sat_streak_max": 0,
        "pump_fullspeed_any_steps": 0,
        "pump_backlog_max_values": [],
        "tail_pump_target_motion_values": [],
        "tail_pump_global_backlog_values": [],
        "pump_near_target_sum": 0.0,
        "pump_quiet_hold_sum": 0.0,
        "pump_quiet_stop_block_count": 0,
        "pump_latch_switch_count": 0,
        "tail_start_s": min(1000.0, max(0.0, n_steps * dt - 500.0)),
        "tail_steps": 0,
        "tail_pitch_sum": 0.0,
        "tail_roll_sum": 0.0,
        "tail_total_rate_sum": 0.0,
        "tail_trim_scale_sum": 0.0,
        "tail_trim_pressure_sum": 0.0,
        "target_mode": "g1",
        "target_table_path": "",
        "target_lookup_key": "state_id",
        "target_lookup_total": 0,
        "target_lookup_hit": 0,
        "target_lookup_fallback": 0,
        "fallback_reason_missing_state": 0,
        "fallback_reason_missing_table": 0,
        "fallback_reason_missing_columns": 0,
        "fallback_reason_invalid_state_id": 0,
        "preview_primary_safety_fallback_steps": 0,
        "preview_primary_safety_active_steps": 0,
        "preview_primary_safety_transition_count": 0,
        "preview_primary_safety_prev_active": 0,
    }

    target_cmd = plant.current_ballast_mass.copy()
    last_info = None
    timeseries = []

    for i in range(n_steps):
        t_sim = i * dt
        if not has_wind:
            wind_obs = {"ws": 0.0, "wd_deg": 0.0, "state_id": None, "ws_state": None, "wd_state": None}
            thrust_n = 0.0
        else:
            wind_obs = wind_source.step()
            thrust_n = float(wind_obs["thrust_n"])

        if control_enabled:
            policy_info_prev = {} if last_info is None else dict(last_info)
            policy_info_prev.setdefault("tank_masses", plant.current_ballast_mass.copy())
            m_cmds_applied, dbg = policy.compute(plant.state, wind_obs, policy_info_prev, t_sim)
        else:
            m_cmds_applied = target_cmd.copy()
            dbg = _build_open_loop_debug(plant, ctrl)

        ctrl_diag = ctrl.last_debug_info if isinstance(ctrl.last_debug_info, dict) else {}
        cmd_gap = float(dbg.get("cmd_gap_kg", 0.0))

        plant.set_ballast_target(*m_cmds_applied)
        _, info = plant.step(thrust_n, wind_obs["wd_deg"], dt, t_sim)

        # Contract: plant feedback at step k is consumed by controller at step k+1.
        info_for_policy = info.copy()
        info_for_policy["cmd_gap_kg"] = cmd_gap
        info_for_policy["ctrl_chain_clipped"] = int(
            dbg.get("ctrl_chain_clipped", dbg.get("ctrl_clipped", 0))
        )
        info_for_policy["ctrl_base_clipped"] = int(dbg.get("ctrl_base_clipped", 0))
        info_for_policy["alloc_clip_any"] = int(dbg.get("alloc_clip_any", 0))
        info_for_policy["post_chain_adjusted"] = int(dbg.get("post_chain_adjusted", 0))
        last_info = info_for_policy

        obs = _extract_step_obs(plant=plant, info=info, dbg=dbg, ctrl_diag=ctrl_diag, limits=limits, wind_obs=wind_obs)
        _update_case_accumulators(acc=acc, obs=obs, dbg=dbg, t_sim=t_sim)

        if record_timeseries:
            _append_timeseries_row(
                timeseries=timeseries,
                t_sim=t_sim,
                wind_obs=wind_obs,
                thrust_n=thrust_n,
                plant=plant,
                dbg=dbg,
                info=info,
                obs=obs,
                control_enabled=control_enabled,
            )

    summary = _finalize_case_summary(
        case_name=case_name,
        n_steps=n_steps,
        dt=dt,
        acc=acc,
        limits=limits,
        target_shape_cfg=target_shape_cfg,
        trim_governor=trim_governor,
        heave_balancer=heave_balancer,
        heave_init_info=heave_init_info,
        control_enabled=control_enabled,
        has_wind=has_wind,
        wind_seed=wind_seed,
        wind_update_interval_s=wind_update_interval_s,
        wind_mean_lpf_tau_s=wind_mean_lpf_tau_s,
        rate_limit_enabled=rate_limit_enabled,
        protocol_meta=protocol_meta,
    )
    return summary, timeseries

def write_timeseries_csv(case_name, rows):
    if not rows:
        return
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"timeseries_{case_name}_{ts}.csv"
    fields = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved time-series: {out_path}")


def write_results_csv(rows):
    if not rows:
        return
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"validation_{ts}.csv"
    preferred_fields = [
        "case",
        "delta_ton",
        "n_steps",
        "dt",
        "reach_time_s",
        "final_err_kg",
        "switches",
        "tv_m3min",
        "sat_ratio",
        "pitch_rms_deg",
        "roll_rms_deg",
        "pitch_abs_max_deg",
        "roll_abs_max_deg",
        "wind_on",
        "pass",
    ]
    extra_fields = []
    keyset = set()
    for r in rows:
        keyset.update(r.keys())
    for k in sorted(keyset):
        if k not in preferred_fields:
            extra_fields.append(k)
    fields = preferred_fields + extra_fields
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved metrics: {out_path}")


def write_generic_csv(rows, filename_prefix):
    if not rows:
        return None
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"{filename_prefix}_{ts}.csv"
    fields = []
    keyset = set()
    for r in rows:
        keyset.update(r.keys())
    fields = sorted(list(keyset))
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {filename_prefix}: {out_path}")
    return out_path


def _z_norm(values):
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - mean) / std


def run_feedforward_gain_sweep(
    excel_path,
    wind,
    dt=0.1,
    n_steps=3000,
    ws_mps=11.5,
    k_candidates=(0.0, 30.0, 60.0, 90.0, 120.0),
    wind_dirs=(0.0, 45.0, 90.0, 135.0, 180.0, 270.0),
):
    print("\n=== Test G: Feedforward Gain Sweep (sync pitch/roll) ===")
    scan_rows = []
    for k in k_candidates:
        print(f"   [G] scanning k_ff={k:.1f}")
        for wd in wind_dirs:
            case_name = f"G_ffk{int(k)}_wd{int(wd)}"
            trace = build_constant_wind_trace(
                n_steps=n_steps, ws_mps=ws_mps, wd_deg=wd, dt=dt, seed=RANDOM_SEED, wind_gen=wind
            )
            row, _ = run_closed_loop_case(
                excel_path=excel_path,
                case_name=case_name,
                dt=dt,
                n_steps=n_steps,
                wind_gen=wind,
                wind_trace=trace,
                trim_cfg=None,
                controller_cfg={
                    "k_wind_comp_pitch": float(k),
                    "k_wind_comp_roll": float(k),
                },
                control_enabled=True,
                record_timeseries=False,
            )
            row["group"] = "ff_scan_case"
            row["ff_gain_pitch"] = float(k)
            row["ff_gain_roll"] = float(k)
            row["test_wind_dir_deg"] = float(wd)
            scan_rows.append(row)

    # Aggregate by k and build compromise score from key metrics.
    summary_rows = []
    for k in k_candidates:
        subset = [r for r in scan_rows if abs(float(r["ff_gain_pitch"]) - float(k)) < 1e-9]
        if not subset:
            continue
        mean_pitch = float(np.mean([float(r["pitch_rms_deg"]) for r in subset]))
        mean_roll = float(np.mean([float(r["roll_rms_deg"]) for r in subset]))
        mean_sat = float(np.mean([float(r["sat_ratio"]) for r in subset]))
        mean_gap95 = float(np.mean([float(r["cmd_gap_p95_kg"]) for r in subset]))
        mean_switch = float(np.mean([float(r["switch_per_min"]) for r in subset]))
        pass_ratio = float(np.mean([float(r["pass"]) for r in subset]))
        summary_rows.append(
            {
                "group": "ff_scan_summary",
                "ff_gain_pitch": float(k),
                "ff_gain_roll": float(k),
                "mean_pitch_rms_deg": mean_pitch,
                "mean_roll_rms_deg": mean_roll,
                "mean_sat_ratio": mean_sat,
                "mean_cmd_gap_p95_kg": mean_gap95,
                "mean_switch_per_min": mean_switch,
                "pass_ratio": pass_ratio,
            }
        )

    if summary_rows:
        z_pitch = _z_norm([r["mean_pitch_rms_deg"] for r in summary_rows])
        z_roll = _z_norm([r["mean_roll_rms_deg"] for r in summary_rows])
        z_sat = _z_norm([r["mean_sat_ratio"] for r in summary_rows])
        z_gap = _z_norm([r["mean_cmd_gap_p95_kg"] for r in summary_rows])
        for i, r in enumerate(summary_rows):
            # Balanced compromise score across control effect + actuator pressure.
            r["compromise_score"] = float(z_pitch[i] + z_roll[i] + z_sat[i] + z_gap[i])

        best = min(summary_rows, key=lambda x: x["compromise_score"])
        best_k = float(best["ff_gain_pitch"])
        print(
            "   [G] Best compromise k="
            f"{best_k:.1f} | score={best['compromise_score']:.3f} | "
            f"mean_pitch={best['mean_pitch_rms_deg']:.3f}, "
            f"mean_roll={best['mean_roll_rms_deg']:.3f}, "
            f"mean_sat={best['mean_sat_ratio']:.3f}, "
            f"mean_gap95={best['mean_cmd_gap_p95_kg']:.1f}"
        )

        wd135 = next(
            (
                r
                for r in scan_rows
                if abs(float(r["ff_gain_pitch"]) - best_k) < 1e-9
                and abs(float(r["test_wind_dir_deg"]) - 135.0) < 1e-9
            ),
            None,
        )
        if wd135 is not None:
            p_rms = float(wd135["pitch_rms_deg"])
            r_rms = float(wd135["roll_rms_deg"])
            sat = float(wd135["sat_ratio"])
            if p_rms < 1.5 and r_rms < 1.5 and sat > 0.85:
                diag = "maintenance_saturation_suspected_consider_lower_k"
            elif (p_rms > 2.0 or r_rms > 2.0) and sat > 0.85:
                diag = "control_insufficient_consider_static_axis_weighting"
            else:
                diag = "balanced_no_weighting_needed"
            summary_rows.append(
                {
                    "group": "ff_scan_wd135_diagnosis",
                    "ff_gain_pitch": best_k,
                    "ff_gain_roll": best_k,
                    "mean_pitch_rms_deg": np.nan,
                    "mean_roll_rms_deg": np.nan,
                    "mean_sat_ratio": np.nan,
                    "mean_cmd_gap_p95_kg": np.nan,
                    "mean_switch_per_min": np.nan,
                    "pass_ratio": np.nan,
                    "compromise_score": float(best["compromise_score"]),
                    "wd135_pitch_rms_deg": p_rms,
                    "wd135_roll_rms_deg": r_rms,
                    "wd135_sat_ratio": sat,
                    "wd135_cmd_gap_p95_kg": float(wd135["cmd_gap_p95_kg"]),
                    "wd135_switch_per_min": float(wd135["switch_per_min"]),
                    "wd135_diagnosis": diag,
                }
            )
            print(
                "   [G] wd135 diagnosis @best k: "
                f"pitch_rms={p_rms:.3f}, roll_rms={r_rms:.3f}, sat={sat:.3f} -> {diag}"
            )

    return scan_rows, summary_rows


def run_slowing_ablation_suite(
    excel_path,
    wind,
    wind_trace,
    dt=0.1,
    n_steps=18000,
):
    print("\n=== Test H: Slowing Layer Ablation (trim off, fixed wind trace) ===")
    if wind_trace is None:
        print("[WARN] Test H skipped: missing shared wind trace.")
        return [], []

    # Keep all comparisons on the same wind input and control objective.
    # Apply changes incrementally so each step isolates one layer.
    cfg_base_shape = {
        "actuator_authority": "plant",
        "enable_rate_limit": False,
        "rate_limit_m3_min": 10.0,
        "enable_setpoint_shaping": True,
        "setpoint_alpha": 0.98,
        "setpoint_rate_deg_s": 0.02,
    }
    plan = [
        {
            "case": "H0_ref_no_rate_limit",
            "desc": "reference stack (rate limiter off)",
            "controller_cfg": {},
            "shape_cfg": dict(cfg_base_shape),
        },
        {
            "case": "H1_no_rate_no_sp_shape",
            "desc": "disable setpoint shaper",
            "controller_cfg": {},
            "shape_cfg": {
                **cfg_base_shape,
                "enable_setpoint_shaping": False,
            },
        },
        {
            "case": "H2_plus_fast_update_1s",
            "desc": "H1 + controller update_interval=1s",
            "controller_cfg": {"update_interval": 1.0},
            "shape_cfg": {
                **cfg_base_shape,
                "enable_setpoint_shaping": False,
            },
        },
        {
            "case": "H3_plus_filter_tau_5s",
            "desc": "H2 + filter_tau=5s",
            "controller_cfg": {"update_interval": 1.0, "filter_tau": 5.0},
            "shape_cfg": {
                **cfg_base_shape,
                "enable_setpoint_shaping": False,
            },
        },
    ]

    case_rows = []
    summary_rows = []
    ref_row = None

    for idx, cfg in enumerate(plan):
        print(f"   [H{idx}] {cfg['case']}: {cfg['desc']}")
        row, ts = run_closed_loop_case(
            excel_path=excel_path,
            case_name=cfg["case"],
            dt=dt,
            n_steps=n_steps,
            wind_gen=wind,
            wind_trace=wind_trace,
            trim_cfg=None,
            controller_cfg=cfg["controller_cfg"],
            target_shape_cfg=cfg["shape_cfg"],
            control_enabled=True,
            record_timeseries=True,
        )
        write_timeseries_csv(cfg["case"], ts)
        row["group"] = "slowing_ablation_case"
        row["ablation_step"] = int(idx)
        row["ablation_desc"] = cfg["desc"]
        row["trim_mode"] = "off"
        row["cfg_update_interval_s"] = float(cfg["controller_cfg"].get("update_interval", 5.0))
        row["cfg_filter_tau_s"] = float(cfg["controller_cfg"].get("filter_tau", 10.0))
        row["cfg_rate_limit_enabled"] = int(bool(cfg["shape_cfg"].get("enable_rate_limit", False)))
        row["cfg_rate_limit_m3_min"] = float(cfg["shape_cfg"].get("rate_limit_m3_min", np.nan))
        row["cfg_setpoint_shaping_enabled"] = int(
            bool(cfg["shape_cfg"].get("enable_setpoint_shaping", False))
        )
        row["cfg_setpoint_alpha"] = float(cfg["shape_cfg"].get("setpoint_alpha", np.nan))
        row["cfg_setpoint_rate_deg_s"] = float(cfg["shape_cfg"].get("setpoint_rate_deg_s", np.nan))
        case_rows.append(row)

        if ref_row is None:
            ref_row = row
            continue

        print(
            "      delta vs H0: "
            f"pitch_rms={row['pitch_rms_deg'] - ref_row['pitch_rms_deg']:+.3f}, "
            f"roll_rms={row['roll_rms_deg'] - ref_row['roll_rms_deg']:+.3f}, "
            f"sat={row['sat_ratio'] - ref_row['sat_ratio']:+.3f}, "
            f"switch/min={row['switch_per_min'] - ref_row['switch_per_min']:+.2f}, "
            f"gap95={row['cmd_gap_p95_kg'] - ref_row['cmd_gap_p95_kg']:+.1f}"
        )

    if ref_row is not None:
        for r in case_rows:
            summary_rows.append(
                {
                    "group": "slowing_ablation_summary",
                    "case": r["case"],
                    "ablation_step": int(r["ablation_step"]),
                    "ablation_desc": r["ablation_desc"],
                    "pass": int(r["pass"]),
                    "pitch_rms_deg": float(r["pitch_rms_deg"]),
                    "roll_rms_deg": float(r["roll_rms_deg"]),
                    "sat_ratio": float(r["sat_ratio"]),
                    "switch_per_min": float(r["switch_per_min"]),
                    "cmd_gap_p95_kg": float(r["cmd_gap_p95_kg"]),
                    "delta_pitch_rms_vs_h0": float(r["pitch_rms_deg"] - ref_row["pitch_rms_deg"]),
                    "delta_roll_rms_vs_h0": float(r["roll_rms_deg"] - ref_row["roll_rms_deg"]),
                    "delta_sat_vs_h0": float(r["sat_ratio"] - ref_row["sat_ratio"]),
                    "delta_switch_vs_h0": float(r["switch_per_min"] - ref_row["switch_per_min"]),
                    "delta_gap95_vs_h0": float(r["cmd_gap_p95_kg"] - ref_row["cmd_gap_p95_kg"]),
                    "cfg_update_interval_s": float(r["cfg_update_interval_s"]),
                    "cfg_filter_tau_s": float(r["cfg_filter_tau_s"]),
                    "cfg_rate_limit_enabled": int(r["cfg_rate_limit_enabled"]),
                    "cfg_setpoint_shaping_enabled": int(r["cfg_setpoint_shaping_enabled"]),
                }
            )

    return case_rows, summary_rows


def run_tests():
    print(f">>> [Validation] numpy random seed fixed at {RANDOM_SEED}")
    excel_path = discover_stiffness_file()
    if not excel_path:
        print("!!! Warning: stiffness xlsx not found in ./data. Validation uses linear fallback.")
    wind_matrix_path = discover_wind_matrix_file()

    run_test_a(excel_path)
    rows = []
    sign_rows = run_sign_tests(excel_path)
    rows.extend(sign_rows)
    sign_ok = all(int(r.get("pass", 0)) == 1 for r in sign_rows)
    if not sign_ok:
        print("[WARN] Sign sanity test failed. Closed-loop Test E will be skipped.")

    print("\n=== Test B: Incremental Ballast Pump Dynamics ===")
    print("Case format: Delta, ReachTime[s], FinalErr[kg], SwitchCount, TV[m3/min], SatRatio")
    dt = 0.1
    heave_cfg = {
        "enabled": True,
        "target_heave_m": "auto",
        "kp": 0.005,
        "ki": 0.0,
        "max_correction_per_tank_kg": 2.0,
        "m_min_op_kg": 200000.0,
        "m_max_op_ratio": 0.90,
        "history_window_s": 30.0,
    }
    trim_cfg_e = clone_cfg(TRIM_CFG_TEST_E)
    trim_cfg_f = clone_cfg(TRIM_CFG_TEST_F)
    # Wind Markov transitions use 1h bins (3600s).
    # D uses >=1 transition; E/H use a longer horizon to span multiple 1h intervals.
    wind_eval_steps = 54000  # 5400s at dt=0.1 (1.5h)
    closed_loop_steps = 72000  # 7200s at dt=0.1 (2h)
    step_plan = [
        ("B_no_wind_5t", 5000.0, 3000),
        ("B_no_wind_15t", 15000.0, 3000),
        ("B_no_wind_30t", 30000.0, 4000),
    ]

    for case_name, delta_target, n_steps in step_plan:
        rows.append(run_step_case(excel_path, delta_target, n_steps, dt, case_name, wind_gen=None))

    wind, wind_ok = run_test_c(wind_matrix_path)
    wind_trace_step = None
    wind_trace_closed = None
    if wind is not None and wind_ok:
        wind_builder = WindEnvMarkov(
            wind_gen=wind,
            dt=dt,
            update_interval_s=3600.0,
            lpf_tau_s=120.0,
            seed=RANDOM_SEED,
        )
        wind_trace_step = wind_builder.generate_trace(n_steps=wind_eval_steps, seed=RANDOM_SEED)
        wind_trace_closed = wind_builder.generate_trace(n_steps=closed_loop_steps, seed=RANDOM_SEED)
        print(
            ">>> [Validation] Shared wind traces prepared: "
            f"D_steps={wind_trace_step['n_steps']}, E_steps={wind_trace_closed['n_steps']}, "
            f"seed={wind_trace_step['seed']}"
        )

    print("\n=== Test D: Wind On vs Off Comparison (+15t) ===")
    off_row = run_step_case(excel_path, 15000.0, wind_eval_steps, dt, "D_off_15t", wind_gen=None)
    rows.append(off_row)
    if wind is not None and wind_ok:
        on_row = run_step_case(
            excel_path,
            15000.0,
            wind_eval_steps,
            dt,
            "D_on_15t",
            wind_gen=wind,
            wind_trace=wind_trace_step,
        )
        rows.append(on_row)
        print(
            "   Comparison (on-off): "
            f"sat_ratio={on_row['sat_ratio'] - off_row['sat_ratio']:+.3f}, "
            f"tv={on_row['tv_m3min'] - off_row['tv_m3min']:+.1f}, "
            f"pitch_rms={on_row['pitch_rms_deg'] - off_row['pitch_rms_deg']:+.3f}, "
            f"roll_rms={on_row['roll_rms_deg'] - off_row['roll_rms_deg']:+.3f}"
        )
    else:
        print("[WARN] Skipped wind-on case due to missing/invalid wind matrix.")

    print("\n=== Test E: Closed-Loop Controller Integration ===")
    if sign_ok:
        e_open_row, e_open_ts = run_closed_loop_case(
            excel_path=excel_path,
            case_name="E_open_on",
            dt=dt,
            n_steps=closed_loop_steps,
            wind_gen=wind if (wind is not None and wind_ok) else None,
            wind_trace=wind_trace_closed if (wind is not None and wind_ok) else None,
            control_enabled=False,
            record_timeseries=True if (wind is not None and wind_ok) else False,
        )
        rows.append(e_open_row)
        e_off_row, _ = run_closed_loop_case(
            excel_path=excel_path,
            case_name="E_closed_off",
            dt=dt,
            n_steps=closed_loop_steps,
            wind_gen=None,
            heave_cfg=heave_cfg,
            record_timeseries=False,
        )
        rows.append(e_off_row)
        if wind is not None and wind_ok:
            e_on_row, e_on_ts = run_closed_loop_case(
                excel_path=excel_path,
                case_name="E_closed_on",
                dt=dt,
                n_steps=closed_loop_steps,
                wind_gen=wind,
                wind_trace=wind_trace_closed,
                heave_cfg=heave_cfg,
                record_timeseries=True,
            )
            rows.append(e_on_row)
            trim_row, trim_ts = run_closed_loop_case(
                excel_path=excel_path,
                case_name="E_closed_trim",
                dt=dt,
                n_steps=closed_loop_steps,
                wind_gen=wind,
                wind_trace=wind_trace_closed,
                trim_cfg=trim_cfg_e,
                heave_cfg=heave_cfg,
                record_timeseries=True,
            )
            rows.append(trim_row)
            print(
                "   Closed-loop on-off delta: "
                f"pitch_rms={e_on_row['pitch_rms_deg'] - e_off_row['pitch_rms_deg']:+.3f}, "
                f"roll_rms={e_on_row['roll_rms_deg'] - e_off_row['roll_rms_deg']:+.3f}, "
                f"clip={e_on_row['ctrl_clip_ratio'] - e_off_row['ctrl_clip_ratio']:+.3f}"
            )
            print(
                "   Closed-loop trim gain: "
                f"pitch_rms={trim_row['pitch_rms_deg'] - e_on_row['pitch_rms_deg']:+.3f}, "
                f"roll_rms={trim_row['roll_rms_deg'] - e_on_row['roll_rms_deg']:+.3f}, "
                f"sat_any={trim_row['sat_ratio'] - e_on_row['sat_ratio']:+.3f}, "
                f"cmd_gap={trim_row['cmd_gap_mean_kg'] - e_on_row['cmd_gap_mean_kg']:+.1f}"
            )
            print(
                "   Open-loop baseline delta (closed_on-open_on): "
                f"pitch_rms={e_on_row['pitch_rms_deg'] - e_open_row['pitch_rms_deg']:+.3f}, "
                f"roll_rms={e_on_row['roll_rms_deg'] - e_open_row['roll_rms_deg']:+.3f}"
            )
            write_timeseries_csv("E_open_on", e_open_ts)
            write_timeseries_csv("E_closed_on", e_on_ts)
            write_timeseries_csv("E_closed_trim", trim_ts)
        else:
            print("[WARN] Skipped closed-loop wind-on case due to missing/invalid wind matrix.")
    else:
        print("[WARN] Closed-loop Test E skipped due to failed sign sanity gate.")

    print("\n=== Test F: Directional Wind Coverage (Open / Closed / Trim) ===")
    # Keep the same trim settings used in Test E for fair comparison.
    directional_plan = [0.0, 45.0, 90.0, 135.0, 180.0, 270.0]
    f_steps = 15000
    f_ws = 15.5
    if wind is not None and wind_ok:
        for wd in directional_plan:
            wd_tag = int(wd)
            const_trace = build_constant_wind_trace(
                n_steps=f_steps,
                ws_mps=f_ws,
                wd_deg=wd,
                dt=dt,
                seed=RANDOM_SEED,
                wind_gen=wind,
            )
            open_case = f"F_open_wd{wd_tag}"
            open_row, open_ts = run_closed_loop_case(
                excel_path=excel_path,
                case_name=open_case,
                dt=dt,
                n_steps=f_steps,
                wind_gen=wind,
                wind_trace=const_trace,
                control_enabled=False,
                trim_cfg=None,
                record_timeseries=True,
            )
            rows.append(open_row)
            write_timeseries_csv(open_case, open_ts)

            on_case = f"F_closed_wd{wd_tag}"
            on_row, on_ts = run_closed_loop_case(
                excel_path=excel_path,
                case_name=on_case,
                dt=dt,
                n_steps=f_steps,
                wind_gen=wind,
                wind_trace=const_trace,
                control_enabled=True,
                trim_cfg=None,
                heave_cfg=heave_cfg,
                record_timeseries=True,
            )
            rows.append(on_row)
            write_timeseries_csv(on_case, on_ts)

            trim_case = f"F_trim_wd{wd_tag}"
            trim_row, trim_ts = run_closed_loop_case(
                excel_path=excel_path,
                case_name=trim_case,
                dt=dt,
                n_steps=f_steps,
                wind_gen=wind,
                wind_trace=const_trace,
                control_enabled=True,
                trim_cfg=trim_cfg_f,
                heave_cfg=heave_cfg,
                record_timeseries=True,
            )
            rows.append(trim_row)
            write_timeseries_csv(trim_case, trim_ts)
    else:
        print("[WARN] Skipped Test F due to missing/invalid wind matrix.")

    print("\n=== Test I: Extreme Wind Heave-Bias Capability ===")
    if wind is not None and wind_ok:
        ws_values = [float(s[0]) for s in wind.next_states] if wind.next_states else []
        if len(ws_values) > 0:
            ws_extreme = float(np.percentile(np.asarray(ws_values, dtype=float), 90.0))
        else:
            ws_extreme = 15.0
        ws_extreme = max(ws_extreme, 12.0)
        # Continuous multi-scenario chain: one run, segmented wind directions.
        # This keeps plant/policy states continuous within the run (no mid-run reset).
        seg_directions = (45.0, 135.0, 270.0)
        seg_duration_s = 1200.0
        seg_steps = int(round(seg_duration_s / dt))
        i_steps = seg_steps * len(seg_directions)
        ws_arr = np.full(i_steps, ws_extreme, dtype=float)
        wd_arr = np.concatenate(
            [np.full(seg_steps, wd, dtype=float) for wd in seg_directions]
        )
        thrust_arr = np.full(i_steps, float(wind_speed_to_thrust_n(ws_extreme)), dtype=float)
        i_trace = {
            "ws": ws_arr,
            "wd": wd_arr,
            "thrust_n": thrust_arr,
            "seed": int(RANDOM_SEED),
            "n_steps": int(i_steps),
            "dt": float(dt),
            "update_interval_s": 0.0,
            "mean_lpf_tau_s": 0.0,
        }

        i_on_case = f"I_chain_closed_ws{ws_extreme:.1f}"
        i_on_row, i_on_ts = run_closed_loop_case(
            excel_path=excel_path,
            case_name=i_on_case,
            dt=dt,
            n_steps=i_steps,
            wind_gen=wind,
            wind_trace=i_trace,
            control_enabled=True,
            trim_cfg=None,
            heave_cfg=heave_cfg,
            record_timeseries=True,
        )
        rows.append(i_on_row)
        write_timeseries_csv(i_on_case, i_on_ts)

        i_trim_case = f"I_chain_trim_ws{ws_extreme:.1f}"
        i_trim_row, i_trim_ts = run_closed_loop_case(
            excel_path=excel_path,
            case_name=i_trim_case,
            dt=dt,
            n_steps=i_steps,
            wind_gen=wind,
            wind_trace=i_trace,
            control_enabled=True,
            trim_cfg=trim_cfg_f,
            heave_cfg=heave_cfg,
            record_timeseries=True,
        )
        rows.append(i_trim_row)
        write_timeseries_csv(i_trim_case, i_trim_ts)
    else:
        print("[WARN] Skipped Test I due to missing/invalid wind matrix.")

    if sign_ok and wind is not None and wind_ok and wind_trace_closed is not None:
        h_rows, h_summary = run_slowing_ablation_suite(
            excel_path=excel_path,
            wind=wind,
            wind_trace=wind_trace_closed,
            dt=dt,
            n_steps=closed_loop_steps,
        )
        rows.extend(h_rows)
        write_generic_csv(h_rows, "slowing_ablation_cases")
        write_generic_csv(h_summary, "slowing_ablation_summary")
    else:
        print("[WARN] Skipped Test H slowing ablation due to missing prerequisites.")

    if wind is not None and wind_ok:
        scan_rows, scan_summary = run_feedforward_gain_sweep(
            excel_path=excel_path,
            wind=wind,
            dt=dt,
            n_steps=3000,
            ws_mps=11.5,
            k_candidates=(0.0, 30.0, 60.0, 90.0, 120.0),
            wind_dirs=(0.0, 45.0, 90.0, 135.0, 180.0, 270.0),
        )
        write_generic_csv(scan_rows, "ff_scan_cases")
        write_generic_csv(scan_summary, "ff_scan_summary")
    else:
        print("[WARN] Skipped Test G feedforward scan due to missing/invalid wind matrix.")

    write_results_csv(rows)



if __name__ == "__main__":
    run_tests()



'''1'''
