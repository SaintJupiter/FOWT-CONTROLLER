from copy import deepcopy


DEFAULT_CONTROLLER_CFG = {
    "kp_p": 7000.0,
    "ki_p": 80.0,
    "kd_p": 0.0,
    "kp_r": 5500.0,
    "ki_r": 70.0,
    "kd_r": 0.0,
    "kp_h": 0.0,
    "ki_h": 0.0,
    "kd_h": 0.0,
    "deadband_pitch": 1.0,
    "deadband_roll": 0.8,
    "deadband_heave": 0.1,
    "deadband_exit_ratio": 0.5,
    "filter_tau": 25.0,
    "update_interval": 10.0,
    "k_wind_comp_pitch": 0.0,
    "k_wind_comp_roll": 0.0,
}

DEFAULT_CONTROLLER_CFG_ENGINEERING_V1 = {
    **DEFAULT_CONTROLLER_CFG,
}


DEFAULT_TRIM_CFG_LEGACY = {
    "k_trim_deg_per_mps": 0.03,
    "max_trim_deg": 1.0,
    "sat_gate_on": 0.95,
    "sat_gate_off": 0.85,
    "cmd_gap_decay_start_kg": 5000.0,
    "cmd_gap_decay_full_kg": 12000.0,
    "sat_window_s": 90.0,
    "cmd_gap_window_s": 180.0,
    "trim_min_scale": 0.0,
    "trim_decay_per_step": 0.995,
    "trim_recover_per_step": 0.05,
    "wind_lpf_tau_s": 60.0,
    "steady_window_s": 45.0,
    "steady_mean_on_deg": 0.15,
    "steady_mean_off_deg": 0.12,
    "steady_std_on_deg": 0.30,
    "steady_std_off_deg": 0.35,
    "trim_update_s": 5.0,
    "trim_logic": "legacy",
    # Explicit trim-map selection and regime config.
    "trim_map_version": "v1",
    "ws_rated_mps": 11.5,
    "ws_cutout_mps": 25.0,
    "post_rated_mode": "plateau",
    "post_rated_decay_ratio": 0.8,
    "protect_trim_scale": 0.0,
    "pressure_backlog_start_kg": 600.0,
    "pressure_backlog_full_kg": 2000.0,
    "simple_decay_on_clip": 0.97,
    "simple_recover_per_step": 0.05,
}


DEFAULT_TRIM_CFG_SIMPLE_DEFAULT = {
    **DEFAULT_TRIM_CFG_LEGACY,
    "k_trim_deg_per_mps": 0.05,
    "max_trim_deg": 1.5,
    "trim_logic": "simple_default",
    "trim_map_version": "v2",
    "post_rated_mode": "plateau",
    "simple_pressure_hold_th": 0.35,
    "simple_pressure_decay_th": 0.70,
    "simple_freeze_enabled": False,
    "simple_pressure_freeze_th": 0.75,
    "simple_pressure_unfreeze_th": 0.55,
}

DEFAULT_TARGET_SHAPE_CFG = {
    "actuator_authority": "plant",
    "enable_rate_limit": True,
    "rate_limit_m3_min": 10.0,
    "enable_setpoint_shaping": True,
    "setpoint_alpha": 0.98,
    "setpoint_rate_deg_s": 0.02,
}


DEFAULT_PUMP_CFG = {
    "pump_stop_err_kg": 300.0,
    "pump_restart_err_kg": 500.0,
    "pump_min_on_s": 20.0,
    "pump_min_off_s": 12.0,
    "pump_hold_before_stop_s": 10.0,
    # Keep quiet-stop dormant in the default F-only pump profile.
    # The target-motion + quiet-hold gate remains available through custom pump CLI
    # for explicit experiments, but it should not silently hold pumps on by default.
    "pump_global_quiet_backlog_kg": float("inf"),
    "pump_target_quiet_rate_kg_s": float("inf"),
    "pump_global_quiet_hold_s": 0.0,
    "pump_ramp_up_m3_min_per_s": 2.0,
    "pump_ramp_down_m3_min_per_s": 3.0,
}


DEFAULT_PLATFORM_PROFILE = "default"

# Baseline platform values remain in core_model.py; profile entries below only
# override the subset needed for low-order reference mapping.
DEFAULT_PLATFORM_PROFILES = {
    "default": {},
    "reference_mapped_volturnus_s": {
        # Keep the current ballast pattern/capacity untouched and only retune the
        # low-order rigid-body foundation toward the public VolturnUS-S reference.
        "mass_dry": 16879507.61225,
        "I_body": [1.251e10, 1.251e10, 1.2e11],
        "hydro_params": {
            "GM_L": 10.778625795547065,
            "GM_T": 10.778625795547065,
        },
        # The report inertia is a rigid-body quantity; this multiplier absorbs
        # the missing low-order added-inertia contribution in the simplified model.
        "rot_inertia_multiplier_roll_pitch": 3.422090977520233,
        # First-round damping stays conservative; it will be refined in Step B3
        # only if free-decay still deviates materially.
        "zetas": [0.05, 0.05, 0.20, 0.08, 0.08, 0.05],
    },
}

REFERENCE_MAPPED_VOLTURNUS_S_PARAM_SOURCES = {
    "mass_dry": {
        "source_type": "direct_mapping",
        "source_note": "From 76773 displaced volume 20206.34889 m^3 times rho=1025 kg/m^3, minus current default ballast total to preserve the existing ballast pattern.",
    },
    "I_body[0]": {
        "source_type": "direct_mapping",
        "source_note": "76773 Table 2 pitch/roll inertia about center of gravity = 1.251E+10 kg-m^2.",
    },
    "I_body[1]": {
        "source_type": "direct_mapping",
        "source_note": "76773 Table 2 pitch/roll inertia about center of gravity = 1.251E+10 kg-m^2.",
    },
    "hydro_params.GM_L": {
        "source_type": "direct_mapping",
        "source_note": "Back-solved from 76773 Table 3 hydrostatic stiffness 2.190E+09 N-m/rad with displaced volume 20206.34889 m^3.",
    },
    "hydro_params.GM_T": {
        "source_type": "direct_mapping",
        "source_note": "Back-solved from 76773 Table 3 hydrostatic stiffness 2.190E+09 N-m/rad with displaced volume 20206.34889 m^3.",
    },
    "rot_inertia_multiplier_roll_pitch": {
        "source_type": "equivalent_tuning",
        "source_note": "Chosen so that the effective pitch/roll inertia implied by the simplified model matches the 27.78 s reference free-decay period order while keeping the direct-mapped rigid-body inertia value visible.",
    },
    "zetas[3]": {
        "source_type": "equivalent_tuning",
        "source_note": "Initial low-order damping placeholder retained at the baseline value; to be refined only if Step B3 free-decay still needs convergence adjustment.",
    },
    "zetas[4]": {
        "source_type": "equivalent_tuning",
        "source_note": "Initial low-order damping placeholder retained at the baseline value; to be refined only if Step B3 free-decay still needs convergence adjustment.",
    },
}


DEFAULT_MAIN_EXPERIMENT_PROTOCOL = {
    "name": "main_protocol_v1",
    "controller_overrides": {
        "k_wind_comp_pitch": 0.0,
        "k_wind_comp_roll": 0.0,
    },
    "heave_overrides": {
        "enabled": False,
    },
    # Keep shaper on with fixed parameters across main comparisons.
    "target_shape_cfg": {
        "actuator_authority": "plant",
        "enable_rate_limit": True,
        "rate_limit_m3_min": 10.0,
        "enable_setpoint_shaping": True,
        "setpoint_alpha": 0.98,
        "setpoint_rate_deg_s": 0.02,
    },
}



# Keep Test E/F trim setups explicit and centralized.
TRIM_CFG_TEST_E = deepcopy(DEFAULT_TRIM_CFG_SIMPLE_DEFAULT)
TRIM_CFG_TEST_F = deepcopy(DEFAULT_TRIM_CFG_SIMPLE_DEFAULT)


DEFAULT_HEAVE_CFG = {
    "enabled": True,
    "target_heave_m": "auto",
    "kp": 0.005,
    "ki": 0.0,
    "max_correction_per_tank_kg": 2.0,
    "m_min_op_kg": 200000.0,
    "m_max_op_ratio": 0.90,
    "history_window_s": 30.0,
}


def clone_cfg(cfg):
    return deepcopy(cfg)


def clone_platform_profile(profile_name=DEFAULT_PLATFORM_PROFILE):
    name = DEFAULT_PLATFORM_PROFILE if profile_name is None else str(profile_name).strip() or DEFAULT_PLATFORM_PROFILE
    if name not in DEFAULT_PLATFORM_PROFILES:
        raise ValueError(f"unsupported platform_profile={profile_name}")
    return deepcopy(DEFAULT_PLATFORM_PROFILES[name])


def merge_platform_cfg(base_cfg, override_cfg):
    merged = deepcopy(base_cfg)
    for key, value in (override_cfg or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(deepcopy(value))
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_platform_profile(platform_profile=None, platform_cfg=None):
    name = DEFAULT_PLATFORM_PROFILE if platform_profile is None else str(platform_profile).strip() or DEFAULT_PLATFORM_PROFILE
    base_cfg = clone_platform_profile(name)
    return name, merge_platform_cfg(base_cfg, platform_cfg or {})
