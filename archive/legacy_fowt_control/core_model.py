import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

try:
    from scipy.interpolate import interp1d
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    def interp1d(x, y, bounds_error=False, fill_value=None):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.size != y_arr.size:
            raise ValueError("interp1d fallback expects 1D x/y arrays of equal length")
        if x_arr.size == 0:
            raise ValueError("interp1d fallback requires at least one sample")

        def _call(x_new):
            xq = np.asarray(x_new, dtype=float)
            if fill_value is None:
                left = float(y_arr[0])
                right = float(y_arr[-1])
            elif isinstance(fill_value, tuple) and len(fill_value) == 2:
                left = float(fill_value[0])
                right = float(fill_value[1])
            else:
                left = right = float(fill_value)
            return np.interp(xq, x_arr, y_arr, left=left, right=right)

        return _call

from defaults import DEFAULT_PLATFORM_PROFILE, clone_cfg, resolve_platform_profile

try:
    from wind_prediction.ballast_mass_properties import (
        compute_ballast_mass_properties,
        compute_incremental_ballast_mass_properties,
    )
except ModuleNotFoundError:  # Compatibility for direct imports from the legacy folder.
    _SRC_DIR = Path(__file__).resolve().parents[2] / "src"
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from wind_prediction.ballast_mass_properties import (
        compute_ballast_mass_properties,
        compute_incremental_ballast_mass_properties,
    )

# ==============================================================================
#  核心物理引擎 (Plant)
#  对应架构中的: core_model.py (Part 1)
# ==============================================================================

class FloatingPlatform:
    def __init__(
        self,
        stiffness_xlsx_path,
        pump_cfg=None,
        platform_profile=None,
        platform_cfg=None,
        allow_linear_mooring_fallback=False,
        platform_profile_purpose="control",
    ):
        """
        初始化浮式平台物理模型。
        :param stiffness_xlsx_path: 系泊刚度 Excel 文件路径（必须提供）
        """
        # --- 1. 物理常数 ---
        self.rho = 1025.0       # 海水密度 [kg/m^3]
        self.g = 9.81           # 重力加速度 [m/s^2]
        # Coordinate convention (right-handed FLU/ENU): x forward, y left(port), z up.
        # Gravity is applied along -z.
        
        # --- 2. 平台基础参数 ---
        # [Mass] 平台干重（不含压载水，约 1.69 万吨）
        self.mass_dry = 1.6937e7  
        
        # [COG] 平台干重重心 (Center of Gravity)
        # 修正逻辑：
        # 当前压载配置（前舱 1108 / 后舱 1362）会产生约 1.15e8 Nm 的抬头力矩，
        # 对应约 Pitch -3 度。为使平台回到正浮（Pitch 0 度），需要用风机重量
        # 产生的低头力矩与之抵消。按当前建模，平台干重重心需向前移动约 0.8 m。
        self.cog_dry = np.array([0.69, 0.0, -4.0]) 
        
        # [Inertia] 转动惯量
        self.I_body = np.array([8.0e10, 8.0e10, 1.2e11]) 
        self.arm_aero = 127.0   
        self.r_fairlead = np.array([0.0, 0.0, -9.0]) 
        
        # --- 3. 压载系统 (Actuator) ---
        R = 46.2 
        # Tank coordinates use the same frame: +x front, +y port(left), +z up.
        self.tank_pos = np.array([
            [ R,          0.0, -10.0],  # Tank 1 (前舱, Tank 0 in logic)
            [-R/2,    R*0.866, -10.0],  # Tank 2 (左后)
            [-R/2,   -R*0.866, -10.0]   # Tank 3 (右后)
        ])
        
        # [关键设定] 工作工况压载 (Working Condition)
        # 前舱 1108 吨、后舱 1362 吨，用于抵消风机重量，配合 COG 修正实现正浮。
        self.default_ballast = np.array([1108000.0, 1362000.0, 1362000.0])
        
        # 初始化状态
        self.current_ballast_mass = self.default_ballast.copy()
        self.target_ballast_mass  = self.default_ballast.copy()
        
        # [Fix] 物理限制: 1850 m^3 * 海水密度 (约 1896 吨)
        self.tank_capacity = 1850.0 * self.rho 
        
        # 泵速: 15 m^3/min -> kg/s
        self.pump_vol_rate = 15.0 / 60.0
        self.pump_mass_rate = self.pump_vol_rate * self.rho
        pump_cfg_eff = dict(pump_cfg or {})
        self.pump_rate_schedule_m3_min = [
            (3000.0, 15.0),
            (2000.0, 14.0),
            (1000.0, 12.0),
            (700.0, 10.0),
            (500.0, 8.0),
            (300.0, 6.0),
            (200.0, 4.0),
            (0.0, 0.0),
        ]
        self.pump_stop_err_kg = float(pump_cfg_eff.get("pump_stop_err_kg", 150.0))
        self.pump_restart_err_kg = float(pump_cfg_eff.get("pump_restart_err_kg", 300.0))
        self.pump_min_on_s = float(pump_cfg_eff.get("pump_min_on_s", 12.0))
        self.pump_min_off_s = float(pump_cfg_eff.get("pump_min_off_s", 6.0))
        # Keep legacy behavior when pump_cfg=None: no extra stop hold and no global quiet gate.
        self.pump_hold_before_stop_s = float(pump_cfg_eff.get("pump_hold_before_stop_s", 0.0))
        self.pump_global_quiet_backlog_kg = float(
            pump_cfg_eff.get("pump_global_quiet_backlog_kg", np.inf)
        )
        # Quiet-stop keeps pumps latched on while the upper-layer target is still moving.
        self.pump_target_quiet_rate_kg_s = float(
            pump_cfg_eff.get("pump_target_quiet_rate_kg_s", np.inf)
        )
        self.pump_global_quiet_hold_s = float(
            pump_cfg_eff.get("pump_global_quiet_hold_s", 0.0)
        )
        # Keep legacy behavior when pump_cfg=None: infinite ramp means instantaneous rate changes.
        self.pump_ramp_up_m3_min_per_s = float(
            pump_cfg_eff.get("pump_ramp_up_m3_min_per_s", np.inf)
        )
        self.pump_ramp_down_m3_min_per_s = float(
            pump_cfg_eff.get("pump_ramp_down_m3_min_per_s", np.inf)
        )
        # Candidate-only extensions; default to disabled so the frozen baseline is unchanged.
        self.pump_stage_hysteresis_kg = float(
            pump_cfg_eff.get("pump_stage_hysteresis_kg", 0.0)
        )
        self.pump_stage_min_dwell_s = float(
            pump_cfg_eff.get("pump_stage_min_dwell_s", 0.0)
        )
        self.pump_rate_release_tau_s = float(
            pump_cfg_eff.get("pump_rate_release_tau_s", 0.0)
        )
        self.pump_low_end_stage_hysteresis_kg = float(
            pump_cfg_eff.get("pump_low_end_stage_hysteresis_kg", 0.0)
        )
        self.pump_low_end_stage_min_dwell_s = float(
            pump_cfg_eff.get("pump_low_end_stage_min_dwell_s", 0.0)
        )
        self.pump_low_end_stage_max_idx = int(
            max(0, int(pump_cfg_eff.get("pump_low_end_stage_max_idx", 2)))
        )
        self.pump_stage_allow_zero_rate_latched = bool(
            pump_cfg_eff.get("pump_stage_allow_zero_rate_latched", False)
        )
        self._pump_active_latch = np.array([False, False, False], dtype=bool)
        self._pump_on_elapsed_s = np.array([0.0, 0.0, 0.0], dtype=float)
        self._pump_off_elapsed_s = np.array([self.pump_min_off_s] * 3, dtype=float)
        self._pump_near_target_s = np.array([0.0, 0.0, 0.0], dtype=float)
        self._pump_rate_smoothed_m3_min = np.array([0.0, 0.0, 0.0], dtype=float)
        self._pump_rate_released_m3_min = np.array([0.0, 0.0, 0.0], dtype=float)
        self._pump_global_quiet_s = 0.0
        self._pump_quiet_stop_blocked_prev = np.array([False, False, False], dtype=bool)
        self._pump_quiet_stop_block_count = 0
        self._pump_latch_switch_count = 0
        self._pump_stage_idx = np.array([0, 0, 0], dtype=int)
        self._pump_stage_dwell_s = np.array([0.0, 0.0, 0.0], dtype=float)
        self._pump_stage_switch_count = 0
        
        # --- 4. 水动力参数 ---
        self.hydro_params = {
            'r_col': 7.0, 'GM_L': 11.0, 'GM_T': 11.0, 'Cd': 0.8,
            'A_proj_surge': 840.0, 'A_proj_heave': 1360.0
        }
        # Keep the base rigid-body inertia values unchanged for now; only remove the
        # unsupported extra roll/pitch inertia inflation in M_total.
        self.rot_inertia_multiplier_roll_pitch = 1.0
        self.rot_inertia_multiplier_yaw = 1.3
        self.K_hydro = np.zeros(6)
        # K33 (Heave刚度) = rho * g * A_wp
        self.K_hydro[2] = self.rho * self.g * (3 * np.pi * self.hydro_params['r_col']**2)

        # --- 5. 阻尼与 RAO ---
        self.K_yaw_stiffness = 5.0e7 
        self.zetas = np.array([0.05, 0.05, 0.20, 0.08, 0.08, 0.05]) 
        self.C_lin = np.zeros((6,6))
        self.rao_params = {'surge_peak': 2.0e5, 'heave_peak': 5.0e5, 'pitch_peak': 1.0e7, 'cutoff_low': 0.5, 'cutoff_high': 0.8}
        self.K_mooring_lin = np.array([2.0e5, 2.0e5, 0, 0, 0, 0])
        self.platform_profile_name = DEFAULT_PLATFORM_PROFILE
        self.platform_profile_cfg = {}
        self.platform_profile_cfg_effective = {}
        self.platform_profile_status = "runtime"
        self.platform_profile_purpose = str(platform_profile_purpose).strip().lower()
        if self.platform_profile_purpose not in {
            "control",
            "framework_smoke",
            "audit",
        }:
            raise ValueError(
                "platform_profile_purpose must be 'control', "
                "'framework_smoke' or 'audit'"
            )
        self.load_reference_mode = "legacy_mixed"
        self.mass_property_mode = "legacy_diagonal"
        self.mooring_reference_mode = "legacy_raw"
        self.reference_property_mode = "assembled_dry_plus_baseline"
        self.configured_reference_total_mass_kg = None
        self.configured_reference_center_of_mass_m = None
        self.configured_reference_inertia_about_reference_kg_m2 = None
        self._apply_platform_profile(platform_profile=platform_profile, platform_cfg=platform_cfg)
        if (
            self.platform_profile_status == "audit_only_not_for_control_validation"
            and self.platform_profile_purpose != "audit"
        ):
            raise ValueError(
                f"platform profile {self.platform_profile_name!r} is audit-only and "
                "cannot be used for control validation"
            )
        self._pump_prev_target_ballast_mass = self.target_ballast_mass.copy()

        # The research profile treats the initial working ballast as the zero-load
        # reference. The submitted-paper default retains its historical behavior.
        self.reference_ballast_mass = self.default_ballast.copy()
        if self.reference_property_mode == "complete_reference":
            self.reference_mass_properties = (
                compute_incremental_ballast_mass_properties(
                    reference_mass_kg=self.configured_reference_total_mass_kg,
                    reference_center_of_mass_m=(
                        self.configured_reference_center_of_mass_m
                    ),
                    reference_inertia_about_reference_kg_m2=(
                        self.configured_reference_inertia_about_reference_kg_m2
                    ),
                    tank_mass_deltas_kg=np.zeros(3, dtype=float),
                    tank_coordinates_m=self.tank_pos,
                )
            )
        else:
            dry_offset = np.asarray(self.cog_dry, dtype=float)
            dry_inertia_reference = np.diag(
                np.asarray(self.I_body, dtype=float)
            ) + self.mass_dry * (
                float(np.dot(dry_offset, dry_offset)) * np.eye(3)
                - np.outer(dry_offset, dry_offset)
            )
            self.reference_mass_properties = compute_ballast_mass_properties(
                dry_mass_kg=self.mass_dry,
                dry_center_of_mass_m=self.cog_dry,
                dry_inertia_about_reference_kg_m2=dry_inertia_reference,
                tank_masses_kg=self.reference_ballast_mass,
                tank_coordinates_m=self.tank_pos,
            )
        self.reference_total_mass = float(
            self.reference_mass_properties.total_mass_kg
        )
        reference_inertia_diagonal = np.diag(
            self.reference_mass_properties.inertia_about_reference_kg_m2
        )
        self.reference_added_mass_matrix = np.diag(
            [
                0.6 * self.reference_total_mass,
                0.6 * self.reference_total_mass,
                1.0 * self.reference_total_mass,
                max(0.0, self.rot_inertia_multiplier_roll_pitch - 1.0)
                * reference_inertia_diagonal[0],
                max(0.0, self.rot_inertia_multiplier_roll_pitch - 1.0)
                * reference_inertia_diagonal[1],
                max(0.0, self.rot_inertia_multiplier_yaw - 1.0)
                * reference_inertia_diagonal[2],
            ]
        )
        self.mass_properties = None
        
        # 更新质量和阻尼矩阵
        self._update_mass_matrix()
        
        # --- 6. 系泊系统 ---
        self.mooring_bounds = {'x': [-100, 100]}
        self.use_nonlinear_mooring = False
        self.mooring_mode = "LINEAR"
        self.mooring_sign = 1.0
        self.mooring_reference_force_raw = 0.0
        self.mooring_source_path = None
        self.allow_linear_mooring_fallback = bool(allow_linear_mooring_fallback)
        
        if stiffness_xlsx_path and os.path.exists(stiffness_xlsx_path):
            self._load_mooring_data(stiffness_xlsx_path)
        elif self.allow_linear_mooring_fallback:
            print(
                f"!!! [Plant Warning] Mooring file not found at "
                f"{stiffness_xlsx_path}. Using Linear Fallback."
            )
        else:
            raise FileNotFoundError(
                f"Mooring stiffness file not found: {stiffness_xlsx_path}"
            )

        self._update_linear_damping()

        # --- 7. 状态 ---
        self.wave_components = [] 
        self.wave_heading_rad = 0.0 
        self.state = np.zeros(12) # [x,y,z, roll,pitch,yaw, vx,vy,vz, wx,wy,wz]
        # Positive angles follow right-hand rule around +x/+y/+z:
        # +roll=starboard-down, +pitch=bow-down, +yaw=turn-to-port.

    def _apply_platform_profile(self, platform_profile=None, platform_cfg=None):
        override_cfg = dict(platform_cfg or {})
        behavioral_identity_fields = {
            "load_reference_mode",
            "mass_property_mode",
            "mooring_reference_mode",
            "reference_property_mode",
        }
        requested_profile_name = (
            DEFAULT_PLATFORM_PROFILE
            if platform_profile is None
            else str(platform_profile).strip() or DEFAULT_PLATFORM_PROFILE
        )
        identity_overrides = sorted(set(override_cfg) & behavioral_identity_fields)
        if "profile_status" in override_cfg:
            raise ValueError("platform_cfg cannot override profile_status")
        if identity_overrides and requested_profile_name != DEFAULT_PLATFORM_PROFILE:
            raise ValueError(
                "named platform profiles cannot override identity fields: "
                + ", ".join(identity_overrides)
            )
        required_behavior_modes = {
            "load_reference_mode",
            "mass_property_mode",
            "mooring_reference_mode",
        }
        if identity_overrides and not required_behavior_modes.issubset(override_cfg):
            raise ValueError(
                "a custom platform identity must define load, mass-property and "
                "mooring-reference modes together"
            )
        profile_name, cfg = resolve_platform_profile(
            platform_profile=platform_profile,
            platform_cfg=platform_cfg,
        )
        self.platform_profile_base_name = profile_name
        if identity_overrides:
            self.platform_profile_name = "custom"
        else:
            self.platform_profile_name = (
                profile_name if not override_cfg else f"{profile_name}+custom"
            )
        self.platform_profile_cfg = clone_cfg(cfg)
        if not cfg:
            self.platform_profile_cfg_effective = {}
            return

        supported_fields = {
            "mass_dry",
            "cog_dry",
            "I_body",
            "default_ballast",
            "hydro_params",
            "rot_inertia_multiplier_roll_pitch",
            "rot_inertia_multiplier_yaw",
            "zetas",
            "K_mooring_lin",
            "load_reference_mode",
            "mass_property_mode",
            "mooring_reference_mode",
            "reference_property_mode",
            "reference_total_mass_kg",
            "reference_center_of_mass_m",
            "reference_inertia_about_reference_kg_m2",
            "profile_status",
        }
        unknown_fields = sorted(set(cfg) - supported_fields)
        if unknown_fields:
            raise ValueError(
                "unsupported platform configuration fields: "
                + ", ".join(unknown_fields)
            )

        if "mass_dry" in cfg:
            self.mass_dry = float(cfg["mass_dry"])
        if "cog_dry" in cfg:
            self.cog_dry = np.asarray(cfg["cog_dry"], dtype=float)
        if "I_body" in cfg:
            self.I_body = np.asarray(cfg["I_body"], dtype=float)
        if "default_ballast" in cfg:
            self.default_ballast = np.asarray(cfg["default_ballast"], dtype=float)
            self.current_ballast_mass = self.default_ballast.copy()
            self.target_ballast_mass = self.default_ballast.copy()
        if "hydro_params" in cfg:
            for key, value in cfg["hydro_params"].items():
                self.hydro_params[key] = float(value)
        if "rot_inertia_multiplier_roll_pitch" in cfg:
            self.rot_inertia_multiplier_roll_pitch = float(cfg["rot_inertia_multiplier_roll_pitch"])
        if "rot_inertia_multiplier_yaw" in cfg:
            self.rot_inertia_multiplier_yaw = float(cfg["rot_inertia_multiplier_yaw"])
        if "zetas" in cfg:
            self.zetas = np.asarray(cfg["zetas"], dtype=float)
        if "K_mooring_lin" in cfg:
            self.K_mooring_lin = np.asarray(cfg["K_mooring_lin"], dtype=float)
        if "load_reference_mode" in cfg:
            self.load_reference_mode = str(cfg["load_reference_mode"])
        if "mass_property_mode" in cfg:
            self.mass_property_mode = str(cfg["mass_property_mode"])
        if "mooring_reference_mode" in cfg:
            self.mooring_reference_mode = str(cfg["mooring_reference_mode"])
        if "reference_property_mode" in cfg:
            self.reference_property_mode = str(cfg["reference_property_mode"])
        if "reference_total_mass_kg" in cfg:
            self.configured_reference_total_mass_kg = float(
                cfg["reference_total_mass_kg"]
            )
        if "reference_center_of_mass_m" in cfg:
            self.configured_reference_center_of_mass_m = np.asarray(
                cfg["reference_center_of_mass_m"], dtype=float
            )
        if "reference_inertia_about_reference_kg_m2" in cfg:
            self.configured_reference_inertia_about_reference_kg_m2 = np.asarray(
                cfg["reference_inertia_about_reference_kg_m2"], dtype=float
            )
        if "profile_status" in cfg:
            self.platform_profile_status = str(cfg["profile_status"])

        if self.load_reference_mode not in {"legacy_mixed", "reference_incremental"}:
            raise ValueError(
                f"unsupported load_reference_mode: {self.load_reference_mode}"
            )
        if self.mass_property_mode not in {
            "legacy_diagonal",
            "point_mass_diagonal",
            "reference_delta_point_mass",
        }:
            raise ValueError(
                f"unsupported mass_property_mode: {self.mass_property_mode}"
            )
        if self.mooring_reference_mode not in {"legacy_raw", "zero_at_reference"}:
            raise ValueError(
                f"unsupported mooring_reference_mode: {self.mooring_reference_mode}"
            )
        if self.reference_property_mode not in {
            "assembled_dry_plus_baseline",
            "complete_reference",
        }:
            raise ValueError(
                f"unsupported reference_property_mode: {self.reference_property_mode}"
            )
        allowed_mode_sets = {
            (
                "legacy_mixed",
                "legacy_diagonal",
                "legacy_raw",
            ),
            (
                "reference_incremental",
                "reference_delta_point_mass",
                "zero_at_reference",
            ),
        }
        selected_modes = (
            self.load_reference_mode,
            self.mass_property_mode,
            self.mooring_reference_mode,
        )
        if selected_modes not in allowed_mode_sets:
            raise ValueError(
                "incompatible platform profile modes: " + repr(selected_modes)
            )
        if self.reference_property_mode == "complete_reference":
            missing_reference_fields = [
                name
                for name, value in (
                    (
                        "reference_total_mass_kg",
                        self.configured_reference_total_mass_kg,
                    ),
                    (
                        "reference_center_of_mass_m",
                        self.configured_reference_center_of_mass_m,
                    ),
                    (
                        "reference_inertia_about_reference_kg_m2",
                        self.configured_reference_inertia_about_reference_kg_m2,
                    ),
                )
                if value is None
            ]
            if missing_reference_fields:
                raise ValueError(
                    "complete_reference requires: "
                    + ", ".join(missing_reference_fields)
                )

        self.K_hydro[2] = self.rho * self.g * (3 * np.pi * self.hydro_params['r_col']**2)
        self.platform_profile_cfg_effective = {
            "mass_dry": float(self.mass_dry),
            "cog_dry": np.asarray(self.cog_dry, dtype=float).tolist(),
            "I_body": np.asarray(self.I_body, dtype=float).tolist(),
            "default_ballast": np.asarray(self.default_ballast, dtype=float).tolist(),
            "hydro_params": dict(self.hydro_params),
            "rot_inertia_multiplier_roll_pitch": float(self.rot_inertia_multiplier_roll_pitch),
            "rot_inertia_multiplier_yaw": float(self.rot_inertia_multiplier_yaw),
            "zetas": np.asarray(self.zetas, dtype=float).tolist(),
            "load_reference_mode": self.load_reference_mode,
            "mass_property_mode": self.mass_property_mode,
            "mooring_reference_mode": self.mooring_reference_mode,
            "reference_property_mode": self.reference_property_mode,
            "profile_status": self.platform_profile_status,
        }

    # ==========================================================================
    #  对外接口 (API)
    # ==========================================================================

    def resolved_platform_identity(self):
        """Return the complete, JSON-compatible plant configuration in use.

        The profile name alone is not a sufficient run identity because custom
        overrides and actuator settings can change the realised plant.  This
        snapshot is intentionally derived from resolved runtime values rather
        than the originally requested configuration.
        """
        reference_properties = self.reference_mass_properties

        def _finite_number_or_label(value):
            number = float(value)
            if np.isposinf(number):
                return "positive_infinity"
            if np.isneginf(number):
                return "negative_infinity"
            if np.isnan(number):
                raise ValueError("resolved platform identity contains NaN")
            return number
        return {
            "schema_version": "floating_platform_identity.v1",
            "profile": {
                "requested_name": self.platform_profile_name,
                "base_name": self.platform_profile_base_name,
                "status": self.platform_profile_status,
                "purpose": self.platform_profile_purpose,
            },
            "model_modes": {
                "load_reference": self.load_reference_mode,
                "mass_properties": self.mass_property_mode,
                "mooring_reference": self.mooring_reference_mode,
                "reference_properties": self.reference_property_mode,
            },
            "constants": {
                "seawater_density_kg_m3": float(self.rho),
                "gravity_m_s2": float(self.g),
            },
            "structure": {
                "dry_mass_kg": float(self.mass_dry),
                "dry_center_of_mass_m": np.asarray(
                    self.cog_dry, dtype=float
                ).tolist(),
                "dry_inertia_about_center_of_mass_kg_m2": np.asarray(
                    self.I_body, dtype=float
                ).tolist(),
                "aerodynamic_load_arm_m": float(self.arm_aero),
                "fairlead_position_m": np.asarray(
                    self.r_fairlead, dtype=float
                ).tolist(),
            },
            "ballast_system": {
                "tank_positions_m": np.asarray(
                    self.tank_pos, dtype=float
                ).tolist(),
                "reference_tank_masses_kg": np.asarray(
                    self.reference_ballast_mass, dtype=float
                ).tolist(),
                "tank_capacity_kg": float(self.tank_capacity),
                "pump_rate_schedule_m3_min": [
                    [float(error), float(rate)]
                    for error, rate in self.pump_rate_schedule_m3_min
                ],
                "pump_stop_error_kg": float(self.pump_stop_err_kg),
                "pump_restart_error_kg": float(self.pump_restart_err_kg),
                "pump_minimum_on_s": float(self.pump_min_on_s),
                "pump_minimum_off_s": float(self.pump_min_off_s),
                "pump_hold_before_stop_s": float(
                    self.pump_hold_before_stop_s
                ),
                "pump_global_quiet_backlog_kg": _finite_number_or_label(
                    self.pump_global_quiet_backlog_kg
                ),
                "pump_target_quiet_rate_kg_s": _finite_number_or_label(
                    self.pump_target_quiet_rate_kg_s
                ),
                "pump_global_quiet_hold_s": float(
                    self.pump_global_quiet_hold_s
                ),
                "pump_ramp_up_m3_min_per_s": _finite_number_or_label(
                    self.pump_ramp_up_m3_min_per_s
                ),
                "pump_ramp_down_m3_min_per_s": _finite_number_or_label(
                    self.pump_ramp_down_m3_min_per_s
                ),
                "pump_stage_hysteresis_kg": float(
                    self.pump_stage_hysteresis_kg
                ),
                "pump_stage_minimum_dwell_s": float(
                    self.pump_stage_min_dwell_s
                ),
                "pump_rate_release_time_constant_s": float(
                    self.pump_rate_release_tau_s
                ),
                "pump_low_end_stage_hysteresis_kg": float(
                    self.pump_low_end_stage_hysteresis_kg
                ),
                "pump_low_end_stage_minimum_dwell_s": float(
                    self.pump_low_end_stage_min_dwell_s
                ),
                "pump_low_end_stage_maximum_index": int(
                    self.pump_low_end_stage_max_idx
                ),
                "pump_allow_zero_rate_while_latched": bool(
                    self.pump_stage_allow_zero_rate_latched
                ),
            },
            "hydrodynamics": {
                "parameters": {
                    str(key): float(value)
                    for key, value in sorted(self.hydro_params.items())
                },
                "hydrostatic_stiffness": np.asarray(
                    self.K_hydro, dtype=float
                ).tolist(),
                "linear_damping_matrix": np.asarray(
                    self.C_lin, dtype=float
                ).tolist(),
                "damping_ratios": np.asarray(
                    self.zetas, dtype=float
                ).tolist(),
                "rotational_inertia_multiplier_roll_pitch": float(
                    self.rot_inertia_multiplier_roll_pitch
                ),
                "rotational_inertia_multiplier_yaw": float(
                    self.rot_inertia_multiplier_yaw
                ),
                "yaw_stiffness": float(self.K_yaw_stiffness),
            },
            "mooring": {
                "mode": self.mooring_mode,
                "uses_nonlinear_curve": bool(self.use_nonlinear_mooring),
                "reference_sign": float(self.mooring_sign),
                "reference_force_raw_n": float(self.mooring_reference_force_raw),
                "linear_stiffness": np.asarray(
                    self.K_mooring_lin, dtype=float
                ).tolist(),
                "bounds_m": {
                    str(key): [float(item) for item in value]
                    for key, value in sorted(self.mooring_bounds.items())
                },
                "linear_fallback_allowed": bool(
                    self.allow_linear_mooring_fallback
                ),
            },
            "reference_mass_properties": {
                "total_mass_kg": float(reference_properties.total_mass_kg),
                "center_of_mass_m": np.asarray(
                    reference_properties.center_of_mass_m, dtype=float
                ).tolist(),
                "inertia_about_reference_kg_m2": np.asarray(
                    reference_properties.inertia_about_reference_kg_m2,
                    dtype=float,
                ).tolist(),
                "fixed_added_mass_matrix": np.asarray(
                    self.reference_added_mass_matrix, dtype=float
                ).tolist(),
                "effective_mass_matrix": np.asarray(
                    self.M_total, dtype=float
                ).tolist(),
            },
        }
    
    def set_ballast_target(self, m1, m2, m3):
        """[Control Input] 设定目标压载量。"""
        m1 = np.clip(m1, 0, self.tank_capacity)
        m2 = np.clip(m2, 0, self.tank_capacity)
        m3 = np.clip(m3, 0, self.tank_capacity)
        self.target_ballast_mass = np.array([m1, m2, m3])

    def _get_pump_rate_m3_min(self, abs_err_kg):
        # Piecewise-linear interpolation over configured breakpoints to avoid hard jumps.
        if not self.pump_rate_schedule_m3_min:
            return 0.0
        points = self._sorted_stage_points()
        e = float(abs_err_kg)
        if e <= points[0][0]:
            return points[0][1]
        if e >= points[-1][0]:
            return points[-1][1]
        for i in range(len(points) - 1):
            e0, r0 = points[i]
            e1, r1 = points[i + 1]
            if e0 <= e <= e1:
                if abs(e1 - e0) < 1e-12:
                    return r1
                frac = (e - e0) / (e1 - e0)
                return r0 + frac * (r1 - r0)
        return points[-1][1]

    def _sorted_stage_points(self):
        return sorted(
            [(float(th), float(rt)) for th, rt in self.pump_rate_schedule_m3_min],
            key=lambda x: x[0],
        )

    @staticmethod
    def _first_positive_stage_idx(points):
        for idx, (_, rate) in enumerate(points):
            if float(rate) > 1e-12:
                return idx
        return None

    def _find_stage_index(self, abs_err_kg, allow_zero_stage=True):
        if not self.pump_rate_schedule_m3_min:
            return 0
        points = self._sorted_stage_points()
        if len(points) <= 1:
            return 0
        e = float(abs_err_kg)
        if e <= points[0][0]:
            idx = 0
        else:
            idx = len(points) - 2
            for cand in range(len(points) - 1):
                if e <= points[cand + 1][0]:
                    idx = cand
                    break
        if not allow_zero_stage and len(points) > 2:
            idx = max(1, idx)
        return int(idx)

    def _stage_gate_params(self, current_idx, desired_idx, min_idx):
        hyst = max(float(self.pump_stage_hysteresis_kg), 0.0)
        dwell = max(float(self.pump_stage_min_dwell_s), 0.0)
        low_end_hyst = max(float(self.pump_low_end_stage_hysteresis_kg), 0.0)
        low_end_dwell = max(float(self.pump_low_end_stage_min_dwell_s), 0.0)
        if low_end_hyst <= 1e-12 and low_end_dwell <= 1e-12:
            return hyst, dwell
        low_end_max_idx = max(int(min_idx), int(self.pump_low_end_stage_max_idx))
        if int(current_idx) <= low_end_max_idx or int(desired_idx) <= low_end_max_idx:
            hyst = max(hyst, low_end_hyst)
            dwell = max(dwell, low_end_dwell)
        return hyst, dwell

    def _resolve_stage_index(self, pump_idx, abs_err_kg, dt, allow_zero_stage=True):
        desired_idx = self._find_stage_index(abs_err_kg, allow_zero_stage=allow_zero_stage)
        points = self._sorted_stage_points()
        if len(points) <= 1:
            self._pump_stage_idx[pump_idx] = 0
            self._pump_stage_dwell_s[pump_idx] += max(float(dt), 0.0)
            return 0, False

        min_idx = 0
        if not allow_zero_stage and len(points) > 2:
            min_idx = 1
        idx = int(np.clip(self._pump_stage_idx[pump_idx], min_idx, len(points) - 2))
        hyst, dwell_req = self._stage_gate_params(idx, desired_idx, min_idx)
        if hyst <= 1e-12 and dwell_req <= 1e-12:
            self._pump_stage_idx[pump_idx] = int(desired_idx)
            self._pump_stage_dwell_s[pump_idx] += max(float(dt), 0.0)
            return int(desired_idx), False
        self._pump_stage_dwell_s[pump_idx] += max(float(dt), 0.0)
        if self._pump_stage_dwell_s[pump_idx] < dwell_req:
            return idx, False

        changed = False
        while idx < len(points) - 2:
            upper = float(points[idx + 1][0])
            if float(abs_err_kg) >= upper + hyst:
                idx += 1
                changed = True
            else:
                break
        while idx > min_idx:
            lower = float(points[idx][0])
            if float(abs_err_kg) <= lower - hyst:
                idx -= 1
                changed = True
            else:
                break

        if changed:
            self._pump_stage_idx[pump_idx] = int(idx)
            self._pump_stage_dwell_s[pump_idx] = 0.0
            self._pump_stage_switch_count += 1
            return int(idx), True

        self._pump_stage_idx[pump_idx] = int(idx)
        return int(idx), False

    def _get_pump_rate_with_stage_logic(self, pump_idx, abs_err_kg, dt, allow_zero_stage=True):
        if not self.pump_rate_schedule_m3_min:
            return 0.0, 0, False
        if (
            self.pump_stage_hysteresis_kg <= 1e-12
            and self.pump_stage_min_dwell_s <= 1e-12
        ):
            stage_idx = self._find_stage_index(abs_err_kg, allow_zero_stage=allow_zero_stage)
            return float(self._get_pump_rate_m3_min(abs_err_kg)), int(stage_idx), False

        points = self._sorted_stage_points()
        stage_idx, stage_changed = self._resolve_stage_index(
            pump_idx=pump_idx,
            abs_err_kg=abs_err_kg,
            dt=dt,
            allow_zero_stage=allow_zero_stage,
        )
        stage_idx = int(np.clip(stage_idx, 0, len(points) - 2))
        e0, r0 = points[stage_idx]
        e1, r1 = points[stage_idx + 1]
        e = float(np.clip(float(abs_err_kg), e0, e1))
        if abs(e1 - e0) < 1e-12:
            return float(r1), stage_idx, stage_changed
        frac = (e - e0) / (e1 - e0)
        return float(r0 + frac * (r1 - r0)), stage_idx, stage_changed

    def _release_rate_target(self, pump_idx, target_rate_m3_min, dt):
        target = max(0.0, float(target_rate_m3_min))
        if self.pump_rate_release_tau_s <= 1e-12 or dt <= 0.0:
            self._pump_rate_released_m3_min[pump_idx] = target
            return target
        alpha = float(dt / (self.pump_rate_release_tau_s + dt))
        prev = max(0.0, float(self._pump_rate_released_m3_min[pump_idx]))
        released = prev + alpha * (target - prev)
        self._pump_rate_released_m3_min[pump_idx] = float(released)
        return float(released)

    def force_ballast_mass(self, m1, m2, m3):
        """[Initialization] 强制瞬间设置（跳过泵动态）。"""
        raw = np.array([m1, m2, m3])
        clamped = np.clip(raw, 0.0, self.tank_capacity)
        self.current_ballast_mass = clamped
        self.target_ballast_mass = clamped.copy()
        self._pump_active_latch[:] = False
        self._pump_on_elapsed_s[:] = 0.0
        self._pump_off_elapsed_s[:] = self.pump_min_off_s
        self._pump_near_target_s[:] = 0.0
        self._pump_rate_smoothed_m3_min[:] = 0.0
        self._pump_rate_released_m3_min[:] = 0.0
        self._pump_prev_target_ballast_mass = clamped.copy()
        self._pump_global_quiet_s = 0.0
        self._pump_quiet_stop_blocked_prev[:] = False
        self._pump_quiet_stop_block_count = 0
        self._pump_latch_switch_count = 0
        self._pump_stage_idx[:] = 0
        self._pump_stage_dwell_s[:] = 0.0
        self._pump_stage_switch_count = 0
        self._update_mass_matrix()
        self._update_linear_damping()
        
        if np.any(np.abs(raw - clamped) > 1.0):
            print(f"!!! [System Warning] Force input clipped to Capacity {self.tank_capacity:.1f} kg")

    def set_irregular_wave(self, Hs, Tp, direction_deg=0.0, n_freqs=50, seed=None):
        """[Env Input] 设置 JONSWAP 波浪谱。"""
        if n_freqs <= 0 or Hs <= 0:
            self.wave_components = []
            return
        if seed is not None: np.random.seed(seed)
        
        self.wave_heading_rad = np.radians(direction_deg)
        wp = 2 * np.pi / Tp
        w_min, w_max = 0.4 * wp, 3.0 * wp
        freqs, dw = np.linspace(w_min, w_max, n_freqs, endpoint=False, retstep=True)
        
        # JONSWAP Calculation
        gamma = 3.3
        sigma = np.where(freqs <= wp, 0.07, 0.09)
        term1 = 5/16 * (Hs**2 * wp**4) / (freqs**5)
        term2 = np.exp(-1.25 * (freqs/wp)**-4)
        term3 = gamma ** np.exp(-0.5 * ((freqs - wp) / (sigma * wp))**2)
        S_vals = (1 - 0.287 * np.log(gamma)) * term1 * term2 * term3
        
        amps = np.sqrt(2 * S_vals * dw)
        phases = np.random.rand(n_freqs) * 2 * np.pi
        
        # Scaling correction
        t_chk = np.linspace(0, 100*Tp, 1000)
        eta = np.sum([a * np.cos(w*t_chk + p) for w,a,p in zip(freqs, amps, phases)], axis=0)
        scale = Hs / (4 * np.std(eta)) if np.std(eta) > 1e-6 else 1.0
        
        self.wave_components = [{'w': w, 'a': a*scale, 'phi': p} for w,a,p in zip(freqs, amps, phases)]
        print(f">>> [Plant] Wave Init: Hs={Hs}m, Tp={Tp}s")

    @staticmethod
    def _ramp_towards_rate(current_rate, target_rate, ramp_up_m3_min_per_s, ramp_down_m3_min_per_s, dt):
        curr = max(0.0, float(current_rate))
        tgt = max(0.0, float(target_rate))
        dt = max(float(dt), 0.0)
        if tgt >= curr:
            if np.isfinite(float(ramp_up_m3_min_per_s)):
                max_delta = max(0.0, float(ramp_up_m3_min_per_s)) * dt
            else:
                max_delta = np.inf
            return float(min(tgt, curr + max_delta))
        if np.isfinite(float(ramp_down_m3_min_per_s)):
            max_delta = max(0.0, float(ramp_down_m3_min_per_s)) * dt
        else:
            max_delta = np.inf
        return float(max(tgt, curr - max_delta))

    def step(self, thrust_N, wind_dir_deg, dt, current_time):
        """
        [Core Loop] 物理步进
        :return: state (numpy array), info (dict)
        """
        # 1. 执行机构动力学 (Pump Dynamics)
        mass_changed = False
        pump_active_flags = [False, False, False]
        pump_saturated_flags = [False, False, False]
        pump_rate_limited_flags = [False, False, False]
        pump_fullspeed_flags = [False, False, False]
        pump_rate_cmd_m3_min = [0.0, 0.0, 0.0]
        pump_rate_target_m3_min = [0.0, 0.0, 0.0]
        pump_rate_released_m3_min = [0.0, 0.0, 0.0]
        pump_backlog_kg = [0.0, 0.0, 0.0]
        pump_latch_switch_step = [False, False, False]
        pump_quiet_stop_blocked = [False, False, False]
        pump_stage_idx = [0, 0, 0]
        pump_stage_switch_step = [False, False, False]
        previous_mass = self.current_ballast_mass.copy()
        total_backlog_kg = float(
            np.sum(np.abs(self.target_ballast_mass - self.current_ballast_mass))
        )
        if dt > 1e-12:
            target_motion_kg_s = float(
                np.sum(np.abs(self.target_ballast_mass - self._pump_prev_target_ballast_mass)) / dt
            )
        else:
            target_motion_kg_s = 0.0
        # Quiet-stop: keep pumps latched on while the global target is still drifting,
        # otherwise slow target motion plus discrete latching creates tail-end pulse trains.
        global_quiet_raw = (
            total_backlog_kg < self.pump_global_quiet_backlog_kg
            and target_motion_kg_s <= self.pump_target_quiet_rate_kg_s
        )
        if global_quiet_raw:
            self._pump_global_quiet_s += dt
        else:
            self._pump_global_quiet_s = 0.0
        global_quiet_ready = self._pump_global_quiet_s >= self.pump_global_quiet_hold_s
        max_sched_rate_m3_min = max([float(rt) for _, rt in self.pump_rate_schedule_m3_min]) if self.pump_rate_schedule_m3_min else 0.0
        
        for i in range(3):
            latch_prev = bool(self._pump_active_latch[i])
            err = self.target_ballast_mass[i] - self.current_ballast_mass[i]
            abs_err = abs(err)
            pump_backlog_kg[i] = float(abs_err)
            if abs_err < self.pump_stop_err_kg:
                self._pump_near_target_s[i] += dt
            else:
                self._pump_near_target_s[i] = 0.0

            if self._pump_active_latch[i]:
                self._pump_on_elapsed_s[i] += dt
                self._pump_off_elapsed_s[i] = 0.0
            else:
                self._pump_off_elapsed_s[i] += dt
                self._pump_on_elapsed_s[i] = 0.0

            # Latch with hysteresis + dwell constraints to reduce rapid on/off toggling.
            local_stop_ready = (
                abs_err < self.pump_stop_err_kg
                and self._pump_on_elapsed_s[i] >= self.pump_min_on_s
                and self._pump_near_target_s[i] >= self.pump_hold_before_stop_s
            )
            if self._pump_active_latch[i]:
                if local_stop_ready and global_quiet_ready:
                    self._pump_active_latch[i] = False
                    self._pump_on_elapsed_s[i] = 0.0
                    self._pump_off_elapsed_s[i] = 0.0
                    self._pump_near_target_s[i] = 0.0
                elif local_stop_ready:
                    pump_quiet_stop_blocked[i] = True
                    if not self._pump_quiet_stop_blocked_prev[i]:
                        self._pump_quiet_stop_block_count += 1
            else:
                if abs_err > self.pump_restart_err_kg and self._pump_off_elapsed_s[i] >= self.pump_min_off_s:
                    self._pump_active_latch[i] = True
                    self._pump_on_elapsed_s[i] = 0.0
                    self._pump_off_elapsed_s[i] = 0.0
                    self._pump_near_target_s[i] = 0.0
                    self._pump_stage_dwell_s[i] = 0.0
            self._pump_quiet_stop_blocked_prev[i] = bool(pump_quiet_stop_blocked[i])
            if bool(self._pump_active_latch[i]) != latch_prev:
                pump_latch_switch_step[i] = True
                self._pump_latch_switch_count += 1

            if self._pump_active_latch[i]:
                # By default, latched pumps choose among positive-flow bands. The
                # actuator-smoothed baseline can opt into zero-rate latched dwell so
                # stage hysteresis does not keep pumping after the target is reached.
                target_rate_m3_min, stage_idx_i, stage_switch_i = self._get_pump_rate_with_stage_logic(
                    pump_idx=i,
                    abs_err_kg=abs_err,
                    dt=dt,
                    allow_zero_stage=bool(self.pump_stage_allow_zero_rate_latched),
                )
                released_rate_m3_min = self._release_rate_target(
                    pump_idx=i,
                    target_rate_m3_min=target_rate_m3_min,
                    dt=dt,
                )
            else:
                target_rate_m3_min = 0.0
                released_rate_m3_min = self._release_rate_target(
                    pump_idx=i,
                    target_rate_m3_min=0.0,
                    dt=dt,
                )
                stage_idx_i = self._find_stage_index(abs_err)
                stage_switch_i = False
            pump_rate_target_m3_min[i] = target_rate_m3_min
            pump_rate_released_m3_min[i] = released_rate_m3_min
            pump_stage_idx[i] = int(stage_idx_i)
            pump_stage_switch_step[i] = bool(stage_switch_i)
            rate_m3_min = self._ramp_towards_rate(
                current_rate=self._pump_rate_smoothed_m3_min[i],
                target_rate=released_rate_m3_min,
                ramp_up_m3_min_per_s=self.pump_ramp_up_m3_min_per_s,
                ramp_down_m3_min_per_s=self.pump_ramp_down_m3_min_per_s,
                dt=dt,
            )
            self._pump_rate_smoothed_m3_min[i] = rate_m3_min
            pump_rate_cmd_m3_min[i] = rate_m3_min
            pump_active_flags[i] = rate_m3_min > 1e-9
            if rate_m3_min <= 0.0 or abs_err <= 1e-4:
                continue

            max_delta_i = (rate_m3_min / 60.0) * self.rho * dt
            pump_rate_limited_flags[i] = abs_err > max_delta_i
            pump_saturated_flags[i] = pump_rate_limited_flags[i]
            pump_fullspeed_flags[i] = (rate_m3_min >= max_sched_rate_m3_min - 1e-9) and (rate_m3_min > 0.0)
            change = np.clip(err, -max_delta_i, max_delta_i)
            new_m = np.clip(self.current_ballast_mass[i] + change, 0, self.tank_capacity)
            if abs(new_m - self.current_ballast_mass[i]) > 1e-6:
                self.current_ballast_mass[i] = new_m
                mass_changed = True
        
        if mass_changed:
            self._update_mass_matrix()
            self._update_linear_damping()
        self._pump_prev_target_ballast_mass = self.target_ballast_mass.copy()
        pump_mass_delta_kg = self.current_ballast_mass - previous_mass
        if dt > 1e-12:
            pump_net_rate_m3_min = pump_mass_delta_kg * 60.0 / (self.rho * dt)
        else:
            pump_net_rate_m3_min = np.zeros(3, dtype=float)
        pump_inflow_m3_min = np.maximum(pump_net_rate_m3_min, 0.0)
        pump_outflow_m3_min = np.maximum(-pump_net_rate_m3_min, 0.0)
        ballast_total_delta_kg = float(np.sum(pump_mass_delta_kg))

        # 2. 刚体动力学 (RK4)
        y = self.state
        def f(t, s): return self._dynamics(t, s, thrust_N, wind_dir_deg)
        
        k1 = f(current_time, y)
        k2 = f(current_time + 0.5*dt, y + 0.5*dt*k1)
        k3 = f(current_time + 0.5*dt, y + 0.5*dt*k2)
        k4 = f(current_time + dt, y + dt*k3)
        self.state = y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # [Fix B] 增强诊断信息
        info = {
            "tank_masses": self.current_ballast_mass.copy(),
            "target_ballast_mass": self.target_ballast_mass.copy(),
            "ballast_err_kg": self.target_ballast_mass - self.current_ballast_mass,
            "pump_active": pump_active_flags,
            "pump_saturated": pump_saturated_flags,
            "pump_rate_limited": pump_rate_limited_flags,
            "pump_fullspeed": pump_fullspeed_flags,
            "pump_fullspeed_any": int(any(pump_fullspeed_flags)),
            "pump_backlog_kg": np.array(pump_backlog_kg, dtype=float),
            "pump_backlog_max_kg": float(np.max(np.asarray(pump_backlog_kg, dtype=float))),
            "pump_total_backlog_kg": float(total_backlog_kg),
            "pump_target_motion_kg_s": float(target_motion_kg_s),
            "pump_global_quiet_ready": int(global_quiet_ready),
            "pump_global_quiet_raw": int(global_quiet_raw),
            "pump_global_quiet_s": float(self._pump_global_quiet_s),
            "pump_quiet_stop_blocked": np.array(pump_quiet_stop_blocked, dtype=bool),
            "pump_quiet_stop_block_count": int(self._pump_quiet_stop_block_count),
            "pump_latch_switch_step": np.array(pump_latch_switch_step, dtype=bool),
            "pump_latch_switch_count": int(self._pump_latch_switch_count),
            "pump_delta": pump_mass_delta_kg.copy(),
            "tank_mass_delta_kg": pump_mass_delta_kg.copy(),
            "pump_rate_cmd_m3_min": np.array(pump_rate_cmd_m3_min),
            "pump_net_rate_m3_min": pump_net_rate_m3_min.copy(),
            "pump_inflow_m3_min": pump_inflow_m3_min.copy(),
            "pump_outflow_m3_min": pump_outflow_m3_min.copy(),
            "pump_rate_target_m3_min": np.array(pump_rate_target_m3_min, dtype=float),
            "pump_rate_released_m3_min": np.array(pump_rate_released_m3_min, dtype=float),
            "pump_rate_smoothed_m3_min": self._pump_rate_smoothed_m3_min.copy(),
            "pump_stage_idx": np.array(pump_stage_idx, dtype=int),
            "pump_stage_switch_step": np.array(pump_stage_switch_step, dtype=bool),
            "pump_stage_switch_count": int(self._pump_stage_switch_count),
            "pump_latched": self._pump_active_latch.copy(),
            "pump_on_elapsed_s": self._pump_on_elapsed_s.copy(),
            "pump_off_elapsed_s": self._pump_off_elapsed_s.copy(),
            "pump_near_target_s": self._pump_near_target_s.copy(),
            "ballast_total": np.sum(self.current_ballast_mass),
            "ballast_total_kg": float(np.sum(self.current_ballast_mass)),
            "ballast_total_delta_kg": ballast_total_delta_kg,
            "ballast_increment_kg": (
                self.current_ballast_mass - self.reference_ballast_mass
            ).copy(),
            "load_reference_mode": self.load_reference_mode,
            "mass_property_mode": self.mass_property_mode,
            "mooring_reference_mode": self.mooring_reference_mode,
            "platform_total_mass_kg": float(self.mass_properties.total_mass_kg),
            "platform_center_of_mass_m": np.array(
                self.mass_properties.center_of_mass_m,
                dtype=float,
                copy=True,
            ),
            "platform_inertia_diagonal_kg_m2": np.diag(
                self.mass_properties.inertia_about_reference_kg_m2
            ).copy(),
            "platform_inertia_about_reference_kg_m2": np.array(
                self.mass_properties.inertia_about_reference_kg_m2,
                dtype=float,
                copy=True,
            ),
            "platform_effective_mass_matrix": np.array(
                self.M_total,
                dtype=float,
                copy=True,
            ),
            "pitch_deg": np.degrees(self.state[4]),
            "roll_deg":  np.degrees(self.state[3]),
            "wind_thrust": thrust_N
        }
        return self.state, info

    def get_posture(self):
        """返回 [x, y, z, roll_deg, pitch_deg, yaw_deg]。"""
        p = self.state[0:6]
        return [p[0], p[1], p[2], np.degrees(p[3]), np.degrees(p[4]), np.degrees(p[5])]

    # ==========================================================================
    #  内部物理计算 (Internal Physics)
    # ==========================================================================

    @staticmethod
    def _skew(vector):
        x, y, z = np.asarray(vector, dtype=float)
        return np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=float,
        )

    @classmethod
    def _rigid_body_mass_matrix(cls, mass_properties):
        mass = float(mass_properties.total_mass_kg)
        center = np.asarray(mass_properties.center_of_mass_m, dtype=float)
        inertia = np.asarray(
            mass_properties.inertia_about_reference_kg_m2,
            dtype=float,
        )
        center_skew = cls._skew(center)
        matrix = np.zeros((6, 6), dtype=float)
        matrix[:3, :3] = mass * np.eye(3)
        matrix[:3, 3:] = -mass * center_skew
        matrix[3:, :3] = mass * center_skew
        matrix[3:, 3:] = inertia
        return matrix
    
    def _update_mass_matrix(self):
        if self.mass_property_mode == "reference_delta_point_mass":
            self.mass_properties = compute_incremental_ballast_mass_properties(
                reference_mass_kg=self.reference_mass_properties.total_mass_kg,
                reference_center_of_mass_m=(
                    self.reference_mass_properties.center_of_mass_m
                ),
                reference_inertia_about_reference_kg_m2=(
                    self.reference_mass_properties.inertia_about_reference_kg_m2
                ),
                tank_mass_deltas_kg=(
                    self.current_ballast_mass - self.reference_ballast_mass
                ),
                tank_coordinates_m=self.tank_pos,
            )
        else:
            dry_inertia_cog = np.diag(np.asarray(self.I_body, dtype=float))
            dry_offset = np.asarray(self.cog_dry, dtype=float)
            dry_inertia_reference = dry_inertia_cog + self.mass_dry * (
                float(np.dot(dry_offset, dry_offset)) * np.eye(3)
                - np.outer(dry_offset, dry_offset)
            )
            self.mass_properties = compute_ballast_mass_properties(
                dry_mass_kg=self.mass_dry,
                dry_center_of_mass_m=self.cog_dry,
                dry_inertia_about_reference_kg_m2=dry_inertia_reference,
                tank_masses_kg=self.current_ballast_mass,
                tank_coordinates_m=self.tank_pos,
            )
        total_m = float(self.mass_properties.total_mass_kg)
        if self.mass_property_mode == "reference_delta_point_mass":
            self.M_total = (
                self._rigid_body_mass_matrix(self.mass_properties)
                + self.reference_added_mass_matrix
            )
            if not np.allclose(self.M_total, self.M_total.T, atol=1e-8):
                raise ValueError("effective mass matrix must remain symmetric")
            if np.min(np.linalg.eigvalsh(self.M_total)) <= 0.0:
                raise ValueError("effective mass matrix must remain positive definite")
            rotational_inertia = np.diag(
                self.mass_properties.inertia_about_reference_kg_m2
            )
        elif self.mass_property_mode == "point_mass_diagonal":
            rotational_inertia = np.diag(
                self.mass_properties.inertia_about_reference_kg_m2
            )
        else:
            rotational_inertia = np.asarray(self.I_body, dtype=float)

        if self.mass_property_mode != "reference_delta_point_mass":
            self.M_total = np.diag([
                total_m * 1.6, total_m * 1.6, total_m * 2.0,
                rotational_inertia[0] * self.rot_inertia_multiplier_roll_pitch,
                rotational_inertia[1] * self.rot_inertia_multiplier_roll_pitch,
                rotational_inertia[2] * self.rot_inertia_multiplier_yaw,
            ])
        self.M_inv = np.linalg.inv(self.M_total)
        if self.load_reference_mode == "reference_incremental":
            restoring_mass = self.reference_total_mass
        else:
            restoring_mass = total_m
        vol_disp = restoring_mass / self.rho
        self.K_hydro[3] = self.rho * self.g * vol_disp * self.hydro_params['GM_T']
        self.K_hydro[4] = self.rho * self.g * vol_disp * self.hydro_params['GM_L']

    def _update_linear_damping(self):
        # 估算临界阻尼
        k_est = [self.K_mooring_lin[0], self.K_mooring_lin[1], self.K_hydro[2], 
                 self.K_hydro[3], self.K_hydro[4], self.K_yaw_stiffness]
        
        if self.use_nonlinear_mooring and hasattr(self, 'f_surge_raw'):
            try:
                val_1m = float(self._get_mooring_force(1.0, 0)) # Surge
                k_est[0] = abs(val_1m) / 1.0
                k_est[1] = k_est[0]
            except: pass
            
        for i in range(6):
            if k_est[i] > 0:
                self.C_lin[i,i] = 2 * self.zetas[i] * np.sqrt(self.M_total[i,i] * k_est[i])

    def _generalized_load_components(self, t, state, thrust, wind_dir):
        state_arr = np.asarray(state, dtype=float)
        if state_arr.shape != (12,) or not np.all(np.isfinite(state_arr)):
            raise ValueError("state must contain 12 finite values")
        scalar_inputs = np.asarray([t, thrust, wind_dir], dtype=float)
        if not np.all(np.isfinite(scalar_inputs)):
            raise ValueError("time, thrust and wind direction must be finite")

        pos = state_arr[0:6]
        vel = state_arr[6:12]
        wind_rad = np.radians(float(wind_dir))
        wind = np.array(
            [
                thrust * np.cos(wind_rad),
                thrust * np.sin(wind_rad),
                0.0,
                -thrust * np.sin(wind_rad) * self.arm_aero,
                thrust * np.cos(wind_rad) * self.arm_aero,
                0.0,
            ],
            dtype=float,
        )
        wave = np.asarray(self._get_wave_forces(float(t)), dtype=float)

        quadratic_drag = np.zeros(6, dtype=float)
        horizontal_speed = float(np.hypot(vel[0], vel[1]))
        if horizontal_speed > 1e-5:
            drag_scale = (
                0.5
                * self.rho
                * self.hydro_params["Cd"]
                * self.hydro_params["A_proj_surge"]
                * horizontal_speed
            )
            quadratic_drag[0] = -drag_scale * vel[0]
            quadratic_drag[1] = -drag_scale * vel[1]
        quadratic_drag[2] = (
            -0.5
            * self.rho
            * self.hydro_params["Cd"]
            * self.hydro_params["A_proj_heave"]
            * abs(vel[2])
            * vel[2]
        )

        mooring = np.zeros(6, dtype=float)
        force_x = self._get_mooring_force(pos[0], 0)
        force_y = self._get_mooring_force(pos[1], 1)
        mooring[0] = force_x
        mooring[1] = force_y
        mooring_moment = np.cross(self.r_fairlead, [force_x, force_y, 0.0])
        mooring[3] = mooring_moment[0]
        mooring[4] = mooring_moment[1]

        hydrostatic = np.zeros(6, dtype=float)
        hydrostatic[2] = -self.K_hydro[2] * pos[2]
        hydrostatic[3] = -self.K_hydro[3] * pos[3]
        hydrostatic[4] = -self.K_hydro[4] * pos[4]

        yaw_restoring = np.zeros(6, dtype=float)
        yaw_restoring[5] = -self.K_yaw_stiffness * pos[5]

        dry_gravity = np.zeros(6, dtype=float)
        if self.load_reference_mode == "legacy_mixed":
            dry_gravity_force = np.array([0.0, 0.0, -self.mass_dry * self.g])
            dry_gravity_moment = np.cross(self.cog_dry, dry_gravity_force)
            dry_gravity[3] = dry_gravity_moment[0]
            dry_gravity[4] = dry_gravity_moment[1]

        ballast_gravity = np.zeros(6, dtype=float)
        if self.load_reference_mode == "reference_incremental":
            ballast_load_mass = (
                self.current_ballast_mass - self.reference_ballast_mass
            )
        else:
            ballast_load_mass = self.current_ballast_mass
        ballast_gravity[2] = -float(np.sum(ballast_load_mass)) * self.g
        for tank_mass, tank_position in zip(
            ballast_load_mass,
            self.tank_pos,
        ):
            if abs(float(tank_mass)) <= 1e-12:
                continue
            tank_moment = np.cross(
                tank_position,
                [0.0, 0.0, -float(tank_mass) * self.g],
            )
            ballast_gravity[3] += tank_moment[0]
            ballast_gravity[4] += tank_moment[1]

        linear_damping = -self.C_lin @ vel
        components = {
            "wind": wind,
            "wave": wave,
            "quadratic_drag": quadratic_drag,
            "mooring": mooring,
            "hydrostatic": hydrostatic,
            "yaw_restoring": yaw_restoring,
            "dry_gravity": dry_gravity,
            "ballast_gravity": ballast_gravity,
            "linear_damping": linear_damping,
        }
        total = np.zeros(6, dtype=float)
        for values in components.values():
            total += values
        components["total"] = total
        return components

    def generalized_load_components(self, t, state, thrust, wind_dir):
        """Return the generalized-load terms used by the 6-DOF equations.

        The returned vectors follow ``[X, Y, Z, K, M, N]`` in the platform
        frame. This method is free of state mutation, and returned arrays are
        detached from the integrator's internal calculation.
        """

        components = self._generalized_load_components(t, state, thrust, wind_dir)
        return {
            name: np.array(values, dtype=float, copy=True)
            for name, values in components.items()
        }

    def _dynamics(self, t, state, thrust, wind_dir):
        loads = self._generalized_load_components(t, state, thrust, wind_dir)
        acc = self.M_inv @ loads["total"]
        return np.concatenate((np.asarray(state, dtype=float)[6:12], acc))

    def _get_mooring_force(self, disp, dof_idx):
        """[Fix C] 系泊力计算，增加 dof_idx 参数。"""
        if not self.use_nonlinear_mooring: 
            return -self.K_mooring_lin[dof_idx] * disp
        
        d_clamp = np.clip(disp, self.mooring_bounds['x'][0], self.mooring_bounds['x'][1])
        val = float(self.f_surge_raw(d_clamp))
        if self.mooring_reference_mode == "zero_at_reference":
            val -= self.mooring_reference_force_raw
        if self.mooring_mode == "TABLE_ABS":
            return (-1 if disp >=0 else 1) * abs(val)
        return self.mooring_sign * val

    def _load_mooring_data(self, path):
        try:
            df = pd.read_excel(path, engine='openpyxl')
            cols = [str(c).strip().lower() for c in df.columns]
            df.columns = cols
            cx = next((c for c in cols if any(k in c for k in ['offsetx','x-axis','x_disp'])), None)
            cf = next((c for c in cols if any(k in c for k in ['force','tension'])), None)
            
            if not cx or not cf:
                raise ValueError(
                    "mooring workbook must contain displacement and force columns; "
                    f"available columns: {cols}"
                )
            d = df[[cx, cf]].dropna().astype(float).sort_values(by=cx).drop_duplicates(subset=cx)
            if len(d) < 2:
                raise ValueError("mooring workbook must contain at least two valid rows")
            self.f_surge_raw = interp1d(d[cx], d[cf], bounds_error=False, fill_value=(d[cf].iloc[0], d[cf].iloc[-1]))
            self.mooring_reference_force_raw = float(self.f_surge_raw(0.0))
            self.mooring_bounds['x'] = [d[cx].min(), d[cx].max()]
            self.use_nonlinear_mooring = True
            self.mooring_source_path = str(os.path.realpath(path))

            v_pos = float(self.f_surge_raw(10.0))
            v_neg = float(self.f_surge_raw(-10.0))
            if v_pos * v_neg < 0:
                self.mooring_mode = "TABLE_SIGNED"
                self.mooring_sign = -1.0 if v_pos > 0 else 1.0
            else:
                self.mooring_mode = "TABLE_ABS"
                self.mooring_sign = -1.0
        except Exception as e:
            if self.allow_linear_mooring_fallback:
                print(f"!!! Mooring Load Error: {e}. Using Linear Fallback.")
                return
            raise ValueError(f"Invalid mooring stiffness workbook {path}: {e}") from e

    def _get_wave_forces(self, t):
        F = np.zeros(6)
        c, s = np.cos(self.wave_heading_rad), np.sin(self.wave_heading_rad)
        
        def rao(w, idx):
            cut = self.rao_params['cutoff_low'] if idx<2 else self.rao_params['cutoff_high']
            ord = 6 if idx<2 else 4
            return 1.0 / (1.0 + (w/cut)**ord)

        fx, fz, my = 0, 0, 0
        for wc in self.wave_components:
            w = wc['w']
            val = wc['a'] * np.cos(w * t + wc['phi'])
            fx += val * self.rao_params['surge_peak'] * rao(w, 0)
            fz += val * self.rao_params['heave_peak'] * rao(w, 2)
            my += val * self.rao_params['pitch_peak'] * rao(w, 4)
            
        F[0] = fx*c; F[1] = fx*s; F[2] = fz
        F[3] = -my*s; F[4] = my*c
        return F


# ==============================================================================
#  环境生成器 (Environment)
#  对应架构中的: core_model.py (Part 2)
# ==============================================================================


# MarkovWindGenerator has moved to wind_env.py; keep a compatibility re-export.
from wind_env import MarkovWindGenerator
