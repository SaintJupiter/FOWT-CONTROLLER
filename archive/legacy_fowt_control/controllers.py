import numpy as np


class BaseController:
    def compute(self, state, env_info, current_time):
        raise NotImplementedError

    def reset(self):
        pass


class MIMOController(BaseController):
    def __init__(
        self,
        kp_p,
        ki_p,
        kd_p,
        kp_r,
        ki_r,
        kd_r,
        kp_h,
        ki_h,
        kd_h,
        dt,
        max_capacity,
        baselines=None,
        setpoints={"pitch": 0.0, "roll": 0.0, "heave": 0.0},
        deadbands={"pitch": 0.5, "roll": 0.5, "heave": 0.05},
        deadband_exit_ratio=0.5,
        filter_tau=1.0,
        update_interval=5.0,
        tank_pos=None,
        k_wind_comp_pitch=0.0,
        k_wind_comp_roll=0.0,
    ):
        self.dt = float(dt)
        max_mass_arr = np.asarray(max_capacity, dtype=float)
        if max_mass_arr.ndim == 0:
            self.max_mass = np.full(3, float(max_mass_arr), dtype=float)
        elif max_mass_arr.size == 3:
            self.max_mass = max_mass_arr.astype(float)
        else:
            self.max_mass = np.full(3, float(np.max(max_mass_arr)), dtype=float)
        if baselines is not None:
            self.baselines = np.asarray(baselines, dtype=float)
        else:
            self.baselines = 0.5 * self.max_mass.copy()

        self.gains = {
            "pitch": (float(kp_p), float(ki_p), float(kd_p)),
            "roll": (float(kp_r), float(ki_r), float(kd_r)),
            "heave": (float(kp_h), float(ki_h), float(kd_h)),
        }
        self.setpoints = dict(setpoints)
        self.deadbands = dict(deadbands)
        self.deadband_enter = dict(deadbands)
        self.deadband_exit = {
            k: max(0.0, float(v) * float(deadband_exit_ratio))
            for k, v in deadbands.items()
        }
        self.deadband_state = {"pitch": False, "roll": False, "heave": False}
        self.ff_gains = {
            "pitch": float(k_wind_comp_pitch),
            "roll": float(k_wind_comp_roll),
        }
        self.update_interval = float(update_interval)
        self.filter_tau = float(filter_tau)

        self.integrals = {"pitch": 0.0, "roll": 0.0, "heave": 0.0}
        self.prev_errors = {"pitch": 0.0, "roll": 0.0, "heave": 0.0}
        self.filtered_state = {"pitch": None, "roll": None, "heave": None}

        # Allocation and sign sensitivity are built via explicit helpers so the
        # geometry logic stays centralized and testable.
        self.B_alloc = self._build_allocation_matrix(tank_pos)
        self.axis_sensitivity = self._compute_axis_sensitivity(tank_pos)

        # Internal controller state (split semantics).
        self._initialized = False
        self._cached_output = tuple(self.baselines.tolist())
        self._last_compute_time = None
        self._last_filter_time = None

        # Legacy-compatible fields used by existing diagnostics.
        self.last_update_time = -999.0
        self.last_output_cmds = self._cached_output
        self.last_debug_info = {}

        print(f">>> [MIMO Init] Pitch Kp={kp_p}, Roll Kp={kp_r}, Heave Kp={kp_h}")

    def _build_allocation_matrix(self, tank_pos):
        default_alloc = np.array(
            [
                [-1.0, 0.0, 1.0],
                [0.5, 1.0, 1.0],
                [0.5, -1.0, 1.0],
            ],
            dtype=float,
        )
        if tank_pos is None:
            return default_alloc

        tp = np.asarray(tank_pos, dtype=float)
        if tp.shape != (3, 3):
            return default_alloc

        x = tp[:, 0]
        y = tp[:, 1]
        x_norm = np.max(np.abs(x)) if np.max(np.abs(x)) > 1e-9 else 1.0
        y_norm = np.max(np.abs(y)) if np.max(np.abs(y)) > 1e-9 else 1.0
        # Coordinate convention: +x front, +y port(left), +z up.
        # Desired signs: +pitch=bow-down, +roll=starboard-down.
        pitch_col = x / x_norm
        roll_col = -y / y_norm
        heave_col = np.ones(3, dtype=float)
        return np.column_stack([pitch_col, roll_col, heave_col])

    def _compute_axis_sensitivity(self, tank_pos):
        if tank_pos is None:
            return {"pitch": 1.0, "roll": 1.0}
        tp = np.asarray(tank_pos, dtype=float)
        if tp.shape != (3, 3):
            return {"pitch": 1.0, "roll": 1.0}
        x_arr = tp[:, 0]
        y_arr = tp[:, 1]
        return {
            "pitch": float(np.dot(x_arr, self.B_alloc[:, 0])),
            "roll": float(np.dot(-y_arr, self.B_alloc[:, 1])),
        }

    def _extract_raw_vals(self, state):
        return {
            "heave": float(state[2]),
            "roll": float(np.degrees(state[3])),
            "pitch": float(np.degrees(state[4])),
        }

    def _update_filter(self, key, raw_val, alpha):
        if self.filtered_state[key] is None:
            self.filtered_state[key] = raw_val
        else:
            self.filtered_state[key] = alpha * raw_val + (1.0 - alpha) * self.filtered_state[key]
        return self.filtered_state[key]

    def _update_filtered_vals(self, raw_vals, elapsed):
        alpha = elapsed / (self.filter_tau + elapsed)
        curr_vals = {}
        for k in raw_vals:
            curr_vals[k] = self._update_filter(k, raw_vals[k], alpha)
        return curr_vals, float(alpha)

    def compute(self, state, env_info, current_time):
        raw_vals = self._extract_raw_vals(state)

        if self._last_filter_time is not None:
            filter_dt = max(float(current_time) - float(self._last_filter_time), self.dt)
        else:
            filter_dt = self.dt
        curr_vals, filter_alpha = self._update_filtered_vals(raw_vals, filter_dt)
        self._last_filter_time = float(current_time)

        if self._initialized and self._last_compute_time is not None:
            if float(current_time) - float(self._last_compute_time) < self.update_interval:
                info = self.last_debug_info.copy()
                info.update(
                    {
                        "status": "hold",
                        "raw_vals": raw_vals,
                        "filtered_vals": curr_vals,
                        "filter_alpha": float(filter_alpha),
                        "effective_dt": float(filter_dt),
                    }
                )
                return self._cached_output, info

        if self._initialized and self._last_compute_time is not None:
            effective_dt = max(float(current_time) - float(self._last_compute_time), self.dt)
        else:
            effective_dt = self.dt

        u_vec_force = np.zeros(3)
        u_vec_total = np.zeros(3)
        debug_pid = {}
        keys = ["pitch", "roll", "heave"]
        pid_components = {}

        for i, k in enumerate(keys):
            kp, ki, kd = self.gains[k]
            # Standard negative feedback: error = setpoint - measurement.
            error_raw = self.setpoints[k] - curr_vals[k]

            enter_th = self.deadband_enter[k]
            exit_th = self.deadband_exit[k]
            if self.deadband_state[k]:
                if abs(error_raw) >= enter_th:
                    self.deadband_state[k] = False
            else:
                if abs(error_raw) <= exit_th:
                    self.deadband_state[k] = True
            in_deadband = self.deadband_state[k]

            if in_deadband:
                error = 0.0
            else:
                error = np.sign(error_raw) * max(0.0, abs(error_raw) - exit_th)

            p_term = kp * error
            if in_deadband:
                d_term = 0.0
                self.prev_errors[k] = 0.0
            else:
                derivative = (error - self.prev_errors[k]) / effective_dt
                d_term = kd * derivative
                self.prev_errors[k] = error

            ff_term = 0.0
            ws = float(env_info.get("wind_speed", 0.0))
            wd_rad = np.radians(float(env_info.get("wind_dir_deg", 0.0)))
            if k == "pitch" and self.ff_gains["pitch"] != 0.0:
                ff_term = -self.ff_gains["pitch"] * (ws**2) * np.cos(wd_rad)
            elif k == "roll" and self.ff_gains["roll"] != 0.0:
                ff_term = self.ff_gains["roll"] * (ws**2) * np.sin(wd_rad)

            i_term_old = ki * self.integrals[k]
            u_force = p_term + d_term + ff_term
            u_vec_force[i] = u_force

            pid_components[k] = {
                "error_raw": error_raw,
                "error": error,
                "p": p_term,
                "d": d_term,
                "ff": ff_term,
                "i_old": i_term_old,
                "u_force": u_force,
                "in_deadband": in_deadband,
                "deadband_enter": enter_th,
                "deadband_exit": exit_th,
                "ki": ki,
                "saturated": False,
                "int_update": 0.0,
            }

        m_deltas_force = self.B_alloc @ u_vec_force
        m_pred_force = self.baselines + m_deltas_force

        is_saturated = np.zeros(3, dtype=bool)
        for t_idx in range(3):
            if m_pred_force[t_idx] < 0.0 or m_pred_force[t_idx] > self.max_mass[t_idx]:
                is_saturated[t_idx] = True

        for i, k in enumerate(keys):
            comp = pid_components[k]
            if comp["in_deadband"]:
                continue

            stop_integration = False
            if np.any(is_saturated):
                for t_idx in range(3):
                    if is_saturated[t_idx]:
                        contribution = self.B_alloc[t_idx, i] * comp["u_force"]
                        if m_pred_force[t_idx] < 0.0 and contribution < 0.0:
                            stop_integration = True
                        if m_pred_force[t_idx] > self.max_mass[t_idx] and contribution > 0.0:
                            stop_integration = True

            if not stop_integration:
                int_update = comp["error"] * effective_dt
                self.integrals[k] += int_update
                if comp["ki"] > 1e-6:
                    limit = 0.8 * np.max(self.max_mass) / comp["ki"]
                    self.integrals[k] = np.clip(self.integrals[k], -limit, limit)
                comp["int_update"] = int_update

            pid_components[k]["saturated"] = stop_integration

        for i, k in enumerate(keys):
            comp = pid_components[k]
            i_term_final = comp["ki"] * self.integrals[k]
            u_total = comp["u_force"] + i_term_final
            u_vec_total[i] = u_total
            debug_pid[k] = {
                "error_raw": comp["error_raw"],
                "error": comp["error"],
                "p_term": comp["p"],
                "i_term": i_term_final,
                "d_term": comp["d"],
                "ff_term": comp["ff"],
                "total_out": u_total,
                "in_deadband": 1 if comp["in_deadband"] else 0,
                "deadband_enter": comp["deadband_enter"],
                "deadband_exit": comp["deadband_exit"],
                "saturated": 1 if comp["saturated"] else 0,
            }

        m_deltas_final = self.B_alloc @ u_vec_total
        m_raw = self.baselines + m_deltas_final
        m_cmds = np.clip(m_raw, 0.0, self.max_mass)
        is_clipped = np.any(np.abs(m_cmds - m_raw) > 1e-3)
        clipped_low = m_raw < 0.0
        clipped_high = m_raw > self.max_mass

        rollback_axes = [False, False, False]
        if is_clipped:
            for i, k in enumerate(keys):
                comp = pid_components[k]
                if abs(comp["int_update"]) <= 1e-12:
                    continue
                axis_pushes_clip = False
                for t_idx in range(3):
                    contribution = self.B_alloc[t_idx, i] * u_vec_total[i]
                    if clipped_low[t_idx] and contribution < 0.0:
                        axis_pushes_clip = True
                    if clipped_high[t_idx] and contribution > 0.0:
                        axis_pushes_clip = True
                if axis_pushes_clip:
                    rollback_axes[i] = True

            if any(rollback_axes):
                for i, k in enumerate(keys):
                    if rollback_axes[i]:
                        self.integrals[k] -= pid_components[k]["int_update"]
                        pid_components[k]["saturated"] = True

                for i, k in enumerate(keys):
                    comp = pid_components[k]
                    i_term_final = comp["ki"] * self.integrals[k]
                    u_total = comp["u_force"] + i_term_final
                    u_vec_total[i] = u_total
                    debug_pid[k] = {
                        "error_raw": comp["error_raw"],
                        "error": comp["error"],
                        "p_term": comp["p"],
                        "i_term": i_term_final,
                        "d_term": comp["d"],
                        "ff_term": comp["ff"],
                        "total_out": u_total,
                        "in_deadband": 1 if comp["in_deadband"] else 0,
                        "deadband_enter": comp["deadband_enter"],
                        "deadband_exit": comp["deadband_exit"],
                        "saturated": 1 if comp["saturated"] else 0,
                    }
                m_deltas_final = self.B_alloc @ u_vec_total
                m_raw = self.baselines + m_deltas_final
                m_cmds = np.clip(m_raw, 0.0, self.max_mass)
                is_clipped = np.any(np.abs(m_cmds - m_raw) > 1e-3)

        axis_warn = {"pitch": 0, "roll": 0}
        axis_err = {
            "pitch": self.setpoints["pitch"] - curr_vals["pitch"],
            "roll": self.setpoints["roll"] - curr_vals["roll"],
        }
        axis_u = {"pitch": float(u_vec_total[0]), "roll": float(u_vec_total[1])}
        for axis in ["pitch", "roll"]:
            err = float(axis_err[axis])
            sens = float(self.axis_sensitivity.get(axis, 1.0))
            effort = float(axis_u[axis])
            if abs(err) > self.deadband_enter[axis]:
                axis_warn[axis] = 1 if (err * sens * effort) < 0.0 else 0
        sign_check_warn = 1 if (axis_warn["pitch"] or axis_warn["roll"]) else 0

        self._last_compute_time = float(current_time)
        self._cached_output = tuple(m_cmds.tolist())
        self._initialized = True
        self.last_update_time = float(current_time)
        self.last_output_cmds = self._cached_output

        self.last_debug_info = {
            "status": "update",
            "tanks": m_cmds,
            "clipped": 1 if is_clipped else 0,
            "alloc_m_deltas_kg": m_deltas_final.copy(),
            "alloc_m_raw_kg": m_raw.copy(),
            "alloc_m_cmds_kg": m_cmds.copy(),
            "sign_check_warn": sign_check_warn,
            "sign_check_warn_pitch": int(axis_warn["pitch"]),
            "sign_check_warn_roll": int(axis_warn["roll"]),
            "raw_vals": raw_vals,
            "filtered_vals": curr_vals,
            "filter_alpha": float(filter_alpha),
            "effective_dt": float(effective_dt),
            "deadband_state": self.deadband_state.copy(),
            "pid_details": debug_pid,
        }
        return self._cached_output, self.last_debug_info

    def reset(self):
        for k in self.integrals:
            self.integrals[k] = 0.0
        for k in self.prev_errors:
            self.prev_errors[k] = 0.0
        for k in self.filtered_state:
            self.filtered_state[k] = None
        for k in self.deadband_state:
            self.deadband_state[k] = False

        self._initialized = False
        self._cached_output = None
        self._last_compute_time = None
        self._last_filter_time = None

        self.last_update_time = -999.0
        self.last_output_cmds = None
        self.last_debug_info = {}

