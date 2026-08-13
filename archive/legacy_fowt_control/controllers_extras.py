import csv
from pathlib import Path

import numpy as np

from target_execution import describe_target_path
from trim_mapping import build_regime_cfg, compute_trim_setpoints as compute_trim_setpoints_shared


def compute_trim_setpoints(
    ws_mps,
    wd_deg,
    k_trim_deg_per_mps=0.06,
    max_trim_deg=2.0,
    trim_map_version=None,
    regime_cfg=None,
    return_meta=False,
):
    # Keep this wrapper for backward import compatibility; mapping logic is centralized
    # in trim_mapping.py and shared by runtime + offline analysis entries.
    return compute_trim_setpoints_shared(
        ws_mps=ws_mps,
        wd_deg=wd_deg,
        k_trim_deg_per_mps=k_trim_deg_per_mps,
        max_trim_deg=max_trim_deg,
        trim_map_version=trim_map_version,
        regime_cfg=regime_cfg,
        return_meta=return_meta,
    )


class SetpointShaper:
    def __init__(self, dt, alpha=0.98, rate_deg_s=0.02, enabled=True, initial=(0.0, 0.0)):
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.rate_deg_s = float(rate_deg_s)
        self.enabled = bool(enabled)
        self._prev = np.array(initial, dtype=float)

    def reset(self, initial=None):
        if initial is not None:
            self._prev = np.array(initial, dtype=float)

    def update(self, pitch_sp_raw, roll_sp_raw):
        raw = np.array([pitch_sp_raw, roll_sp_raw], dtype=float)
        if not self.enabled:
            self._prev = raw
            return float(raw[0]), float(raw[1])

        lpf = self.alpha * self._prev + (1.0 - self.alpha) * raw
        max_delta = self.rate_deg_s * self.dt
        delta = np.clip(lpf - self._prev, -max_delta, max_delta)
        cmd = self._prev + delta
        self._prev = cmd
        return float(cmd[0]), float(cmd[1])




def _zero_preview_trim_debug(source="none"):
    return {
        "preview_trim_active": 0,
        "preview_trim_source": str(source),
        "preview_pitch_bias_deg": 0.0,
        "preview_roll_bias_deg": 0.0,
    }


def evaluate_forecast_safe_deadband(
    preview,
    state,
    *,
    enabled,
    event_probability_max,
    future_rise_max,
    minimum_direction_dot,
    posture_gate_deg,
    allowed_actions=("hold",),
    gate_mode="conservative",
    require_event_probability=False,
):
    """Decide whether a forecast may admit the pump-saving attitude deadband."""
    preview = preview if isinstance(preview, dict) else {}
    state = np.asarray(state, dtype=float).reshape(-1)
    pitch_abs_deg = abs(float(np.degrees(state[4]))) if state.size > 4 else np.inf
    roll_abs_deg = abs(float(np.degrees(state[3]))) if state.size > 3 else np.inf
    posture_abs_deg = max(pitch_abs_deg, roll_abs_deg)

    pressure0 = float(
        preview.get(
            "preview_raw_pressure_block0_norm",
            preview.get("preview_pressure_block0_norm", 0.0),
        )
    )
    pressure1 = float(
        preview.get(
            "preview_raw_pressure_block1_norm",
            preview.get("preview_pressure_block1_norm", 0.0),
        )
    )
    pressure2 = float(
        preview.get(
            "preview_raw_pressure_block2_norm",
            preview.get("preview_pressure_block2_norm", 0.0),
        )
    )
    future_rise = max(pressure1, pressure2) - pressure0
    direction_dot = float(
        preview.get(
            "preview_raw_pressure_block02_dot",
            preview.get("preview_pressure_block02_dot", 0.0),
        )
    )
    event_probability = float(
        preview.get(
            "preview_forecast_control_trust_event_prob_max",
            max(
                float(preview.get("preview_event_risk_prob_0_20m", 0.0)),
                float(preview.get("preview_event_risk_prob_20_40m", 0.0)),
                float(preview.get("preview_event_risk_prob_40_60m", 0.0)),
            ),
        )
    )
    event_probability_available = bool(
        int(preview.get("preview_forecast_event_probs_available", 0))
    )
    action = str(preview.get("preview_primary_action", ""))

    gate_mode = str(gate_mode).strip().lower()
    if gate_mode not in ("conservative", "forecast_veto"):
        raise ValueError(f"unsupported forecast-safe deadband gate mode: {gate_mode}")

    reason = "forecast_safe_deadband"
    active = True
    if not enabled:
        active, reason = False, "disabled"
    elif not bool(int(preview.get("preview_forecast_has_future", 0))):
        active, reason = False, "no_future_forecast"
    elif bool(require_event_probability) and not event_probability_available:
        active, reason = False, "event_probability_unavailable"
    elif not bool(int(preview.get("preview_forecast_control_trust_ok", 0))):
        active, reason = False, "forecast_untrusted"
    elif event_probability > float(event_probability_max):
        active, reason = False, "event_risk_high"
    elif posture_abs_deg > float(posture_gate_deg):
        active, reason = False, "posture_gate"
    elif gate_mode == "conservative" and action not in tuple(
        str(value) for value in allowed_actions
    ):
        active, reason = False, "action_not_eligible"
    elif gate_mode == "conservative" and future_rise > float(future_rise_max):
        active, reason = False, "future_pressure_rising"
    elif gate_mode == "conservative" and direction_dot < float(minimum_direction_dot):
        active, reason = False, "direction_reversal"

    return {
        "active": bool(active),
        "reason": str(reason),
        "gate_mode": gate_mode,
        "has_future": int(bool(int(preview.get("preview_forecast_has_future", 0)))),
        "trust_ok": int(bool(int(preview.get("preview_forecast_control_trust_ok", 0)))),
        "action": action,
        "event_probability": event_probability,
        "event_probability_available": int(event_probability_available),
        "future_rise": float(future_rise),
        "direction_dot": direction_dot,
        "pitch_abs_deg": pitch_abs_deg,
        "roll_abs_deg": roll_abs_deg,
        "posture_abs_deg": posture_abs_deg,
    }


def _coerce_preview_trim_bias(preview_trim_bias):
    if preview_trim_bias is None:
        return (0.0, 0.0), _zero_preview_trim_debug()

    if isinstance(preview_trim_bias, dict):
        pitch = preview_trim_bias.get(
            "pitch_bias_deg",
            preview_trim_bias.get("preview_pitch_bias_deg", 0.0),
        )
        roll = preview_trim_bias.get(
            "roll_bias_deg",
            preview_trim_bias.get("preview_roll_bias_deg", 0.0),
        )
        source = preview_trim_bias.get("source", preview_trim_bias.get("preview_trim_source", "external"))
        debug = {
            "preview_trim_active": int(abs(float(pitch)) > 1e-12 or abs(float(roll)) > 1e-12),
            "preview_trim_source": str(source),
            "preview_pitch_bias_deg": float(pitch),
            "preview_roll_bias_deg": float(roll),
        }
        for key, value in preview_trim_bias.items():
            if key not in debug:
                debug[key] = value
        return (float(pitch), float(roll)), debug

    arr = np.asarray(preview_trim_bias, dtype=float).reshape(-1)
    if arr.size < 2:
        return (0.0, 0.0), _zero_preview_trim_debug(source="invalid")
    pitch = float(arr[0])
    roll = float(arr[1])
    return (pitch, roll), {
        "preview_trim_active": int(abs(pitch) > 1e-12 or abs(roll) > 1e-12),
        "preview_trim_source": "external",
        "preview_pitch_bias_deg": pitch,
        "preview_roll_bias_deg": roll,
    }


class _TargetTableLookup:
    REQUIRED_COLS = {"state_id", "pitch_sp_target_deg", "roll_sp_target_deg"}

    def __init__(self, table_path=None, strict=False):
        self.table_path = str(table_path or "").strip()
        self.strict = bool(strict)
        self._map = {}
        self.available = False
        self.unavailable_reason = "missing_table"
        self.lookup_total = 0
        self.lookup_hit = 0
        self.fallback_reasons = {
            "missing_state": 0,
            "missing_table": 0,
            "missing_columns": 0,
            "invalid_state_id": 0,
        }
        self._load()

    def _raise_or_flag(self, exc_type, msg, reason):
        if self.strict:
            raise exc_type(msg)
        self.available = False
        self.unavailable_reason = str(reason)

    def _load(self):
        if not self.table_path:
            self.available = False
            self.unavailable_reason = "missing_table"
            return

        p = Path(self.table_path)
        if not p.exists():
            self._raise_or_flag(FileNotFoundError, f"target table not found: {self.table_path}", "missing_table")
            return

        with p.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                self._raise_or_flag(ValueError, f"target table has no header: {self.table_path}", "missing_columns")
                return
            fields = {str(c).strip() for c in reader.fieldnames}
            miss = self.REQUIRED_COLS - fields
            if miss:
                self._raise_or_flag(
                    ValueError,
                    f"target table missing required columns: {sorted(miss)} ({self.table_path})",
                    "missing_columns",
                )
                return

            m = {}
            for row in reader:
                try:
                    sid = int(float(row["state_id"]))
                    pitch = float(row["pitch_sp_target_deg"])
                    roll = float(row["roll_sp_target_deg"])
                except Exception:
                    continue
                m[sid] = (pitch, roll)

        if not m:
            self._raise_or_flag(ValueError, f"target table has no valid rows: {self.table_path}", "missing_table")
            return

        self._map = m
        self.available = True
        self.unavailable_reason = ""

    def _record_fallback(self, reason):
        r = str(reason)
        if r not in self.fallback_reasons:
            self.fallback_reasons[r] = 0
        self.fallback_reasons[r] += 1

    def lookup(self, wind_obs):
        self.lookup_total += 1

        if not self.available:
            reason = self.unavailable_reason or "missing_table"
            self._record_fallback(reason)
            return False, np.nan, np.nan, reason

        if wind_obs is None:
            self._record_fallback("missing_state")
            return False, np.nan, np.nan, "missing_state"

        sid_raw = wind_obs.get("state_id", None)
        if sid_raw is None:
            self._record_fallback("missing_state")
            return False, np.nan, np.nan, "missing_state"

        try:
            if isinstance(sid_raw, float) and not np.isfinite(sid_raw):
                raise ValueError("non-finite")
            sid = int(sid_raw)
        except Exception:
            self._record_fallback("invalid_state_id")
            return False, np.nan, np.nan, "invalid_state_id"

        if sid not in self._map:
            self._record_fallback("missing_state")
            return False, np.nan, np.nan, "missing_state"

        self.lookup_hit += 1
        p, r = self._map[sid]
        return True, float(p), float(r), ""

    def stats(self):
        fallback = int(max(self.lookup_total - self.lookup_hit, 0))
        return {
            "target_lookup_total": int(self.lookup_total),
            "target_lookup_hit": int(self.lookup_hit),
            "target_lookup_fallback": int(fallback),
            "fallback_reason_missing_state": int(self.fallback_reasons.get("missing_state", 0)),
            "fallback_reason_missing_table": int(self.fallback_reasons.get("missing_table", 0)),
            "fallback_reason_missing_columns": int(self.fallback_reasons.get("missing_columns", 0)),
            "fallback_reason_invalid_state_id": int(self.fallback_reasons.get("invalid_state_id", 0)),
        }


class TrimGovernor:
    def __init__(
        self,
        dt,
        k_trim_deg_per_mps=0.03,
        max_trim_deg=1.0,
        sat_gate_on=0.95,
        sat_gate_off=0.85,
        cmd_gap_decay_start_kg=5000.0,
        cmd_gap_decay_full_kg=12000.0,
        sat_window_s=90.0,
        cmd_gap_window_s=180.0,
        trim_min_scale=0.0,
        trim_decay_per_step=0.995,
        trim_recover_per_step=0.05,
        wind_lpf_tau_s=60.0,
        steady_window_s=45.0,
        steady_mean_on_deg=0.25,
        steady_mean_off_deg=0.12,
        steady_std_on_deg=0.20,
        steady_std_off_deg=0.35,
        trim_update_s=10.0,
        pressure_backlog_start_kg=600.0,
        pressure_backlog_full_kg=2000.0,
        trim_logic="legacy",
        simple_decay_on_clip=0.97,
        simple_recover_per_step=0.05,
        simple_pressure_hold_th=0.25,
        simple_pressure_decay_th=0.55,
        simple_freeze_enabled=False,
        simple_pressure_freeze_th=0.75,
        simple_pressure_unfreeze_th=0.55,
        trim_map_version="v1",
        ws_rated_mps=11.5,
        ws_cutout_mps=25.0,
        post_rated_mode="plateau",
        post_rated_decay_ratio=0.8,
        protect_trim_scale=0.0,
        target_mode="g1",
        target_table_path="",
        target_lookup_key="state_id",
        target_table_strict=False,
        preview_scale_mode="unscaled",
        preview_scale=1.0,
    ):
        self.dt = float(dt)
        self.k_trim_deg_per_mps = float(k_trim_deg_per_mps)
        self.max_trim_deg = float(max_trim_deg)
        self.sat_gate_on = float(sat_gate_on)
        self.sat_gate_off = float(sat_gate_off)
        self.cmd_gap_decay_start_kg = float(cmd_gap_decay_start_kg)
        self.cmd_gap_decay_full_kg = float(cmd_gap_decay_full_kg)
        self.sat_window_s = float(sat_window_s)
        self.cmd_gap_window_s = float(cmd_gap_window_s)
        self.trim_min_scale = float(trim_min_scale)
        self.trim_decay_per_step = float(trim_decay_per_step)
        self.trim_recover_per_step = float(trim_recover_per_step)
        self.wind_lpf_tau_s = float(wind_lpf_tau_s)
        self.steady_window_s = float(steady_window_s)
        self.steady_mean_on_deg = float(steady_mean_on_deg)
        self.steady_mean_off_deg = float(steady_mean_off_deg)
        self.steady_std_on_deg = float(steady_std_on_deg)
        self.steady_std_off_deg = float(steady_std_off_deg)
        self.trim_update_steps = max(1, int(round(float(trim_update_s) / self.dt)))
        self.pressure_backlog_start_kg = float(pressure_backlog_start_kg)
        self.pressure_backlog_full_kg = float(pressure_backlog_full_kg)
        self.trim_logic = str(trim_logic).strip().lower() or "legacy"
        self.simple_decay_on_clip = float(simple_decay_on_clip)
        self.simple_recover_per_step = float(simple_recover_per_step)
        self.simple_pressure_hold_th = float(simple_pressure_hold_th)
        self.simple_pressure_decay_th = float(simple_pressure_decay_th)
        self.simple_freeze_enabled = bool(simple_freeze_enabled)
        self.simple_pressure_freeze_th = float(simple_pressure_freeze_th)
        self.simple_pressure_unfreeze_th = float(simple_pressure_unfreeze_th)
        self.trim_map_version = str(trim_map_version)
        self.trim_regime_cfg = build_regime_cfg(
            ws_rated_mps=ws_rated_mps,
            ws_cutout_mps=ws_cutout_mps,
            post_rated_mode=post_rated_mode,
            post_rated_decay_ratio=post_rated_decay_ratio,
            protect_trim_scale=protect_trim_scale,
        )

        self.target_mode = str(target_mode).strip().lower() or "g1"
        if self.target_mode not in ("g1", "g2"):
            if bool(target_table_strict):
                raise ValueError(f"unsupported target_mode: {self.target_mode}")
            self.target_mode = "g1"

        self.target_lookup_key = str(target_lookup_key).strip().lower() or "state_id"
        self.target_table_path = str(target_table_path or "")
        self.target_table_strict = bool(target_table_strict)
        self.preview_scale_mode = str(preview_scale_mode).strip().lower() or "unscaled"
        if self.preview_scale_mode not in ("guarded", "separate", "unscaled"):
            if bool(target_table_strict):
                raise ValueError(f"unsupported preview_scale_mode: {self.preview_scale_mode}")
            self.preview_scale_mode = "unscaled"
        self.preview_scale = float(preview_scale)
        if self.target_lookup_key != "state_id":
            if self.target_table_strict:
                raise ValueError(f"unsupported target_lookup_key: {self.target_lookup_key}")
            self.target_lookup_key = "state_id"

        self._target_lookup = None
        if self.target_mode == "g2":
            self._target_lookup = _TargetTableLookup(
                table_path=self.target_table_path,
                strict=self.target_table_strict,
            )

        self.target_source_last = "g1_formula"
        self.target_fallback_reason_last = ""

        self.sat_hist = []
        self.cmd_gap_hist = []
        self.pressure_clip_hist = []
        self.pressure_cmd_gap_hist = []
        self.pressure_fullspeed_hist = []
        self.pressure_backlog_hist = []
        self.err_hist = []
        self.trim_scale_prev = 1.0
        self.trim_enabled = False
        self.ws_trim_lpf = 0.0
        self.sat_recent_ratio = 0.0
        self.cmd_gap_recent = 0.0
        self.steady_mean_abs = 0.0
        self.steady_std = 0.0
        self._trim_hold = np.array([0.0, 0.0], dtype=float)
        self._step_idx = 0
        self.trim_pressure_recent = 0.0
        self.trim_pressure_clip_recent = 0.0
        self.trim_pressure_cmd_gap_recent = 0.0
        self.trim_pressure_fullspeed_recent = 0.0
        self.trim_pressure_backlog_recent = 0.0
        self.trim_pressure_sample = 0.0
        self.trim_pressure_backlog_max_kg = 0.0
        self.trim_pressure_cmd_gap_kg = 0.0
        self.trim_pressure_clip = 0
        self.trim_pressure_fullspeed = 0
        self.simple_freeze_active = False
        self._current_trim_raw = np.array([0.0, 0.0], dtype=float)
        self._preview_trim_bias = np.array([0.0, 0.0], dtype=float)
        self._combined_trim_raw = np.array([0.0, 0.0], dtype=float)
        self._current_trim_scaled = np.array([0.0, 0.0], dtype=float)
        self._preview_trim_scaled = np.array([0.0, 0.0], dtype=float)
        self._preview_trim_debug = _zero_preview_trim_debug()

    def reset(self, initial_ws=0.0):
        self.sat_hist = []
        self.cmd_gap_hist = []
        self.pressure_clip_hist = []
        self.pressure_cmd_gap_hist = []
        self.pressure_fullspeed_hist = []
        self.pressure_backlog_hist = []
        self.err_hist = []
        self.trim_scale_prev = 1.0
        self.trim_enabled = False
        self.ws_trim_lpf = float(initial_ws)
        self.sat_recent_ratio = 0.0
        self.cmd_gap_recent = 0.0
        self.steady_mean_abs = 0.0
        self.steady_std = 0.0
        self._trim_hold = np.array([0.0, 0.0], dtype=float)
        self._step_idx = 0
        self.trim_pressure_recent = 0.0
        self.trim_pressure_clip_recent = 0.0
        self.trim_pressure_cmd_gap_recent = 0.0
        self.trim_pressure_fullspeed_recent = 0.0
        self.trim_pressure_backlog_recent = 0.0
        self.trim_pressure_sample = 0.0
        self.trim_pressure_backlog_max_kg = 0.0
        self.trim_pressure_cmd_gap_kg = 0.0
        self.trim_pressure_clip = 0
        self.trim_pressure_fullspeed = 0
        self.simple_freeze_active = False
        self._current_trim_raw = np.array([0.0, 0.0], dtype=float)
        self._preview_trim_bias = np.array([0.0, 0.0], dtype=float)
        self._combined_trim_raw = np.array([0.0, 0.0], dtype=float)
        self._current_trim_scaled = np.array([0.0, 0.0], dtype=float)
        self._preview_trim_scaled = np.array([0.0, 0.0], dtype=float)
        self._preview_trim_debug = _zero_preview_trim_debug()
        self.target_source_last = "g1_formula"
        self.target_fallback_reason_last = ""

    @staticmethod
    def _linear_scale(value, start, full):
        if value <= start:
            return 0.0
        if value >= full:
            return 1.0
        return float((value - start) / max(full - start, 1e-6))

    def _compute_pressure_sample(self, plant_info_prev):
        cmd_gap = 0.0
        backlog_max = 0.0
        clip_like = 0
        fullspeed_any = 0

        if plant_info_prev is not None:
            cmd_gap = float(plant_info_prev.get("cmd_gap_kg", 0.0))
            # Prefer controller-base clip as the main semantic signal.
            if "ctrl_base_clipped" in plant_info_prev:
                clip_like = int(bool(plant_info_prev.get("ctrl_base_clipped", 0)))
            else:
                # Fallback only when base clip is unavailable.
                clip_like = int(bool(plant_info_prev.get("ctrl_chain_clipped", 0)))

            fullspeed_any = int(bool(plant_info_prev.get("pump_fullspeed_any", 0)))
            backlog_vals = plant_info_prev.get("pump_backlog_kg", None)
            if backlog_vals is not None:
                arr = np.asarray(backlog_vals, dtype=float).reshape(-1)
                if arr.size > 0:
                    backlog_max = float(np.max(np.abs(arr)))
            else:
                backlog_max = float(plant_info_prev.get("pump_backlog_max_kg", 0.0))

        cmd_gap_factor = self._linear_scale(
            cmd_gap,
            self.cmd_gap_decay_start_kg,
            self.cmd_gap_decay_full_kg,
        )
        backlog_factor = self._linear_scale(
            backlog_max,
            self.pressure_backlog_start_kg,
            self.pressure_backlog_full_kg,
        )
        fullspeed_factor = float(fullspeed_any)
        pressure_sample = float(
            np.clip(
                max(float(clip_like), cmd_gap_factor, fullspeed_factor, backlog_factor),
                0.0,
                1.0,
            )
        )

        self.trim_pressure_sample = pressure_sample
        self.trim_pressure_backlog_max_kg = float(backlog_max)
        self.trim_pressure_cmd_gap_kg = float(cmd_gap)
        self.trim_pressure_clip = int(clip_like)
        self.trim_pressure_fullspeed = int(fullspeed_any)
        return {
            "pressure_sample": float(pressure_sample),
            "cmd_gap_kg": float(cmd_gap),
            "clip": float(clip_like),
            "cmd_gap_factor": float(cmd_gap_factor),
            "fullspeed": float(fullspeed_factor),
            "backlog_factor": float(backlog_factor),
        }

    def _update_pressure_hist(self, pressure_sample):
        sat_win = max(1, int(self.sat_window_s / self.dt))
        self.sat_hist.append(float(pressure_sample["pressure_sample"]))
        if len(self.sat_hist) > sat_win:
            self.sat_hist.pop(0)
        self.pressure_clip_hist.append(float(pressure_sample["clip"]))
        if len(self.pressure_clip_hist) > sat_win:
            self.pressure_clip_hist.pop(0)
        self.pressure_cmd_gap_hist.append(float(pressure_sample["cmd_gap_factor"]))
        if len(self.pressure_cmd_gap_hist) > sat_win:
            self.pressure_cmd_gap_hist.pop(0)
        self.pressure_fullspeed_hist.append(float(pressure_sample["fullspeed"]))
        if len(self.pressure_fullspeed_hist) > sat_win:
            self.pressure_fullspeed_hist.pop(0)
        self.pressure_backlog_hist.append(float(pressure_sample["backlog_factor"]))
        if len(self.pressure_backlog_hist) > sat_win:
            self.pressure_backlog_hist.pop(0)

        self.trim_pressure_clip_recent = float(np.mean(np.asarray(self.pressure_clip_hist, dtype=float)))
        self.trim_pressure_cmd_gap_recent = float(
            np.mean(np.asarray(self.pressure_cmd_gap_hist, dtype=float))
        )
        self.trim_pressure_fullspeed_recent = float(
            np.mean(np.asarray(self.pressure_fullspeed_hist, dtype=float))
        )
        self.trim_pressure_backlog_recent = float(
            np.mean(np.asarray(self.pressure_backlog_hist, dtype=float))
        )
        self.trim_pressure_recent = float(
            max(
                self.trim_pressure_clip_recent,
                self.trim_pressure_cmd_gap_recent,
                self.trim_pressure_fullspeed_recent,
                self.trim_pressure_backlog_recent,
            )
        )
        # Backward-compatible field name for legacy consumers.
        self.sat_recent_ratio = float(self.trim_pressure_recent)

    def _update_cmd_gap_hist(self, cmd_gap):
        gap_win = max(1, int(self.cmd_gap_window_s / self.dt))
        self.cmd_gap_hist.append(float(cmd_gap))
        if len(self.cmd_gap_hist) > gap_win:
            self.cmd_gap_hist.pop(0)
        self.cmd_gap_recent = float(np.mean(np.asarray(self.cmd_gap_hist, dtype=float)))

    def _update_hist(self, plant_info_prev):
        pressure_sample = self._compute_pressure_sample(plant_info_prev)
        cmd_gap = float(pressure_sample["cmd_gap_kg"])
        self._update_pressure_hist(pressure_sample)
        self._update_cmd_gap_hist(cmd_gap)

    def _compute_trim_scale_simple(self, trim_allowed=True):
        p = float(np.clip(self.trim_pressure_recent, 0.0, 1.0))
        if not trim_allowed:
            trim_scale = 0.0
        elif p >= self.simple_pressure_decay_th or self.trim_pressure_clip:
            trim_scale = max(
                self.trim_min_scale,
                self.trim_scale_prev * self.simple_decay_on_clip,
            )
        elif p >= self.simple_pressure_hold_th:
            # Mid-pressure: keep conservative authority recovery.
            trim_scale = min(1.0, self.trim_scale_prev + 0.25 * self.simple_recover_per_step)
        else:
            trim_scale = min(1.0, self.trim_scale_prev + self.simple_recover_per_step)
        self.trim_scale_prev = float(trim_scale)
        return float(trim_scale)

    def _compute_trim_scale_legacy(self, trim_allowed=True):
        if self.sat_recent_ratio <= self.sat_gate_off:
            sat_factor = 1.0
        elif self.sat_recent_ratio >= self.sat_gate_on:
            sat_factor = self.trim_min_scale
        else:
            frac = (self.sat_recent_ratio - self.sat_gate_off) / max(
                self.sat_gate_on - self.sat_gate_off, 1e-6
            )
            sat_factor = 1.0 - frac * (1.0 - self.trim_min_scale)

        if self.cmd_gap_recent <= self.cmd_gap_decay_start_kg:
            gap_factor = 1.0
        elif self.cmd_gap_recent >= self.cmd_gap_decay_full_kg:
            gap_factor = self.trim_min_scale
        else:
            frac = (self.cmd_gap_recent - self.cmd_gap_decay_start_kg) / max(
                self.cmd_gap_decay_full_kg - self.cmd_gap_decay_start_kg, 1e-6
            )
            gap_factor = 1.0 - frac * (1.0 - self.trim_min_scale)

        target = min(sat_factor, gap_factor)
        if not trim_allowed:
            target = 0.0
        if target < self.trim_scale_prev:
            trim_scale = max(target, self.trim_scale_prev * self.trim_decay_per_step)
        else:
            trim_scale = self.trim_scale_prev + self.trim_recover_per_step * (
                target - self.trim_scale_prev
            )
        self.trim_scale_prev = float(trim_scale)
        return float(trim_scale)

    def _compute_trim_scale(self, trim_allowed=True):
        if self.trim_logic == "simple_default":
            return self._compute_trim_scale_simple(trim_allowed=trim_allowed)
        return self._compute_trim_scale_legacy(trim_allowed=trim_allowed)

    def _compute_simple_freeze_update(self):
        if not self.simple_freeze_enabled:
            self.simple_freeze_active = False
            return False

        p = float(np.clip(self.trim_pressure_recent, 0.0, 1.0))
        if self.simple_freeze_active:
            if p <= self.simple_pressure_unfreeze_th:
                self.simple_freeze_active = False
        else:
            if p >= self.simple_pressure_freeze_th:
                self.simple_freeze_active = True
        return bool(self.simple_freeze_active)

    def _update_wind_lpf(self, ws):
        if self.wind_lpf_tau_s > 0.0:
            alpha = self.wind_lpf_tau_s / (self.wind_lpf_tau_s + self.dt)
            self.ws_trim_lpf = alpha * self.ws_trim_lpf + (1.0 - alpha) * ws
        else:
            self.ws_trim_lpf = float(ws)

    def _apply_trim_scaling(self, trim_scale):
        if self.preview_scale_mode == "guarded":
            preview_scale_eff = float(trim_scale)
        elif self.preview_scale_mode == "separate":
            preview_scale_eff = float(self.preview_scale)
        else:
            preview_scale_eff = 1.0

        self._current_trim_scaled[:] = self._current_trim_raw * float(trim_scale)
        self._preview_trim_scaled[:] = self._preview_trim_bias * preview_scale_eff
        self._trim_hold[:] = self._current_trim_scaled + self._preview_trim_scaled
        return float(preview_scale_eff)

    def _update_steady_state_gate(self, state=None, setpoints=None):
        if state is None or setpoints is None:
            if self.trim_logic == "simple_default":
                self.trim_enabled = True
            return
        pitch = float(np.degrees(state[4]))
        roll = float(np.degrees(state[3]))
        e_pitch = float(setpoints.get("pitch", 0.0) - pitch)
        e_roll = float(setpoints.get("roll", 0.0) - roll)
        e_abs = float(np.sqrt(e_pitch * e_pitch + e_roll * e_roll))

        win = max(1, int(self.steady_window_s / self.dt))
        self.err_hist.append(e_abs)
        if len(self.err_hist) > win:
            self.err_hist.pop(0)
        arr = np.asarray(self.err_hist, dtype=float)
        self.steady_mean_abs = float(np.mean(arr))
        self.steady_std = float(np.std(arr))

        if self.trim_logic == "simple_default":
            self.trim_enabled = True
            return

        if not self.trim_enabled:
            if (
                self.steady_mean_abs >= self.steady_mean_on_deg
                and self.steady_std <= self.steady_std_on_deg
                and self.sat_recent_ratio < self.sat_gate_off
            ):
                self.trim_enabled = True
        else:
            if (
                self.steady_mean_abs <= self.steady_mean_off_deg
                or self.steady_std >= self.steady_std_off_deg
                or self.sat_recent_ratio >= self.sat_gate_on
            ):
                self.trim_enabled = False

    def update(
        self,
        wind_obs,
        plant_info_prev=None,
        state=None,
        setpoints=None,
        preview_trim_bias=None,
    ):
        self._update_hist(plant_info_prev)
        self._update_steady_state_gate(state=state, setpoints=setpoints)
        trim_scale = self._compute_trim_scale(trim_allowed=self.trim_enabled)

        ws = float(wind_obs.get("ws", 0.0)) if wind_obs is not None else 0.0
        wd = float(wind_obs.get("wd_deg", 0.0)) if wind_obs is not None else 0.0
        self._update_wind_lpf(ws)
        if self.trim_logic == "simple_default":
            freeze_update = self._compute_simple_freeze_update()
        else:
            freeze_update = (
                self.sat_recent_ratio >= self.sat_gate_on
                or self.cmd_gap_recent >= self.cmd_gap_decay_full_kg
            )
        do_update = (self._step_idx % self.trim_update_steps == 0)
        (preview_pitch, preview_roll), preview_dbg = _coerce_preview_trim_bias(preview_trim_bias)
        self._preview_trim_bias[:] = [preview_pitch, preview_roll]
        self._preview_trim_debug = preview_dbg

        if not self.trim_enabled:
            self._trim_hold[:] = 0.0
            self._current_trim_raw[:] = 0.0
            self._preview_trim_bias[:] = 0.0
            self._combined_trim_raw[:] = 0.0
            self._current_trim_scaled[:] = 0.0
            self._preview_trim_scaled[:] = 0.0
            self._preview_trim_debug = _zero_preview_trim_debug(source="trim_disabled")
            self.target_source_last = "trim_disabled"
            self.target_fallback_reason_last = ""
        elif do_update and not freeze_update:
            pitch_raw_g1, roll_raw_g1 = compute_trim_setpoints(
                self.ws_trim_lpf,
                wd,
                k_trim_deg_per_mps=self.k_trim_deg_per_mps,
                max_trim_deg=self.max_trim_deg,
                trim_map_version=self.trim_map_version,
                regime_cfg=self.trim_regime_cfg,
            )
            pitch_raw = float(pitch_raw_g1)
            roll_raw = float(roll_raw_g1)
            source = "g1_formula"
            fallback_reason = ""

            if self.target_mode == "g2":
                if self._target_lookup is None:
                    source = "g2_fallback_g1"
                    fallback_reason = "missing_table"
                else:
                    hit, p_tab, r_tab, reason = self._target_lookup.lookup(wind_obs)
                    if hit:
                        pitch_raw = float(p_tab)
                        roll_raw = float(r_tab)
                        source = "g2_table"
                    else:
                        source = "g2_fallback_g1"
                        fallback_reason = str(reason)

            self._current_trim_raw[:] = [pitch_raw, roll_raw]
            self.target_source_last = str(source)
            self.target_fallback_reason_last = str(fallback_reason)
        elif freeze_update:
            self.target_source_last = "freeze_hold"
            self.target_fallback_reason_last = ""
        else:
            self.target_source_last = "hold"
            self.target_fallback_reason_last = ""

        if self.trim_enabled:
            self._combined_trim_raw[:] = self._current_trim_raw + self._preview_trim_bias
            preview_scale_eff = self._apply_trim_scaling(trim_scale)
        else:
            preview_scale_eff = 0.0

        self._step_idx += 1

        lookup_stats = {
            "target_lookup_total": 0,
            "target_lookup_hit": 0,
            "target_lookup_fallback": 0,
            "fallback_reason_missing_state": 0,
            "fallback_reason_missing_table": 0,
            "fallback_reason_missing_columns": 0,
            "fallback_reason_invalid_state_id": 0,
        }
        if self._target_lookup is not None:
            lookup_stats.update(self._target_lookup.stats())

        return {
            "pitch_sp_raw_deg": float(self._trim_hold[0]),
            "roll_sp_raw_deg": float(self._trim_hold[1]),
            "current_pitch_trim_raw_deg": float(self._current_trim_raw[0]),
            "current_roll_trim_raw_deg": float(self._current_trim_raw[1]),
            "preview_pitch_bias_deg": float(self._preview_trim_bias[0]),
            "preview_roll_bias_deg": float(self._preview_trim_bias[1]),
            "combined_pitch_trim_raw_deg": float(self._combined_trim_raw[0]),
            "combined_roll_trim_raw_deg": float(self._combined_trim_raw[1]),
            "current_pitch_trim_scaled_deg": float(self._current_trim_scaled[0]),
            "current_roll_trim_scaled_deg": float(self._current_trim_scaled[1]),
            "preview_pitch_scaled_deg": float(self._preview_trim_scaled[0]),
            "preview_roll_scaled_deg": float(self._preview_trim_scaled[1]),
            "preview_scale_mode": str(self.preview_scale_mode),
            "preview_scale_eff": float(preview_scale_eff),
            "preview_trim_active": int(self._preview_trim_debug.get("preview_trim_active", 0)),
            "preview_trim_source": str(self._preview_trim_debug.get("preview_trim_source", "none")),
            "trim_scale": float(trim_scale),
            "trim_enabled": int(self.trim_enabled),
            "sat_recent_ratio": float(self.sat_recent_ratio),
            "trim_pressure_recent": float(self.trim_pressure_recent),
            "trim_pressure_clip_recent": float(self.trim_pressure_clip_recent),
            "trim_pressure_cmd_gap_recent": float(self.trim_pressure_cmd_gap_recent),
            "trim_pressure_fullspeed_recent": float(self.trim_pressure_fullspeed_recent),
            "trim_pressure_backlog_recent": float(self.trim_pressure_backlog_recent),
            "trim_pressure_sample": float(self.trim_pressure_sample),
            "trim_pressure_backlog_max_kg": float(self.trim_pressure_backlog_max_kg),
            "trim_pressure_cmd_gap_kg": float(self.trim_pressure_cmd_gap_kg),
            "trim_pressure_clip": int(self.trim_pressure_clip),
            "trim_pressure_fullspeed": int(self.trim_pressure_fullspeed),
            "cmd_gap_recent_kg": float(self.cmd_gap_recent),
            "steady_mean_abs_err_deg": float(self.steady_mean_abs),
            "steady_std_err_deg": float(self.steady_std),
            "trim_freeze": int(freeze_update),
            "trim_update_tick": int(do_update),
            "trim_map_version": str(self.trim_map_version),
            "target_mode": str(self.target_mode),
            "target_source": str(self.target_source_last),
            "target_fallback_reason_last": str(self.target_fallback_reason_last),
            "target_table_path": str(self.target_table_path),
            "target_lookup_key": str(self.target_lookup_key),
            "target_lookup_total": int(lookup_stats.get("target_lookup_total", 0)),
            "target_lookup_hit": int(lookup_stats.get("target_lookup_hit", 0)),
            "target_lookup_fallback": int(lookup_stats.get("target_lookup_fallback", 0)),
            "fallback_reason_missing_state": int(lookup_stats.get("fallback_reason_missing_state", 0)),
            "fallback_reason_missing_table": int(lookup_stats.get("fallback_reason_missing_table", 0)),
            "fallback_reason_missing_columns": int(lookup_stats.get("fallback_reason_missing_columns", 0)),
            "fallback_reason_invalid_state_id": int(lookup_stats.get("fallback_reason_invalid_state_id", 0)),
        }


class HeaveMonitor:
    def __init__(self, dt, history_window_s=30.0):
        self.dt = float(dt)
        self.history_window_s = float(history_window_s)
        self._max_len = max(1, int(round(self.history_window_s / self.dt)))
        self._hist = []

    def reset(self):
        self._hist = []

    def update(self, heave_m):
        h = float(heave_m)
        self._hist.append(h)
        if len(self._hist) > self._max_len:
            self._hist.pop(0)
        return float(np.mean(np.asarray(self._hist, dtype=float)))


class HeaveBiasBalancer:
    def __init__(
        self,
        dt,
        rho,
        g,
        a_waterplane_m2,
        target_heave_m=-8.1,
        kp=0.005,
        ki=0.0,
        max_correction_per_tank_kg=2.0,
        m_min_op_kg=200000.0,
        m_max_op_ratio=0.90,
        history_window_s=30.0,
        enabled=True,
    ):
        self.dt = float(dt)
        self.rho = float(rho)
        self.g = float(g)
        self.a_waterplane_m2 = max(float(a_waterplane_m2), 0.0)
        self.target_heave_m = float(target_heave_m)
        self.kp = float(kp)
        self.ki = float(ki)
        self.max_correction_per_tank_kg = max(float(max_correction_per_tank_kg), 0.0)
        self.m_min_op_kg = max(float(m_min_op_kg), 0.0)
        self.m_max_op_ratio = float(m_max_op_ratio)
        self.enabled = bool(enabled)
        self.monitor = HeaveMonitor(dt=self.dt, history_window_s=history_window_s)
        self.int_mass_err = 0.0

    def reset(self):
        self.int_mass_err = 0.0
        self.monitor.reset()

    def update(self, cmd_raw, state, tank_masses, max_capacity):
        raw = np.asarray(cmd_raw, dtype=float).copy()
        if not self.enabled:
            return raw, {
                "hm_smoothed_heave_m": float(state[2]),
                "hm_heave_error_m": 0.0,
                "hm_total_ballast_kg": float(np.sum(raw)),
                "ballast_total_kg": float(np.sum(np.asarray(tank_masses, dtype=float))),
                "heave_correction_per_tank_kg": 0.0,
                "heave_balancer_active": 0,
            }

        capacities = np.asarray(max_capacity, dtype=float)
        masses = np.asarray(tank_masses, dtype=float)
        smoothed_heave = self.monitor.update(float(state[2]))
        heave_error = smoothed_heave - self.target_heave_m

        # Convert heave offset to an equivalent displaced mass offset.
        mass_err = self.rho * self.a_waterplane_m2 * heave_error
        self.int_mass_err += mass_err * self.dt

        total_correction = self.kp * mass_err + self.ki * self.int_mass_err
        per_tank = float(total_correction / 3.0)
        per_tank = float(
            np.clip(
                per_tank,
                -self.max_correction_per_tank_kg,
                self.max_correction_per_tank_kg,
            )
        )

        # Runtime safety belt: keep commanded operation inside an engineering band.
        m_min = np.full(3, self.m_min_op_kg, dtype=float)
        m_max = np.minimum(capacities * self.m_max_op_ratio, capacities)
        add_limit = float(np.min(np.maximum(0.0, m_max - masses)))
        remove_limit = float(np.min(np.maximum(0.0, masses - m_min)))
        per_tank = float(np.clip(per_tank, -remove_limit, add_limit))

        adjusted = np.clip(raw + per_tank, 0.0, capacities)
        return adjusted, {
            "hm_smoothed_heave_m": float(smoothed_heave),
            "hm_heave_error_m": float(heave_error),
            "hm_total_ballast_kg": float(np.sum(adjusted)),
            "ballast_total_kg": float(np.sum(masses)),
            "heave_correction_per_tank_kg": float(per_tank),
            "heave_balancer_active": 1,
        }


class ClosedLoopPolicy:
    def __init__(
        self,
        controller,
        trim_governor=None,
        setpoint_shaper=None,
        target_slew_limiter=None,
        heave_balancer=None,
        preview_trim_provider=None,
        primary_safety_cfg=None,
    ):
        self.controller = controller
        self.trim_governor = trim_governor
        self.setpoint_shaper = setpoint_shaper
        self.target_slew_limiter = target_slew_limiter
        self.heave_balancer = heave_balancer
        self.preview_trim_provider = preview_trim_provider
        cfg = dict(primary_safety_cfg or {})
        self.primary_safety_enabled = self._cfg_bool(
            cfg, "primary_safety_fallback_enabled", True
        )
        self.primary_safety_pitch_enter_deg = float(
            cfg.get("primary_safety_pitch_enter_deg", 6.5)
        )
        self.primary_safety_roll_enter_deg = float(
            cfg.get("primary_safety_roll_enter_deg", 5.5)
        )
        self.primary_safety_pitch_exit_deg = float(
            cfg.get("primary_safety_pitch_exit_deg", 6.0)
        )
        self.primary_safety_roll_exit_deg = float(
            cfg.get("primary_safety_roll_exit_deg", 5.0)
        )
        self.primary_safety_use_envelope = self._cfg_bool(
            cfg, "primary_safety_use_envelope", False
        )
        self.primary_safety_envelope_enter_norm = max(
            float(cfg.get("primary_safety_envelope_enter_norm", 1.5)), 1e-6
        )
        self.primary_safety_enter_hold_s = max(
            float(cfg.get("primary_safety_enter_hold_s", 30.0)), 0.0
        )
        self.primary_safety_emergency_pitch_enter_deg = float(
            cfg.get("primary_safety_emergency_pitch_enter_deg", 9.0)
        )
        self.primary_safety_emergency_roll_enter_deg = float(
            cfg.get("primary_safety_emergency_roll_enter_deg", 7.5)
        )
        self.primary_safety_exit_hold_s = max(
            float(cfg.get("primary_safety_exit_hold_s", 0.0)), 0.0
        )
        self.primary_safety_exit_required_windows = max(
            int(cfg.get("primary_safety_exit_required_windows", 1)), 1
        )
        self.primary_safety_bucket_guard_enabled = self._cfg_bool(
            cfg, "primary_safety_bucket_guard_enabled", False
        )
        self.primary_safety_bucket_guard_bucket_s = max(
            float(cfg.get("primary_safety_bucket_guard_bucket_s", 600.0)), 1e-6
        )
        self.primary_safety_bucket_guard_pitch_enter_deg = float(
            cfg.get("primary_safety_bucket_guard_pitch_enter_deg", 4.5)
        )
        self.primary_safety_bucket_guard_roll_enter_deg = float(
            cfg.get("primary_safety_bucket_guard_roll_enter_deg", 3.8)
        )
        self.primary_safety_bucket_guard_pitch_exit_deg = float(
            cfg.get("primary_safety_bucket_guard_pitch_exit_deg", 4.0)
        )
        self.primary_safety_bucket_guard_roll_exit_deg = float(
            cfg.get("primary_safety_bucket_guard_roll_exit_deg", 3.2)
        )
        self.primary_safety_bucket_guard_improve_tol_deg = max(
            float(cfg.get("primary_safety_bucket_guard_improve_tol_deg", 0.05)), 0.0
        )
        self.primary_safety_bucket_guard_exit_required_windows = max(
            int(cfg.get("primary_safety_bucket_guard_exit_required_windows", 1)), 1
        )
        self.primary_safety_bucket_guard_max_active_windows = max(
            int(cfg.get("primary_safety_bucket_guard_max_active_windows", 1)), 1
        )
        self.primary_safety_bucket_guard_hold_only = self._cfg_bool(
            cfg, "primary_safety_bucket_guard_hold_only", True
        )
        self.primary_safety_bucket_guard_require_current_high = self._cfg_bool(
            cfg, "primary_safety_bucket_guard_require_current_high", True
        )
        self.primary_safety_hold_risk_gate_enabled = self._cfg_bool(
            cfg, "primary_safety_hold_risk_gate_enabled", False
        )
        self.primary_safety_hold_risk_gate_action = str(
            cfg.get("primary_safety_hold_risk_gate_action", "hold")
        )
        self.deadband_target_release_enabled = self._cfg_bool(
            cfg, "deadband_target_release_enabled", False
        )
        self.deadband_target_release_pitch_deg = float(
            cfg.get("deadband_target_release_pitch_deg", 0.0)
        )
        self.deadband_target_release_roll_deg = float(
            cfg.get("deadband_target_release_roll_deg", 0.0)
        )
        self.deadband_target_release_near_zero_deg = max(
            float(cfg.get("deadband_target_release_near_zero_deg", 0.15)), 0.0
        )
        self.deadband_target_release_rate_eps_deg_s = max(
            float(cfg.get("deadband_target_release_rate_eps_deg_s", 0.002)), 0.0
        )
        self.deadband_target_release_blend = float(
            np.clip(float(cfg.get("deadband_target_release_blend", 1.0)), 0.0, 1.0)
        )
        self.deadband_target_release_require_both_axes = self._cfg_bool(
            cfg, "deadband_target_release_require_both_axes", True
        )
        self.deadband_target_release_use_pid_deadband = self._cfg_bool(
            cfg, "deadband_target_release_use_pid_deadband", True
        )
        self.deadband_target_release_require_forecast_safe = self._cfg_bool(
            cfg, "deadband_target_release_require_forecast_safe", False
        )
        self.deadband_target_release_reset_limiter = self._cfg_bool(
            cfg, "deadband_target_release_reset_limiter", True
        )
        self.deadband_target_release_exit_pitch_deg = max(
            float(cfg.get("deadband_target_release_exit_pitch_deg", 0.0)), 0.0
        )
        self.deadband_target_release_exit_roll_deg = max(
            float(cfg.get("deadband_target_release_exit_roll_deg", 0.0)), 0.0
        )
        self._deadband_target_release_latched = False
        self.forecast_safe_deadband_enabled = self._cfg_bool(
            cfg, "forecast_safe_deadband_enabled", False
        )
        self.forecast_safe_deadband_sync_provider_target = self._cfg_bool(
            cfg, "forecast_safe_deadband_sync_provider_target", False
        )
        self.forecast_safe_deadband_apply_pid_deadband = self._cfg_bool(
            cfg, "forecast_safe_deadband_apply_pid_deadband", True
        )
        self.forecast_safe_deadband_release_primary_target = self._cfg_bool(
            cfg, "forecast_safe_deadband_release_primary_target", True
        )
        self.forecast_safe_deadband_min_target_error_kg = max(
            float(cfg.get("forecast_safe_deadband_min_target_error_kg", 1.0)), 0.0
        )
        self.forecast_safe_deadband_sync_require_recovering = self._cfg_bool(
            cfg, "forecast_safe_deadband_sync_require_recovering", False
        )
        self.forecast_safe_deadband_sync_require_pump_demand = self._cfg_bool(
            cfg, "forecast_safe_deadband_sync_require_pump_demand", False
        )
        self.forecast_safe_deadband_sync_refresh_current = self._cfg_bool(
            cfg, "forecast_safe_deadband_sync_refresh_current", False
        )
        self.forecast_safe_deadband_sync_startup_delay_s = max(
            float(cfg.get("forecast_safe_deadband_sync_startup_delay_s", 0.0)), 0.0
        )
        self.forecast_safe_deadband_sync_reentry_cooldown_s = max(
            float(cfg.get("forecast_safe_deadband_sync_reentry_cooldown_s", 0.0)), 0.0
        )
        self.forecast_safe_deadband_sync_min_blend = float(
            np.clip(cfg.get("forecast_safe_deadband_sync_min_blend", 1.0), 0.0, 1.0)
        )
        self.forecast_safe_deadband_sync_full_pressure_norm = float(
            cfg.get("forecast_safe_deadband_sync_full_pressure_norm", 1.0)
        )
        self.forecast_safe_deadband_sync_min_pressure_norm = max(
            float(cfg.get("forecast_safe_deadband_sync_min_pressure_norm", 1.5)),
            self.forecast_safe_deadband_sync_full_pressure_norm + 1e-9,
        )
        self.forecast_safe_deadband_sync_future_rise_max = float(
            cfg.get(
                "forecast_safe_deadband_sync_future_rise_max",
                self.forecast_safe_deadband_future_rise_max
                if hasattr(self, "forecast_safe_deadband_future_rise_max")
                else 0.10,
            )
        )
        self.forecast_safe_deadband_sync_near_zero_deg = max(
            float(cfg.get("forecast_safe_deadband_sync_near_zero_deg", 0.15)), 0.0
        )
        self.forecast_safe_deadband_sync_rate_eps_deg_s = max(
            float(cfg.get("forecast_safe_deadband_sync_rate_eps_deg_s", 0.002)), 0.0
        )
        self.forecast_safe_deadband_gate_mode = str(
            cfg.get("forecast_safe_deadband_gate_mode", "conservative")
        )
        self.forecast_safe_deadband_pitch_enter_deg = max(
            float(cfg.get("forecast_safe_deadband_pitch_enter_deg", 1.5)), 0.0
        )
        self.forecast_safe_deadband_roll_enter_deg = max(
            float(cfg.get("forecast_safe_deadband_roll_enter_deg", 1.5)), 0.0
        )
        self.forecast_safe_deadband_pitch_exit_deg = max(
            float(cfg.get("forecast_safe_deadband_pitch_exit_deg", 1.2)), 0.0
        )
        self.forecast_safe_deadband_roll_exit_deg = max(
            float(cfg.get("forecast_safe_deadband_roll_exit_deg", 1.2)), 0.0
        )
        self.forecast_safe_deadband_event_probability_max = float(
            cfg.get("forecast_safe_deadband_event_probability_max", 0.75)
        )
        self.forecast_safe_deadband_require_event_probability = self._cfg_bool(
            cfg, "forecast_safe_deadband_require_event_probability", False
        )
        self.forecast_safe_deadband_future_rise_max = float(
            cfg.get("forecast_safe_deadband_future_rise_max", 0.05)
        )
        self.forecast_safe_deadband_minimum_direction_dot = float(
            cfg.get("forecast_safe_deadband_minimum_direction_dot", 0.0)
        )
        self.forecast_safe_deadband_posture_gate_deg = max(
            float(cfg.get("forecast_safe_deadband_posture_gate_deg", 1.2)), 0.0
        )
        self.forecast_safe_deadband_allowed_actions = tuple(
            str(value)
            for value in cfg.get("forecast_safe_deadband_allowed_actions", ("hold",))
        )
        self.forecast_safe_deadband_reject_unsafe_hold = self._cfg_bool(
            cfg, "forecast_safe_deadband_reject_unsafe_hold", False
        )
        self._forecast_safe_deadband_base_enter = {
            axis: float(self.controller.deadband_enter.get(axis, 0.0))
            for axis in ("pitch", "roll")
        }
        self._forecast_safe_deadband_base_exit = {
            axis: float(self.controller.deadband_exit.get(axis, 0.0))
            for axis in ("pitch", "roll")
        }
        self._forecast_safe_deadband_dbg = {
            "active": False,
            "reason": "init",
            "gate_mode": self.forecast_safe_deadband_gate_mode,
            "has_future": 0,
            "trust_ok": 0,
            "action": "",
            "event_probability": 0.0,
            "event_probability_available": 0,
            "future_rise": 0.0,
            "direction_dot": 0.0,
            "pitch_abs_deg": 0.0,
            "roll_abs_deg": 0.0,
            "posture_abs_deg": 0.0,
            "provider_target_synced": 0,
            "provider_sync_started": 0,
            "provider_replan_requested": 0,
            "provider_hold_active": 0,
            "provider_sync_eligible": 0,
            "provider_sync_reason": "init",
            "posture_recovering": 0,
            "sync_future_rise_ok": 0,
            "target_error_mean_kg": 0.0,
            "pump_demand_active": 0,
            "actionable_pump_demand": 0,
            "startup_ready": 0,
            "future_pressure_peak_norm": 0.0,
            "provider_sync_blend": 0.0,
        }
        self._forecast_safe_deadband_provider_hold_active = False
        self._forecast_safe_deadband_reentry_after_s = 0.0
        self.reactive_pump_suppression_enabled = self._cfg_bool(
            cfg, "reactive_pump_suppression_enabled", False
        )
        self.reactive_pump_suppression_pitch_enter_deg = max(
            float(cfg.get("reactive_pump_suppression_pitch_enter_deg", 1.2)), 0.0
        )
        self.reactive_pump_suppression_roll_enter_deg = max(
            float(cfg.get("reactive_pump_suppression_roll_enter_deg", 1.0)), 0.0
        )
        self.reactive_pump_suppression_pitch_exit_deg = max(
            float(cfg.get("reactive_pump_suppression_pitch_exit_deg", 1.6)), 0.0
        )
        self.reactive_pump_suppression_roll_exit_deg = max(
            float(cfg.get("reactive_pump_suppression_roll_exit_deg", 1.3)), 0.0
        )
        self.reactive_pump_suppression_restart_err_kg = max(
            float(cfg.get("reactive_pump_suppression_restart_err_kg", 1100.0)), 0.0
        )
        self.reactive_pump_suppression_require_not_fullspeed = self._cfg_bool(
            cfg, "reactive_pump_suppression_require_not_fullspeed", True
        )
        self._reactive_pump_suppression_latched = False
        self._primary_safety_active = False
        self._primary_safety_enter_elapsed_s = 0.0
        self._primary_safety_last_time_s = None
        self._primary_safety_exit_window_start_s = None
        self._primary_safety_exit_window_max_pitch_abs_deg = 0.0
        self._primary_safety_exit_window_max_roll_abs_deg = 0.0
        self._primary_safety_exit_clean_windows = 0
        self._primary_bucket_guard_active = False
        self._primary_bucket_guard_window_start_s = None
        self._primary_bucket_guard_pitch_max_deg = 0.0
        self._primary_bucket_guard_roll_max_deg = 0.0
        self._primary_bucket_guard_prev_pitch_max_deg = np.nan
        self._primary_bucket_guard_prev_roll_max_deg = np.nan
        self._primary_bucket_guard_clean_windows = 0
        self._primary_bucket_guard_active_windows = 0
        self._primary_bucket_guard_completed_windows = 0
        self._primary_bucket_guard_reason = "init"
        self._deadband_target_release_latched = False
        self._reactive_pump_suppression_latched = False

    def reset(self, initial_cmd, initial_ws=0.0):
        self._restore_forecast_safe_deadband()
        self.controller.reset()
        if self.trim_governor is not None:
            self.trim_governor.reset(initial_ws=initial_ws)
        if self.setpoint_shaper is not None:
            self.setpoint_shaper.reset(
                initial=(self.controller.setpoints["pitch"], self.controller.setpoints["roll"])
            )
        if self.target_slew_limiter is not None:
            self.target_slew_limiter.reset(initial_target=initial_cmd)
        if self.heave_balancer is not None:
            self.heave_balancer.reset()
        if self.preview_trim_provider is not None and hasattr(self.preview_trim_provider, "reset"):
            self.preview_trim_provider.reset()
        self._primary_safety_active = False
        self._primary_safety_enter_elapsed_s = 0.0
        self._primary_safety_last_time_s = None
        self._primary_safety_exit_window_start_s = None
        self._primary_safety_exit_window_max_pitch_abs_deg = 0.0
        self._primary_safety_exit_window_max_roll_abs_deg = 0.0
        self._primary_safety_exit_clean_windows = 0
        self._reset_primary_bucket_guard()
        self._deadband_target_release_latched = False
        self._reactive_pump_suppression_latched = False
        self._forecast_safe_deadband_dbg = {
            **self._forecast_safe_deadband_dbg,
            "active": False,
            "reason": "reset",
            "has_future": 0,
            "trust_ok": 0,
            "event_probability": 0.0,
            "event_probability_available": 0,
            "future_rise": 0.0,
            "direction_dot": 0.0,
            "pitch_abs_deg": 0.0,
            "roll_abs_deg": 0.0,
            "posture_abs_deg": 0.0,
        }
        self._forecast_safe_deadband_provider_hold_active = False
        self._forecast_safe_deadband_reentry_after_s = 0.0

    @staticmethod
    def _cfg_bool(cfg, key, default):
        value = cfg.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() not in ("", "0", "false", "no", "off")
        return bool(value)

    @staticmethod
    def _safe_norm_axis(value, scale):
        scale = abs(float(scale))
        if scale <= 1e-9:
            return 0.0
        return float(value) / scale

    def _restore_forecast_safe_deadband(self):
        for axis in ("pitch", "roll"):
            self.controller.deadband_enter[axis] = self._forecast_safe_deadband_base_enter[axis]
            self.controller.deadband_exit[axis] = self._forecast_safe_deadband_base_exit[axis]

    def _apply_forecast_safe_deadband(
        self,
        state,
        preview,
        *,
        plant_info_prev=None,
        current_time=0.0,
    ):
        decision = evaluate_forecast_safe_deadband(
            preview,
            state,
            enabled=self.forecast_safe_deadband_enabled,
            event_probability_max=self.forecast_safe_deadband_event_probability_max,
            future_rise_max=self.forecast_safe_deadband_future_rise_max,
            minimum_direction_dot=self.forecast_safe_deadband_minimum_direction_dot,
            posture_gate_deg=self.forecast_safe_deadband_posture_gate_deg,
            allowed_actions=self.forecast_safe_deadband_allowed_actions,
            gate_mode=self.forecast_safe_deadband_gate_mode,
            require_event_probability=self.forecast_safe_deadband_require_event_probability,
        )
        decision.update(
            {
                "provider_target_synced": 0,
                "provider_sync_started": 0,
                "provider_replan_requested": 0,
                "provider_hold_active": int(
                    self._forecast_safe_deadband_provider_hold_active
                ),
                "provider_sync_eligible": 0,
                "provider_sync_reason": "not_evaluated",
                "posture_recovering": 0,
                "sync_future_rise_ok": 0,
                "target_error_mean_kg": 0.0,
                "pump_demand_active": 0,
                "actionable_pump_demand": 0,
                "startup_ready": 0,
                "future_pressure_peak_norm": 0.0,
                "provider_sync_blend": 0.0,
                "reentry_ready": 0,
            }
        )

        preview_dict = preview if isinstance(preview, dict) else None
        masses = None
        if plant_info_prev is not None and "tank_masses" in plant_info_prev:
            arr = np.asarray(plant_info_prev["tank_masses"], dtype=float).reshape(-1)
            if arr.size >= 3:
                masses = arr[:3]

        target_error_mean_kg = 0.0
        if masses is not None and preview_dict is not None:
            target_raw = preview_dict.get("preview_primary_target_kg")
            if target_raw is not None:
                target = np.asarray(target_raw, dtype=float).reshape(-1)
                if target.size >= 3:
                    target_error_mean_kg = float(np.mean(np.abs(target[:3] - masses)))
        pump_demand_active = False
        if plant_info_prev is not None:
            rates = np.asarray(
                plant_info_prev.get("pump_rate_cmd_m3_min", np.zeros(3)),
                dtype=float,
            ).reshape(-1)
            pump_demand_active = bool(
                np.any(np.abs(rates) > 1e-9)
                or int(plant_info_prev.get("pump_fullspeed_any", 0)) > 0
            )
        decision["target_error_mean_kg"] = float(target_error_mean_kg)
        decision["pump_demand_active"] = int(pump_demand_active)
        pressure_values = []
        if preview_dict is not None:
            for key in (
                "preview_pressure_block0_norm",
                "preview_pressure_block1_norm",
                "preview_pressure_block2_norm",
            ):
                value = float(preview_dict.get(key, np.nan))
                if np.isfinite(value):
                    pressure_values.append(value)
        future_pressure_peak_norm = max(pressure_values) if pressure_values else 0.0
        full_pressure = self.forecast_safe_deadband_sync_full_pressure_norm
        min_pressure = self.forecast_safe_deadband_sync_min_pressure_norm
        if future_pressure_peak_norm <= full_pressure:
            provider_sync_blend = 1.0
        elif future_pressure_peak_norm >= min_pressure:
            provider_sync_blend = self.forecast_safe_deadband_sync_min_blend
        else:
            fraction = (future_pressure_peak_norm - full_pressure) / (
                min_pressure - full_pressure
            )
            provider_sync_blend = 1.0 - fraction * (
                1.0 - self.forecast_safe_deadband_sync_min_blend
            )
        startup_ready = bool(
            float(current_time) >= self.forecast_safe_deadband_sync_startup_delay_s
        )
        reentry_ready = bool(
            float(current_time) >= self._forecast_safe_deadband_reentry_after_s
        )
        decision["startup_ready"] = int(startup_ready)
        decision["reentry_ready"] = int(reentry_ready)
        decision["future_pressure_peak_norm"] = float(future_pressure_peak_norm)
        decision["provider_sync_blend"] = float(provider_sync_blend)

        pitch_deg = float(np.degrees(state[4]))
        roll_deg = float(np.degrees(state[3]))
        pitch_rate_deg_s = float(np.degrees(state[10])) if len(state) > 10 else 0.0
        roll_rate_deg_s = float(np.degrees(state[9])) if len(state) > 9 else 0.0

        def axis_recovering(value_deg, rate_deg_s):
            near_zero = abs(float(value_deg)) <= self.forecast_safe_deadband_sync_near_zero_deg
            moving_to_zero = (
                float(value_deg) * float(rate_deg_s)
                <= self.forecast_safe_deadband_sync_rate_eps_deg_s
            )
            return bool(near_zero or moving_to_zero)

        posture_recovering = bool(
            axis_recovering(pitch_deg, pitch_rate_deg_s)
            and axis_recovering(roll_deg, roll_rate_deg_s)
        )
        sync_future_rise_ok = bool(
            float(decision.get("future_rise", np.inf))
            <= self.forecast_safe_deadband_sync_future_rise_max
        )
        provider_sync_entry_eligible = bool(
            decision["active"]
            and reentry_ready
            and (
                not self.forecast_safe_deadband_sync_require_recovering
                or posture_recovering
            )
            and sync_future_rise_ok
        )
        provider_sync_maintain_eligible = bool(
            decision["active"]
            and self._forecast_safe_deadband_provider_hold_active
            and sync_future_rise_ok
        )
        provider_sync_eligible = bool(
            provider_sync_entry_eligible or provider_sync_maintain_eligible
        )
        decision["posture_recovering"] = int(posture_recovering)
        decision["sync_future_rise_ok"] = int(sync_future_rise_ok)
        decision["provider_sync_eligible"] = int(provider_sync_eligible)

        target_error_active = bool(
            target_error_mean_kg > self.forecast_safe_deadband_min_target_error_kg
        )
        actionable = bool(
            target_error_active
            and startup_ready
            and (
                pump_demand_active
                if self.forecast_safe_deadband_sync_require_pump_demand
                else True
            )
        )
        decision["actionable_pump_demand"] = int(actionable)
        provider = self.preview_trim_provider
        if (
            self.forecast_safe_deadband_sync_provider_target
            and provider_sync_eligible
            and masses is not None
            and preview_dict is not None
            and (actionable or self._forecast_safe_deadband_provider_hold_active)
            and hasattr(provider, "synchronize_forecast_deadband_hold")
        ):
            sync_started = not self._forecast_safe_deadband_provider_hold_active
            original_action = str(preview_dict.get("preview_primary_action", ""))
            synced = provider.synchronize_forecast_deadband_hold(
                plant_info_prev,
                float(current_time),
                blend=float(provider_sync_blend),
                start_new=bool(
                    sync_started or self.forecast_safe_deadband_sync_refresh_current
                ),
            )
            synced = np.asarray(synced, dtype=float).reshape(-1)[:3]
            preview_dict["preview_forecast_safe_deadband_original_action"] = original_action
            preview_dict["preview_primary_action"] = "hold"
            preview_dict["preview_primary_target_kg"] = synced.tolist()
            preview_dict["preview_primary_delta_kg"] = (synced - masses).tolist()
            decision["provider_target_synced"] = 1
            decision["provider_sync_started"] = int(sync_started)
            decision["provider_sync_reason"] = (
                "actionable_target" if sync_started else "maintain_synchronized_hold"
            )
            self._forecast_safe_deadband_provider_hold_active = True
        elif (
            self.forecast_safe_deadband_sync_provider_target
            and self._forecast_safe_deadband_provider_hold_active
            and not provider_sync_eligible
        ):
            if hasattr(provider, "exit_forecast_deadband_hold"):
                provider.exit_forecast_deadband_hold(
                    plant_info_prev,
                    float(current_time),
                )
            if preview_dict is not None:
                preview_dict["preview_primary_enabled"] = 0
                preview_dict["preview_primary_active"] = 0
            decision["provider_replan_requested"] = 1
            if not decision["active"]:
                decision["provider_sync_reason"] = str(
                    decision.get("reason", "forecast_gate_closed")
                )
            elif not posture_recovering and not self._forecast_safe_deadband_provider_hold_active:
                decision["provider_sync_reason"] = "posture_not_recovering"
            else:
                decision["provider_sync_reason"] = "future_pressure_rising"
            self._forecast_safe_deadband_provider_hold_active = False
            self._forecast_safe_deadband_reentry_after_s = (
                float(current_time)
                + self.forecast_safe_deadband_sync_reentry_cooldown_s
            )
        elif self.forecast_safe_deadband_sync_provider_target:
            if not decision["active"]:
                decision["provider_sync_reason"] = str(decision.get("reason", "inactive"))
            elif not posture_recovering:
                decision["provider_sync_reason"] = "posture_not_recovering"
            elif not sync_future_rise_ok:
                decision["provider_sync_reason"] = "future_pressure_rising"
            elif masses is None or preview_dict is None:
                decision["provider_sync_reason"] = "missing_target_state"
            elif not actionable:
                decision["provider_sync_reason"] = (
                    "startup_delay" if not startup_ready else "no_pump_demand"
                )
            elif not reentry_ready:
                decision["provider_sync_reason"] = "reentry_cooldown"
            else:
                decision["provider_sync_reason"] = "provider_sync_unavailable"
        decision["provider_hold_active"] = int(
            self._forecast_safe_deadband_provider_hold_active
        )
        if decision["active"] and self.forecast_safe_deadband_apply_pid_deadband:
            self.controller.deadband_enter["pitch"] = (
                self.forecast_safe_deadband_pitch_enter_deg
            )
            self.controller.deadband_enter["roll"] = (
                self.forecast_safe_deadband_roll_enter_deg
            )
            self.controller.deadband_exit["pitch"] = (
                self.forecast_safe_deadband_pitch_exit_deg
            )
            self.controller.deadband_exit["roll"] = (
                self.forecast_safe_deadband_roll_exit_deg
            )
        else:
            self._restore_forecast_safe_deadband()
        self._forecast_safe_deadband_dbg = decision
        return decision

    def _primary_safety_env_norm(self, pitch_abs_deg, roll_abs_deg):
        p = self._safe_norm_axis(pitch_abs_deg, self.primary_safety_pitch_enter_deg)
        r = self._safe_norm_axis(roll_abs_deg, self.primary_safety_roll_enter_deg)
        return float(np.sqrt(p * p + r * r))

    def _primary_safety_exit_env_norm(self):
        p = self._safe_norm_axis(
            self._primary_safety_exit_window_max_pitch_abs_deg,
            self.primary_safety_pitch_exit_deg,
        )
        r = self._safe_norm_axis(
            self._primary_safety_exit_window_max_roll_abs_deg,
            self.primary_safety_roll_exit_deg,
        )
        return float(np.sqrt(p * p + r * r))

    def _primary_safety_update_window(self, current_time, pitch_abs_deg, roll_abs_deg):
        if self._primary_safety_exit_window_start_s is None:
            self._primary_safety_exit_window_start_s = float(current_time)
            self._primary_safety_exit_window_max_pitch_abs_deg = float(pitch_abs_deg)
            self._primary_safety_exit_window_max_roll_abs_deg = float(roll_abs_deg)
            return
        self._primary_safety_exit_window_max_pitch_abs_deg = max(
            self._primary_safety_exit_window_max_pitch_abs_deg,
            float(pitch_abs_deg),
        )
        self._primary_safety_exit_window_max_roll_abs_deg = max(
            self._primary_safety_exit_window_max_roll_abs_deg,
            float(roll_abs_deg),
        )

    def _primary_safety_reset_exit_window(self, current_time, pitch_abs_deg, roll_abs_deg):
        self._primary_safety_exit_window_start_s = float(current_time)
        self._primary_safety_exit_window_max_pitch_abs_deg = float(pitch_abs_deg)
        self._primary_safety_exit_window_max_roll_abs_deg = float(roll_abs_deg)

    def _primary_safety_step_dt(self, current_time):
        current = float(current_time)
        if self._primary_safety_last_time_s is None:
            self._primary_safety_last_time_s = current
            return 0.0
        dt = max(current - float(self._primary_safety_last_time_s), 0.0)
        self._primary_safety_last_time_s = current
        return float(dt)

    @staticmethod
    def _primary_safety_reason(prefix, pitch_enter, roll_enter, env_enter):
        if pitch_enter:
            return f"{prefix}_pitch"
        if roll_enter:
            return f"{prefix}_roll"
        if env_enter:
            return f"{prefix}_envelope"
        return str(prefix)

    def _reset_primary_bucket_guard(self):
        self._primary_bucket_guard_active = False
        self._primary_bucket_guard_window_start_s = None
        self._primary_bucket_guard_pitch_max_deg = 0.0
        self._primary_bucket_guard_roll_max_deg = 0.0
        self._primary_bucket_guard_prev_pitch_max_deg = np.nan
        self._primary_bucket_guard_prev_roll_max_deg = np.nan
        self._primary_bucket_guard_clean_windows = 0
        self._primary_bucket_guard_active_windows = 0
        self._primary_bucket_guard_completed_windows = 0
        self._primary_bucket_guard_reason = "reset"

    def _primary_bucket_guard_dbg(
        self,
        reason,
        high=False,
        not_improving=False,
        exit_clean=False,
    ):
        return {
            "enabled": int(
                self.primary_safety_enabled
                and self.primary_safety_bucket_guard_enabled
            ),
            "active": int(self._primary_bucket_guard_active),
            "reason": str(reason),
            "bucket_s": float(self.primary_safety_bucket_guard_bucket_s),
            "pitch_enter_deg": float(self.primary_safety_bucket_guard_pitch_enter_deg),
            "roll_enter_deg": float(self.primary_safety_bucket_guard_roll_enter_deg),
            "pitch_exit_deg": float(self.primary_safety_bucket_guard_pitch_exit_deg),
            "roll_exit_deg": float(self.primary_safety_bucket_guard_roll_exit_deg),
            "improve_tol_deg": float(self.primary_safety_bucket_guard_improve_tol_deg),
            "exit_required_windows": int(
                self.primary_safety_bucket_guard_exit_required_windows
            ),
            "max_active_windows": int(
                self.primary_safety_bucket_guard_max_active_windows
            ),
            "current_pitch_max_deg": float(self._primary_bucket_guard_pitch_max_deg),
            "current_roll_max_deg": float(self._primary_bucket_guard_roll_max_deg),
            "prev_pitch_max_deg": float(self._primary_bucket_guard_prev_pitch_max_deg),
            "prev_roll_max_deg": float(self._primary_bucket_guard_prev_roll_max_deg),
            "clean_windows": int(self._primary_bucket_guard_clean_windows),
            "active_windows": int(self._primary_bucket_guard_active_windows),
            "completed_windows": int(self._primary_bucket_guard_completed_windows),
            "high": int(bool(high)),
            "not_improving": int(bool(not_improving)),
            "exit_clean": int(bool(exit_clean)),
            "current_high": 0,
            "hold_only": int(self.primary_safety_bucket_guard_hold_only),
            "require_current_high": int(
                self.primary_safety_bucket_guard_require_current_high
            ),
            "primary_action": "",
        }

    def _primary_safety_bucket_guard_update(
        self,
        current_time,
        pitch_abs_deg,
        roll_abs_deg,
        primary_enabled,
        primary_candidate_applied,
        primary_action="",
    ):
        if not (self.primary_safety_enabled and self.primary_safety_bucket_guard_enabled):
            self._reset_primary_bucket_guard()
            return self._primary_bucket_guard_dbg("disabled")
        if not (primary_enabled and primary_candidate_applied):
            self._reset_primary_bucket_guard()
            return self._primary_bucket_guard_dbg("primary_not_applied")
        action = str(primary_action or "")
        if self.primary_safety_bucket_guard_hold_only and action != "hold":
            self._reset_primary_bucket_guard()
            dbg = self._primary_bucket_guard_dbg("non_hold_action")
            dbg["primary_action"] = action
            return dbg

        current = float(current_time)
        pitch_abs = float(pitch_abs_deg)
        roll_abs = float(roll_abs_deg)
        if self._primary_bucket_guard_window_start_s is None:
            self._primary_bucket_guard_window_start_s = current
            self._primary_bucket_guard_pitch_max_deg = pitch_abs
            self._primary_bucket_guard_roll_max_deg = roll_abs
            reason = "latched" if self._primary_bucket_guard_active else "collecting"
            self._primary_bucket_guard_reason = reason
            return self._primary_bucket_guard_dbg(reason)

        self._primary_bucket_guard_pitch_max_deg = max(
            self._primary_bucket_guard_pitch_max_deg, pitch_abs
        )
        self._primary_bucket_guard_roll_max_deg = max(
            self._primary_bucket_guard_roll_max_deg, roll_abs
        )

        elapsed = current - float(self._primary_bucket_guard_window_start_s)
        if elapsed < self.primary_safety_bucket_guard_bucket_s:
            reason = "latched" if self._primary_bucket_guard_active else "collecting"
            self._primary_bucket_guard_reason = reason
            return self._primary_bucket_guard_dbg(reason)

        cur_pitch_max = float(self._primary_bucket_guard_pitch_max_deg)
        cur_roll_max = float(self._primary_bucket_guard_roll_max_deg)
        prev_pitch_max = float(self._primary_bucket_guard_prev_pitch_max_deg)
        prev_roll_max = float(self._primary_bucket_guard_prev_roll_max_deg)
        have_prev = bool(
            np.isfinite(prev_pitch_max) and np.isfinite(prev_roll_max)
        )
        pitch_high = cur_pitch_max > self.primary_safety_bucket_guard_pitch_enter_deg
        roll_high = cur_roll_max > self.primary_safety_bucket_guard_roll_enter_deg
        high = bool(pitch_high or roll_high)
        current_pitch_high = pitch_abs > self.primary_safety_bucket_guard_pitch_enter_deg
        current_roll_high = roll_abs > self.primary_safety_bucket_guard_roll_enter_deg
        current_high = bool(current_pitch_high or current_roll_high)
        if have_prev:
            pitch_not_improving = (
                pitch_high
                and cur_pitch_max
                >= prev_pitch_max - self.primary_safety_bucket_guard_improve_tol_deg
            )
            roll_not_improving = (
                roll_high
                and cur_roll_max
                >= prev_roll_max - self.primary_safety_bucket_guard_improve_tol_deg
            )
            not_improving = bool(pitch_not_improving or roll_not_improving)
        else:
            not_improving = False

        exit_clean = (
            cur_pitch_max < self.primary_safety_bucket_guard_pitch_exit_deg
            and cur_roll_max < self.primary_safety_bucket_guard_roll_exit_deg
        )
        enter_current_ok = (
            current_high or not self.primary_safety_bucket_guard_require_current_high
        )

        reason = "bucket_complete"
        if self._primary_bucket_guard_active:
            self._primary_bucket_guard_active_windows += 1
            if (
                self._primary_bucket_guard_active_windows
                >= self.primary_safety_bucket_guard_max_active_windows
            ):
                self._primary_bucket_guard_active = False
                self._primary_bucket_guard_clean_windows = 0
                self._primary_bucket_guard_active_windows = 0
                reason = "exit_bucket_timeout"
            elif exit_clean:
                self._primary_bucket_guard_clean_windows += 1
                if (
                    self._primary_bucket_guard_clean_windows
                    >= self.primary_safety_bucket_guard_exit_required_windows
                ):
                    self._primary_bucket_guard_active = False
                    self._primary_bucket_guard_clean_windows = 0
                    reason = "exit_bucket_clean"
                else:
                    reason = "exit_bucket_counting"
            else:
                self._primary_bucket_guard_clean_windows = 0
                reason = "latched_bucket_dirty"
        elif high and enter_current_ok and not_improving:
            self._primary_bucket_guard_active = True
            self._primary_bucket_guard_clean_windows = 0
            self._primary_bucket_guard_active_windows = 0
            reason = "enter_bucket_not_improving"
        elif high and not current_high:
            reason = "high_peak_resolved"
        elif high:
            reason = "high_but_improving"
        else:
            self._primary_bucket_guard_clean_windows = 0
            reason = "free"

        dbg = self._primary_bucket_guard_dbg(
            reason,
            high=high,
            not_improving=not_improving,
            exit_clean=exit_clean,
        )
        dbg["current_high"] = int(current_high)
        dbg["hold_only"] = int(self.primary_safety_bucket_guard_hold_only)
        dbg["primary_action"] = action
        self._primary_bucket_guard_prev_pitch_max_deg = cur_pitch_max
        self._primary_bucket_guard_prev_roll_max_deg = cur_roll_max
        self._primary_bucket_guard_completed_windows += 1
        self._primary_bucket_guard_window_start_s = current
        self._primary_bucket_guard_pitch_max_deg = pitch_abs
        self._primary_bucket_guard_roll_max_deg = roll_abs
        self._primary_bucket_guard_reason = reason
        return dbg

    def _primary_safety_update(
        self,
        state,
        current_time,
        primary_enabled,
        primary_candidate_applied,
        primary_action="",
    ):
        pitch_deg = float(np.degrees(state[4]))
        roll_deg = float(np.degrees(state[3]))
        pitch_abs = abs(pitch_deg)
        roll_abs = abs(roll_deg)
        env_norm = self._primary_safety_env_norm(pitch_abs, roll_abs)
        step_dt = self._primary_safety_step_dt(current_time)

        pitch_enter = pitch_abs > self.primary_safety_pitch_enter_deg
        roll_enter = roll_abs > self.primary_safety_roll_enter_deg
        env_enter = (
            self.primary_safety_use_envelope
            and env_norm > self.primary_safety_envelope_enter_norm
        )
        normal_enter = bool(pitch_enter or roll_enter or env_enter)
        if self.primary_safety_hold_risk_gate_enabled:
            action_gate = str(primary_action or "") == self.primary_safety_hold_risk_gate_action
            normal_enter = bool(normal_enter and action_gate)
        emergency_pitch_enter = (
            self.primary_safety_emergency_pitch_enter_deg > 0.0
            and pitch_abs > self.primary_safety_emergency_pitch_enter_deg
        )
        emergency_roll_enter = (
            self.primary_safety_emergency_roll_enter_deg > 0.0
            and roll_abs > self.primary_safety_emergency_roll_enter_deg
        )
        emergency_enter = bool(emergency_pitch_enter or emergency_roll_enter)

        reason = "disabled"
        if not self.primary_safety_enabled:
            self._primary_safety_active = False
            self._primary_safety_enter_elapsed_s = 0.0
            self._primary_safety_exit_clean_windows = 0
        elif not (primary_enabled and primary_candidate_applied):
            self._primary_safety_active = False
            self._primary_safety_enter_elapsed_s = 0.0
            self._primary_safety_exit_clean_windows = 0
            reason = "primary_not_applied"
        else:
            if self._primary_safety_active:
                reason = "latched"
                self._primary_safety_enter_elapsed_s = 0.0
                if self.primary_safety_exit_hold_s <= 0.0:
                    hold_elapsed = 0.0
                    exit_clean = (
                        pitch_abs < self.primary_safety_pitch_exit_deg
                        and roll_abs < self.primary_safety_roll_exit_deg
                    )
                    if exit_clean:
                        self._primary_safety_active = False
                        reason = "exit_instant_clean"
                        self._primary_safety_exit_clean_windows = 0
                    else:
                        reason = "exit_instant_dirty"
                    self._primary_safety_reset_exit_window(current_time, pitch_abs, roll_abs)
                else:
                    self._primary_safety_update_window(current_time, pitch_abs, roll_abs)
                    hold_elapsed = (
                        0.0
                        if self._primary_safety_exit_window_start_s is None
                        else float(current_time) - float(self._primary_safety_exit_window_start_s)
                    )
                    if hold_elapsed >= self.primary_safety_exit_hold_s:
                        exit_clean = (
                            self._primary_safety_exit_window_max_pitch_abs_deg
                            < self.primary_safety_pitch_exit_deg
                            and self._primary_safety_exit_window_max_roll_abs_deg
                            < self.primary_safety_roll_exit_deg
                        )
                        if exit_clean:
                            self._primary_safety_exit_clean_windows += 1
                        else:
                            self._primary_safety_exit_clean_windows = 0
                        if self._primary_safety_exit_clean_windows >= self.primary_safety_exit_required_windows:
                            self._primary_safety_active = False
                            reason = "exit_hysteresis_clean"
                            self._primary_safety_exit_clean_windows = 0
                        else:
                            reason = "exit_window_dirty" if not exit_clean else "exit_window_counting"
                        self._primary_safety_reset_exit_window(current_time, pitch_abs, roll_abs)
            elif emergency_enter:
                self._primary_safety_active = True
                self._primary_safety_enter_elapsed_s = 0.0
                self._primary_safety_exit_clean_windows = 0
                self._primary_safety_reset_exit_window(current_time, pitch_abs, roll_abs)
                if emergency_pitch_enter:
                    reason = "enter_emergency_pitch"
                else:
                    reason = "enter_emergency_roll"
            elif normal_enter:
                if self.primary_safety_enter_hold_s <= 0.0:
                    self._primary_safety_enter_elapsed_s = self.primary_safety_enter_hold_s
                else:
                    self._primary_safety_enter_elapsed_s += step_dt
                if self._primary_safety_enter_elapsed_s >= self.primary_safety_enter_hold_s:
                    self._primary_safety_active = True
                    self._primary_safety_enter_elapsed_s = 0.0
                    self._primary_safety_exit_clean_windows = 0
                    self._primary_safety_reset_exit_window(current_time, pitch_abs, roll_abs)
                    reason = self._primary_safety_reason(
                        "enter_persistent", pitch_enter, roll_enter, env_enter
                    )
                else:
                    reason = self._primary_safety_reason(
                        "enter_counting", pitch_enter, roll_enter, env_enter
                    )
            else:
                reason = "free"
                self._primary_safety_enter_elapsed_s = 0.0
                self._primary_safety_exit_clean_windows = 0

        bucket_guard_dbg = self._primary_safety_bucket_guard_update(
            current_time=current_time,
            pitch_abs_deg=pitch_abs,
            roll_abs_deg=roll_abs,
            primary_enabled=bool(primary_enabled),
            primary_candidate_applied=bool(primary_candidate_applied),
            primary_action=primary_action,
        )
        hard_fallback = bool(self.primary_safety_enabled and self._primary_safety_active)
        bucket_fallback = bool(bucket_guard_dbg["active"])
        fallback = bool(hard_fallback or bucket_fallback)
        if hard_fallback and bucket_fallback:
            fallback_source = "hard+bucket_guard"
            combined_reason = f"{reason}+bucket_guard:{bucket_guard_dbg['reason']}"
        elif bucket_fallback:
            fallback_source = "bucket_guard"
            combined_reason = f"bucket_guard:{bucket_guard_dbg['reason']}"
        elif hard_fallback:
            fallback_source = "hard"
            combined_reason = str(reason)
        else:
            fallback_source = "none"
            combined_reason = str(reason)

        return {
            "enabled": int(self.primary_safety_enabled),
            "active": int(fallback),
            "fallback": int(fallback),
            "fallback_source": str(fallback_source),
            "reason": str(combined_reason),
            "hard_active": int(hard_fallback),
            "pitch_deg": float(pitch_deg),
            "roll_deg": float(roll_deg),
            "pitch_abs_deg": float(pitch_abs),
            "roll_abs_deg": float(roll_abs),
            "env_norm": float(env_norm),
            "use_envelope": int(self.primary_safety_use_envelope),
            "envelope_enter_norm": float(self.primary_safety_envelope_enter_norm),
            "normal_enter": int(normal_enter),
            "emergency_enter": int(emergency_enter),
            "enter_elapsed_s": float(self._primary_safety_enter_elapsed_s),
            "enter_hold_s": float(self.primary_safety_enter_hold_s),
            "pitch_enter_deg": float(self.primary_safety_pitch_enter_deg),
            "roll_enter_deg": float(self.primary_safety_roll_enter_deg),
            "emergency_pitch_enter_deg": float(self.primary_safety_emergency_pitch_enter_deg),
            "emergency_roll_enter_deg": float(self.primary_safety_emergency_roll_enter_deg),
            "pitch_exit_deg": float(self.primary_safety_pitch_exit_deg),
            "roll_exit_deg": float(self.primary_safety_roll_exit_deg),
            "exit_window_max_pitch_abs_deg": float(
                self._primary_safety_exit_window_max_pitch_abs_deg
            ),
            "exit_window_max_roll_abs_deg": float(
                self._primary_safety_exit_window_max_roll_abs_deg
            ),
            "exit_env_norm": float(self._primary_safety_exit_env_norm()),
            "exit_clean_windows": int(self._primary_safety_exit_clean_windows),
            "exit_hold_s": float(self.primary_safety_exit_hold_s),
            "exit_required_windows": int(self.primary_safety_exit_required_windows),
            "bucket_guard": bucket_guard_dbg,
        }

    def _deadband_target_release_update(
        self,
        state,
        plant_info_prev,
        ctrl_dbg,
        m_cmd_applied,
        m_cmd_reference,
    ):
        zero = {
            "active": 0,
            "reason": "disabled" if not self.deadband_target_release_enabled else "inactive",
            "delta_mean_kg": 0.0,
            "blend": float(self.deadband_target_release_blend),
            "pitch_ok": 0,
            "roll_ok": 0,
            "exit_pitch_ok": 0,
            "exit_roll_ok": 0,
            "latched": int(self._deadband_target_release_latched),
            "reset_limiter": 0,
        }
        if not self.deadband_target_release_enabled:
            self._deadband_target_release_latched = False
            return m_cmd_applied, zero
        if (
            self.deadband_target_release_require_forecast_safe
            and not bool(self._forecast_safe_deadband_dbg.get("active", False))
        ):
            self._deadband_target_release_latched = False
            zero["reason"] = "forecast_not_safe"
            zero["latched"] = 0
            return m_cmd_applied, zero
        if plant_info_prev is None or "tank_masses" not in plant_info_prev:
            zero["reason"] = "missing_tank_feedback"
            return m_cmd_applied, zero

        masses = np.asarray(plant_info_prev["tank_masses"], dtype=float).reshape(-1)
        if masses.size < 3:
            zero["reason"] = "invalid_tank_feedback"
            return m_cmd_applied, zero
        masses = masses[:3]

        pid_details = ctrl_dbg.get("pid_details", {}) if isinstance(ctrl_dbg, dict) else {}
        pitch_pid = pid_details.get("pitch", {})
        roll_pid = pid_details.get("roll", {})
        pitch_limit = self.deadband_target_release_pitch_deg
        roll_limit = self.deadband_target_release_roll_deg
        if pitch_limit <= 0.0:
            pitch_limit = float(pitch_pid.get("deadband_exit", 0.5))
        if roll_limit <= 0.0:
            roll_limit = float(roll_pid.get("deadband_exit", 0.4))
        exit_pitch_limit = max(
            self.deadband_target_release_exit_pitch_deg,
            pitch_limit,
        )
        exit_roll_limit = max(
            self.deadband_target_release_exit_roll_deg,
            roll_limit,
        )

        pitch_deg = float(np.degrees(state[4]))
        roll_deg = float(np.degrees(state[3]))
        pitch_rate_deg_s = float(np.degrees(state[10])) if len(state) > 10 else 0.0
        roll_rate_deg_s = float(np.degrees(state[9])) if len(state) > 9 else 0.0

        def axis_ok(value_deg, rate_deg_s, in_deadband, limit_deg):
            pid_ready = bool(in_deadband) or not self.deadband_target_release_use_pid_deadband
            inside = pid_ready and abs(float(value_deg)) <= max(float(limit_deg), 1e-9)
            near_zero = abs(float(value_deg)) <= self.deadband_target_release_near_zero_deg
            moving_to_zero = float(value_deg) * float(rate_deg_s) <= self.deadband_target_release_rate_eps_deg_s
            return bool(inside and (near_zero or moving_to_zero))

        pitch_ok = axis_ok(
            pitch_deg,
            pitch_rate_deg_s,
            int(pitch_pid.get("in_deadband", 0)) > 0,
            pitch_limit,
        )
        roll_ok = axis_ok(
            roll_deg,
            roll_rate_deg_s,
            int(roll_pid.get("in_deadband", 0)) > 0,
            roll_limit,
        )
        entry_active = (
            (pitch_ok and roll_ok)
            if self.deadband_target_release_require_both_axes
            else (pitch_ok or roll_ok)
        )
        exit_pitch_ok = abs(pitch_deg) <= max(exit_pitch_limit, 1e-9)
        exit_roll_ok = abs(roll_deg) <= max(exit_roll_limit, 1e-9)
        exit_active = (
            (exit_pitch_ok and exit_roll_ok)
            if self.deadband_target_release_require_both_axes
            else (exit_pitch_ok or exit_roll_ok)
        )
        latched_active = bool(self._deadband_target_release_latched and exit_active)
        active = bool(entry_active or latched_active)
        if not active:
            reason = "exit_deadband_hysteresis" if self._deadband_target_release_latched else "axis_not_ready"
            self._deadband_target_release_latched = False
            zero.update(
                {
                    "reason": reason,
                    "pitch_ok": int(pitch_ok),
                    "roll_ok": int(roll_ok),
                    "exit_pitch_ok": int(exit_pitch_ok),
                    "exit_roll_ok": int(exit_roll_ok),
                    "latched": 0,
                }
            )
            return m_cmd_applied, zero

        self._deadband_target_release_latched = True
        cmd = np.asarray(m_cmd_applied, dtype=float).reshape(-1)[:3]
        blend = float(self.deadband_target_release_blend)
        released = masses + (1.0 - blend) * (cmd - masses)
        released = np.clip(released, 0.0, self.controller.max_mass)
        delta_mean = float(np.mean(np.abs(cmd - released)))
        if self.deadband_target_release_reset_limiter and self.target_slew_limiter is not None:
            self.target_slew_limiter.reset(initial_target=released)
            reset_limiter = 1
        else:
            reset_limiter = 0
        return released, {
            "active": 1,
            "reason": "inside_deadband_moving_to_zero" if entry_active else "latched_deadband_hysteresis",
            "delta_mean_kg": delta_mean,
            "blend": blend,
            "pitch_ok": int(pitch_ok),
            "roll_ok": int(roll_ok),
            "exit_pitch_ok": int(exit_pitch_ok),
            "exit_roll_ok": int(exit_roll_ok),
            "latched": 1,
            "reset_limiter": int(reset_limiter),
            "pitch_deg": float(pitch_deg),
            "roll_deg": float(roll_deg),
            "pitch_rate_deg_s": float(pitch_rate_deg_s),
            "roll_rate_deg_s": float(roll_rate_deg_s),
            "pitch_limit_deg": float(pitch_limit),
            "roll_limit_deg": float(roll_limit),
            "reference_gap_mean_kg": float(np.mean(np.abs(np.asarray(m_cmd_reference, dtype=float).reshape(-1)[:3] - masses))),
        }

    def _reactive_pump_suppression_update(
        self,
        state,
        plant_info_prev,
        m_cmd_reference,
        primary_applied=False,
    ):
        cmd = np.asarray(m_cmd_reference, dtype=float).reshape(-1)
        if cmd.size < 3:
            cmd3 = np.zeros(3, dtype=float)
        else:
            cmd3 = cmd[:3].copy()
        pitch_abs = abs(float(np.degrees(state[4])))
        roll_abs = abs(float(np.degrees(state[3])))
        zero = {
            "active": 0,
            "reason": "disabled" if not self.reactive_pump_suppression_enabled else "inactive",
            "blocked_tanks": 0,
            "blocked_mass_kg": 0.0,
            "delta_mean_kg": 0.0,
            "pitch_abs_deg": float(pitch_abs),
            "roll_abs_deg": float(roll_abs),
            "pitch_enter_deg": float(self.reactive_pump_suppression_pitch_enter_deg),
            "roll_enter_deg": float(self.reactive_pump_suppression_roll_enter_deg),
            "pitch_exit_deg": float(self.reactive_pump_suppression_pitch_exit_deg),
            "roll_exit_deg": float(self.reactive_pump_suppression_roll_exit_deg),
            "restart_err_kg": float(self.reactive_pump_suppression_restart_err_kg),
            "safe_zone": 0,
            "latched": int(self._reactive_pump_suppression_latched),
            "fullspeed_block": 0,
            "mask_t1": 0,
            "mask_t2": 0,
            "mask_t3": 0,
        }
        if not self.reactive_pump_suppression_enabled:
            self._reactive_pump_suppression_latched = False
            return cmd3, zero
        if bool(primary_applied):
            self._reactive_pump_suppression_latched = False
            zero["reason"] = "preview_primary_applied"
            zero["latched"] = 0
            return cmd3, zero
        if plant_info_prev is None or "tank_masses" not in plant_info_prev:
            self._reactive_pump_suppression_latched = False
            zero["reason"] = "missing_tank_feedback"
            zero["latched"] = 0
            return cmd3, zero
        if (
            self.reactive_pump_suppression_require_not_fullspeed
            and int(plant_info_prev.get("pump_fullspeed_any", 0)) > 0
        ):
            self._reactive_pump_suppression_latched = False
            zero.update({"reason": "fullspeed_guard", "fullspeed_block": 1, "latched": 0})
            return cmd3, zero

        masses = np.asarray(plant_info_prev["tank_masses"], dtype=float).reshape(-1)
        if masses.size < 3:
            self._reactive_pump_suppression_latched = False
            zero["reason"] = "invalid_tank_feedback"
            zero["latched"] = 0
            return cmd3, zero
        masses = masses[:3]

        entry_safe = (
            pitch_abs <= self.reactive_pump_suppression_pitch_enter_deg
            and roll_abs <= self.reactive_pump_suppression_roll_enter_deg
        )
        exit_safe = (
            pitch_abs <= max(
                self.reactive_pump_suppression_pitch_exit_deg,
                self.reactive_pump_suppression_pitch_enter_deg,
            )
            and roll_abs <= max(
                self.reactive_pump_suppression_roll_exit_deg,
                self.reactive_pump_suppression_roll_enter_deg,
            )
        )
        safe_zone = bool(entry_safe or (self._reactive_pump_suppression_latched and exit_safe))
        if not safe_zone:
            reason = (
                "exit_safe_zone_hysteresis"
                if self._reactive_pump_suppression_latched
                else "attitude_not_safe"
            )
            self._reactive_pump_suppression_latched = False
            zero.update({"reason": reason, "safe_zone": 0, "latched": 0})
            return cmd3, zero

        self._reactive_pump_suppression_latched = True
        latched_raw = plant_info_prev.get("pump_latched", [False, False, False])
        latched = np.asarray(latched_raw, dtype=bool).reshape(-1)
        if latched.size < 3:
            latched = np.pad(latched, (0, 3 - latched.size), constant_values=False)
        latched = latched[:3]
        err = cmd3 - masses
        abs_err = np.abs(err)
        restart_err = float(self.reactive_pump_suppression_restart_err_kg)
        mask = (~latched) & (abs_err > 1e-6) & (abs_err < restart_err)
        suppressed = cmd3
        blocked_mass = 0.0
        if np.any(mask):
            suppressed = cmd3.copy()
            blocked_mass = float(np.sum(np.abs(err[mask])))
            suppressed[mask] = masses[mask]
        delta_mean = float(np.mean(np.abs(suppressed - cmd3)))
        reason = "safe_zone_small_unlatched_error" if np.any(mask) else "safe_zone_no_small_error"
        return suppressed, {
            "active": 1,
            "reason": reason,
            "blocked_tanks": int(np.sum(mask)),
            "blocked_mass_kg": float(blocked_mass),
            "delta_mean_kg": float(delta_mean),
            "pitch_abs_deg": float(pitch_abs),
            "roll_abs_deg": float(roll_abs),
            "pitch_enter_deg": float(self.reactive_pump_suppression_pitch_enter_deg),
            "roll_enter_deg": float(self.reactive_pump_suppression_roll_enter_deg),
            "pitch_exit_deg": float(self.reactive_pump_suppression_pitch_exit_deg),
            "roll_exit_deg": float(self.reactive_pump_suppression_roll_exit_deg),
            "restart_err_kg": float(restart_err),
            "safe_zone": 1,
            "latched": 1,
            "fullspeed_block": 0,
            "mask_t1": int(mask[0]) if mask.size > 0 else 0,
            "mask_t2": int(mask[1]) if mask.size > 1 else 0,
            "mask_t3": int(mask[2]) if mask.size > 2 else 0,
        }

    def _commit_target_transaction(
        self,
        *,
        state,
        plant_info_prev,
        ctrl_dbg,
        preview_trim_bias,
        primary_safety_dbg,
        primary_applied,
        primary_target,
    ):
        """Apply ordered target adjustments and return one committed target."""
        preview = preview_trim_bias if isinstance(preview_trim_bias, dict) else {}

        mass_ff_raw = preview.get("preview_mass_ff_kg")
        if mass_ff_raw is not None:
            mass_ff = np.asarray(mass_ff_raw, dtype=float).reshape(-1)
            if mass_ff.size >= 3:
                mass_ff = mass_ff[:3]
                target_with_ff = np.clip(
                    primary_target + mass_ff,
                    0.0,
                    self.controller.max_mass,
                )
            else:
                mass_ff = np.zeros(3, dtype=float)
                target_with_ff = primary_target
        else:
            mass_ff = np.zeros(3, dtype=float)
            target_with_ff = primary_target
        if primary_safety_dbg["fallback"]:
            mass_ff = np.zeros(3, dtype=float)
            target_with_ff = primary_target

        suppression_active = int(preview.get("preview_pump_suppression_active", 0))
        suppression_restart_err_kg = float(
            preview.get("preview_pump_restart_err_kg", 0.0)
        )
        suppression_reason = str(preview.get("preview_pump_suppression_reason", ""))
        suppression_blocked_tanks = 0
        suppression_blocked_mass_kg = 0.0
        suppression_mask = np.zeros(3, dtype=bool)
        target_after_preview_suppression = target_with_ff
        if (
            suppression_active
            and suppression_restart_err_kg > 0.0
            and plant_info_prev is not None
            and "tank_masses" in plant_info_prev
        ):
            current_masses = np.asarray(
                plant_info_prev["tank_masses"], dtype=float
            ).reshape(-1)
            if current_masses.size >= 3:
                current_masses = current_masses[:3]
                latched_raw = plant_info_prev.get(
                    "pump_latched", [False, False, False]
                )
                latched = np.asarray(latched_raw, dtype=bool).reshape(-1)
                if latched.size < 3:
                    latched = np.pad(
                        latched,
                        (0, 3 - latched.size),
                        constant_values=False,
                    )
                latched = latched[:3]
                err = target_with_ff - current_masses
                abs_err = np.abs(err)
                suppression_mask = (
                    (~latched)
                    & (abs_err > 1e-6)
                    & (abs_err < suppression_restart_err_kg)
                )
                if np.any(suppression_mask):
                    target_after_preview_suppression = target_with_ff.copy()
                    suppression_blocked_mass_kg = float(
                        np.sum(np.abs(err[suppression_mask]))
                    )
                    target_after_preview_suppression[suppression_mask] = current_masses[
                        suppression_mask
                    ]
                    suppression_blocked_tanks = int(np.sum(suppression_mask))

        target_pre_execution, reactive_suppression_dbg = (
            self._reactive_pump_suppression_update(
                state=state,
                plant_info_prev=plant_info_prev,
                m_cmd_reference=target_after_preview_suppression,
                primary_applied=bool(primary_applied),
            )
        )

        if self.target_slew_limiter is not None:
            target_after_slew_limiter, cmd_gap = self.target_slew_limiter.update(
                target_pre_execution
            )
        else:
            target_after_slew_limiter = target_pre_execution
            cmd_gap = 0.0
        target_after_slew_limiter = np.asarray(
            target_after_slew_limiter, dtype=float
        ).copy()

        final_target, release_dbg = self._deadband_target_release_update(
            state=state,
            plant_info_prev=plant_info_prev,
            ctrl_dbg=ctrl_dbg,
            m_cmd_applied=target_after_slew_limiter,
            m_cmd_reference=target_pre_execution,
        )
        final_target = np.asarray(final_target, dtype=float)
        if int(release_dbg.get("active", 0)):
            cmd_gap = float(np.mean(np.abs(target_pre_execution - final_target)))

        feedforward_delta_mean = float(np.mean(np.abs(mass_ff)))
        preview_suppression_delta_mean = float(
            np.mean(np.abs(target_after_preview_suppression - target_with_ff))
        )
        reactive_suppression_delta_mean = float(
            reactive_suppression_dbg.get("delta_mean_kg", 0.0)
        )
        suppression_delta_mean = float(
            np.mean(np.abs(target_pre_execution - target_with_ff))
        )
        target_slew_delta_mean = float(
            np.mean(np.abs(target_after_slew_limiter - target_pre_execution))
        )
        trace = describe_target_path(
            primary_applied=bool(primary_applied),
            primary_safety_fallback=bool(primary_safety_dbg["fallback"]),
            feedforward_delta_kg=feedforward_delta_mean,
            preview_suppression_delta_kg=preview_suppression_delta_mean,
            reactive_suppression_delta_kg=reactive_suppression_delta_mean,
            target_slew_delta_kg=target_slew_delta_mean,
            deadband_release_active=bool(release_dbg.get("active", 0)),
        )

        return {
            "final_target": final_target,
            "target_pre_execution": np.asarray(
                target_pre_execution, dtype=float
            ).copy(),
            "target_after_slew_limiter": target_after_slew_limiter,
            "mass_ff": mass_ff,
            "cmd_gap": float(cmd_gap),
            "suppression_active": suppression_active,
            "suppression_restart_err_kg": suppression_restart_err_kg,
            "suppression_reason": suppression_reason,
            "suppression_blocked_tanks": suppression_blocked_tanks,
            "suppression_blocked_mass_kg": suppression_blocked_mass_kg,
            "suppression_mask": suppression_mask,
            "reactive_suppression_dbg": reactive_suppression_dbg,
            "release_dbg": release_dbg,
            "feedforward_delta_mean_kg": feedforward_delta_mean,
            "suppression_delta_mean_kg": suppression_delta_mean,
            "preview_suppression_delta_mean_kg": preview_suppression_delta_mean,
            "reactive_suppression_delta_mean_kg": reactive_suppression_delta_mean,
            "target_slew_delta_mean_kg": target_slew_delta_mean,
            "trace": trace,
        }

    def _preview_trim_bias(self, state, wind_obs, plant_info_prev, current_time):
        if self.preview_trim_provider is None:
            return None
        provider = self.preview_trim_provider
        if hasattr(provider, "compute"):
            return provider.compute(
                state=state,
                wind_obs=wind_obs,
                plant_info_prev=plant_info_prev,
                current_time=current_time,
            )
        return provider(
            state=state,
            wind_obs=wind_obs,
            plant_info_prev=plant_info_prev,
            current_time=current_time,
        )

    def _resolve_cycle_setpoints(
        self,
        *,
        state,
        wind_obs,
        plant_info_prev,
        current_time,
    ):
        preview_trim_bias = self._preview_trim_bias(
            state=state,
            wind_obs=wind_obs,
            plant_info_prev=plant_info_prev,
            current_time=current_time,
        )
        forecast_safe_deadband_dbg = self._apply_forecast_safe_deadband(
            state,
            preview_trim_bias,
            plant_info_prev=plant_info_prev,
            current_time=current_time,
        )
        if self.trim_governor is not None:
            trim_dbg = self.trim_governor.update(
                wind_obs,
                plant_info_prev=plant_info_prev,
                state=state,
                setpoints=self.controller.setpoints,
                preview_trim_bias=preview_trim_bias,
            )
            pitch_sp_raw = trim_dbg["pitch_sp_raw_deg"]
            roll_sp_raw = trim_dbg["roll_sp_raw_deg"]
        else:
            (preview_pitch, preview_roll), preview_dbg = _coerce_preview_trim_bias(preview_trim_bias)
            trim_dbg = {
                "trim_scale": 0.0,
                "trim_enabled": 0,
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
                "steady_mean_abs_err_deg": 0.0,
                "steady_std_err_deg": 0.0,
                "trim_freeze": 0,
                "trim_update_tick": 0,
                "target_mode": "g1",
                "target_source": "no_trim",
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
                "current_pitch_trim_raw_deg": 0.0,
                "current_roll_trim_raw_deg": 0.0,
                "preview_pitch_bias_deg": float(preview_pitch),
                "preview_roll_bias_deg": float(preview_roll),
                "combined_pitch_trim_raw_deg": float(preview_pitch),
                "combined_roll_trim_raw_deg": float(preview_roll),
                "current_pitch_trim_scaled_deg": 0.0,
                "current_roll_trim_scaled_deg": 0.0,
                "preview_pitch_scaled_deg": float(preview_pitch),
                "preview_roll_scaled_deg": float(preview_roll),
                "preview_scale_mode": "none",
                "preview_scale_eff": 1.0 if (abs(preview_pitch) > 1e-12 or abs(preview_roll) > 1e-12) else 0.0,
                "preview_trim_active": int(abs(preview_pitch) > 1e-12 or abs(preview_roll) > 1e-12),
                "preview_trim_source": str(preview_dbg.get("preview_trim_source", preview_dbg.get("source", "no_trim"))),
            }
            for key, value in preview_dbg.items():
                if key not in trim_dbg:
                    trim_dbg[key] = value
            pitch_sp_raw = float(preview_dbg.get("pitch_sp_deg", preview_pitch))
            roll_sp_raw = float(preview_dbg.get("roll_sp_deg", preview_roll))
            trim_dbg["target_source"] = str(trim_dbg.get("selected_plan", "preview_direct"))

        if self.setpoint_shaper is not None:
            pitch_sp, roll_sp = self.setpoint_shaper.update(pitch_sp_raw, roll_sp_raw)
        else:
            pitch_sp, roll_sp = float(pitch_sp_raw), float(roll_sp_raw)

        self.controller.setpoints["pitch"] = float(pitch_sp)
        self.controller.setpoints["roll"] = float(roll_sp)

        return {
            "preview": preview_trim_bias,
            "forecast_safe_deadband": forecast_safe_deadband_dbg,
            "trim_debug": trim_dbg,
            "pitch_sp_raw": float(pitch_sp_raw),
            "roll_sp_raw": float(roll_sp_raw),
            "pitch_sp": float(pitch_sp),
            "roll_sp": float(roll_sp),
        }

    def _compute_feedback_target(
        self,
        *,
        state,
        wind_obs,
        plant_info_prev,
        current_time,
    ):
        # The posture controller remains warm while prediction owns the target,
        # so safety fallback always has a current feedback command available.

        env_info = {
            "wind_speed": float(wind_obs.get("ws", 0.0)) if wind_obs is not None else 0.0,
            "wind_dir_deg": float(wind_obs.get("wd_deg", 0.0)) if wind_obs is not None else 0.0,
        }
        m_cmd_raw, ctrl_dbg = self.controller.compute(state, env_info, current_time)
        m_cmd_raw = np.array(m_cmd_raw, dtype=float)
        if self.heave_balancer is not None:
            masses = m_cmd_raw
            if plant_info_prev is not None and "tank_masses" in plant_info_prev:
                masses = np.asarray(plant_info_prev["tank_masses"], dtype=float)
            m_cmd_heave, hm_dbg = self.heave_balancer.update(
                cmd_raw=m_cmd_raw,
                state=state,
                tank_masses=masses,
                max_capacity=self.controller.max_mass,
            )
        else:
            m_cmd_heave = m_cmd_raw
            hm_dbg = {
                "hm_smoothed_heave_m": float(state[2]),
                "hm_heave_error_m": 0.0,
                "hm_total_ballast_kg": float(np.sum(m_cmd_raw)),
                "ballast_total_kg": float(
                    np.sum(np.asarray(plant_info_prev.get("tank_masses", m_cmd_raw), dtype=float))
                )
                if plant_info_prev is not None
                else float(np.sum(m_cmd_raw)),
                "heave_correction_per_tank_kg": 0.0,
                "heave_balancer_active": 0,
            }

        return {
            "raw_target": m_cmd_raw,
            "feedback_target": m_cmd_heave,
            "controller_debug": ctrl_dbg,
            "heave_debug": hm_dbg,
        }

    def _select_primary_target(
        self,
        *,
        state,
        plant_info_prev,
        current_time,
        preview_trim_bias,
        forecast_safe_deadband_dbg,
        feedback_target,
    ):

        primary_enabled = int(
            preview_trim_bias.get("preview_primary_enabled", 0)
            if isinstance(preview_trim_bias, dict)
            else 0
        )
        primary_active = int(
            preview_trim_bias.get("preview_primary_active", 0)
            if isinstance(preview_trim_bias, dict)
            else 0
        )
        primary_target_raw = (
            preview_trim_bias.get("preview_primary_target_kg")
            if isinstance(preview_trim_bias, dict)
            else None
        )
        primary_delta_raw = (
            preview_trim_bias.get("preview_primary_delta_kg")
            if isinstance(preview_trim_bias, dict)
            else None
        )
        primary_target = np.zeros(3, dtype=float)
        primary_delta = np.zeros(3, dtype=float)
        primary_applied = 0
        primary_candidate_applied = 0
        m_cmd_primary_candidate = feedback_target
        # enabled means the provider owns a valid target, including a zero-change
        # hold. active is diagnostic only and must not decide target ownership.
        if primary_enabled and primary_target_raw is not None:
            arr = np.asarray(primary_target_raw, dtype=float).reshape(-1)
            if arr.size >= 3:
                primary_target = np.clip(arr[:3], 0.0, self.controller.max_mass)
                m_cmd_primary_candidate = primary_target
                primary_candidate_applied = 1
            else:
                m_cmd_primary_candidate = feedback_target
        else:
            m_cmd_primary_candidate = feedback_target
        if primary_delta_raw is not None:
            arr_delta = np.asarray(primary_delta_raw, dtype=float).reshape(-1)
            if arr_delta.size >= 3:
                primary_delta = arr_delta[:3]

        primary_action = (
            str(preview_trim_bias.get("preview_primary_action", ""))
            if isinstance(preview_trim_bias, dict)
            else ""
        )
        forecast_safe_deadband_target_released = 0
        if (
            forecast_safe_deadband_dbg["active"]
            and not self.forecast_safe_deadband_sync_provider_target
            and self.forecast_safe_deadband_release_primary_target
            and primary_action in self.forecast_safe_deadband_allowed_actions
            and plant_info_prev is not None
            and "tank_masses" in plant_info_prev
        ):
            current_masses = np.asarray(
                plant_info_prev["tank_masses"], dtype=float
            ).reshape(-1)
            if current_masses.size >= 3:
                primary_target = np.clip(
                    current_masses[:3], 0.0, self.controller.max_mass
                )
                m_cmd_primary_candidate = primary_target.copy()
                primary_candidate_applied = 1
                forecast_safe_deadband_target_released = 1
        forecast_safe_deadband_hold_rejected = int(
            self.forecast_safe_deadband_enabled
            and self.forecast_safe_deadband_reject_unsafe_hold
            and primary_action in self.forecast_safe_deadband_allowed_actions
            and not forecast_safe_deadband_dbg["active"]
        )
        if forecast_safe_deadband_hold_rejected:
            m_cmd_primary_candidate = feedback_target
            primary_candidate_applied = 0

        primary_safety_dbg = self._primary_safety_update(
            state=state,
            current_time=current_time,
            primary_enabled=bool(primary_enabled),
            primary_candidate_applied=bool(primary_candidate_applied),
            primary_action=(
                primary_action
            ),
        )
        if primary_safety_dbg["fallback"]:
            m_cmd_primary = feedback_target
            primary_applied = 0
        else:
            m_cmd_primary = m_cmd_primary_candidate
            primary_applied = int(primary_candidate_applied)

        return {
            "enabled": primary_enabled,
            "active": primary_active,
            "target": primary_target,
            "delta": primary_delta,
            "action": primary_action,
            "candidate_target": m_cmd_primary_candidate,
            "candidate_applied": primary_candidate_applied,
            "selected_target": m_cmd_primary,
            "applied": primary_applied,
            "safety_debug": primary_safety_dbg,
            "deadband_target_released": forecast_safe_deadband_target_released,
            "deadband_hold_rejected": forecast_safe_deadband_hold_rejected,
        }

    def compute(self, state, wind_obs, plant_info_prev, current_time):
        setpoint_cycle = self._resolve_cycle_setpoints(
            state=state,
            wind_obs=wind_obs,
            plant_info_prev=plant_info_prev,
            current_time=current_time,
        )
        preview_trim_bias = setpoint_cycle["preview"]
        forecast_safe_deadband_dbg = setpoint_cycle["forecast_safe_deadband"]
        trim_dbg = setpoint_cycle["trim_debug"]
        pitch_sp_raw = setpoint_cycle["pitch_sp_raw"]
        roll_sp_raw = setpoint_cycle["roll_sp_raw"]
        pitch_sp = setpoint_cycle["pitch_sp"]
        roll_sp = setpoint_cycle["roll_sp"]

        feedback_cycle = self._compute_feedback_target(
            state=state,
            wind_obs=wind_obs,
            plant_info_prev=plant_info_prev,
            current_time=current_time,
        )
        m_cmd_raw = feedback_cycle["raw_target"]
        m_cmd_heave = feedback_cycle["feedback_target"]
        ctrl_dbg = feedback_cycle["controller_debug"]
        hm_dbg = feedback_cycle["heave_debug"]

        primary_cycle = self._select_primary_target(
            state=state,
            plant_info_prev=plant_info_prev,
            current_time=current_time,
            preview_trim_bias=preview_trim_bias,
            forecast_safe_deadband_dbg=forecast_safe_deadband_dbg,
            feedback_target=m_cmd_heave,
        )
        primary_enabled = primary_cycle["enabled"]
        primary_active = primary_cycle["active"]
        primary_target = primary_cycle["target"]
        primary_delta = primary_cycle["delta"]
        primary_action = primary_cycle["action"]
        m_cmd_primary_candidate = primary_cycle["candidate_target"]
        primary_candidate_applied = primary_cycle["candidate_applied"]
        m_cmd_primary = primary_cycle["selected_target"]
        primary_applied = primary_cycle["applied"]
        primary_safety_dbg = primary_cycle["safety_debug"]
        forecast_safe_deadband_target_released = primary_cycle[
            "deadband_target_released"
        ]
        forecast_safe_deadband_hold_rejected = primary_cycle[
            "deadband_hold_rejected"
        ]

        target_tx = self._commit_target_transaction(
            state=state,
            plant_info_prev=plant_info_prev,
            ctrl_dbg=ctrl_dbg,
            preview_trim_bias=preview_trim_bias,
            primary_safety_dbg=primary_safety_dbg,
            primary_applied=bool(primary_applied),
            primary_target=m_cmd_primary,
        )
        m_cmd_applied = target_tx["final_target"]
        m_cmd_suppressed = target_tx["target_pre_execution"]
        m_cmd_after_slew_limiter = target_tx["target_after_slew_limiter"]
        mass_ff = target_tx["mass_ff"]
        cmd_gap = target_tx["cmd_gap"]
        suppression_active = target_tx["suppression_active"]
        suppression_restart_err_kg = target_tx["suppression_restart_err_kg"]
        suppression_reason = target_tx["suppression_reason"]
        suppression_blocked_tanks = target_tx["suppression_blocked_tanks"]
        suppression_blocked_mass_kg = target_tx["suppression_blocked_mass_kg"]
        suppression_mask = target_tx["suppression_mask"]
        reactive_suppression_dbg = target_tx["reactive_suppression_dbg"]
        release_dbg = target_tx["release_dbg"]
        ff_delta_mean = target_tx["feedforward_delta_mean_kg"]
        suppression_delta_mean = target_tx["suppression_delta_mean_kg"]
        preview_suppression_delta_mean = target_tx[
            "preview_suppression_delta_mean_kg"
        ]
        reactive_suppression_delta_mean = target_tx[
            "reactive_suppression_delta_mean_kg"
        ]
        target_slew_delta_mean = target_tx["target_slew_delta_mean_kg"]
        target_trace = target_tx["trace"]

        heave_bias_delta_mean = float(np.mean(np.abs(m_cmd_heave - m_cmd_raw)))
        primary_delta_mean = float(np.mean(np.abs(m_cmd_primary - m_cmd_heave)))
        primary_candidate_delta_mean = float(np.mean(np.abs(m_cmd_primary_candidate - m_cmd_heave)))
        post_chain_delta_mean = float(np.mean(np.abs(m_cmd_applied - m_cmd_raw)))
        post_chain_adjusted = int(post_chain_delta_mean > 1e-6)
        ctrl_base_clipped = int(ctrl_dbg.get("clipped", 0))
        ctrl_chain_clipped = int(ctrl_base_clipped or post_chain_adjusted)
        raw_map = ctrl_dbg.get("raw_vals", {})
        filtered_map = ctrl_dbg.get("filtered_vals", {})
        pid_details = ctrl_dbg.get("pid_details", {})
        deadband_state = ctrl_dbg.get("deadband_state", {})
        pitch_pid = pid_details.get("pitch", {})
        roll_pid = pid_details.get("roll", {})
        heave_pid = pid_details.get("heave", {})
        alloc_raw = np.asarray(ctrl_dbg.get("alloc_m_raw_kg", m_cmd_raw), dtype=float).reshape(-1)
        alloc_cmd = np.asarray(ctrl_dbg.get("alloc_m_cmds_kg", m_cmd_raw), dtype=float).reshape(-1)
        n_alloc = min(alloc_raw.size, alloc_cmd.size)
        if n_alloc > 0:
            alloc_clip_any = int(np.any(np.abs(alloc_raw[:n_alloc] - alloc_cmd[:n_alloc]) > 1e-6))
        else:
            alloc_clip_any = int(ctrl_base_clipped)

        dbg = {
            **target_trace,
            "target_feedback_kg": np.asarray(m_cmd_heave, dtype=float).copy(),
            "target_prediction_candidate_kg": np.asarray(
                m_cmd_primary_candidate, dtype=float
            ).copy(),
            "target_intent_kg": np.asarray(m_cmd_primary, dtype=float).copy(),
            "target_pre_execution_kg": np.asarray(m_cmd_suppressed, dtype=float).copy(),
            "target_after_slew_limiter_kg": m_cmd_after_slew_limiter.copy(),
            # Retained for readers of historical time-series files.
            "target_after_rate_limiter_kg": m_cmd_after_slew_limiter.copy(),
            "target_final_kg": np.asarray(m_cmd_applied, dtype=float).copy(),
            "pitch_sp_deg": float(pitch_sp),
            "roll_sp_deg": float(roll_sp),
            "pitch_sp_raw_deg": float(pitch_sp_raw),
            "roll_sp_raw_deg": float(roll_sp_raw),
            "current_pitch_trim_raw_deg": float(trim_dbg.get("current_pitch_trim_raw_deg", 0.0)),
            "current_roll_trim_raw_deg": float(trim_dbg.get("current_roll_trim_raw_deg", 0.0)),
            "preview_pitch_bias_deg": float(trim_dbg.get("preview_pitch_bias_deg", 0.0)),
            "preview_roll_bias_deg": float(trim_dbg.get("preview_roll_bias_deg", 0.0)),
            "combined_pitch_trim_raw_deg": float(trim_dbg.get("combined_pitch_trim_raw_deg", 0.0)),
            "combined_roll_trim_raw_deg": float(trim_dbg.get("combined_roll_trim_raw_deg", 0.0)),
            "current_pitch_trim_scaled_deg": float(trim_dbg.get("current_pitch_trim_scaled_deg", 0.0)),
            "current_roll_trim_scaled_deg": float(trim_dbg.get("current_roll_trim_scaled_deg", 0.0)),
            "preview_pitch_scaled_deg": float(trim_dbg.get("preview_pitch_scaled_deg", 0.0)),
            "preview_roll_scaled_deg": float(trim_dbg.get("preview_roll_scaled_deg", 0.0)),
            "preview_scale_mode": str(trim_dbg.get("preview_scale_mode", "none")),
            "preview_scale_eff": float(trim_dbg.get("preview_scale_eff", 0.0)),
            "preview_trim_active": int(trim_dbg.get("preview_trim_active", 0)),
            "preview_trim_source": str(trim_dbg.get("preview_trim_source", "none")),
            "preview_event_risk_prob_0_20m": float(
                trim_dbg.get("preview_event_risk_prob_0_20m", 0.0)
            ),
            "preview_event_risk_prob_20_40m": float(
                trim_dbg.get("preview_event_risk_prob_20_40m", 0.0)
            ),
            "preview_event_risk_prob_40_60m": float(
                trim_dbg.get("preview_event_risk_prob_40_60m", 0.0)
            ),
            "trim_scale": float(trim_dbg["trim_scale"]),
            "trim_enabled": int(trim_dbg.get("trim_enabled", 0)),
            "sat_recent_ratio": float(trim_dbg["sat_recent_ratio"]),
            "trim_pressure_recent": float(
                trim_dbg.get("trim_pressure_recent", trim_dbg.get("sat_recent_ratio", 0.0))
            ),
            "trim_pressure_clip_recent": float(trim_dbg.get("trim_pressure_clip_recent", 0.0)),
            "trim_pressure_cmd_gap_recent": float(
                trim_dbg.get("trim_pressure_cmd_gap_recent", 0.0)
            ),
            "trim_pressure_fullspeed_recent": float(
                trim_dbg.get("trim_pressure_fullspeed_recent", 0.0)
            ),
            "trim_pressure_backlog_recent": float(
                trim_dbg.get("trim_pressure_backlog_recent", 0.0)
            ),
            "trim_pressure_sample": float(trim_dbg.get("trim_pressure_sample", 0.0)),
            "trim_pressure_backlog_max_kg": float(
                trim_dbg.get("trim_pressure_backlog_max_kg", 0.0)
            ),
            "trim_pressure_cmd_gap_kg": float(
                trim_dbg.get("trim_pressure_cmd_gap_kg", 0.0)
            ),
            "trim_pressure_clip": int(trim_dbg.get("trim_pressure_clip", 0)),
            "trim_pressure_fullspeed": int(trim_dbg.get("trim_pressure_fullspeed", 0)),
            "cmd_gap_recent_kg": float(trim_dbg["cmd_gap_recent_kg"]),
            "steady_mean_abs_err_deg": float(trim_dbg.get("steady_mean_abs_err_deg", 0.0)),
            "steady_std_err_deg": float(trim_dbg.get("steady_std_err_deg", 0.0)),
            "trim_freeze": int(trim_dbg.get("trim_freeze", 0)),
            "trim_update_tick": int(trim_dbg.get("trim_update_tick", 0)),
            "target_mode": str(trim_dbg.get("target_mode", "g1")),
            "target_source": str(trim_dbg.get("target_source", "g1_formula")),
            "target_fallback_reason_last": str(trim_dbg.get("target_fallback_reason_last", "")),
            "target_table_path": str(trim_dbg.get("target_table_path", "")),
            "target_lookup_key": str(trim_dbg.get("target_lookup_key", "state_id")),
            "target_lookup_total": int(trim_dbg.get("target_lookup_total", 0)),
            "target_lookup_hit": int(trim_dbg.get("target_lookup_hit", 0)),
            "target_lookup_fallback": int(trim_dbg.get("target_lookup_fallback", 0)),
            "fallback_reason_missing_state": int(trim_dbg.get("fallback_reason_missing_state", 0)),
            "fallback_reason_missing_table": int(trim_dbg.get("fallback_reason_missing_table", 0)),
            "fallback_reason_missing_columns": int(trim_dbg.get("fallback_reason_missing_columns", 0)),
            "fallback_reason_invalid_state_id": int(trim_dbg.get("fallback_reason_invalid_state_id", 0)),
            "cmd_gap_kg": float(cmd_gap),
            "ctrl_status": str(ctrl_dbg.get("status", "")),
            "ctrl_clipped": int(ctrl_chain_clipped),
            "ctrl_base_clipped": int(ctrl_base_clipped),
            "ctrl_chain_clipped": int(ctrl_chain_clipped),
            "alloc_clip_any": int(alloc_clip_any),
            "ctrl_sign_warn": int(ctrl_dbg.get("sign_check_warn", 0)),
            "ctrl_filter_alpha": float(ctrl_dbg.get("filter_alpha", 0.0)),
            "ctrl_effective_dt": float(ctrl_dbg.get("effective_dt", 0.0)),
            "ctrl_update_tick": int(str(ctrl_dbg.get("status", "")) == "update"),
            "ctrl_raw_pitch_deg": float(raw_map.get("pitch", np.degrees(state[4]))),
            "ctrl_raw_roll_deg": float(raw_map.get("roll", np.degrees(state[3]))),
            "ctrl_raw_heave_m": float(raw_map.get("heave", state[2])),
            "ctrl_filtered_pitch_deg": float(filtered_map.get("pitch", np.degrees(state[4]))),
            "ctrl_filtered_roll_deg": float(filtered_map.get("roll", np.degrees(state[3]))),
            "ctrl_filtered_heave_m": float(filtered_map.get("heave", state[2])),
            "ctrl_pitch_err_raw_deg": float(pitch_pid.get("error_raw", 0.0)),
            "ctrl_roll_err_raw_deg": float(roll_pid.get("error_raw", 0.0)),
            "ctrl_heave_err_raw_m": float(heave_pid.get("error_raw", 0.0)),
            "ctrl_pitch_err_deg": float(pitch_pid.get("error", 0.0)),
            "ctrl_roll_err_deg": float(roll_pid.get("error", 0.0)),
            "ctrl_heave_err_m": float(heave_pid.get("error", 0.0)),
            "ctrl_pitch_u_total": float(pitch_pid.get("total_out", 0.0)),
            "ctrl_roll_u_total": float(roll_pid.get("total_out", 0.0)),
            "ctrl_heave_u_total": float(heave_pid.get("total_out", 0.0)),
            "ctrl_pitch_i_term": float(pitch_pid.get("i_term", 0.0)),
            "ctrl_roll_i_term": float(roll_pid.get("i_term", 0.0)),
            "ctrl_heave_i_term": float(heave_pid.get("i_term", 0.0)),
            "ctrl_pitch_integral_state": float(pitch_pid.get("integral_state", 0.0)),
            "ctrl_roll_integral_state": float(roll_pid.get("integral_state", 0.0)),
            "ctrl_heave_integral_state": float(heave_pid.get("integral_state", 0.0)),
            "ctrl_pitch_integral_decay": float(pitch_pid.get("integral_decay", 0.0)),
            "ctrl_roll_integral_decay": float(roll_pid.get("integral_decay", 0.0)),
            "ctrl_heave_integral_decay": float(heave_pid.get("integral_decay", 0.0)),
            "ctrl_pitch_integral_decay_reason": str(pitch_pid.get("integral_decay_reason", "")),
            "ctrl_roll_integral_decay_reason": str(roll_pid.get("integral_decay_reason", "")),
            "ctrl_heave_integral_decay_reason": str(heave_pid.get("integral_decay_reason", "")),
            "ctrl_pitch_in_deadband": int(pitch_pid.get("in_deadband", 0)),
            "ctrl_roll_in_deadband": int(roll_pid.get("in_deadband", 0)),
            "ctrl_heave_in_deadband": int(heave_pid.get("in_deadband", 0)),
            "ctrl_deadband_pitch_state": int(bool(deadband_state.get("pitch", False))),
            "ctrl_deadband_roll_state": int(bool(deadband_state.get("roll", False))),
            "ctrl_deadband_heave_state": int(bool(deadband_state.get("heave", False))),
            "forecast_safe_deadband_enabled": int(self.forecast_safe_deadband_enabled),
            "forecast_safe_deadband_active": int(
                forecast_safe_deadband_dbg.get("active", False)
            ),
            "forecast_safe_deadband_reason": str(
                forecast_safe_deadband_dbg.get("reason", "")
            ),
            "forecast_safe_deadband_has_future": int(
                forecast_safe_deadband_dbg.get("has_future", 0)
            ),
            "forecast_safe_deadband_trust_ok": int(
                forecast_safe_deadband_dbg.get("trust_ok", 0)
            ),
            "forecast_safe_deadband_hold_rejected": int(
                forecast_safe_deadband_hold_rejected
            ),
            "forecast_safe_deadband_target_released": int(
                forecast_safe_deadband_target_released
            ),
            "forecast_safe_deadband_provider_target_synced": int(
                forecast_safe_deadband_dbg.get("provider_target_synced", 0)
            ),
            "forecast_safe_deadband_provider_sync_started": int(
                forecast_safe_deadband_dbg.get("provider_sync_started", 0)
            ),
            "forecast_safe_deadband_provider_replan_requested": int(
                forecast_safe_deadband_dbg.get("provider_replan_requested", 0)
            ),
            "forecast_safe_deadband_provider_hold_active": int(
                forecast_safe_deadband_dbg.get("provider_hold_active", 0)
            ),
            "forecast_safe_deadband_provider_sync_eligible": int(
                forecast_safe_deadband_dbg.get("provider_sync_eligible", 0)
            ),
            "forecast_safe_deadband_provider_sync_reason": str(
                forecast_safe_deadband_dbg.get("provider_sync_reason", "")
            ),
            "forecast_safe_deadband_posture_recovering": int(
                forecast_safe_deadband_dbg.get("posture_recovering", 0)
            ),
            "forecast_safe_deadband_sync_future_rise_ok": int(
                forecast_safe_deadband_dbg.get("sync_future_rise_ok", 0)
            ),
            "forecast_safe_deadband_target_error_mean_kg": float(
                forecast_safe_deadband_dbg.get("target_error_mean_kg", 0.0)
            ),
            "forecast_safe_deadband_pump_demand_active": int(
                forecast_safe_deadband_dbg.get("pump_demand_active", 0)
            ),
            "forecast_safe_deadband_actionable_pump_demand": int(
                forecast_safe_deadband_dbg.get("actionable_pump_demand", 0)
            ),
            "forecast_safe_deadband_startup_ready": int(
                forecast_safe_deadband_dbg.get("startup_ready", 0)
            ),
            "forecast_safe_deadband_future_pressure_peak_norm": float(
                forecast_safe_deadband_dbg.get("future_pressure_peak_norm", 0.0)
            ),
            "forecast_safe_deadband_provider_sync_blend": float(
                forecast_safe_deadband_dbg.get("provider_sync_blend", 0.0)
            ),
            "forecast_safe_deadband_reentry_ready": int(
                forecast_safe_deadband_dbg.get("reentry_ready", 0)
            ),
            "forecast_safe_deadband_event_probability": float(
                forecast_safe_deadband_dbg.get("event_probability", 0.0)
            ),
            "forecast_safe_deadband_event_probability_available": int(
                forecast_safe_deadband_dbg.get("event_probability_available", 0)
            ),
            "forecast_safe_deadband_future_rise": float(
                forecast_safe_deadband_dbg.get("future_rise", 0.0)
            ),
            "forecast_safe_deadband_direction_dot": float(
                forecast_safe_deadband_dbg.get("direction_dot", 0.0)
            ),
            "forecast_safe_deadband_posture_abs_deg": float(
                forecast_safe_deadband_dbg.get("posture_abs_deg", 0.0)
            ),
            "forecast_safe_deadband_pitch_enter_deg": float(
                self.controller.deadband_enter.get("pitch", 0.0)
            ),
            "forecast_safe_deadband_roll_enter_deg": float(
                self.controller.deadband_enter.get("roll", 0.0)
            ),
            "heave_bias_delta_mean_kg": float(heave_bias_delta_mean),
            "target_slew_delta_mean_kg": float(target_slew_delta_mean),
            "limiter_delta_mean_kg": float(target_slew_delta_mean),
            "post_chain_delta_mean_kg": float(post_chain_delta_mean),
            "post_chain_adjusted": int(post_chain_adjusted),
            "deadband_target_release_active": int(release_dbg.get("active", 0)),
            "deadband_target_release_reason": str(release_dbg.get("reason", "")),
            "deadband_target_release_delta_mean_kg": float(release_dbg.get("delta_mean_kg", 0.0)),
            "deadband_target_release_blend": float(release_dbg.get("blend", 0.0)),
            "deadband_target_release_pitch_ok": int(release_dbg.get("pitch_ok", 0)),
            "deadband_target_release_roll_ok": int(release_dbg.get("roll_ok", 0)),
            "deadband_target_release_exit_pitch_ok": int(release_dbg.get("exit_pitch_ok", 0)),
            "deadband_target_release_exit_roll_ok": int(release_dbg.get("exit_roll_ok", 0)),
            "deadband_target_release_latched": int(release_dbg.get("latched", 0)),
            "deadband_target_release_reset_limiter": int(release_dbg.get("reset_limiter", 0)),
            "reactive_pump_suppression_active": int(reactive_suppression_dbg.get("active", 0)),
            "reactive_pump_suppression_reason": str(reactive_suppression_dbg.get("reason", "")),
            "reactive_pump_suppression_safe_zone": int(
                reactive_suppression_dbg.get("safe_zone", 0)
            ),
            "reactive_pump_suppression_latched": int(
                reactive_suppression_dbg.get("latched", 0)
            ),
            "reactive_pump_suppression_fullspeed_block": int(
                reactive_suppression_dbg.get("fullspeed_block", 0)
            ),
            "reactive_pump_suppression_pitch_abs_deg": float(
                reactive_suppression_dbg.get("pitch_abs_deg", 0.0)
            ),
            "reactive_pump_suppression_roll_abs_deg": float(
                reactive_suppression_dbg.get("roll_abs_deg", 0.0)
            ),
            "reactive_pump_suppression_restart_err_kg": float(
                reactive_suppression_dbg.get("restart_err_kg", 0.0)
            ),
            "reactive_suppression_blocked_tanks": int(
                reactive_suppression_dbg.get("blocked_tanks", 0)
            ),
            "reactive_suppression_blocked_mass_kg": float(
                reactive_suppression_dbg.get("blocked_mass_kg", 0.0)
            ),
            "reactive_suppression_delta_mean_kg": float(reactive_suppression_delta_mean),
            "reactive_suppression_mask_t1": int(reactive_suppression_dbg.get("mask_t1", 0)),
            "reactive_suppression_mask_t2": int(reactive_suppression_dbg.get("mask_t2", 0)),
            "reactive_suppression_mask_t3": int(reactive_suppression_dbg.get("mask_t3", 0)),
            "hm_smoothed_heave_m": float(hm_dbg.get("hm_smoothed_heave_m", state[2])),
            "hm_heave_error_m": float(hm_dbg.get("hm_heave_error_m", 0.0)),
            "hm_total_ballast_kg": float(hm_dbg.get("hm_total_ballast_kg", np.sum(m_cmd_applied))),
            "ballast_total_kg": float(
                hm_dbg.get("ballast_total_kg", np.sum(m_cmd_applied))
            ),
            "heave_correction_per_tank_kg": float(
                hm_dbg.get("heave_correction_per_tank_kg", 0.0)
            ),
            "heave_balancer_active": int(hm_dbg.get("heave_balancer_active", 0)),
            "preview_primary_enabled": int(primary_enabled),
            "preview_primary_active": int(primary_active),
            "preview_primary_applied": int(primary_applied),
            "preview_primary_candidate_applied": int(primary_candidate_applied),
            "preview_primary_target_t1_kg": float(primary_target[0]),
            "preview_primary_target_t2_kg": float(primary_target[1]),
            "preview_primary_target_t3_kg": float(primary_target[2]),
            "preview_primary_delta_t1_kg": float(primary_delta[0]),
            "preview_primary_delta_t2_kg": float(primary_delta[1]),
            "preview_primary_delta_t3_kg": float(primary_delta[2]),
            "preview_primary_delta_mean_kg": float(primary_delta_mean),
            "preview_primary_candidate_delta_mean_kg": float(primary_candidate_delta_mean),
            "preview_primary_action": str(
                preview_trim_bias.get("preview_primary_action", "")
                if isinstance(preview_trim_bias, dict)
                else ""
            ),
            "preview_primary_event_reset": int(
                preview_trim_bias.get("preview_primary_event_reset", 0)
                if isinstance(preview_trim_bias, dict)
                else 0
            ),
            "preview_primary_target_refreshed": int(
                preview_trim_bias.get("preview_primary_target_refreshed", 0)
                if isinstance(preview_trim_bias, dict)
                else 0
            ),
            "preview_primary_target_reused": int(
                preview_trim_bias.get("preview_primary_target_reused", 0)
                if isinstance(preview_trim_bias, dict)
                else 0
            ),
            "preview_primary_target_resumed": int(
                preview_trim_bias.get("preview_primary_target_resumed", 0)
                if isinstance(preview_trim_bias, dict)
                else 0
            ),
            "preview_primary_hold_target_mode": str(
                preview_trim_bias.get("preview_primary_hold_target_mode", "")
                if isinstance(preview_trim_bias, dict)
                else ""
            ),
            "preview_primary_target_age_s": float(
                preview_trim_bias.get("preview_primary_target_age_s", 0.0)
                if isinstance(preview_trim_bias, dict)
                else 0.0
            ),
            "preview_primary_safety_enabled": int(primary_safety_dbg["enabled"]),
            "preview_primary_safety_active": int(primary_safety_dbg["active"]),
            "preview_primary_safety_fallback": int(primary_safety_dbg["fallback"]),
            "preview_primary_safety_fallback_source": str(
                primary_safety_dbg["fallback_source"]
            ),
            "preview_primary_safety_reason": str(primary_safety_dbg["reason"]),
            "preview_primary_safety_hard_active": int(primary_safety_dbg["hard_active"]),
            "preview_primary_safety_pitch_abs_deg": float(primary_safety_dbg["pitch_abs_deg"]),
            "preview_primary_safety_roll_abs_deg": float(primary_safety_dbg["roll_abs_deg"]),
            "preview_primary_safety_env_norm": float(primary_safety_dbg["env_norm"]),
            "preview_primary_safety_use_envelope": int(primary_safety_dbg["use_envelope"]),
            "preview_primary_safety_envelope_enter_norm": float(
                primary_safety_dbg["envelope_enter_norm"]
            ),
            "preview_primary_safety_normal_enter": int(primary_safety_dbg["normal_enter"]),
            "preview_primary_safety_emergency_enter": int(
                primary_safety_dbg["emergency_enter"]
            ),
            "preview_primary_safety_enter_elapsed_s": float(
                primary_safety_dbg["enter_elapsed_s"]
            ),
            "preview_primary_safety_enter_hold_s": float(primary_safety_dbg["enter_hold_s"]),
            "preview_primary_safety_pitch_enter_deg": float(primary_safety_dbg["pitch_enter_deg"]),
            "preview_primary_safety_roll_enter_deg": float(primary_safety_dbg["roll_enter_deg"]),
            "preview_primary_safety_emergency_pitch_enter_deg": float(
                primary_safety_dbg["emergency_pitch_enter_deg"]
            ),
            "preview_primary_safety_emergency_roll_enter_deg": float(
                primary_safety_dbg["emergency_roll_enter_deg"]
            ),
            "preview_primary_safety_pitch_exit_deg": float(primary_safety_dbg["pitch_exit_deg"]),
            "preview_primary_safety_roll_exit_deg": float(primary_safety_dbg["roll_exit_deg"]),
            "preview_primary_safety_exit_window_max_pitch_abs_deg": float(
                primary_safety_dbg["exit_window_max_pitch_abs_deg"]
            ),
            "preview_primary_safety_exit_window_max_roll_abs_deg": float(
                primary_safety_dbg["exit_window_max_roll_abs_deg"]
            ),
            "preview_primary_safety_exit_env_norm": float(primary_safety_dbg["exit_env_norm"]),
            "preview_primary_safety_exit_clean_windows": int(
                primary_safety_dbg["exit_clean_windows"]
            ),
            "preview_primary_safety_exit_hold_s": float(primary_safety_dbg["exit_hold_s"]),
            "preview_primary_safety_exit_required_windows": int(
                primary_safety_dbg["exit_required_windows"]
            ),
            "preview_primary_bucket_guard_enabled": int(
                primary_safety_dbg["bucket_guard"]["enabled"]
            ),
            "preview_primary_bucket_guard_active": int(
                primary_safety_dbg["bucket_guard"]["active"]
            ),
            "preview_primary_bucket_guard_reason": str(
                primary_safety_dbg["bucket_guard"]["reason"]
            ),
            "preview_primary_bucket_guard_bucket_s": float(
                primary_safety_dbg["bucket_guard"]["bucket_s"]
            ),
            "preview_primary_bucket_guard_pitch_enter_deg": float(
                primary_safety_dbg["bucket_guard"]["pitch_enter_deg"]
            ),
            "preview_primary_bucket_guard_roll_enter_deg": float(
                primary_safety_dbg["bucket_guard"]["roll_enter_deg"]
            ),
            "preview_primary_bucket_guard_pitch_exit_deg": float(
                primary_safety_dbg["bucket_guard"]["pitch_exit_deg"]
            ),
            "preview_primary_bucket_guard_roll_exit_deg": float(
                primary_safety_dbg["bucket_guard"]["roll_exit_deg"]
            ),
            "preview_primary_bucket_guard_improve_tol_deg": float(
                primary_safety_dbg["bucket_guard"]["improve_tol_deg"]
            ),
            "preview_primary_bucket_guard_exit_required_windows": int(
                primary_safety_dbg["bucket_guard"]["exit_required_windows"]
            ),
            "preview_primary_bucket_guard_max_active_windows": int(
                primary_safety_dbg["bucket_guard"]["max_active_windows"]
            ),
            "preview_primary_bucket_guard_current_pitch_max_deg": float(
                primary_safety_dbg["bucket_guard"]["current_pitch_max_deg"]
            ),
            "preview_primary_bucket_guard_current_roll_max_deg": float(
                primary_safety_dbg["bucket_guard"]["current_roll_max_deg"]
            ),
            "preview_primary_bucket_guard_prev_pitch_max_deg": float(
                primary_safety_dbg["bucket_guard"]["prev_pitch_max_deg"]
            ),
            "preview_primary_bucket_guard_prev_roll_max_deg": float(
                primary_safety_dbg["bucket_guard"]["prev_roll_max_deg"]
            ),
            "preview_primary_bucket_guard_clean_windows": int(
                primary_safety_dbg["bucket_guard"]["clean_windows"]
            ),
            "preview_primary_bucket_guard_active_windows": int(
                primary_safety_dbg["bucket_guard"]["active_windows"]
            ),
            "preview_primary_bucket_guard_completed_windows": int(
                primary_safety_dbg["bucket_guard"]["completed_windows"]
            ),
            "preview_primary_bucket_guard_high": int(
                primary_safety_dbg["bucket_guard"]["high"]
            ),
            "preview_primary_bucket_guard_not_improving": int(
                primary_safety_dbg["bucket_guard"]["not_improving"]
            ),
            "preview_primary_bucket_guard_exit_clean": int(
                primary_safety_dbg["bucket_guard"]["exit_clean"]
            ),
            "preview_mass_ff_t1_kg": float(mass_ff[0]) if mass_ff.size > 0 else 0.0,
            "preview_mass_ff_t2_kg": float(mass_ff[1]) if mass_ff.size > 1 else 0.0,
            "preview_mass_ff_t3_kg": float(mass_ff[2]) if mass_ff.size > 2 else 0.0,
            "preview_mass_ff_abs_mean_kg": ff_delta_mean,
            "preview_ff_channel_active": int(ff_delta_mean > 1.0),
            "preview_pump_suppression_active": int(suppression_active),
            "preview_pump_restart_err_kg": float(suppression_restart_err_kg),
            "preview_pump_suppression_reason": suppression_reason,
            "preview_pressure_block0_norm": float(
                preview_trim_bias.get("preview_pressure_block0_norm", 0.0)
                if isinstance(preview_trim_bias, dict)
                else 0.0
            ),
            "preview_pressure_block1_norm": float(
                preview_trim_bias.get("preview_pressure_block1_norm", 0.0)
                if isinstance(preview_trim_bias, dict)
                else 0.0
            ),
            "preview_pressure_block2_norm": float(
                preview_trim_bias.get("preview_pressure_block2_norm", 0.0)
                if isinstance(preview_trim_bias, dict)
                else 0.0
            ),
            "preview_pressure_block02_dot": float(
                preview_trim_bias.get("preview_pressure_block02_dot", 0.0)
                if isinstance(preview_trim_bias, dict)
                else 0.0
            ),
            "preview_event_risk_pressure_floor_enabled": int(
                preview_trim_bias.get("preview_event_risk_pressure_floor_enabled", 0)
                if isinstance(preview_trim_bias, dict)
                else 0
            ),
            "preview_event_risk_floor_active_0_20m": int(
                preview_trim_bias.get("preview_event_risk_floor_active_0_20m", 0)
                if isinstance(preview_trim_bias, dict)
                else 0
            ),
            "preview_event_risk_floor_active_20_40m": int(
                preview_trim_bias.get("preview_event_risk_floor_active_20_40m", 0)
                if isinstance(preview_trim_bias, dict)
                else 0
            ),
            "preview_event_risk_floor_active_40_60m": int(
                preview_trim_bias.get("preview_event_risk_floor_active_40_60m", 0)
                if isinstance(preview_trim_bias, dict)
                else 0
            ),
            "preview_event_risk_floor_norm_0_20m": float(
                preview_trim_bias.get("preview_event_risk_floor_norm_0_20m", 0.0)
                if isinstance(preview_trim_bias, dict)
                else 0.0
            ),
            "preview_event_risk_floor_norm_20_40m": float(
                preview_trim_bias.get("preview_event_risk_floor_norm_20_40m", 0.0)
                if isinstance(preview_trim_bias, dict)
                else 0.0
            ),
            "preview_event_risk_floor_norm_40_60m": float(
                preview_trim_bias.get("preview_event_risk_floor_norm_40_60m", 0.0)
                if isinstance(preview_trim_bias, dict)
                else 0.0
            ),
            "preview_lead_action_enabled": int(
                preview_trim_bias.get("preview_lead_action_enabled", 0)
                if isinstance(preview_trim_bias, dict)
                else 0
            ),
            "preview_lead_action_active": int(
                preview_trim_bias.get("preview_lead_action_active", 0)
                if isinstance(preview_trim_bias, dict)
                else 0
            ),
            "preview_lead_action_reason": str(
                preview_trim_bias.get("preview_lead_action_reason", "")
                if isinstance(preview_trim_bias, dict)
                else ""
            ),
            "preview_lead_action_block_index": int(
                preview_trim_bias.get("preview_lead_action_block_index", -1)
                if isinstance(preview_trim_bias, dict)
                else -1
            ),
            "preview_lead_action_name": str(
                preview_trim_bias.get("preview_lead_action_name", "")
                if isinstance(preview_trim_bias, dict)
                else ""
            ),
            "preview_relief_medium_cap_enabled": int(
                preview_trim_bias.get("preview_relief_medium_cap_enabled", 0)
                if isinstance(preview_trim_bias, dict)
                else 0
            ),
            "preview_relief_medium_cap_active": int(
                preview_trim_bias.get("preview_relief_medium_cap_active", 0)
                if isinstance(preview_trim_bias, dict)
                else 0
            ),
            "preview_relief_medium_cap_reason": str(
                preview_trim_bias.get("preview_relief_medium_cap_reason", "")
                if isinstance(preview_trim_bias, dict)
                else ""
            ),
            "preview_relief_medium_cap_margin": float(
                preview_trim_bias.get("preview_relief_medium_cap_margin", 0.0)
                if isinstance(preview_trim_bias, dict)
                else 0.0
            ),
            "suppression_blocked_tanks": int(suppression_blocked_tanks),
            "suppression_blocked_mass_kg": float(suppression_blocked_mass_kg),
            "suppression_delta_mean_kg": float(suppression_delta_mean),
            "preview_suppression_delta_mean_kg": float(preview_suppression_delta_mean),
            "suppression_mask_t1": int(suppression_mask[0]) if suppression_mask.size > 0 else 0,
            "suppression_mask_t2": int(suppression_mask[1]) if suppression_mask.size > 1 else 0,
            "suppression_mask_t3": int(suppression_mask[2]) if suppression_mask.size > 2 else 0,
        }
        return m_cmd_applied, dbg
