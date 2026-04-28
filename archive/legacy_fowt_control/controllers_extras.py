import csv
from pathlib import Path

import numpy as np

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




class CommandRateLimiter:
    def __init__(self, dt, rho, max_capacity, rate_limit_m3_min=12.0, enabled=True):
        self.dt = float(dt)
        self.rho = float(rho)
        self.max_capacity = max_capacity
        self.rate_limit_m3_min = float(rate_limit_m3_min)
        self.enabled = bool(enabled)
        self._target = None

    def reset(self, initial_cmd):
        self._target = np.array(initial_cmd, dtype=float)

    def _clip_capacity(self, values):
        return np.clip(values, 0.0, self.max_capacity)

    def update(self, cmd_raw):
        raw = np.array(cmd_raw, dtype=float)
        if self._target is None:
            self._target = self._clip_capacity(raw)

        if not self.enabled:
            applied = self._clip_capacity(raw)
            self._target = applied.copy()
            gap = float(np.mean(np.abs(raw - applied)))
            return applied, gap

        max_delta = (self.rate_limit_m3_min / 60.0) * self.rho * self.dt
        delta = np.clip(raw - self._target, -max_delta, max_delta)
        self._target = self._clip_capacity(self._target + delta)
        gap = float(np.mean(np.abs(raw - self._target)))
        return self._target.copy(), gap


def _zero_preview_trim_debug(source="none"):
    return {
        "preview_trim_active": 0,
        "preview_trim_source": str(source),
        "preview_pitch_bias_deg": 0.0,
        "preview_roll_bias_deg": 0.0,
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
        command_rate_limiter=None,
        heave_balancer=None,
        preview_trim_provider=None,
    ):
        self.controller = controller
        self.trim_governor = trim_governor
        self.setpoint_shaper = setpoint_shaper
        self.command_rate_limiter = command_rate_limiter
        self.heave_balancer = heave_balancer
        self.preview_trim_provider = preview_trim_provider

    def reset(self, initial_cmd, initial_ws=0.0):
        self.controller.reset()
        if self.trim_governor is not None:
            self.trim_governor.reset(initial_ws=initial_ws)
        if self.setpoint_shaper is not None:
            self.setpoint_shaper.reset(
                initial=(self.controller.setpoints["pitch"], self.controller.setpoints["roll"])
            )
        if self.command_rate_limiter is not None:
            self.command_rate_limiter.reset(initial_cmd=initial_cmd)
        if self.heave_balancer is not None:
            self.heave_balancer.reset()
        if self.preview_trim_provider is not None and hasattr(self.preview_trim_provider, "reset"):
            self.preview_trim_provider.reset()

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

    def compute(self, state, wind_obs, plant_info_prev, current_time):
        preview_trim_bias = self._preview_trim_bias(
            state=state,
            wind_obs=wind_obs,
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
                "preview_trim_source": "no_trim",
            }
            pitch_sp_raw = float(self.controller.setpoints["pitch"])
            roll_sp_raw = float(self.controller.setpoints["roll"])

        if self.setpoint_shaper is not None:
            pitch_sp, roll_sp = self.setpoint_shaper.update(pitch_sp_raw, roll_sp_raw)
        else:
            pitch_sp, roll_sp = float(pitch_sp_raw), float(roll_sp_raw)

        self.controller.setpoints["pitch"] = float(pitch_sp)
        self.controller.setpoints["roll"] = float(roll_sp)

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

        if self.command_rate_limiter is not None:
            m_cmd_applied, cmd_gap = self.command_rate_limiter.update(m_cmd_heave)
        else:
            m_cmd_applied = m_cmd_heave
            cmd_gap = 0.0

        heave_bias_delta_mean = float(np.mean(np.abs(m_cmd_heave - m_cmd_raw)))
        limiter_delta_mean = float(np.mean(np.abs(m_cmd_applied - m_cmd_heave)))
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
            "ctrl_pitch_in_deadband": int(pitch_pid.get("in_deadband", 0)),
            "ctrl_roll_in_deadband": int(roll_pid.get("in_deadband", 0)),
            "ctrl_heave_in_deadband": int(heave_pid.get("in_deadband", 0)),
            "ctrl_deadband_pitch_state": int(bool(deadband_state.get("pitch", False))),
            "ctrl_deadband_roll_state": int(bool(deadband_state.get("roll", False))),
            "ctrl_deadband_heave_state": int(bool(deadband_state.get("heave", False))),
            "heave_bias_delta_mean_kg": float(heave_bias_delta_mean),
            "limiter_delta_mean_kg": float(limiter_delta_mean),
            "post_chain_delta_mean_kg": float(post_chain_delta_mean),
            "post_chain_adjusted": int(post_chain_adjusted),
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
        }
        return m_cmd_applied, dbg
