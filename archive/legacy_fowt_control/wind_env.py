import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_THRUST_CURVE = Path("data/reference/thrust_curve_15mw_hub_height.xlsx")
_THRUST_CURVE_CACHE = {}


class ThrustCurve:
    """Hub-height wind speed to rotor thrust lookup table."""

    def __init__(self, wind_speed_ms, thrust_n, source):
        wind_speed = np.asarray(wind_speed_ms, dtype=float)
        thrust = np.asarray(thrust_n, dtype=float)
        valid = np.isfinite(wind_speed) & np.isfinite(thrust)
        wind_speed = wind_speed[valid]
        thrust = thrust[valid]
        if len(wind_speed) < 2:
            raise ValueError(f"thrust curve needs at least two valid points: {source}")
        order = np.argsort(wind_speed)
        self.wind_speed_ms = wind_speed[order]
        self.thrust_n = thrust[order]
        self.source = str(source)

    def thrust(self, ws_mps):
        ws = np.asarray(ws_mps, dtype=float)
        values = np.interp(
            ws,
            self.wind_speed_ms,
            self.thrust_n,
            left=0.0,
            right=float(self.thrust_n[-1]),
        )
        if np.isscalar(ws_mps):
            return float(values)
        return values


def discover_thrust_curve_file():
    env_path = os.environ.get("FOWT_THRUST_CURVE_FILE", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            DEFAULT_THRUST_CURVE,
            Path(__file__).resolve().parents[2] / DEFAULT_THRUST_CURVE,
        ]
    )
    for candidate in candidates:
        if candidate and candidate.exists() and not candidate.name.startswith("~$"):
            return candidate
    return None


def load_thrust_curve(path=None):
    curve_path = Path(path).expanduser() if path else discover_thrust_curve_file()
    if curve_path is None:
        return None
    curve_path = curve_path.resolve()
    cache_key = str(curve_path)
    if cache_key in _THRUST_CURVE_CACHE:
        return _THRUST_CURVE_CACHE[cache_key]

    frame = pd.read_excel(curve_path)
    columns = {str(col).strip().lower(): col for col in frame.columns}
    wind_col = next((col for key, col in columns.items() if "wind speed" in key), None)
    thrust_col = next((col for key, col in columns.items() if "thrust" in key), None)
    if wind_col is None or thrust_col is None:
        raise ValueError(f"missing wind speed or thrust column in {curve_path}")

    clean = frame[[wind_col, thrust_col]].copy()
    clean[wind_col] = pd.to_numeric(clean[wind_col], errors="coerce")
    clean[thrust_col] = pd.to_numeric(clean[thrust_col], errors="coerce")
    clean = clean.dropna().groupby(wind_col, as_index=False)[thrust_col].mean()

    thrust_values = clean[thrust_col].to_numpy(dtype=float)
    if "kn" in str(thrust_col).lower():
        thrust_values = thrust_values * 1000.0

    curve = ThrustCurve(
        clean[wind_col].to_numpy(dtype=float),
        thrust_values,
        source=curve_path,
    )
    _THRUST_CURVE_CACHE[cache_key] = curve
    return curve


def wind_speed_to_thrust_n(ws_mps, rho_air=1.225, rotor_radius=63.0, ct=0.80, thrust_curve="auto"):
    """Return rotor thrust [N].

    Prefer a hub-height thrust curve when available. Fall back to
    T ~= 0.5 * rho_air * A_rotor * Ct * V^2 for legacy runs or missing data.
    """
    if thrust_curve is not False:
        if isinstance(thrust_curve, ThrustCurve):
            return thrust_curve.thrust(ws_mps)
        curve = load_thrust_curve(None if thrust_curve == "auto" else thrust_curve)
        if curve is not None:
            return curve.thrust(ws_mps)

    rotor_area = np.pi * rotor_radius * rotor_radius
    return 0.5 * rho_air * rotor_area * ct * (ws_mps ** 2)


def pick_initial_wind_state(wind_gen, target_ws=11.5, target_wd=0.0):
    if wind_gen is None or len(wind_gen.curr_states) == 0:
        return 10.0, 0.0
    dists = [(ws - target_ws) ** 2 + (wd - target_wd) ** 2 for ws, wd in wind_gen.curr_states]
    idx = int(np.argmin(np.array(dists)))
    return wind_gen.curr_states[idx]


class WindEnvMarkov:
    """
    Runtime wind environment:
    - Markov state transition at fixed interval
    - LPF on wind speed and wind direction (vector form)
    """

    def __init__(
        self,
        wind_gen,
        dt,
        update_interval_s=3600.0,
        lpf_tau_s=120.0,
        target_ws=11.5,
        target_wd=0.0,
        seed=42,
        rho_air=1.225,
        rotor_radius=63.0,
        ct=0.80,
        thrust_curve_path="auto",
    ):
        self.wind_gen = wind_gen
        self.dt = float(dt)
        self.update_interval_s = float(update_interval_s)
        self.lpf_tau_s = float(lpf_tau_s)
        self.target_ws = float(target_ws)
        self.target_wd = float(target_wd)
        self.update_steps = max(1, int(round(self.update_interval_s / self.dt)))
        self.seed = int(seed)
        self.rho_air = float(rho_air)
        self.rotor_radius = float(rotor_radius)
        self.ct = float(ct)
        self.thrust_curve = load_thrust_curve(None if thrust_curve_path == "auto" else thrust_curve_path)
        self.rng = np.random.default_rng(self.seed)
        self._step_idx = 0
        self.ws_state = 0.0
        self.wd_state = 0.0
        self.state_id = None
        self.ws_eff = 0.0
        self.u_eff = 1.0
        self.v_eff = 0.0
        self.reset(seed=self.seed)

    def _resolve_state_id(self, ws, wd):
        if self.wind_gen is None:
            return None
        try:
            return int(self.wind_gen.get_state_id(float(ws), float(wd)))
        except Exception:
            return None

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)

        ws0, wd0 = pick_initial_wind_state(
            self.wind_gen, target_ws=self.target_ws, target_wd=self.target_wd
        )
        wd0_rad = np.radians(wd0)
        self._step_idx = 0
        self.ws_state = float(ws0)
        self.wd_state = float(wd0)
        self.state_id = self._resolve_state_id(self.ws_state, self.wd_state)
        self.ws_eff = float(ws0)
        self.u_eff = float(np.cos(wd0_rad))
        self.v_eff = float(np.sin(wd0_rad))

    def _alpha(self):
        if self.lpf_tau_s <= 0.0:
            return 0.0
        return self.lpf_tau_s / (self.lpf_tau_s + self.dt)

    def _transition_if_needed(self):
        if self._step_idx > 0 and (self._step_idx % self.update_steps == 0):
            ws_next, wd_next = self.wind_gen.get_next_wind(
                self.ws_state, self.wd_state, rng=self.rng
            )
            self.ws_state = float(ws_next)
            self.wd_state = float(wd_next)
            self.state_id = self._resolve_state_id(self.ws_state, self.wd_state)

    def _lpf_update(self):
        alpha = self._alpha()
        one_minus = 1.0 - alpha

        self.ws_eff = alpha * self.ws_eff + one_minus * self.ws_state

        wd_target_rad = np.radians(self.wd_state)
        u_target = np.cos(wd_target_rad)
        v_target = np.sin(wd_target_rad)
        self.u_eff = alpha * self.u_eff + one_minus * u_target
        self.v_eff = alpha * self.v_eff + one_minus * v_target

        uv_norm = float(np.hypot(self.u_eff, self.v_eff))
        if uv_norm > 1e-12:
            self.u_eff /= uv_norm
            self.v_eff /= uv_norm

    def _to_thrust(self, ws):
        return float(
            wind_speed_to_thrust_n(
                ws,
                rho_air=self.rho_air,
                rotor_radius=self.rotor_radius,
                ct=self.ct,
                thrust_curve=self.thrust_curve if self.thrust_curve is not None else False,
            )
        )

    def step(self):
        if self.wind_gen is None:
            return {
                "ws": 0.0,
                "wd_deg": 0.0,
                "thrust_n": 0.0,
                "state_id": None,
                "ws_state": None,
                "wd_state": None,
            }

        self._transition_if_needed()
        self._lpf_update()

        wd_eff = float(np.degrees(np.arctan2(self.v_eff, self.u_eff)))
        obs = {
            "ws": float(self.ws_eff),
            "wd_deg": wd_eff,
            "thrust_n": self._to_thrust(self.ws_eff),
            "state_id": int(self.state_id) if self.state_id is not None else None,
            "ws_state": float(self.ws_state),
            "wd_state": float(self.wd_state),
        }
        self._step_idx += 1
        return obs

    def generate_trace(self, n_steps, seed=None):
        n_steps = int(n_steps)
        if n_steps <= 0:
            return {
                "ws": np.array([], dtype=float),
                "wd": np.array([], dtype=float),
                "thrust_n": np.array([], dtype=float),
                "seed": int(self.seed if seed is None else seed),
                "n_steps": 0,
                "dt": float(self.dt),
                "update_interval_s": float(self.update_interval_s),
                "mean_lpf_tau_s": float(self.lpf_tau_s),
            }

        saved = {
            "seed": self.seed,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "step_idx": self._step_idx,
            "ws_state": self.ws_state,
            "wd_state": self.wd_state,
            "ws_eff": self.ws_eff,
            "u_eff": self.u_eff,
            "v_eff": self.v_eff,
        }

        self.reset(seed=self.seed if seed is None else seed)
        ws_vals = np.zeros(n_steps, dtype=float)
        wd_vals = np.zeros(n_steps, dtype=float)
        thrust_vals = np.zeros(n_steps, dtype=float)
        state_ids = np.full(n_steps, -1, dtype=int)
        ws_state_vals = np.zeros(n_steps, dtype=float)
        wd_state_vals = np.zeros(n_steps, dtype=float)
        for i in range(n_steps):
            obs = self.step()
            ws_vals[i] = obs["ws"]
            wd_vals[i] = obs["wd_deg"]
            thrust_vals[i] = obs["thrust_n"]
            sid = obs.get("state_id", None)
            state_ids[i] = int(sid) if sid is not None else -1
            ws_state_vals[i] = float(obs.get("ws_state", np.nan))
            wd_state_vals[i] = float(obs.get("wd_state", np.nan))

        trace = {
            "ws": ws_vals,
            "wd": wd_vals,
            "thrust_n": thrust_vals,
            "state_id": state_ids,
            "ws_state": ws_state_vals,
            "wd_state": wd_state_vals,
            "seed": int(self.seed if seed is None else seed),
            "n_steps": int(n_steps),
            "dt": float(self.dt),
            "update_interval_s": float(self.update_interval_s),
            "mean_lpf_tau_s": float(self.lpf_tau_s),
        }

        self.seed = int(saved["seed"])
        self.rng = np.random.default_rng(self.seed)
        self.rng.bit_generator.state = saved["rng_state"]
        self._step_idx = int(saved["step_idx"])
        self.ws_state = float(saved["ws_state"])
        self.wd_state = float(saved["wd_state"])
        self.ws_eff = float(saved["ws_eff"])
        self.u_eff = float(saved["u_eff"])
        self.v_eff = float(saved["v_eff"])
        return trace


class WindTracePlayer:
    """Replay a prebuilt wind trace exactly."""

    def __init__(self, trace):
        self.trace = trace
        self.ws = np.asarray(trace["ws"], dtype=float)
        self.wd = np.asarray(trace["wd"], dtype=float)
        self.thrust = (
            np.asarray(trace["thrust_n"], dtype=float)
            if "thrust_n" in trace
            else np.array([], dtype=float)
        )
        self.state_ids = (
            np.asarray(trace["state_id"]) if "state_id" in trace else np.array([], dtype=object)
        )
        self.ws_state = (
            np.asarray(trace["ws_state"], dtype=float) if "ws_state" in trace else np.array([], dtype=float)
        )
        self.wd_state = (
            np.asarray(trace["wd_state"], dtype=float) if "wd_state" in trace else np.array([], dtype=float)
        )
        self.n_steps = int(len(self.ws))
        self.idx = 0

    def reset(self):
        self.idx = 0

    def step(self):
        if self.idx >= self.n_steps:
            raise IndexError(f"wind trace exhausted: idx={self.idx}, n_steps={self.n_steps}")
        ws = float(self.ws[self.idx])
        wd = float(self.wd[self.idx])
        if len(self.thrust) == self.n_steps:
            thrust = float(self.thrust[self.idx])
        else:
            thrust = float(wind_speed_to_thrust_n(ws))
        sid = None
        if len(self.state_ids) == self.n_steps:
            raw_sid = self.state_ids[self.idx]
            if raw_sid is not None:
                try:
                    sid = int(raw_sid)
                except Exception:
                    sid = None

        ws_state = float(self.ws_state[self.idx]) if len(self.ws_state) == self.n_steps else None
        wd_state = float(self.wd_state[self.idx]) if len(self.wd_state) == self.n_steps else None
        obs = {
            "ws": ws,
            "wd_deg": wd,
            "thrust_n": thrust,
            "state_id": sid,
            "ws_state": ws_state,
            "wd_state": wd_state,
        }
        self.idx += 1
        return obs



class MarkovWindGenerator:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._load_data()

    def _load_data(self):
        try:
            ext = os.path.splitext(self.csv_path)[1].lower()
            if ext in [".xlsx", ".xls"]:
                df = pd.read_excel(self.csv_path, header=None)
            else:
                df = pd.read_csv(self.csv_path, header=None, low_memory=False)

            next_speed = pd.to_numeric(df.iloc[0, 2:], errors="coerce").ffill()
            next_dir = pd.to_numeric(df.iloc[1, 2:], errors="coerce")

            valid_cols = next_dir.notna() & next_speed.notna()
            next_speed = next_speed[valid_cols].to_numpy()
            next_dir = next_dir[valid_cols].to_numpy()

            curr_speed = pd.to_numeric(df.iloc[3:, 0], errors="coerce").ffill()
            curr_dir = pd.to_numeric(df.iloc[3:, 1], errors="coerce")

            valid_rows = curr_dir.notna() & curr_speed.notna()
            curr_speed = curr_speed[valid_rows].to_numpy()
            curr_dir = curr_dir[valid_rows].to_numpy()

            counts = df.iloc[3:, 2:]
            counts = counts.loc[valid_rows, valid_cols].fillna(0.0).to_numpy(dtype=float)

            self.next_states = list(zip(next_speed.tolist(), next_dir.tolist()))
            self.curr_states = list(zip(curr_speed.tolist(), curr_dir.tolist()))

            sums = counts.sum(axis=1)
            sums[sums == 0.0] = 1.0
            self.probs = counts / sums[:, None]

            self.state_map = {s: i for i, s in enumerate(self.curr_states)}

            n_states = len(self.curr_states)
            print(">>> [WindEnv] Data Loading Check:")
            print(f"    - States: {n_states}, Matrix: {self.probs.shape}")

            if self.probs.shape != (n_states, len(self.next_states)):
                raise ValueError(
                    f"Shape mismatch: probs={self.probs.shape}, curr={n_states}, next={len(self.next_states)}"
                )

            if n_states != len(self.next_states):
                print(
                    "!!! [WindEnv Warning] curr_states != next_states (non-square Markov). "
                    "This is allowed, but confirm it's intended."
                )

            if np.isnan(self.probs).any():
                raise ValueError("NaN in probs after cleaning. Check table offsets / non-numeric cells.")

            rs = self.probs.sum(axis=1)
            print(f"    - RowSum: min={rs.min():.4f}, max={rs.max():.4f}, mean={rs.mean():.4f}")

        except Exception as e:
            print(f"!!! [WindEnv] Error loading {self.csv_path}: {e}")
            self.next_states = [(10.0, 0.0)]
            self.curr_states = [(10.0, 0.0)]
            self.probs = np.array([[1.0]])
            self.state_map = {(10.0, 0.0): 0}

    def get_state_id(self, curr_ws, curr_wd):
        key = (curr_ws, curr_wd)
        if key in self.state_map:
            return int(self.state_map[key])
        dists = np.array([(s - curr_ws) ** 2 + (d - curr_wd) ** 2 for s, d in self.curr_states])
        return int(np.argmin(dists))

    def get_next_wind(self, curr_ws, curr_wd, rng=None):
        row_idx = self.get_state_id(curr_ws, curr_wd)

        p_row = self.probs[row_idx]
        if p_row.sum() == 0:
            return curr_ws, curr_wd

        p_row = p_row / p_row.sum()
        if rng is None:
            col_idx = int(np.random.choice(len(self.next_states), p=p_row))
        else:
            col_idx = int(rng.choice(len(self.next_states), p=p_row))
        return self.next_states[col_idx]
