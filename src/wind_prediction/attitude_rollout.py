"""Lightweight pitch/roll rollout for candidate-action evaluation.

The submitted controller does not import this module.  It is intentionally an
opt-in model that can later replace the planner's attitude-residual proxy after
its parameters have been calibrated against the maintained six-DOF plant.

The model keeps the two rotational axes independent:

    I * angle_ddot + C * angle_dot + K * angle = wind_moment + ballast_moment

Angles use the plant's right-handed convention.  Inputs and outputs are in
degrees, while the internal dynamics are evaluated in radians.  A stage's
ballast input is an *actual incremental tank-mass change*.  Its moment remains
active in later stages, matching the persistence of ballast mass in the plant.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


# Frozen-paper plant, ``platform_profile="default"``.  Keep these values
# traceable to ``archive/legacy_fowt_control/core_model.py`` instead of using a
# separately tuned reduced-order parameter set:
#   mass_dry (line 58), tank_pos/default_ballast (lines 73-83), I_body (line 68),
#   hydro_params/zetas (lines 167-181), and M/K/C assembly (lines 716-746).
_LEGACY_DEFAULT_PARAMETER_SOURCE = "legacy_fowt_default_working_condition_v1"
_LEGACY_DEFAULT_DRY_MASS_KG = 16_937_000.0
_LEGACY_DEFAULT_BALLAST_MASS_KG = (1_108_000.0, 1_362_000.0, 1_362_000.0)
_LEGACY_DEFAULT_TOTAL_MASS_KG = (
    _LEGACY_DEFAULT_DRY_MASS_KG + sum(_LEGACY_DEFAULT_BALLAST_MASS_KG)
)
_LEGACY_DEFAULT_GRAVITY_M_S2 = 9.81
_LEGACY_DEFAULT_GM_M = 11.0
_LEGACY_DEFAULT_RESTORING_STIFFNESS_NM_RAD = (
    _LEGACY_DEFAULT_TOTAL_MASS_KG
    * _LEGACY_DEFAULT_GRAVITY_M_S2
    * _LEGACY_DEFAULT_GM_M
)
_LEGACY_DEFAULT_ROLL_PITCH_INERTIA_KG_M2 = 8.0e10
_LEGACY_DEFAULT_ROLL_PITCH_DAMPING_RATIO = 0.08
_LEGACY_DEFAULT_TANK_XY_M = (
    (46.2, 0.0),
    (-23.1, 46.2 * 0.866),
    (-23.1, -46.2 * 0.866),
)


@dataclass(frozen=True)
class AttitudeRolloutParameters:
    """Explicit physical and numerical parameters for the reduced model.

    ``wind_moment_per_effect_unit_nm`` maps each supplied forecast-effect unit
    to a generalized pitch/roll moment.  Setting it to ``(1, 1)`` means the
    caller already supplies moments in N m.  The default inertia, stiffness,
    damping, and tank coordinates reproduce the frozen paper run's ``default``
    platform at its initial working ballast condition.  The model remains a
    local pitch/roll approximation, not an independent plant calibration.
    """

    enabled: bool = False
    parameter_source: str = _LEGACY_DEFAULT_PARAMETER_SOURCE
    pitch_inertia_kg_m2: float = _LEGACY_DEFAULT_ROLL_PITCH_INERTIA_KG_M2
    roll_inertia_kg_m2: float = _LEGACY_DEFAULT_ROLL_PITCH_INERTIA_KG_M2
    pitch_stiffness_nm_rad: float = _LEGACY_DEFAULT_RESTORING_STIFFNESS_NM_RAD
    roll_stiffness_nm_rad: float = _LEGACY_DEFAULT_RESTORING_STIFFNESS_NM_RAD
    pitch_damping_ratio: float = _LEGACY_DEFAULT_ROLL_PITCH_DAMPING_RATIO
    roll_damping_ratio: float = _LEGACY_DEFAULT_ROLL_PITCH_DAMPING_RATIO
    tank_xy_m: tuple[tuple[float, float], ...] = _LEGACY_DEFAULT_TANK_XY_M
    gravity_m_s2: float = _LEGACY_DEFAULT_GRAVITY_M_S2
    ballast_moment_scale: float = 1.0
    wind_moment_per_effect_unit_nm: tuple[float, float] = (1.0, 1.0)
    constant_moment_nm: tuple[float, float] = (0.0, 0.0)
    metric_sample_interval_s: float = 5.0
    thresholds_deg: tuple[float, ...] = (2.0, 3.0, 5.0)

    def __post_init__(self) -> None:
        if not str(self.parameter_source).strip():
            raise ValueError("parameter_source must be a non-empty identifier")
        positive = {
            "pitch_inertia_kg_m2": self.pitch_inertia_kg_m2,
            "roll_inertia_kg_m2": self.roll_inertia_kg_m2,
            "pitch_stiffness_nm_rad": self.pitch_stiffness_nm_rad,
            "roll_stiffness_nm_rad": self.roll_stiffness_nm_rad,
            "gravity_m_s2": self.gravity_m_s2,
            "metric_sample_interval_s": self.metric_sample_interval_s,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("pitch_damping_ratio", self.pitch_damping_ratio),
            ("roll_damping_ratio", self.roll_damping_ratio),
            ("ballast_moment_scale", self.ballast_moment_scale),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        tank_xy = np.asarray(self.tank_xy_m, dtype=float)
        if tank_xy.ndim != 2 or tank_xy.shape[1] != 2 or tank_xy.shape[0] == 0:
            raise ValueError("tank_xy_m must have shape (n_tanks, 2)")
        if not np.all(np.isfinite(tank_xy)):
            raise ValueError("tank_xy_m must contain only finite values")

        for name, values in (
            ("wind_moment_per_effect_unit_nm", self.wind_moment_per_effect_unit_nm),
            ("constant_moment_nm", self.constant_moment_nm),
        ):
            array = np.asarray(values, dtype=float)
            if array.shape != (2,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain finite pitch/roll values")

        thresholds = np.asarray(self.thresholds_deg, dtype=float)
        if thresholds.ndim != 1 or not np.all(np.isfinite(thresholds)):
            raise ValueError("thresholds_deg must be a finite one-dimensional sequence")
        if np.any(thresholds <= 0.0) or np.any(np.diff(thresholds) <= 0.0):
            raise ValueError("thresholds_deg must be positive and strictly increasing")


@dataclass(frozen=True)
class AttitudeState:
    pitch_deg: float
    roll_deg: float
    pitch_rate_deg_s: float
    roll_rate_deg_s: float


@dataclass(frozen=True)
class AttitudeMetrics:
    duration_s: float
    pitch_rms_deg: float
    roll_rms_deg: float
    vector_rms_deg: float
    pitch_peak_abs_deg: float
    roll_peak_abs_deg: float
    dominant_peak_deg: float
    pitch_exceedance_s: Mapping[float, float]
    roll_exceedance_s: Mapping[float, float]
    dominant_exceedance_s: Mapping[float, float]


@dataclass(frozen=True)
class AttitudeStageResult:
    stage_index: int
    start_time_s: float
    end_time_s: float
    start_state: AttitudeState
    end_state: AttitudeState
    wind_moment_nm: tuple[float, float]
    ballast_moment_nm: tuple[float, float]
    total_moment_nm: tuple[float, float]
    metrics: AttitudeMetrics


@dataclass(frozen=True)
class AttitudeRolloutResult:
    stages: tuple[AttitudeStageResult, ...]
    times_s: np.ndarray
    pitch_deg: np.ndarray
    roll_deg: np.ndarray
    pitch_rate_deg_s: np.ndarray
    roll_rate_deg_s: np.ndarray
    metrics: AttitudeMetrics


def _state_from_values(angle_rad: np.ndarray, rate_rad_s: np.ndarray) -> AttitudeState:
    return AttitudeState(
        pitch_deg=float(np.degrees(angle_rad[0])),
        roll_deg=float(np.degrees(angle_rad[1])),
        pitch_rate_deg_s=float(np.degrees(rate_rad_s[0])),
        roll_rate_deg_s=float(np.degrees(rate_rad_s[1])),
    )


def _axis_response(
    angle_rad: float,
    rate_rad_s: float,
    moment_nm: float,
    times_s: np.ndarray,
    *,
    inertia_kg_m2: float,
    stiffness_nm_rad: float,
    damping_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact constant-moment response of one damped second-order axis."""

    omega_n = math.sqrt(stiffness_nm_rad / inertia_kg_m2)
    equilibrium_rad = moment_nm / stiffness_nm_rad
    displacement = float(angle_rad) - equilibrium_rad
    velocity = float(rate_rad_s)
    zeta = float(damping_ratio)
    t = np.asarray(times_s, dtype=float)

    if zeta < 1.0 - 1e-10:
        decay_rate = zeta * omega_n
        omega_d = omega_n * math.sqrt(1.0 - zeta * zeta)
        decay = np.exp(-decay_rate * t)
        cosine = np.cos(omega_d * t)
        sine = np.sin(omega_d * t)
        phi11 = decay * (cosine + (decay_rate / omega_d) * sine)
        phi12 = decay * sine / omega_d
        phi21 = -decay * (omega_n * omega_n / omega_d) * sine
        phi22 = decay * (cosine - (decay_rate / omega_d) * sine)
        relative_angle = phi11 * displacement + phi12 * velocity
        response_rate = phi21 * displacement + phi22 * velocity
    elif zeta <= 1.0 + 1e-10:
        decay = np.exp(-omega_n * t)
        relative_angle = decay * (
            (1.0 + omega_n * t) * displacement + t * velocity
        )
        response_rate = decay * (
            -(omega_n * omega_n) * t * displacement
            + (1.0 - omega_n * t) * velocity
        )
    else:
        root = math.sqrt(zeta * zeta - 1.0)
        r1 = -omega_n * (zeta - root)
        r2 = -omega_n * (zeta + root)
        denominator = r1 - r2
        c1 = (velocity - r2 * displacement) / denominator
        c2 = (r1 * displacement - velocity) / denominator
        e1 = np.exp(r1 * t)
        e2 = np.exp(r2 * t)
        relative_angle = c1 * e1 + c2 * e2
        response_rate = r1 * c1 * e1 + r2 * c2 * e2

    return equilibrium_rad + relative_angle, response_rate


def _threshold_duration_s(
    times_s: np.ndarray,
    nonnegative_signal: np.ndarray,
    threshold: float,
) -> float:
    """Estimate strict-threshold exposure using linear crossing locations."""

    total = 0.0
    for idx in range(len(times_s) - 1):
        left = float(nonnegative_signal[idx])
        right = float(nonnegative_signal[idx + 1])
        dt = float(times_s[idx + 1] - times_s[idx])
        left_above = left > threshold
        right_above = right > threshold
        if left_above and right_above:
            total += dt
        elif left_above != right_above and abs(right - left) > 1e-15:
            crossing = float(np.clip((threshold - left) / (right - left), 0.0, 1.0))
            total += dt * (crossing if left_above else 1.0 - crossing)
    return total


def _metrics(
    times_s: np.ndarray,
    pitch_deg: np.ndarray,
    roll_deg: np.ndarray,
    thresholds_deg: Sequence[float],
) -> AttitudeMetrics:
    duration = float(times_s[-1] - times_s[0])
    if duration <= 0.0:
        raise ValueError("metric duration must be positive")

    pitch_abs = np.abs(pitch_deg)
    roll_abs = np.abs(roll_deg)
    dominant = np.maximum(pitch_abs, roll_abs)
    pitch_mean_square = float(np.trapezoid(pitch_deg * pitch_deg, times_s) / duration)
    roll_mean_square = float(np.trapezoid(roll_deg * roll_deg, times_s) / duration)

    def exposure(signal: np.ndarray) -> dict[float, float]:
        return {
            float(threshold): _threshold_duration_s(
                times_s, signal, float(threshold)
            )
            for threshold in thresholds_deg
        }

    return AttitudeMetrics(
        duration_s=duration,
        pitch_rms_deg=math.sqrt(max(pitch_mean_square, 0.0)),
        roll_rms_deg=math.sqrt(max(roll_mean_square, 0.0)),
        vector_rms_deg=math.sqrt(max(pitch_mean_square + roll_mean_square, 0.0)),
        pitch_peak_abs_deg=float(np.max(pitch_abs)),
        roll_peak_abs_deg=float(np.max(roll_abs)),
        dominant_peak_deg=float(np.max(dominant)),
        pitch_exceedance_s=exposure(pitch_abs),
        roll_exceedance_s=exposure(roll_abs),
        dominant_exceedance_s=exposure(dominant),
    )


def _validate_rollout_inputs(
    initial_pitch_roll_deg: Sequence[float],
    initial_pitch_roll_rate_deg_s: Sequence[float] | None,
    stage_ballast_mass_delta_kg: Sequence[Sequence[float]],
    stage_wind_effect: Sequence[Sequence[float]],
    stage_durations_s: Sequence[float],
    parameters: AttitudeRolloutParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    initial = np.asarray(initial_pitch_roll_deg, dtype=float)
    rates = np.zeros(2, dtype=float) if initial_pitch_roll_rate_deg_s is None else np.asarray(
        initial_pitch_roll_rate_deg_s, dtype=float
    )
    mass_delta = np.asarray(stage_ballast_mass_delta_kg, dtype=float)
    wind_effect = np.asarray(stage_wind_effect, dtype=float)
    durations = np.asarray(stage_durations_s, dtype=float)
    n_tanks = len(parameters.tank_xy_m)

    if initial.shape != (2,) or rates.shape != (2,):
        raise ValueError("initial pitch/roll attitude and rate must each have shape (2,)")
    if durations.ndim != 1 or durations.size == 0:
        raise ValueError("stage_durations_s must be a non-empty one-dimensional sequence")
    n_stages = durations.size
    if mass_delta.shape != (n_stages, n_tanks):
        raise ValueError(
            f"stage_ballast_mass_delta_kg must have shape ({n_stages}, {n_tanks})"
        )
    if wind_effect.shape != (n_stages, 2):
        raise ValueError(f"stage_wind_effect must have shape ({n_stages}, 2)")
    for name, values in (
        ("initial_pitch_roll_deg", initial),
        ("initial_pitch_roll_rate_deg_s", rates),
        ("stage_ballast_mass_delta_kg", mass_delta),
        ("stage_wind_effect", wind_effect),
        ("stage_durations_s", durations),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
    if np.any(durations <= 0.0):
        raise ValueError("all stage durations must be positive")
    return initial, rates, mass_delta, wind_effect, durations


def rollout_attitude(
    initial_pitch_roll_deg: Sequence[float],
    stage_ballast_mass_delta_kg: Sequence[Sequence[float]],
    stage_wind_effect: Sequence[Sequence[float]],
    stage_durations_s: Sequence[float],
    *,
    initial_pitch_roll_rate_deg_s: Sequence[float] | None = None,
    parameters: AttitudeRolloutParameters | None = None,
) -> AttitudeRolloutResult | None:
    """Roll out pitch/roll attitude over constant-input forecast stages.

    Inputs use ``[pitch, roll]`` order.  ``stage_ballast_mass_delta_kg`` is the
    actual incremental mass delivered to each tank in each stage, not a nominal
    command.  ``stage_wind_effect`` is mapped to generalized moments through
    ``parameters.wind_moment_per_effect_unit_nm``.

    The default parameter set is disabled and returns ``None``.  This explicit
    opt-in keeps the frozen v1 candidate-ranking path unchanged.
    """

    params = parameters or AttitudeRolloutParameters()
    if not params.enabled:
        return None

    initial, rates, mass_delta, wind_effect, durations = _validate_rollout_inputs(
        initial_pitch_roll_deg,
        initial_pitch_roll_rate_deg_s,
        stage_ballast_mass_delta_kg,
        stage_wind_effect,
        stage_durations_s,
        params,
    )

    angle_rad = np.radians(initial)
    rate_rad_s = np.radians(rates)
    tank_xy = np.asarray(params.tank_xy_m, dtype=float)
    wind_scale = np.asarray(params.wind_moment_per_effect_unit_nm, dtype=float)
    constant_moment = np.asarray(params.constant_moment_nm, dtype=float)
    cumulative_mass_delta = np.zeros(tank_xy.shape[0], dtype=float)

    all_times = [0.0]
    all_pitch = [float(initial[0])]
    all_roll = [float(initial[1])]
    all_pitch_rate = [float(rates[0])]
    all_roll_rate = [float(rates[1])]
    stage_results: list[AttitudeStageResult] = []
    elapsed = 0.0

    for stage_index, duration in enumerate(durations):
        stage_start_state = _state_from_values(angle_rad, rate_rad_s)
        cumulative_mass_delta += mass_delta[stage_index]

        # cross([x, y, z], [0, 0, -dm*g]) gives roll=-y*dm*g,
        # pitch=x*dm*g.  This is the maintained plant's sign convention.
        pitch_ballast_moment = float(
            params.gravity_m_s2 * np.dot(tank_xy[:, 0], cumulative_mass_delta)
        )
        roll_ballast_moment = float(
            -params.gravity_m_s2 * np.dot(tank_xy[:, 1], cumulative_mass_delta)
        )
        ballast_moment = params.ballast_moment_scale * np.array(
            [pitch_ballast_moment, roll_ballast_moment], dtype=float
        )
        wind_moment = wind_effect[stage_index] * wind_scale
        total_moment = constant_moment + wind_moment + ballast_moment

        sample_count = max(
            1, int(math.ceil(float(duration) / params.metric_sample_interval_s))
        )
        local_times = np.linspace(0.0, float(duration), sample_count + 1)
        pitch_rad, pitch_rate_rad_s = _axis_response(
            angle_rad[0],
            rate_rad_s[0],
            total_moment[0],
            local_times,
            inertia_kg_m2=params.pitch_inertia_kg_m2,
            stiffness_nm_rad=params.pitch_stiffness_nm_rad,
            damping_ratio=params.pitch_damping_ratio,
        )
        roll_rad, roll_rate_rad_s = _axis_response(
            angle_rad[1],
            rate_rad_s[1],
            total_moment[1],
            local_times,
            inertia_kg_m2=params.roll_inertia_kg_m2,
            stiffness_nm_rad=params.roll_stiffness_nm_rad,
            damping_ratio=params.roll_damping_ratio,
        )
        local_pitch_deg = np.degrees(pitch_rad)
        local_roll_deg = np.degrees(roll_rad)
        local_pitch_rate_deg_s = np.degrees(pitch_rate_rad_s)
        local_roll_rate_deg_s = np.degrees(roll_rate_rad_s)
        stage_metrics = _metrics(
            local_times,
            local_pitch_deg,
            local_roll_deg,
            params.thresholds_deg,
        )

        angle_rad = np.array([pitch_rad[-1], roll_rad[-1]], dtype=float)
        rate_rad_s = np.array(
            [pitch_rate_rad_s[-1], roll_rate_rad_s[-1]], dtype=float
        )
        stage_results.append(
            AttitudeStageResult(
                stage_index=stage_index,
                start_time_s=elapsed,
                end_time_s=elapsed + float(duration),
                start_state=stage_start_state,
                end_state=_state_from_values(angle_rad, rate_rad_s),
                wind_moment_nm=(float(wind_moment[0]), float(wind_moment[1])),
                ballast_moment_nm=(
                    float(ballast_moment[0]),
                    float(ballast_moment[1]),
                ),
                total_moment_nm=(float(total_moment[0]), float(total_moment[1])),
                metrics=stage_metrics,
            )
        )

        all_times.extend((elapsed + local_times[1:]).tolist())
        all_pitch.extend(local_pitch_deg[1:].tolist())
        all_roll.extend(local_roll_deg[1:].tolist())
        all_pitch_rate.extend(local_pitch_rate_deg_s[1:].tolist())
        all_roll_rate.extend(local_roll_rate_deg_s[1:].tolist())
        elapsed += float(duration)

    times = np.asarray(all_times, dtype=float)
    pitch = np.asarray(all_pitch, dtype=float)
    roll = np.asarray(all_roll, dtype=float)
    pitch_rate = np.asarray(all_pitch_rate, dtype=float)
    roll_rate = np.asarray(all_roll_rate, dtype=float)
    return AttitudeRolloutResult(
        stages=tuple(stage_results),
        times_s=times,
        pitch_deg=pitch,
        roll_deg=roll,
        pitch_rate_deg_s=pitch_rate,
        roll_rate_deg_s=roll_rate,
        metrics=_metrics(times, pitch, roll, params.thresholds_deg),
    )


__all__ = [
    "AttitudeMetrics",
    "AttitudeRolloutParameters",
    "AttitudeRolloutResult",
    "AttitudeStageResult",
    "AttitudeState",
    "rollout_attitude",
]
