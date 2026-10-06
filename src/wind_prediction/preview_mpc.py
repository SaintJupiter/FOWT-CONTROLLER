"""Control-oriented preview model for slow three-tank ballast redistribution.

This module is deliberately narrower than the physical execution chain.  It
uses the frozen small-angle platform matrices for one control cycle and plans
only the two zero-net differential ballast coordinates.  Each forecast block
assumes a constant generalized disturbance load while the requested modal
mass increment is applied as a linear ramp over the block.

The preview state is not the complete actuator state.  Pump latches, timers,
unfinished target error, and any common tank-mass change remain owned by the
physical execution model.  The optimizer built on this model must therefore
execute only its first block and pass that target through the existing pump
model before the next cycle is replanned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import osqp
from scipy.linalg import expm
from scipy import sparse

from fowt_platform import IncrementalState, PlatformMatrices, ThreeTankDifferentialModes
from fowt_platform.generalized_load_forecast import GeneralizedLoadForecast


_PLATFORM_STATE_SIZE = 12
_MODAL_STATE_SIZE = 2
_PREVIEW_STATE_SIZE = _PLATFORM_STATE_SIZE + _MODAL_STATE_SIZE
_DEFAULT_SAMPLE_FRACTIONS = tuple(np.arange(1, 121, dtype=float) / 120.0)
# The QP is solved in dimensionless scaled coordinates.  Keep the solver's
# requested precision distinct from the reconstructed-plan acceptance bound.
OSQP_FEASIBILITY_TOLERANCE = 1.0e-6
MAX_SCALED_RECONSTRUCTION_VIOLATION = 5.0 * OSQP_FEASIBILITY_TOLERANCE
# Solver-scaled residuals alone do not give a direct physical guarantee.  A
# solved plan is accepted only when its reconstructed posture and mass
# constraints also meet explicit physical-unit tolerances.  The 0.05 kg mass
# bound is consistent with the 5e-6 scaled acceptance bound at the controller's
# 10,250 kg reference movement scale; it remains negligible beside one pump
# block while avoiding a contradictory, tighter second solver tolerance.
MAX_POSTURE_RECONSTRUCTION_VIOLATION_RAD = 1.0e-6
MAX_MASS_RECONSTRUCTION_VIOLATION_KG = 5.0e-2


def _finite_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    return np.array(array, dtype=float, copy=True)


def _positive_scalar(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _readonly(value: Any) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PreviewState:
    """Platform state and planned redistribution relative to current tanks."""

    platform: IncrementalState
    modal_redistribution_kg: Any

    def __post_init__(self) -> None:
        if not isinstance(self.platform, IncrementalState):
            raise TypeError("platform must be IncrementalState")
        object.__setattr__(
            self,
            "modal_redistribution_kg",
            _readonly(
                _finite_array(
                    "modal_redistribution_kg",
                    self.modal_redistribution_kg,
                    (_MODAL_STATE_SIZE,),
                )
            ),
        )

    @classmethod
    def from_current_platform(cls, platform: IncrementalState) -> "PreviewState":
        """Start a new receding-horizon cycle at the measured tank state."""

        return cls(platform=platform, modal_redistribution_kg=np.zeros(2))

    def as_vector(self) -> np.ndarray:
        return _readonly(
            np.concatenate(
                (
                    self.platform.position,
                    self.platform.velocity,
                    self.modal_redistribution_kg,
                )
            )
        )

    @classmethod
    def from_vector(cls, value: Any) -> "PreviewState":
        vector = _finite_array("preview_state", value, (_PREVIEW_STATE_SIZE,))
        return cls(
            platform=IncrementalState(position=vector[:6], velocity=vector[6:12]),
            modal_redistribution_kg=vector[12:14],
        )


@dataclass(frozen=True)
class PreviewBlockSamples:
    """Exact samples under one constant-load, linear-modal-ramp block."""

    elapsed_time_s: Any
    state_vectors: Any

    def __post_init__(self) -> None:
        time = np.asarray(self.elapsed_time_s, dtype=float)
        states = np.asarray(self.state_vectors, dtype=float)
        if time.ndim != 1 or time.size == 0 or not np.all(np.isfinite(time)):
            raise ValueError("elapsed_time_s must be a non-empty finite vector")
        if states.shape != (time.size, _PREVIEW_STATE_SIZE):
            raise ValueError(
                "state_vectors must have one shape-(14,) row per sample time"
            )
        if np.any(time <= 0.0) or np.any(np.diff(time) <= 0.0):
            raise ValueError("elapsed_time_s must be positive and strictly increasing")
        if not np.all(np.isfinite(states)):
            raise ValueError("state_vectors must contain finite values")
        object.__setattr__(self, "elapsed_time_s", _readonly(time))
        object.__setattr__(self, "state_vectors", _readonly(states))

    @property
    def endpoint(self) -> PreviewState:
        return PreviewState.from_vector(self.state_vectors[-1])

    @property
    def roll_rad(self) -> np.ndarray:
        return _readonly(self.state_vectors[:, 3])

    @property
    def pitch_rad(self) -> np.ndarray:
        return _readonly(self.state_vectors[:, 4])

    @property
    def dominant_tilt_rad(self) -> np.ndarray:
        return _readonly(np.maximum(np.abs(self.roll_rad), np.abs(self.pitch_rad)))


@dataclass(frozen=True)
class PreviewTrajectory:
    """Forecast-block rollout with samples retained inside every block."""

    initial_state: PreviewState
    blocks: tuple[PreviewBlockSamples, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.initial_state, PreviewState):
            raise TypeError("initial_state must be PreviewState")
        blocks = tuple(self.blocks)
        if not blocks or not all(isinstance(block, PreviewBlockSamples) for block in blocks):
            raise ValueError("blocks must contain at least one PreviewBlockSamples")
        object.__setattr__(self, "blocks", blocks)

    @property
    def endpoint_states(self) -> tuple[PreviewState, ...]:
        return tuple(block.endpoint for block in self.blocks)

    @property
    def maximum_dominant_tilt_rad(self) -> float:
        return max(float(np.max(block.dominant_tilt_rad)) for block in self.blocks)

    def sampled_posture_energy(self) -> float:
        """Trapezoidal integral of squared roll and pitch over the horizon."""

        total = 0.0
        previous_time = 0.0
        previous_value = float(
            self.initial_state.platform.position[3] ** 2
            + self.initial_state.platform.position[4] ** 2
        )
        block_offset = 0.0
        for block in self.blocks:
            values = block.roll_rad**2 + block.pitch_rad**2
            for local_time, value in zip(block.elapsed_time_s, values):
                absolute_time = block_offset + float(local_time)
                total += 0.5 * (previous_value + float(value)) * (
                    absolute_time - previous_time
                )
                previous_time = absolute_time
                previous_value = float(value)
            block_offset += float(block.elapsed_time_s[-1])
        return float(total)


@dataclass(frozen=True)
class PreviewDisturbanceBlocks:
    """Physical interval loads assembled from endpoint forecasts and current ballast."""

    generalized_loads: Any
    block_duration_s: float
    endpoint_mapping: str = (
        "piecewise_linear_endpoints_to_interval_average_constant_load"
    )

    def __post_init__(self) -> None:
        loads = np.asarray(self.generalized_loads, dtype=float)
        if loads.ndim != 2 or loads.shape[0] == 0 or loads.shape[1] != 6:
            raise ValueError("generalized_loads must have shape (block_count, 6)")
        if not np.all(np.isfinite(loads)):
            raise ValueError("generalized_loads must contain finite values")
        object.__setattr__(self, "generalized_loads", _readonly(loads))
        object.__setattr__(
            self,
            "block_duration_s",
            _positive_scalar("block_duration_s", self.block_duration_s),
        )
        mapping = str(self.endpoint_mapping).strip()
        if not mapping:
            raise ValueError("endpoint_mapping must be non-empty")
        object.__setattr__(self, "endpoint_mapping", mapping)


def assemble_preview_disturbance_blocks(
    *,
    rotor_load_forecast: GeneralizedLoadForecast,
    current_incremental_ballast_load: Any,
    block_duration_s: Any,
    wave_interval_loads: Any | None = None,
    other_interval_loads: Any | None = None,
) -> PreviewDisturbanceBlocks:
    """Map discrete rotor-load endpoints to interval-average preview loads.

    Forecast points at ``+T, +2T, ...`` are treated as endpoints of a
    piecewise-linear load path beginning at the measured current rotor load.
    The interval load is the exact average of that linear path, namely the
    trapezoid of its two endpoints.  Current ballast gravity load is then
    added unchanged to every block.  Wave and other inputs are already
    interval loads and are never inferred here.
    """

    if not isinstance(rotor_load_forecast, GeneralizedLoadForecast):
        raise TypeError("rotor_load_forecast must be GeneralizedLoadForecast")
    duration = _positive_scalar("block_duration_s", block_duration_s)
    expected_leads = (
        np.arange(1, rotor_load_forecast.horizon_steps + 1, dtype=float) * duration
    )
    if not np.allclose(
        rotor_load_forecast.lead_times_s,
        expected_leads,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError(
            "rotor load forecast points must lie at consecutive block endpoints"
        )
    ballast = _finite_array(
        "current_incremental_ballast_load",
        current_incremental_ballast_load,
        (6,),
    )
    block_count = rotor_load_forecast.horizon_steps

    def interval_loads(name: str, value: Any | None) -> np.ndarray:
        if value is None:
            return np.zeros((block_count, 6), dtype=float)
        return _finite_array(name, value, (block_count, 6))

    wave = interval_loads("wave_interval_loads", wave_interval_loads)
    other = interval_loads("other_interval_loads", other_interval_loads)
    endpoint_rotor = np.vstack(
        (
            rotor_load_forecast.current_generalized_load,
            rotor_load_forecast.future_generalized_loads,
        )
    )
    rotor_interval_average = 0.5 * (
        endpoint_rotor[:-1] + endpoint_rotor[1:]
    )
    total = rotor_interval_average + wave + other + ballast
    return PreviewDisturbanceBlocks(
        generalized_loads=total,
        block_duration_s=duration,
    )


class PreviewBlockModel:
    """Exact block lift of frozen linear platform dynamics and modal ballast."""

    def __init__(
        self,
        *,
        matrices: PlatformMatrices,
        generalized_load_per_mode_kg: Any,
        block_duration_s: Any = 600.0,
        sample_fractions: Iterable[float] = _DEFAULT_SAMPLE_FRACTIONS,
    ) -> None:
        if not isinstance(matrices, PlatformMatrices):
            raise TypeError("matrices must be PlatformMatrices")
        modal_load = _finite_array(
            "generalized_load_per_mode_kg",
            generalized_load_per_mode_kg,
            (6, 2),
        )
        duration = _positive_scalar("block_duration_s", block_duration_s)
        fractions = np.asarray(tuple(sample_fractions), dtype=float)
        if (
            fractions.ndim != 1
            or fractions.size == 0
            or not np.all(np.isfinite(fractions))
            or np.any(fractions <= 0.0)
            or np.any(fractions > 1.0)
            or np.any(np.diff(fractions) <= 0.0)
            or not np.isclose(fractions[-1], 1.0, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError(
                "sample_fractions must be strictly increasing in (0, 1] and end at 1"
            )
        self._matrices = matrices
        self._modal_load = _readonly(modal_load)
        self._block_duration_s = duration
        self._sample_fractions = _readonly(fractions)
        self._continuous_state, self._continuous_control, self._continuous_load = (
            self._assemble_continuous_matrices()
        )
        self._discrete_cache: dict[
            float, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}

    @property
    def block_duration_s(self) -> float:
        return self._block_duration_s

    @property
    def matrices(self) -> PlatformMatrices:
        return self._matrices

    @property
    def sample_fractions(self) -> np.ndarray:
        return self._sample_fractions

    def _assemble_continuous_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mass = self._matrices.mass
        state = np.zeros((_PREVIEW_STATE_SIZE, _PREVIEW_STATE_SIZE), dtype=float)
        state[:6, 6:12] = np.eye(6)
        state[6:12, :6] = -np.linalg.solve(
            mass, self._matrices.restoring_stiffness
        )
        state[6:12, 6:12] = -np.linalg.solve(mass, self._matrices.damping)
        state[6:12, 12:14] = np.linalg.solve(mass, self._modal_load)

        control = np.zeros((_PREVIEW_STATE_SIZE, 2), dtype=float)
        control[12:14] = np.eye(2) / self._block_duration_s
        load = np.zeros((_PREVIEW_STATE_SIZE, 6), dtype=float)
        load[6:12] = np.linalg.solve(mass, np.eye(6))
        return _readonly(state), _readonly(control), _readonly(load)

    def discrete_matrices(
        self, duration_s: Any | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``Ad, Bu, Bd`` for constant increment and load over a duration."""

        duration = (
            self._block_duration_s
            if duration_s is None
            else _positive_scalar("duration_s", duration_s)
        )
        cached = self._discrete_cache.get(duration)
        if cached is not None:
            return cached
        augmented = np.zeros((22, 22), dtype=float)
        augmented[:14, :14] = self._continuous_state
        augmented[:14, 14:16] = self._continuous_control
        augmented[:14, 16:22] = self._continuous_load
        transition = expm(augmented * duration)
        result = (
            _readonly(transition[:14, :14]),
            _readonly(transition[:14, 14:16]),
            _readonly(transition[:14, 16:22]),
        )
        self._discrete_cache[duration] = result
        return result

    def propagate_block(
        self,
        *,
        initial_state: PreviewState,
        modal_increment_kg: Any,
        generalized_disturbance_load: Any,
    ) -> PreviewBlockSamples:
        """Propagate one block with constant load and a linear modal ramp."""

        if not isinstance(initial_state, PreviewState):
            raise TypeError("initial_state must be PreviewState")
        increment = _finite_array("modal_increment_kg", modal_increment_kg, (2,))
        load = _finite_array(
            "generalized_disturbance_load", generalized_disturbance_load, (6,)
        )
        initial = initial_state.as_vector()
        samples = []
        elapsed = self._block_duration_s * self._sample_fractions
        for duration in elapsed:
            transition, control_map, load_map = self.discrete_matrices(duration)
            samples.append(transition @ initial + control_map @ increment + load_map @ load)
        return PreviewBlockSamples(elapsed_time_s=elapsed, state_vectors=np.vstack(samples))

    def rollout(
        self,
        *,
        initial_state: PreviewState,
        modal_increments_kg: Any,
        generalized_disturbance_loads: Any,
    ) -> PreviewTrajectory:
        increments = np.asarray(modal_increments_kg, dtype=float)
        loads = np.asarray(generalized_disturbance_loads, dtype=float)
        if increments.ndim != 2 or increments.shape[1:] != (2,):
            raise ValueError("modal_increments_kg must have shape (block_count, 2)")
        if loads.shape != (increments.shape[0], 6):
            raise ValueError(
                "generalized_disturbance_loads must have shape (block_count, 6)"
            )
        if increments.shape[0] == 0:
            raise ValueError("rollout requires at least one forecast block")
        if not np.all(np.isfinite(increments)) or not np.all(np.isfinite(loads)):
            raise ValueError("rollout inputs must contain finite values")

        state = initial_state
        blocks = []
        for increment, load in zip(increments, loads):
            block = self.propagate_block(
                initial_state=state,
                modal_increment_kg=increment,
                generalized_disturbance_load=load,
            )
            blocks.append(block)
            state = block.endpoint
        return PreviewTrajectory(initial_state=initial_state, blocks=tuple(blocks))


@dataclass(frozen=True)
class PreviewMPCWeights:
    """Physical-unit coefficients for dimensionless preview-control costs."""

    roll: float
    pitch: float
    tank_movement: float
    tank_throughput: float = 0.0
    movement_change: float = 0.0
    posture_slack: float = 0.0
    terminal_roll: float = 0.0
    terminal_pitch: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "roll",
            "pitch",
            "tank_movement",
            "tank_throughput",
            "movement_change",
            "posture_slack",
            "terminal_roll",
            "terminal_pitch",
        ):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be finite and non-negative") from exc
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.roll == 0.0 and self.pitch == 0.0:
            raise ValueError("at least one posture weight must be positive")
        if (
            self.tank_movement == 0.0
            and self.tank_throughput == 0.0
            and self.movement_change == 0.0
        ):
            raise ValueError("at least one movement regularization weight must be positive")

    @classmethod
    def from_normalized_scales(
        cls,
        *,
        roll_scale_rad: Any,
        pitch_scale_rad: Any,
        tank_movement_scale_kg: Any,
        posture_slack_scale_rad: Any,
        running_posture_priority: Any = 1.0,
        terminal_posture_priority: Any = 1.0,
        tank_movement_priority: Any = 1.0e-2,
        tank_throughput_priority: Any = 0.0,
        movement_change_priority: Any = 1.0e-3,
        posture_slack_priority: Any = 1.0e2,
    ) -> "PreviewMPCWeights":
        """Construct coefficients from physical scales and dimensionless priorities.

        A normalized roll, pitch, tank movement, or slack equal to one has the
        corresponding dimensionless priority.  This keeps the design meaning
        independent of whether angles are represented in radians or tank
        motion in kilograms, while retaining the physical-unit coefficients
        required by the QP.
        """

        roll_scale = _positive_scalar("roll_scale_rad", roll_scale_rad)
        pitch_scale = _positive_scalar("pitch_scale_rad", pitch_scale_rad)
        movement_scale = _positive_scalar(
            "tank_movement_scale_kg", tank_movement_scale_kg
        )
        slack_scale = _positive_scalar(
            "posture_slack_scale_rad", posture_slack_scale_rad
        )

        def priority(name: str, value: Any) -> float:
            try:
                result = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be finite and non-negative") from exc
            if not np.isfinite(result) or result < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            return result

        running = priority("running_posture_priority", running_posture_priority)
        terminal = priority("terminal_posture_priority", terminal_posture_priority)
        movement = priority("tank_movement_priority", tank_movement_priority)
        throughput = priority(
            "tank_throughput_priority", tank_throughput_priority
        )
        change = priority("movement_change_priority", movement_change_priority)
        slack = priority("posture_slack_priority", posture_slack_priority)
        return cls(
            roll=running / roll_scale**2,
            pitch=running / pitch_scale**2,
            tank_movement=movement / movement_scale**2,
            tank_throughput=throughput / movement_scale,
            movement_change=change / movement_scale**2,
            posture_slack=slack / slack_scale**2,
            terminal_roll=terminal / roll_scale**2,
            terminal_pitch=terminal / pitch_scale**2,
        )


@dataclass(frozen=True)
class PreviewMPCConstraints:
    """Hard posture, tank-capacity, and per-block execution envelopes."""

    maximum_abs_roll_rad: float
    maximum_abs_pitch_rad: float
    maximum_abs_tank_mass_change_per_block_kg: Any
    maximum_posture_slack_rad: Any = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maximum_abs_roll_rad",
            _positive_scalar("maximum_abs_roll_rad", self.maximum_abs_roll_rad),
        )
        object.__setattr__(
            self,
            "maximum_abs_pitch_rad",
            _positive_scalar("maximum_abs_pitch_rad", self.maximum_abs_pitch_rad),
        )
        movement = _finite_array(
            "maximum_abs_tank_mass_change_per_block_kg",
            self.maximum_abs_tank_mass_change_per_block_kg,
            (3,),
        )
        if np.any(movement <= 0.0):
            raise ValueError(
                "maximum_abs_tank_mass_change_per_block_kg values must be positive"
            )
        object.__setattr__(
            self,
            "maximum_abs_tank_mass_change_per_block_kg",
            _readonly(movement),
        )
        slack = np.asarray(self.maximum_posture_slack_rad, dtype=float)
        if slack.ndim == 0:
            slack = np.full(2, float(slack))
        if slack.shape != (2,) or not np.all(np.isfinite(slack)):
            raise ValueError(
                "maximum_posture_slack_rad must be a finite scalar or shape-(2,) vector"
            )
        if np.any(slack < 0.0):
            raise ValueError("maximum_posture_slack_rad must be non-negative")
        object.__setattr__(self, "maximum_posture_slack_rad", _readonly(slack))


@dataclass(frozen=True)
class PreviewActuatorEnvelope:
    """Current-block axis-aligned motion outer envelope and target reference."""

    first_block_lower_tank_delta_kg: Any
    first_block_upper_tank_delta_kg: Any
    outstanding_target_tank_delta_kg: Any
    source: str

    def __post_init__(self) -> None:
        lower = _finite_array(
            "first_block_lower_tank_delta_kg",
            self.first_block_lower_tank_delta_kg,
            (3,),
        )
        upper = _finite_array(
            "first_block_upper_tank_delta_kg",
            self.first_block_upper_tank_delta_kg,
            (3,),
        )
        if np.any(lower > upper):
            raise ValueError("first-block actuator lower bounds must not exceed upper bounds")
        outstanding = _finite_array(
            "outstanding_target_tank_delta_kg",
            self.outstanding_target_tank_delta_kg,
            (3,),
        )
        source = str(self.source).strip()
        if not source:
            raise ValueError("source must be non-empty")
        object.__setattr__(self, "first_block_lower_tank_delta_kg", _readonly(lower))
        object.__setattr__(self, "first_block_upper_tank_delta_kg", _readonly(upper))
        object.__setattr__(self, "outstanding_target_tank_delta_kg", _readonly(outstanding))
        object.__setattr__(self, "source", source)


@dataclass(frozen=True)
class PreviewQPProblem:
    """Condensed convex QP solved by the deterministic preview controller."""

    hessian: Any
    linear: Any
    constraint_matrix: Any
    lower_bounds: Any
    upper_bounds: Any
    decision_scaling: Any
    constraint_row_scaling: Any
    objective_scale: float

    def __post_init__(self) -> None:
        hessian = np.asarray(self.hessian, dtype=float)
        if (
            hessian.ndim != 2
            or hessian.shape[0] == 0
            or hessian.shape[0] != hessian.shape[1]
            or not np.all(np.isfinite(hessian))
        ):
            raise ValueError("hessian must be a finite non-empty square matrix")
        if not np.allclose(hessian, hessian.T, rtol=0.0, atol=1.0e-10):
            raise ValueError("hessian must be symmetric")
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hessian)))
        if minimum_eigenvalue < -1.0e-9:
            raise ValueError("hessian must be positive semidefinite")
        decision_size = hessian.shape[0]
        linear = _finite_array("linear", self.linear, (decision_size,))
        matrix = np.asarray(self.constraint_matrix, dtype=float)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != decision_size
            or matrix.shape[0] == 0
            or not np.all(np.isfinite(matrix))
        ):
            raise ValueError(
                "constraint_matrix must be finite with one column per decision"
            )
        lower = np.asarray(self.lower_bounds, dtype=float)
        upper = np.asarray(self.upper_bounds, dtype=float)
        if lower.shape != (matrix.shape[0],) or upper.shape != lower.shape:
            raise ValueError("constraint bounds must match constraint_matrix rows")
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            raise ValueError("constraint bounds must not contain NaN")
        if np.any(lower > upper):
            raise ValueError("constraint lower bounds must not exceed upper bounds")
        decision_scaling = _finite_array(
            "decision_scaling", self.decision_scaling, (decision_size,)
        )
        if np.any(decision_scaling <= 0.0):
            raise ValueError("decision_scaling must be positive")
        row_scaling = _finite_array(
            "constraint_row_scaling",
            self.constraint_row_scaling,
            (matrix.shape[0],),
        )
        if np.any(row_scaling <= 0.0):
            raise ValueError("constraint_row_scaling must be positive")
        scale = _positive_scalar("objective_scale", self.objective_scale)
        object.__setattr__(self, "hessian", _readonly(hessian))
        object.__setattr__(self, "linear", _readonly(linear))
        object.__setattr__(self, "constraint_matrix", _readonly(matrix))
        object.__setattr__(self, "lower_bounds", _readonly(lower))
        object.__setattr__(self, "upper_bounds", _readonly(upper))
        object.__setattr__(self, "decision_scaling", _readonly(decision_scaling))
        object.__setattr__(self, "constraint_row_scaling", _readonly(row_scaling))
        object.__setattr__(self, "objective_scale", scale)


@dataclass(frozen=True)
class PreviewMPCObjectiveBreakdown:
    """Physical cost terms evaluated at the selected preview solution.

    ``absolute_total`` includes the fixed disturbance-only offset that the QP
    omits.  Subtracting ``constant_offset`` therefore reconstructs the reduced
    objective reported by the solver.  Per-block arrays expose where the
    horizon accumulates posture and ballast-motion cost without changing the
    optimization itself.
    """

    running_posture_by_block: Any
    tank_movement_by_block: Any
    tank_throughput_by_block: Any
    movement_change_by_block: Any
    terminal_posture: float
    posture_slack: float
    numerical_regularization: float
    constant_offset: float
    absolute_total: float

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        block_count: int | None = None
        for name in (
            "running_posture_by_block",
            "tank_movement_by_block",
            "tank_throughput_by_block",
            "movement_change_by_block",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a non-empty finite vector")
            if np.any(value < -1.0e-12):
                raise ValueError(f"{name} must be non-negative")
            if block_count is None:
                block_count = value.size
            elif value.size != block_count:
                raise ValueError("objective per-block arrays must have equal length")
            arrays[name] = np.maximum(value, 0.0)
        for name, value in arrays.items():
            object.__setattr__(self, name, _readonly(value))
        for name in (
            "terminal_posture",
            "posture_slack",
            "numerical_regularization",
            "constant_offset",
            "absolute_total",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < -1.0e-12:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, max(value, 0.0))

    @property
    def running_posture(self) -> float:
        return float(np.sum(self.running_posture_by_block))

    @property
    def tank_movement(self) -> float:
        return float(np.sum(self.tank_movement_by_block))

    @property
    def tank_throughput(self) -> float:
        return float(np.sum(self.tank_throughput_by_block))

    @property
    def movement_change(self) -> float:
        return float(np.sum(self.movement_change_by_block))

    @property
    def reduced_total(self) -> float:
        return float(self.absolute_total - self.constant_offset)


@dataclass(frozen=True)
class PreviewMPCResult:
    """One explicit preview solution; only the first target may be executed."""

    success: bool
    status: int
    message: str
    objective_value: float
    modal_increments_kg: Any
    planned_target_tank_masses_kg: Any
    current_target_tank_masses_kg: Any
    predicted_trajectory: PreviewTrajectory
    qp_problem: PreviewQPProblem
    objective_breakdown: PreviewMPCObjectiveBreakdown
    solver_name: str
    solver_primal_residual: float
    solver_dual_residual: float
    solver_iterations: int
    posture_slack_rad: Any
    maximum_posture_constraint_violation_rad: float
    maximum_nominal_posture_exceedance_rad: float
    maximum_tank_constraint_violation_kg: float
    maximum_block_tank_movement_violation_kg: float
    maximum_first_block_actuator_violation_kg: float
    maximum_tank_throughput_epigraph_violation_kg: float
    maximum_scaled_constraint_violation: float
    fixed_first_modal_increment_kg: Any | None = None

    def __post_init__(self) -> None:
        if type(self.success) is not bool:
            raise ValueError("success must be a boolean")
        object.__setattr__(self, "status", int(self.status))
        message = str(self.message).strip()
        if not message:
            raise ValueError("message must be non-empty")
        object.__setattr__(self, "message", message)
        for name in (
            "objective_value",
            "maximum_posture_constraint_violation_rad",
            "maximum_nominal_posture_exceedance_rad",
            "maximum_tank_constraint_violation_kg",
            "maximum_block_tank_movement_violation_kg",
            "maximum_first_block_actuator_violation_kg",
            "maximum_tank_throughput_epigraph_violation_kg",
            "maximum_scaled_constraint_violation",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        increments = np.asarray(self.modal_increments_kg, dtype=float)
        if increments.ndim != 2 or increments.shape[1:] != (2,):
            raise ValueError("modal_increments_kg must have shape (block_count, 2)")
        targets = np.asarray(self.planned_target_tank_masses_kg, dtype=float)
        if targets.shape != (increments.shape[0], 3):
            raise ValueError(
                "planned_target_tank_masses_kg must have shape (block_count, 3)"
            )
        current = _finite_array(
            "current_target_tank_masses_kg",
            self.current_target_tank_masses_kg,
            (3,),
        )
        fixed_first = self.fixed_first_modal_increment_kg
        if fixed_first is not None:
            fixed_first = _finite_array(
                "fixed_first_modal_increment_kg",
                fixed_first,
                (2,),
            )
            # Feasibility of the fixed row is already checked in the same
            # dimensionless, row-scaled constraint space used by OSQP.  Do
            # not impose a second kilogram-space tolerance here because its
            # equivalent depends on the modal and row scaling.
        if not isinstance(self.predicted_trajectory, PreviewTrajectory):
            raise TypeError("predicted_trajectory must be PreviewTrajectory")
        if not isinstance(self.qp_problem, PreviewQPProblem):
            raise TypeError("qp_problem must be PreviewQPProblem")
        if not isinstance(self.objective_breakdown, PreviewMPCObjectiveBreakdown):
            raise TypeError(
                "objective_breakdown must be PreviewMPCObjectiveBreakdown"
            )
        solver_name = str(self.solver_name).strip()
        if not solver_name:
            raise ValueError("solver_name must be non-empty")
        object.__setattr__(self, "solver_name", solver_name)
        for name in ("solver_primal_residual", "solver_dual_residual"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        iterations = int(self.solver_iterations)
        if iterations < 0:
            raise ValueError("solver_iterations must be non-negative")
        object.__setattr__(self, "solver_iterations", iterations)
        slack = _finite_array("posture_slack_rad", self.posture_slack_rad, (2,))
        if np.any(slack < -1.0e-10):
            raise ValueError("posture_slack_rad must be non-negative")
        object.__setattr__(self, "posture_slack_rad", _readonly(np.maximum(slack, 0.0)))
        object.__setattr__(self, "modal_increments_kg", _readonly(increments))
        object.__setattr__(
            self, "planned_target_tank_masses_kg", _readonly(targets)
        )
        object.__setattr__(self, "current_target_tank_masses_kg", _readonly(current))
        object.__setattr__(
            self,
            "fixed_first_modal_increment_kg",
            None if fixed_first is None else _readonly(fixed_first),
        )


class PreviewMPCController:
    """Minimal deterministic convex preview controller.

    The optimization variables are two modal mass increments per forecast
    block.  Disturbance loads enter only through the physical preview model;
    reliability and event probabilities are intentionally absent from this
    interface.  The resulting program has a quadratic objective and linear
    constraints and is solved by the dedicated OSQP convex-QP backend.
    """

    def __init__(
        self,
        *,
        block_model: PreviewBlockModel,
        ballast_modes: ThreeTankDifferentialModes,
        weights: PreviewMPCWeights,
        constraints: PreviewMPCConstraints,
        running_posture_reference_duration_s: Any | None = None,
    ) -> None:
        if not isinstance(block_model, PreviewBlockModel):
            raise TypeError("block_model must be PreviewBlockModel")
        if not isinstance(ballast_modes, ThreeTankDifferentialModes):
            raise TypeError("ballast_modes must be ThreeTankDifferentialModes")
        if not isinstance(weights, PreviewMPCWeights):
            raise TypeError("weights must be PreviewMPCWeights")
        if not isinstance(constraints, PreviewMPCConstraints):
            raise TypeError("constraints must be PreviewMPCConstraints")
        if not np.allclose(
            block_model._modal_load,
            ballast_modes.generalized_load_per_mode_kg,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError(
                "block_model and ballast_modes must use the same modal load map"
            )
        self._model = block_model
        self._modes = ballast_modes
        self._weights = weights
        self._constraints = constraints
        self._running_posture_reference_duration_s = (
            None
            if running_posture_reference_duration_s is None
            else _positive_scalar(
                "running_posture_reference_duration_s",
                running_posture_reference_duration_s,
            )
        )

    @property
    def weights(self) -> PreviewMPCWeights:
        """Return the immutable physical-unit objective weights."""

        return self._weights

    @property
    def constraints(self) -> PreviewMPCConstraints:
        """Return the immutable posture and tank-movement constraints."""

        return self._constraints

    @property
    def block_model(self) -> PreviewBlockModel:
        return self._model

    @property
    def ballast_modes(self) -> ThreeTankDifferentialModes:
        return self._modes

    @property
    def running_posture_reference_duration_s(self) -> float | None:
        """Fixed time basis for running-posture cost, when configured."""

        return self._running_posture_reference_duration_s

    @staticmethod
    def _sampled_roll_pitch(trajectory: PreviewTrajectory) -> np.ndarray:
        rows = []
        for block in trajectory.blocks:
            rows.extend(np.column_stack((block.roll_rad, block.pitch_rad)))
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _posture_time_average_weights(
        trajectory: PreviewTrajectory,
        *,
        normalization_duration_s: float | None = None,
    ) -> tuple[float, np.ndarray]:
        """Return initial-state and sampled trapezoid weights on one time basis."""

        sample_times: list[float] = []
        offset = 0.0
        for block in trajectory.blocks:
            sample_times.extend(
                offset + np.asarray(block.elapsed_time_s, dtype=float)
            )
            offset += float(block.elapsed_time_s[-1])
        times = np.asarray(sample_times, dtype=float)
        extended = np.concatenate(([0.0], times))
        intervals = np.diff(extended)
        weights = np.empty(times.size, dtype=float)
        if times.size == 1:
            weights[0] = 0.5 * intervals[0]
        else:
            weights[:-1] = 0.5 * (intervals[:-1] + intervals[1:])
            weights[-1] = 0.5 * intervals[-1]
        normalization_duration = (
            float(times[-1])
            if normalization_duration_s is None
            else _positive_scalar(
                "running_posture_normalization_duration_s",
                normalization_duration_s,
            )
        )
        if normalization_duration + 1.0e-12 < float(times[-1]):
            raise ValueError(
                "running posture normalization duration cannot be shorter "
                "than the optimized trajectory"
            )
        initial_weight = 0.5 * intervals[0] / normalization_duration
        return initial_weight, weights / normalization_duration

    def _condense_posture(
        self,
        *,
        initial_state: PreviewState,
        forecast_loads: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        block_count = forecast_loads.shape[0]
        zero_controls = np.zeros((block_count, 2))
        baseline = self._sampled_roll_pitch(
            self._model.rollout(
                initial_state=initial_state,
                modal_increments_kg=zero_controls,
                generalized_disturbance_loads=forecast_loads,
            )
        ).reshape(-1)
        response = np.empty((baseline.size, 2 * block_count), dtype=float)
        for column in range(2 * block_count):
            controls = np.zeros((block_count, 2))
            controls.reshape(-1)[column] = 1.0
            sampled = self._sampled_roll_pitch(
                self._model.rollout(
                    initial_state=initial_state,
                    modal_increments_kg=controls,
                    generalized_disturbance_loads=forecast_loads,
                )
            ).reshape(-1)
            response[:, column] = sampled - baseline
        return baseline, response

    @staticmethod
    def _cumulative_modal_map(block_count: int) -> np.ndarray:
        return np.kron(np.tril(np.ones((block_count, block_count))), np.eye(2))

    @staticmethod
    def _block_tank_movement_map(
        block_count: int, tank_basis: np.ndarray
    ) -> np.ndarray:
        return np.kron(np.eye(block_count), tank_basis)

    def solve(
        self,
        *,
        initial_platform_state: IncrementalState,
        actual_tank_masses_kg: Any,
        tank_capacities_kg: Any,
        generalized_disturbance_loads: Any,
        actuator_envelope: PreviewActuatorEnvelope | None = None,
        fixed_first_modal_increment_kg: Any | None = None,
        previous_tank_movement_kg: Any | None = None,
        running_posture_normalization_duration_s: Any | None = None,
        minimum_posture_slack_rad: Any | None = None,
    ) -> PreviewMPCResult:
        """Solve one horizon and return the first executable tank target.

        ``fixed_first_modal_increment_kg`` is used when an outer target-lifecycle
        decision has already replayed one physical first-block request.  It
        fixes that request's reachable differential movement in the same QP
        while leaving all later blocks free to re-optimise.  The option does
        not choose a lifecycle or add a discrete decision to the convex model.

        ``previous_tank_movement_kg`` supplies the actual preceding three-tank
        movement when this solve starts after a physically replayed prefix.
        It affects only the first movement-change term of the new horizon.

        A tail solve can retain the original full-horizon posture scale through
        ``running_posture_normalization_duration_s``.  Its slack variable can
        likewise inherit posture exceedance already reached in the physical
        prefix through ``minimum_posture_slack_rad``.
        """

        if not isinstance(initial_platform_state, IncrementalState):
            raise TypeError("initial_platform_state must be IncrementalState")
        actual = _finite_array("actual_tank_masses_kg", actual_tank_masses_kg, (3,))
        capacities = _finite_array("tank_capacities_kg", tank_capacities_kg, (3,))
        if np.any(capacities <= 0.0):
            raise ValueError("tank_capacities_kg values must be positive")
        if np.any(actual < 0.0) or np.any(actual > capacities):
            raise ValueError("actual_tank_masses_kg must remain within capacities")
        loads = np.asarray(generalized_disturbance_loads, dtype=float)
        if loads.ndim != 2 or loads.shape[1:] != (6,) or loads.shape[0] == 0:
            raise ValueError(
                "generalized_disturbance_loads must have shape (block_count, 6)"
            )
        if not np.all(np.isfinite(loads)):
            raise ValueError("generalized_disturbance_loads must contain finite values")

        if actuator_envelope is not None and not isinstance(
            actuator_envelope, PreviewActuatorEnvelope
        ):
            raise TypeError("actuator_envelope must be PreviewActuatorEnvelope")
        fixed_first_modal = None
        if fixed_first_modal_increment_kg is not None:
            fixed_first_modal = _finite_array(
                "fixed_first_modal_increment_kg",
                fixed_first_modal_increment_kg,
                (2,),
            )
        previous_movement = None
        if previous_tank_movement_kg is not None:
            previous_movement = _finite_array(
                "previous_tank_movement_kg",
                previous_tank_movement_kg,
                (3,),
            )
        minimum_slack = np.zeros(2, dtype=float)
        if minimum_posture_slack_rad is not None:
            minimum_slack = _finite_array(
                "minimum_posture_slack_rad",
                minimum_posture_slack_rad,
                (2,),
            )
            if np.any(minimum_slack < 0.0):
                raise ValueError("minimum_posture_slack_rad must be non-negative")
            if np.any(
                minimum_slack
                > self._constraints.maximum_posture_slack_rad + 1.0e-12
            ):
                raise ValueError(
                    "minimum_posture_slack_rad cannot exceed the configured "
                    "maximum posture slack"
                )

        block_count = loads.shape[0]
        control_size = 2 * block_count
        slack_size = 2
        throughput_size = (
            3 * block_count if self._weights.tank_throughput > 0.0 else 0
        )
        slack_slice = slice(control_size, control_size + slack_size)
        throughput_slice = slice(
            control_size + slack_size,
            control_size + slack_size + throughput_size,
        )
        decision_size = control_size + slack_size + throughput_size
        initial = PreviewState.from_current_platform(initial_platform_state)
        initial_roll_pitch = np.abs(initial_platform_state.position[[3, 4]])
        initial_limits = np.array(
            [
                self._constraints.maximum_abs_roll_rad,
                self._constraints.maximum_abs_pitch_rad,
            ]
        )
        hard_initial_limits = (
            initial_limits + self._constraints.maximum_posture_slack_rad
        )
        # The current measured posture may already lie outside the sampled
        # horizon envelope.  It remains the fixed initial condition rather
        # than an optimization constraint; the sampled future path must either
        # recover inside the bounded soft envelope or return infeasible.
        posture_offset, posture_map = self._condense_posture(
            initial_state=initial,
            forecast_loads=loads,
        )
        baseline_trajectory = self._model.rollout(
            initial_state=initial,
            modal_increments_kg=np.zeros((block_count, 2)),
            generalized_disturbance_loads=loads,
        )
        normalization_duration_s = running_posture_normalization_duration_s
        if normalization_duration_s is None:
            normalization_duration_s = self._running_posture_reference_duration_s
        initial_time_weight, time_weights = self._posture_time_average_weights(
            baseline_trajectory,
            normalization_duration_s=normalization_duration_s,
        )
        posture_weights = np.tile(
            np.array([self._weights.roll, self._weights.pitch]),
            posture_offset.size // 2,
        ) * np.repeat(time_weights, 2)
        weighted_posture_map = posture_weights[:, None] * posture_map
        weighted_posture_offset = posture_weights * posture_offset

        tank_basis = self._modes.tank_mass_basis
        movement_map = self._block_tank_movement_map(block_count, tank_basis)
        difference = np.eye(control_size)
        if block_count > 1:
            for block_index in range(1, block_count):
                rows = slice(2 * block_index, 2 * block_index + 2)
                previous = slice(2 * (block_index - 1), 2 * block_index)
                difference[rows, previous] = -np.eye(2)
        movement_change_map = movement_map @ difference

        control_hessian = (
            posture_map.T @ (posture_weights[:, None] * posture_map)
            + self._weights.tank_movement * (movement_map.T @ movement_map)
            + self._weights.movement_change
            * (movement_change_map.T @ movement_change_map)
        )
        control_linear = posture_map.T @ weighted_posture_offset
        terminal_posture_map = posture_map[-2:]
        terminal_posture_offset = posture_offset[-2:]
        terminal_posture_weights = np.array(
            [self._weights.terminal_roll, self._weights.terminal_pitch]
        )
        control_hessian += terminal_posture_map.T @ (
            terminal_posture_weights[:, None] * terminal_posture_map
        )
        control_linear += terminal_posture_map.T @ (
            terminal_posture_weights * terminal_posture_offset
        )
        movement_change_reference = np.zeros(3 * block_count, dtype=float)
        if previous_movement is not None:
            movement_change_reference[:3] = previous_movement
        control_linear -= (
            self._weights.movement_change
            * movement_change_map.T
            @ movement_change_reference
        )

        hessian = np.zeros((decision_size, decision_size), dtype=float)
        hessian[:control_size, :control_size] = 0.5 * (
            control_hessian + control_hessian.T
        )
        hessian[slack_slice, slack_slice] += (
            self._weights.posture_slack * np.eye(slack_size)
        )
        linear = np.zeros(decision_size, dtype=float)
        linear[:control_size] = control_linear
        if throughput_size:
            linear[throughput_slice] = self._weights.tank_throughput

        def objective(decision: np.ndarray) -> float:
            return float(0.5 * decision @ hessian @ decision + linear @ decision)

        # Physical posture sensitivities can be several orders smaller than
        # one kilogram.  A positive scalar rescales the full objective without
        # changing the optimizer or any physical constraint.
        lower: list[np.ndarray] = []
        upper: list[np.ndarray] = []
        matrices: list[np.ndarray] = []

        posture_limit = np.tile(
            np.array(
                [
                    self._constraints.maximum_abs_roll_rad,
                    self._constraints.maximum_abs_pitch_rad,
                ]
            ),
            posture_offset.size // 2,
        )
        posture_axis_selector = np.tile(
            np.eye(2), (posture_offset.size // 2, 1)
        )
        throughput_zeros_for_posture = np.zeros(
            (posture_map.shape[0], throughput_size)
        )
        posture_upper_map = np.hstack(
            (posture_map, -posture_axis_selector, throughput_zeros_for_posture)
        )
        matrices.append(posture_upper_map)
        lower.append(np.full(posture_offset.size, -np.inf))
        upper.append(posture_limit - posture_offset)

        posture_lower_map = np.hstack(
            (posture_map, posture_axis_selector, throughput_zeros_for_posture)
        )
        matrices.append(posture_lower_map)
        lower.append(-posture_limit - posture_offset)
        upper.append(np.full(posture_offset.size, np.inf))

        slack_map = np.hstack(
            (
                np.zeros((slack_size, control_size)),
                np.eye(2),
                np.zeros((slack_size, throughput_size)),
            )
        )
        matrices.append(slack_map)
        lower.append(minimum_slack)
        upper.append(self._constraints.maximum_posture_slack_rad)

        if fixed_first_modal is not None:
            fixed_first_map = np.zeros((2, decision_size), dtype=float)
            fixed_first_map[:, :2] = np.eye(2)
            matrices.append(fixed_first_map)
            lower.append(fixed_first_modal)
            upper.append(fixed_first_modal)

        cumulative_modal = self._cumulative_modal_map(block_count)
        target_map = np.kron(np.eye(block_count), tank_basis) @ cumulative_modal
        repeated_actual = np.tile(actual, block_count)
        repeated_capacity = np.tile(capacities, block_count)
        target_map_with_slack = np.hstack(
            (
                target_map,
                np.zeros((target_map.shape[0], slack_size + throughput_size)),
            )
        )
        matrices.append(target_map_with_slack)
        lower.append(-repeated_actual)
        upper.append(repeated_capacity - repeated_actual)

        repeated_movement_limit = np.tile(
            self._constraints.maximum_abs_tank_mass_change_per_block_kg,
            block_count,
        )
        movement_lower = -repeated_movement_limit
        movement_upper = repeated_movement_limit
        if actuator_envelope is not None:
            # The current block is governed by measured latch, dwell and ramp
            # state.  Its physical envelope supersedes the generic steady
            # per-block limit, which remains the approximation for later
            # forecast blocks.
            movement_lower[:3] = (
                actuator_envelope.first_block_lower_tank_delta_kg
            )
            movement_upper[:3] = (
                actuator_envelope.first_block_upper_tank_delta_kg
            )
        movement_map_with_slack = np.hstack(
            (
                movement_map,
                np.zeros((movement_map.shape[0], slack_size + throughput_size)),
            )
        )
        matrices.append(movement_map_with_slack)
        lower.append(movement_lower)
        upper.append(movement_upper)

        if throughput_size:
            throughput_identity = np.eye(throughput_size)
            throughput_slack_zeros = np.zeros((throughput_size, slack_size))
            matrices.append(
                np.hstack(
                    (
                        movement_map,
                        throughput_slack_zeros,
                        -throughput_identity,
                    )
                )
            )
            lower.append(np.full(throughput_size, -np.inf))
            upper.append(np.zeros(throughput_size))
            matrices.append(
                np.hstack(
                    (
                        -movement_map,
                        throughput_slack_zeros,
                        -throughput_identity,
                    )
                )
            )
            lower.append(np.full(throughput_size, -np.inf))
            upper.append(np.zeros(throughput_size))

        constraint_matrix = np.vstack(matrices)
        constraint_lower = np.concatenate(lower)
        constraint_upper = np.concatenate(upper)

        modal_scale = np.empty(2, dtype=float)
        movement_limit = self._constraints.maximum_abs_tank_mass_change_per_block_kg
        for mode_index in range(2):
            active = np.abs(tank_basis[:, mode_index]) > 1.0e-12
            modal_scale[mode_index] = float(
                np.min(
                    movement_limit[active]
                    / np.abs(tank_basis[active, mode_index])
                )
            )
        slack_scale = np.where(
            self._constraints.maximum_posture_slack_rad > 0.0,
            self._constraints.maximum_posture_slack_rad,
            initial_limits,
        )
        decision_scaling = np.concatenate(
            (
                np.tile(modal_scale, block_count),
                slack_scale,
                repeated_movement_limit if throughput_size else np.empty(0),
            )
        )
        scaled_hessian = (
            decision_scaling[:, None]
            * hessian
            * decision_scaling[None, :]
        )
        scaled_linear = decision_scaling * linear
        scaled_constraint_matrix = constraint_matrix * decision_scaling[None, :]
        finite_bound_magnitude = np.maximum(
            np.where(np.isfinite(constraint_lower), np.abs(constraint_lower), 0.0),
            np.where(np.isfinite(constraint_upper), np.abs(constraint_upper), 0.0),
        )
        constraint_characteristic = np.maximum(
            np.max(np.abs(scaled_constraint_matrix), axis=1),
            finite_bound_magnitude,
        )
        constraint_row_scaling = 1.0 / np.maximum(
            constraint_characteristic, 1.0e-12
        )
        solver_constraint_matrix = (
            constraint_row_scaling[:, None] * scaled_constraint_matrix
        )
        solver_lower = constraint_row_scaling * constraint_lower
        solver_upper = constraint_row_scaling * constraint_upper
        objective_scale = 1.0 / max(
            float(np.max(np.abs(scaled_hessian))),
            float(np.max(np.abs(scaled_linear))),
            1.0e-12,
        )
        qp_problem = PreviewQPProblem(
            hessian=hessian,
            linear=linear,
            constraint_matrix=constraint_matrix,
            lower_bounds=constraint_lower,
            upper_bounds=constraint_upper,
            decision_scaling=decision_scaling,
            constraint_row_scaling=constraint_row_scaling,
            objective_scale=objective_scale,
        )
        # Add numerical curvature only after decisions and the objective are
        # dimensionlessly scaled. It is a solver tie-breaker, not a physical
        # control cost, and is therefore absent from the reported objective.
        solver_hessian = (
            objective_scale * scaled_hessian
            + 1.0e-12 * np.eye(decision_size)
        )
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix(np.triu(solver_hessian)),
            q=objective_scale * scaled_linear,
            A=sparse.csc_matrix(solver_constraint_matrix),
            l=solver_lower,
            u=solver_upper,
            # The rows and decisions are dimensionlessly scaled.  A 1e-6
            # solver tolerance remains far below one kilogram and one
            # microradian in the reconstructed physical constraints, while
            # avoiding artificial non-convergence of the L1 epigraph at 1e-8.
            eps_abs=OSQP_FEASIBILITY_TOLERANCE,
            eps_rel=OSQP_FEASIBILITY_TOLERANCE,
            max_iter=100_000,
            polishing=True,
            verbose=False,
        )
        solved = solver.solve(raise_error=False)
        solver_success = int(solved.info.status_val) == 1
        if (
            not solver_success
            or solved.x is None
            or not np.all(np.isfinite(solved.x))
        ):
            decision = np.zeros(decision_size, dtype=float)
        else:
            decision = decision_scaling * np.asarray(solved.x, dtype=float)
        control = decision[:control_size]
        posture_slack = np.maximum(decision[slack_slice], 0.0)
        if throughput_size:
            throughput_auxiliary = np.maximum(
                decision[throughput_slice].reshape(block_count, 3), 0.0
            )
        else:
            throughput_auxiliary = np.zeros((block_count, 3), dtype=float)
        increments = control.reshape(block_count, 2)
        cumulative = (cumulative_modal @ control).reshape(block_count, 2)
        targets = np.vstack(
            [
                self._modes.target_tank_masses(
                    reference_tank_masses_kg=actual,
                    modal_masses_kg=modal,
                )
                for modal in cumulative
            ]
        )
        trajectory = self._model.rollout(
            initial_state=initial,
            modal_increments_kg=increments,
            generalized_disturbance_loads=loads,
        )
        sampled_posture = self._sampled_roll_pitch(trajectory)
        nominal_posture_exceedance = np.maximum(
            np.abs(sampled_posture)
            - np.array(
                [
                    self._constraints.maximum_abs_roll_rad,
                    self._constraints.maximum_abs_pitch_rad,
                ]
            ),
            0.0,
        )
        posture_violation = np.maximum(
            np.abs(sampled_posture)
            - (
                np.array(
                    [
                        self._constraints.maximum_abs_roll_rad,
                        self._constraints.maximum_abs_pitch_rad,
                    ]
                )
                + posture_slack
            ),
            0.0,
        )
        tank_violation = np.maximum(
            np.maximum(-targets, targets - capacities),
            0.0,
        )
        block_tank_movements = (movement_map @ control).reshape(block_count, 3)
        posture_cost_by_sample = 0.5 * time_weights * (
            self._weights.roll * sampled_posture[:, 0] ** 2
            + self._weights.pitch * sampled_posture[:, 1] ** 2
        )
        running_posture_by_block = np.empty(block_count, dtype=float)
        sample_cursor = 0
        for block_index, block in enumerate(trajectory.blocks):
            next_cursor = sample_cursor + block.elapsed_time_s.size
            running_posture_by_block[block_index] = float(
                np.sum(posture_cost_by_sample[sample_cursor:next_cursor])
            )
            sample_cursor = next_cursor
        initial_posture = initial_platform_state.position[[3, 4]]
        initial_posture_cost = 0.5 * initial_time_weight * float(
            self._weights.roll * initial_posture[0] ** 2
            + self._weights.pitch * initial_posture[1] ** 2
        )
        running_posture_by_block[0] += initial_posture_cost
        tank_movement_by_block = 0.5 * self._weights.tank_movement * np.sum(
            block_tank_movements**2, axis=1
        )
        tank_throughput_by_block = self._weights.tank_throughput * np.sum(
            throughput_auxiliary, axis=1
        )
        movement_change_residual = (
            movement_change_map @ control - movement_change_reference
        ).reshape(block_count, 3)
        movement_change_by_block = 0.5 * self._weights.movement_change * np.sum(
            movement_change_residual**2, axis=1
        )
        terminal_posture = 0.5 * float(
            np.sum(terminal_posture_weights * sampled_posture[-1] ** 2)
        )
        posture_slack_cost = 0.5 * self._weights.posture_slack * float(
            posture_slack @ posture_slack
        )
        numerical_regularization = 0.0
        constant_offset = 0.5 * float(
            posture_offset @ (posture_weights * posture_offset)
            + terminal_posture_offset
            @ (terminal_posture_weights * terminal_posture_offset)
            + self._weights.movement_change
            * movement_change_reference
            @ movement_change_reference
        ) + initial_posture_cost
        absolute_total = float(
            np.sum(running_posture_by_block)
            + np.sum(tank_movement_by_block)
            + np.sum(tank_throughput_by_block)
            + np.sum(movement_change_by_block)
            + terminal_posture
            + posture_slack_cost
            + numerical_regularization
        )
        objective_breakdown = PreviewMPCObjectiveBreakdown(
            running_posture_by_block=running_posture_by_block,
            tank_movement_by_block=tank_movement_by_block,
            tank_throughput_by_block=tank_throughput_by_block,
            movement_change_by_block=movement_change_by_block,
            terminal_posture=terminal_posture,
            posture_slack=posture_slack_cost,
            numerical_regularization=numerical_regularization,
            constant_offset=constant_offset,
            absolute_total=absolute_total,
        )
        movement_violation = np.maximum(
            np.maximum(
                movement_lower.reshape(block_count, 3) - block_tank_movements,
                block_tank_movements - movement_upper.reshape(block_count, 3),
            ),
            0.0,
        )
        if throughput_size:
            throughput_epigraph_violation = np.maximum(
                np.abs(block_tank_movements) - throughput_auxiliary,
                0.0,
            )
        else:
            throughput_epigraph_violation = np.zeros_like(block_tank_movements)
        if actuator_envelope is None:
            first_block_actuator_violation = np.zeros(3, dtype=float)
        else:
            first_block_actuator_violation = np.maximum(
                np.maximum(
                    actuator_envelope.first_block_lower_tank_delta_kg
                    - block_tank_movements[0],
                    block_tank_movements[0]
                    - actuator_envelope.first_block_upper_tank_delta_kg,
                ),
                0.0,
            )
        reconstructed_constraint_value = constraint_matrix @ decision
        scaled_constraint_violation = constraint_row_scaling * np.maximum(
            np.maximum(
                constraint_lower - reconstructed_constraint_value,
                reconstructed_constraint_value - constraint_upper,
            ),
            0.0,
        )
        maximum_posture_violation = float(np.max(posture_violation))
        maximum_tank_violation = float(np.max(tank_violation))
        maximum_movement_violation = float(np.max(movement_violation))
        maximum_first_block_actuator_violation = float(
            np.max(first_block_actuator_violation)
        )
        maximum_throughput_epigraph_violation = float(
            np.max(throughput_epigraph_violation)
        )
        # OSQP solves a row-scaled problem, but the controller consumes
        # radians and kilograms.  Require both numerical convergence and a
        # physically meaningful reconstruction check before accepting a plan.
        success = bool(
            solver_success
            and float(np.max(scaled_constraint_violation))
            <= MAX_SCALED_RECONSTRUCTION_VIOLATION
            and maximum_posture_violation
            <= MAX_POSTURE_RECONSTRUCTION_VIOLATION_RAD
            and maximum_tank_violation <= MAX_MASS_RECONSTRUCTION_VIOLATION_KG
            and maximum_movement_violation
            <= MAX_MASS_RECONSTRUCTION_VIOLATION_KG
            and maximum_first_block_actuator_violation
            <= MAX_MASS_RECONSTRUCTION_VIOLATION_KG
            and maximum_throughput_epigraph_violation
            <= MAX_MASS_RECONSTRUCTION_VIOLATION_KG
        )
        message = str(solved.info.status)
        if solver_success and not success:
            message = "solver reported success but the reconstructed plan violates constraints"
        return PreviewMPCResult(
            success=success,
            status=int(solved.info.status_val),
            message=message,
            objective_value=objective(decision),
            modal_increments_kg=increments,
            planned_target_tank_masses_kg=targets,
            current_target_tank_masses_kg=targets[0],
            predicted_trajectory=trajectory,
            qp_problem=qp_problem,
            objective_breakdown=objective_breakdown,
            solver_name="OSQP",
            solver_primal_residual=float(abs(solved.info.prim_res)),
            solver_dual_residual=float(abs(solved.info.dual_res)),
            solver_iterations=int(solved.info.iter),
            posture_slack_rad=posture_slack,
            maximum_posture_constraint_violation_rad=maximum_posture_violation,
            maximum_nominal_posture_exceedance_rad=float(
                np.max(nominal_posture_exceedance)
            ),
            maximum_tank_constraint_violation_kg=maximum_tank_violation,
            maximum_block_tank_movement_violation_kg=maximum_movement_violation,
            maximum_first_block_actuator_violation_kg=(
                maximum_first_block_actuator_violation
            ),
            maximum_tank_throughput_epigraph_violation_kg=(
                maximum_throughput_epigraph_violation
            ),
            maximum_scaled_constraint_violation=float(
                np.max(scaled_constraint_violation)
            ),
            fixed_first_modal_increment_kg=fixed_first_modal,
        )


__all__ = [
    "PreviewBlockModel",
    "PreviewBlockSamples",
    "PreviewDisturbanceBlocks",
    "PreviewMPCController",
    "PreviewMPCConstraints",
    "PreviewMPCObjectiveBreakdown",
    "PreviewActuatorEnvelope",
    "PreviewMPCResult",
    "PreviewMPCWeights",
    "PreviewQPProblem",
    "PreviewState",
    "PreviewTrajectory",
    "OSQP_FEASIBILITY_TOLERANCE",
    "MAX_SCALED_RECONSTRUCTION_VIOLATION",
    "assemble_preview_disturbance_blocks",
]
