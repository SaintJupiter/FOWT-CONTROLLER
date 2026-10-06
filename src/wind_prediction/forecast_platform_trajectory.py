"""Shadow platform-state rollout under an explicit forecast-load convention.

``ForecastGeneralizedLoadAssembly`` holds discrete rotor load points rather
than a continuous forcing history.  This module makes the missing interval
assumption explicit and advances the frozen small-angle platform model through
those intervals.  It is a diagnostic bridge only: tank masses remain fixed,
the supplied wave and other loads are caller-owned, and no result enters
candidate ranking, target generation or pump execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fowt_platform.ballast_snapshot import BallastModelSnapshot
from fowt_platform.incremental import IncrementalState
from fowt_platform.open_loop_step import (
    OpenLoopPlatformStep,
    advance_frozen_open_loop_step,
)
from fowt_platform.rotor_input import RotorGeneralizedLoad

from .forecast_physical_load import (
    ForecastGeneralizedLoadAssembly,
    FuturePlatformKinematicsMode,
)


def _readonly_load_matrix(name: str, value: Any) -> np.ndarray:
    try:
        loads = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must have shape (H, 6) with H > 0 and finite values"
        ) from exc
    if (
        loads.ndim != 2
        or loads.shape[0] == 0
        or loads.shape[1] != 6
        or not np.all(np.isfinite(loads))
    ):
        raise ValueError(
            f"{name} must have shape (H, 6) with H > 0 and finite values"
        )
    result = np.array(loads, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _readonly_lead_times(
    name: str,
    value: Any,
    *,
    interval_count: int,
) -> np.ndarray:
    """Copy an explicit positive, strictly increasing interval-end grid."""

    try:
        leads = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain one finite positive lead per interval"
        ) from exc
    if (
        leads.shape != (interval_count,)
        or not np.all(np.isfinite(leads))
        or np.any(leads <= 0.0)
        or np.any(np.diff(leads) <= 0.0)
    ):
        raise ValueError(
            f"{name} must contain one finite positive strictly increasing lead per interval"
        )
    result = np.array(leads, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _readonly_six(name: str, value: Any) -> np.ndarray:
    try:
        load = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (6,) and finite values") from exc
    if load.shape != (6,) or not np.all(np.isfinite(load)):
        raise ValueError(f"{name} must have shape (6,) and finite values")
    result = np.array(load, dtype=float, copy=True)
    result.setflags(write=False)
    return result


FORECAST_INTERVAL_LOAD_CONVENTION = "future_point_backward_hold"


def _validated_initial_state_time(
    load_assembly: ForecastGeneralizedLoadAssembly,
    initial_state_time: Any,
) -> str:
    """Bind a frozen rollout state to the physical forecast origin.

    The preassembled rotor loads may include a caller-supplied future platform
    motion path.  Such a path cannot be combined with a second internally
    propagated trajectory without a coupled aerodynamic iteration.  The first
    shadow rollout therefore accepts only the explicit frozen-zero convention.
    """

    if (
        load_assembly.input_identity.future_platform_kinematics_mode
        is not FuturePlatformKinematicsMode.FROZEN_ZERO
    ):
        raise ValueError(
            "frozen forecast trajectory only supports frozen_zero future "
            "platform kinematics"
        )
    origin = load_assembly.input_identity.forecast_origin_time
    if origin is None:
        raise ValueError(
            "frozen forecast trajectory requires a forecast_origin_time"
        )
    state_time = str(initial_state_time).strip()
    if not state_time:
        raise ValueError("initial_state_time must be a non-empty label")
    if state_time != origin:
        raise ValueError(
            "initial_state_time must match load_assembly forecast_origin_time"
        )
    return state_time


@dataclass(frozen=True)
class ForecastTrajectoryEnvironment:
    """Explicit current and interval non-rotor loads for a forecast shadow.

    Current and future wave and other generalized loads remain separate because
    the origin RHS must not infer its values from the first forecast interval.
    They must be supplied even when the caller intentionally assumes zero.
    ``provenance`` records that shared environmental assumption instead of
    introducing an implicit zero channel.
    """

    current_wave_load: Any
    current_other_load: Any
    wave_interval_loads: Any
    other_interval_loads: Any
    provenance: str
    forecast_origin_time: str | None = None
    lead_times_s: Any | None = None

    def __post_init__(self) -> None:
        current_wave = _readonly_six("current_wave_load", self.current_wave_load)
        current_other = _readonly_six("current_other_load", self.current_other_load)
        wave = _readonly_load_matrix("wave_interval_loads", self.wave_interval_loads)
        other = _readonly_load_matrix(
            "other_interval_loads", self.other_interval_loads
        )
        if wave.shape != other.shape:
            raise ValueError(
                "wave_interval_loads and other_interval_loads must have equal shapes"
            )
        provenance = str(self.provenance).strip()
        if not provenance:
            raise ValueError("provenance must be a non-empty label")
        has_origin = self.forecast_origin_time is not None
        has_leads = self.lead_times_s is not None
        if has_origin != has_leads:
            raise ValueError(
                "forecast_origin_time and lead_times_s must be provided together"
            )
        if has_origin:
            forecast_origin_time = str(self.forecast_origin_time).strip()
            if not forecast_origin_time:
                raise ValueError("forecast_origin_time must be a non-empty label")
            lead_times_s = _readonly_lead_times(
                "lead_times_s",
                self.lead_times_s,
                interval_count=int(wave.shape[0]),
            )
        else:
            forecast_origin_time = None
            lead_times_s = None
        object.__setattr__(self, "current_wave_load", current_wave)
        object.__setattr__(self, "current_other_load", current_other)
        object.__setattr__(self, "wave_interval_loads", wave)
        object.__setattr__(self, "other_interval_loads", other)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "forecast_origin_time", forecast_origin_time)
        object.__setattr__(self, "lead_times_s", lead_times_s)

    @property
    def interval_count(self) -> int:
        return int(self.wave_interval_loads.shape[0])


def _validate_environment_forecast_binding(
    load_assembly: ForecastGeneralizedLoadAssembly,
    environment: ForecastTrajectoryEnvironment,
) -> None:
    """Check an optional non-rotor schedule identity against rotor loads."""

    if environment.forecast_origin_time is None:
        return
    forecast_origin_time = load_assembly.input_identity.forecast_origin_time
    if environment.forecast_origin_time != forecast_origin_time:
        raise ValueError(
            "environment forecast_origin_time must match load_assembly forecast origin"
        )
    expected_leads = np.asarray(
        [
            load_assembly.load_forecast.lead_time_at(index)
            for index in range(load_assembly.load_forecast.horizon_steps)
        ],
        dtype=float,
    )
    if not np.allclose(
        environment.lead_times_s,
        expected_leads,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError(
            "environment lead_times_s must match load_assembly forecast lead times"
        )


def _states_match(left: IncrementalState, right: IncrementalState) -> bool:
    return bool(
        np.allclose(left.position, right.position, rtol=0.0, atol=1.0e-10)
        and np.allclose(left.velocity, right.velocity, rtol=0.0, atol=1.0e-10)
    )


@dataclass(frozen=True)
class ForecastPlatformTrajectoryStep:
    """One fixed-tank, fixed-load interval ending at a forecast lead time."""

    lead_index: int
    start_time_s: float
    end_time_s: float
    rotor_load: RotorGeneralizedLoad
    wave_load: Any
    other_load: Any
    start_state: IncrementalState
    platform_step: OpenLoopPlatformStep

    def __post_init__(self) -> None:
        if not isinstance(self.lead_index, int) or isinstance(self.lead_index, bool):
            raise TypeError("lead_index must be an integer")
        if self.lead_index < 0:
            raise ValueError("lead_index must be non-negative")
        for name in ("start_time_s", "end_time_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.start_time_s < 0.0 or self.end_time_s <= self.start_time_s:
            raise ValueError("trajectory step times must form a positive interval")
        if not isinstance(self.rotor_load, RotorGeneralizedLoad):
            raise TypeError("rotor_load must be RotorGeneralizedLoad")
        object.__setattr__(self, "wave_load", _readonly_six("wave_load", self.wave_load))
        object.__setattr__(self, "other_load", _readonly_six("other_load", self.other_load))
        if not isinstance(self.start_state, IncrementalState):
            raise TypeError("start_state must be IncrementalState")
        if not isinstance(self.platform_step, OpenLoopPlatformStep):
            raise TypeError("platform_step must be OpenLoopPlatformStep")
        if not np.allclose(
            self.platform_step.loads.wind,
            self.rotor_load.generalized_load_platform,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError("platform_step wind load must match rotor_load")
        if not np.allclose(
            self.platform_step.loads.wave,
            self.wave_load,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError("platform_step wave load must match wave_load")
        if not np.allclose(
            self.platform_step.loads.other,
            self.other_load,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError("platform_step other load must match other_load")

    @property
    def duration_s(self) -> float:
        return float(self.end_time_s - self.start_time_s)

    @property
    def end_state(self) -> IncrementalState:
        return self.platform_step.next_state


@dataclass(frozen=True)
class ForecastPlatformTrajectory:
    """One state trajectory with fixed tank masses and explicit load holds.

    This object exposes a consistent future platform state at each discrete
    prediction lead.  It does not establish a new static equilibrium, update
    tank masses, recompute aerodynamic loading from the evolved velocity, or
    turn its states into a control demand.
    """

    load_assembly: ForecastGeneralizedLoadAssembly
    platform_snapshot: BallastModelSnapshot
    initial_state: IncrementalState
    initial_state_time: str
    environment: ForecastTrajectoryEnvironment
    steps: tuple[ForecastPlatformTrajectoryStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.load_assembly, ForecastGeneralizedLoadAssembly):
            raise TypeError("load_assembly must be ForecastGeneralizedLoadAssembly")
        if not isinstance(self.platform_snapshot, BallastModelSnapshot):
            raise TypeError("platform_snapshot must be BallastModelSnapshot")
        if not isinstance(self.initial_state, IncrementalState):
            raise TypeError("initial_state must be IncrementalState")
        initial_state_time = _validated_initial_state_time(
            self.load_assembly,
            self.initial_state_time,
        )
        if not isinstance(self.environment, ForecastTrajectoryEnvironment):
            raise TypeError("environment must be ForecastTrajectoryEnvironment")
        steps = tuple(self.steps)
        horizon = self.load_assembly.load_forecast.horizon_steps
        if len(steps) != horizon:
            raise ValueError("steps must contain one entry per physical forecast lead")
        if self.environment.interval_count != horizon:
            raise ValueError(
                "environment interval loads must contain one entry per physical forecast lead"
            )
        _validate_environment_forecast_binding(self.load_assembly, self.environment)

        previous_end = 0.0
        previous_state = self.initial_state
        for index, step in enumerate(steps):
            if not isinstance(step, ForecastPlatformTrajectoryStep):
                raise TypeError("steps must contain ForecastPlatformTrajectoryStep values")
            expected_end = self.load_assembly.load_forecast.lead_time_at(index)
            if step.lead_index != index:
                raise ValueError("trajectory lead indices must be contiguous")
            if not np.isclose(step.start_time_s, previous_end, rtol=0.0, atol=1.0e-9):
                raise ValueError("trajectory step start times must be contiguous")
            if not np.isclose(step.end_time_s, expected_end, rtol=0.0, atol=1.0e-9):
                raise ValueError("trajectory end time must match the forecast lead")
            if not _states_match(step.start_state, previous_state):
                raise ValueError("trajectory states must remain contiguous")
            expected_endpoint = self.load_assembly.future_rotor_loads[index]
            if step.rotor_load is not expected_endpoint:
                raise ValueError(
                    "trajectory rotor load must be the matching source-bound forecast load"
                )
            if not np.allclose(
                step.wave_load,
                self.environment.wave_interval_loads[index],
                rtol=0.0,
                atol=1.0e-8,
            ) or not np.allclose(
                step.other_load,
                self.environment.other_interval_loads[index],
                rtol=0.0,
                atol=1.0e-8,
            ):
                raise ValueError("trajectory external interval loads must match environment")
            previous_end = step.end_time_s
            previous_state = step.end_state
        object.__setattr__(self, "initial_state_time", initial_state_time)
        object.__setattr__(self, "steps", steps)

    @property
    def interval_load_convention(self) -> str:
        """Return the fixed right-endpoint hold convention used by this trace."""

        return FORECAST_INTERVAL_LOAD_CONVENTION

    @property
    def terminal_state(self) -> IncrementalState:
        return self.steps[-1].end_state


def rollout_frozen_forecast_platform_trajectory(
    *,
    load_assembly: ForecastGeneralizedLoadAssembly,
    platform_snapshot: BallastModelSnapshot,
    initial_state: IncrementalState,
    initial_state_time: str,
    environment: ForecastTrajectoryEnvironment,
) -> ForecastPlatformTrajectory:
    """Advance one fixed-tank platform shadow through forecast-load intervals.

    Every future rotor load is held over the interval immediately before its
    own timestamp, i.e. ``tau(+600 s)`` is applied on ``(0, +600 s]``.  This
    fixed right-endpoint hold is recorded by ``interval_load_convention``.
    Callers still supply non-rotor loads for every interval explicitly.  The
    existing frozen platform step is reused unchanged.  In particular, a
    preassembled rotor forecast remains frozen while the state evolves; this
    does not yet close the aerodynamic platform-motion feedback loop.
    """

    if not isinstance(load_assembly, ForecastGeneralizedLoadAssembly):
        raise TypeError("load_assembly must be ForecastGeneralizedLoadAssembly")
    if not isinstance(platform_snapshot, BallastModelSnapshot):
        raise TypeError("platform_snapshot must be BallastModelSnapshot")
    if not isinstance(initial_state, IncrementalState):
        raise TypeError("initial_state must be IncrementalState")
    normalized_initial_state_time = _validated_initial_state_time(
        load_assembly,
        initial_state_time,
    )
    if not isinstance(environment, ForecastTrajectoryEnvironment):
        raise TypeError("environment must be ForecastTrajectoryEnvironment")
    horizon = load_assembly.load_forecast.horizon_steps
    if environment.interval_count != horizon:
        raise ValueError(
            "environment interval loads must contain one entry per physical forecast lead"
        )
    _validate_environment_forecast_binding(load_assembly, environment)

    steps: list[ForecastPlatformTrajectoryStep] = []
    state = initial_state
    start_time_s = 0.0
    for index in range(horizon):
        end_time_s = load_assembly.load_forecast.lead_time_at(index)
        held_rotor = load_assembly.future_rotor_loads[index]
        step = advance_frozen_open_loop_step(
            snapshot=platform_snapshot,
            state=state,
            rotor_load=held_rotor,
            wave_load=environment.wave_interval_loads[index],
            other_load=environment.other_interval_loads[index],
            duration_s=end_time_s - start_time_s,
        )
        steps.append(
            ForecastPlatformTrajectoryStep(
                lead_index=index,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
                rotor_load=held_rotor,
                wave_load=environment.wave_interval_loads[index],
                other_load=environment.other_interval_loads[index],
                start_state=state,
                platform_step=step,
            )
        )
        state = step.next_state
        start_time_s = end_time_s
    return ForecastPlatformTrajectory(
        load_assembly=load_assembly,
        platform_snapshot=platform_snapshot,
        initial_state=initial_state,
        initial_state_time=normalized_initial_state_time,
        environment=environment,
        steps=tuple(steps),
    )


__all__ = [
    "FORECAST_INTERVAL_LOAD_CONVENTION",
    "ForecastPlatformTrajectory",
    "ForecastPlatformTrajectoryStep",
    "ForecastTrajectoryEnvironment",
    "rollout_frozen_forecast_platform_trajectory",
]
