"""Causal two-axis posture filtering, independent of any control objective.

Times are absolute seconds; posture columns are roll and pitch in radians.
The caller owns initialization and the physical time constant. No task band
or change to the raw-posture M0 objective is defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _readonly_array(name: str, values: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _time_vector(values: Any, *, includes_root: bool) -> np.ndarray:
    times = np.asarray(values, dtype=float)
    if (
        times.ndim != 1
        or (includes_root and times.size == 0)
        or not np.all(np.isfinite(times))
    ):
        raise ValueError("times must be a finite one-dimensional vector")
    if np.any(times[1:] <= times[:-1]):
        raise ValueError("times must be strictly increasing")
    return _readonly_array("times", times, times.shape)


@dataclass(frozen=True)
class PreviewPostureFilter:
    """One explicit, positive time constant shared by roll and pitch."""

    time_constant_s: float

    def __post_init__(self) -> None:
        value = float(self.time_constant_s)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("time_constant_s must be positive and finite")
        object.__setattr__(self, "time_constant_s", value)

    def advance(
        self,
        *,
        initial_state: PreviewPostureFilterState,
        sample_times_s: Any,
        posture_rad: Any,
    ) -> PreviewPostureFilterTrace:
        """Consume each future sample once and return root plus filtered samples.

        Continue another segment from the returned end_state, not from its
        root. Empty shape-(0, 2) samples return the unchanged root alone.
        """

        posture = np.asarray(posture_rad, dtype=float)
        if posture.ndim != 2 or posture.shape[1] != 2:
            raise ValueError("posture_rad must have shape (n, 2)")
        affine = filter_affine_task_signal(
            config=self,
            initial_state=initial_state,
            sample_times_s=sample_times_s,
            constant_rad=posture,
            response_rad=np.zeros((posture.shape[0], 2, 0)),
        )
        return PreviewPostureFilterTrace(
            time_s=affine.time_s,
            posture_rad=affine.constant_rad,
        )


@dataclass(frozen=True)
class PreviewPostureFilterState:
    """Immutable filtered posture after consuming the sample at time_s."""

    time_s: float
    roll_rad: float
    pitch_rad: float

    def __post_init__(self) -> None:
        for name in ("time_s", "roll_rad", "pitch_rad"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PreviewPostureFilterTrace:
    """Read-only filtered samples, including the fixed root at row zero."""

    time_s: Any
    posture_rad: Any

    def __post_init__(self) -> None:
        times = _time_vector(self.time_s, includes_root=True)
        object.__setattr__(self, "time_s", times)
        object.__setattr__(
            self,
            "posture_rad",
            _readonly_array("posture_rad", self.posture_rad, (times.size, 2)),
        )

    @property
    def initial_state(self) -> PreviewPostureFilterState:
        return PreviewPostureFilterState(
            time_s=self.time_s[0],
            roll_rad=self.posture_rad[0, 0],
            pitch_rad=self.posture_rad[0, 1],
        )

    @property
    def end_state(self) -> PreviewPostureFilterState:
        return PreviewPostureFilterState(
            time_s=self.time_s[-1],
            roll_rad=self.posture_rad[-1, 0],
            pitch_rad=self.posture_rad[-1, 1],
        )

    @property
    def roll_rad(self) -> np.ndarray:
        return self.posture_rad[:, 0]

    @property
    def pitch_rad(self) -> np.ndarray:
        return self.posture_rad[:, 1]


@dataclass(frozen=True)
class PreviewPostureFilterAffineTrace:
    """Filtered constant + response @ decision, with a decision-free root."""

    time_s: Any
    constant_rad: Any
    response_rad: Any

    def __post_init__(self) -> None:
        times = _time_vector(self.time_s, includes_root=True)
        response = np.asarray(self.response_rad, dtype=float)
        if response.ndim != 3 or response.shape[:2] != (times.size, 2):
            raise ValueError("response_rad must have shape (n + 1, 2, d)")
        response = _readonly_array("response_rad", response, response.shape)
        if np.any(response[0] != 0.0):
            raise ValueError("root response_rad must be zero")
        object.__setattr__(self, "time_s", times)
        object.__setattr__(
            self,
            "constant_rad",
            _readonly_array("constant_rad", self.constant_rad, (times.size, 2)),
        )
        object.__setattr__(self, "response_rad", response)

    @property
    def initial_state(self) -> PreviewPostureFilterState:
        return PreviewPostureFilterState(
            time_s=self.time_s[0],
            roll_rad=self.constant_rad[0, 0],
            pitch_rad=self.constant_rad[0, 1],
        )

    def evaluate(self, decision: Any) -> PreviewPostureFilterTrace:
        """Resolve one decision into a numerical trace and its end_state."""

        vector = _readonly_array("decision", decision, (self.response_rad.shape[2],))
        return PreviewPostureFilterTrace(
            time_s=self.time_s,
            posture_rad=self.constant_rad
            + np.einsum("nad,d->na", self.response_rad, vector),
        )


def filter_affine_task_signal(
    *,
    config: PreviewPostureFilter,
    initial_state: PreviewPostureFilterState,
    sample_times_s: Any,
    constant_rad: Any,
    response_rad: Any,
) -> PreviewPostureFilterAffineTrace:
    """Filter future shape-(n, 2)/(n, 2, d) affine posture predictions.

    Inputs exclude the root and use strictly increasing absolute times after
    initial_state.time_s. The returned arrays include the root, whose constant
    is the inherited filter state and whose decision response is always zero.
    For each sample, z_next = a*z_prev + (1-a)*theta, a = exp(-dt/T).
    """

    if not isinstance(config, PreviewPostureFilter):
        raise TypeError("config must be PreviewPostureFilter")
    if not isinstance(initial_state, PreviewPostureFilterState):
        raise TypeError("initial_state must be PreviewPostureFilterState")
    times = _time_vector(sample_times_s, includes_root=False)
    if times.size and times[0] <= initial_state.time_s:
        raise ValueError("sample times must be strictly after initial_state.time_s")
    constant = _readonly_array("constant_rad", constant_rad, (times.size, 2))
    response = np.asarray(response_rad, dtype=float)
    if response.ndim != 3 or response.shape[:2] != (times.size, 2):
        raise ValueError("response_rad must have shape (n, 2, d)")
    response = _readonly_array("response_rad", response, response.shape)

    filtered_constant = np.empty((times.size + 1, 2))
    filtered_response = np.zeros((times.size + 1, 2, response.shape[2]))
    filtered_constant[0] = (initial_state.roll_rad, initial_state.pitch_rad)
    previous_time = initial_state.time_s
    for index, time in enumerate(times):
        dt = float(time) - previous_time
        decay = float(np.exp(-dt / config.time_constant_s))
        weight = 1.0 - decay
        filtered_constant[index + 1] = (
            decay * filtered_constant[index] + weight * constant[index]
        )
        filtered_response[index + 1] = (
            decay * filtered_response[index] + weight * response[index]
        )
        previous_time = float(time)

    return PreviewPostureFilterAffineTrace(
        time_s=np.concatenate(([initial_state.time_s], times)),
        constant_rad=filtered_constant,
        response_rad=filtered_response,
    )
