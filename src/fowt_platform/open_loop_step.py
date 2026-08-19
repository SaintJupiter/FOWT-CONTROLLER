"""One frozen open-loop platform step assembled from explicit load sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ballast_snapshot import BallastModelSnapshot
from .incremental import (
    IncrementalLoads,
    IncrementalPlatformModel,
    IncrementalState,
)
from .rotor_input import RotorGeneralizedLoad


def _six_vector(name: str, value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (6,):
        raise ValueError(f"{name} must have shape (6,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class OpenLoopPlatformStep:
    """Loads and state at the end of one frozen open-loop platform step."""

    loads: IncrementalLoads
    next_state: IncrementalState

    def __post_init__(self) -> None:
        if not isinstance(self.loads, IncrementalLoads):
            raise TypeError("loads must be an IncrementalLoads")
        if not isinstance(self.next_state, IncrementalState):
            raise TypeError("next_state must be an IncrementalState")


def advance_frozen_open_loop_step(
    *,
    snapshot: BallastModelSnapshot,
    state: IncrementalState,
    rotor_load: RotorGeneralizedLoad,
    wave_load: Any,
    other_load: Any,
    duration_s: float,
) -> OpenLoopPlatformStep:
    """Advance one step using a same-time tank snapshot and rotor load.

    ``snapshot``, ``state`` and ``rotor_load`` all represent the beginning of
    the step.  The snapshot supplies both the frozen matrices and the actual
    tank-mass increment load.  The rotor input supplies the wind channel.
    Wave and other loads remain explicit so that no environmental model is
    selected implicitly.  This function neither changes tank masses nor
    recalculates the rotor load after the state advances.
    """

    if not isinstance(snapshot, BallastModelSnapshot):
        raise TypeError("snapshot must be a BallastModelSnapshot")
    if not isinstance(state, IncrementalState):
        raise TypeError("state must be an IncrementalState")
    if not isinstance(rotor_load, RotorGeneralizedLoad):
        raise TypeError("rotor_load must be a RotorGeneralizedLoad")
    loads = IncrementalLoads(
        wind=rotor_load.generalized_load_platform,
        wave=_six_vector("wave_load", wave_load),
        ballast=snapshot.incremental_ballast_load,
        other=_six_vector("other_load", other_load),
    )
    next_state = IncrementalPlatformModel(snapshot.matrices).advance_frozen_step(
        state,
        loads,
        duration_s,
    )
    return OpenLoopPlatformStep(loads=loads, next_state=next_state)
