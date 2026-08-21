"""Compare explicit candidate ballast states without entering controller ranking.

This narrow diagnostic evaluates the same frozen environmental loads from one
platform state twice: once with the measured tank masses and once with a
hypothetical candidate tank state.  It exposes the resulting matrix, ballast
load and short-horizon state differences, but does not select a candidate,
model pump execution, or alter controller demand formation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ballast_snapshot import (
    BallastModelSnapshot,
    BallastRuntimeAssembly,
    assemble_ballast_model_snapshot,
)
from .incremental import IncrementalState
from .open_loop_step import OpenLoopPlatformStep, advance_frozen_open_loop_step
from .rotor_input import RotorGeneralizedLoad


@dataclass(frozen=True)
class CandidateResponseDiagnostic:
    """Two frozen responses from the same state and explicit environmental load.

    ``actual_response`` uses the measured tank masses. ``candidate_response``
    uses a hypothetical already-achieved candidate tank state.  The comparison
    is therefore a plant-response diagnostic, not a claim that a pump can
    achieve the candidate within the supplied duration.
    """

    actual_snapshot: BallastModelSnapshot
    candidate_snapshot: BallastModelSnapshot
    actual_response: OpenLoopPlatformStep
    candidate_response: OpenLoopPlatformStep

    @property
    def candidate_minus_actual_position(self) -> np.ndarray:
        """Return the candidate-response displacement difference in six DOF."""

        result = (
            self.candidate_response.next_state.position
            - self.actual_response.next_state.position
        )
        result.setflags(write=False)
        return result

    @property
    def candidate_minus_actual_velocity(self) -> np.ndarray:
        """Return the candidate-response velocity difference in six DOF."""

        result = (
            self.candidate_response.next_state.velocity
            - self.actual_response.next_state.velocity
        )
        result.setflags(write=False)
        return result


def compare_hypothetical_candidate_response(
    *,
    runtime_assembly: BallastRuntimeAssembly,
    platform_state: IncrementalState,
    actual_tank_masses_kg: Any,
    candidate_tank_masses_kg: Any,
    reference_tank_masses_kg: Any,
    tank_capacities_kg: Any,
    tank_coordinates_m: Any,
    rotor_load: RotorGeneralizedLoad,
    wave_load: Any,
    other_load: Any,
    duration_s: float,
) -> CandidateResponseDiagnostic:
    """Compare actual and hypothetical candidate states over one frozen step.

    The explicit ``rotor_load`` may represent a forecast operating point, but
    this function deliberately does not create that load from a forecast or
    decide whether the candidate should be selected.  Both branches receive
    exactly the same wind, wave and other load inputs.
    """

    if not isinstance(runtime_assembly, BallastRuntimeAssembly):
        raise TypeError("runtime_assembly must be BallastRuntimeAssembly")
    if not isinstance(platform_state, IncrementalState):
        raise TypeError("platform_state must be an IncrementalState")
    if not isinstance(rotor_load, RotorGeneralizedLoad):
        raise TypeError("rotor_load must be a RotorGeneralizedLoad")

    actual_snapshot = assemble_ballast_model_snapshot(
        runtime_assembly=runtime_assembly,
        actual_tank_masses_kg=actual_tank_masses_kg,
        reference_tank_masses_kg=reference_tank_masses_kg,
        tank_capacities_kg=tank_capacities_kg,
        tank_coordinates_m=tank_coordinates_m,
    )
    candidate_snapshot = assemble_ballast_model_snapshot(
        runtime_assembly=runtime_assembly,
        actual_tank_masses_kg=candidate_tank_masses_kg,
        reference_tank_masses_kg=reference_tank_masses_kg,
        tank_capacities_kg=tank_capacities_kg,
        tank_coordinates_m=tank_coordinates_m,
    )
    actual_response = advance_frozen_open_loop_step(
        snapshot=actual_snapshot,
        state=platform_state,
        rotor_load=rotor_load,
        wave_load=wave_load,
        other_load=other_load,
        duration_s=duration_s,
    )
    candidate_response = advance_frozen_open_loop_step(
        snapshot=candidate_snapshot,
        state=platform_state,
        rotor_load=rotor_load,
        wave_load=wave_load,
        other_load=other_load,
        duration_s=duration_s,
    )
    return CandidateResponseDiagnostic(
        actual_snapshot=actual_snapshot,
        candidate_snapshot=candidate_snapshot,
        actual_response=actual_response,
        candidate_response=candidate_response,
    )


__all__ = [
    "CandidateResponseDiagnostic",
    "compare_hypothetical_candidate_response",
]
