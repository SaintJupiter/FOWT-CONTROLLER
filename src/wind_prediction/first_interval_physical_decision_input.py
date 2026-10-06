"""Unranked physical inputs for one first-forecast-interval decision.

The modules below this boundary already establish three factual target
lifecycle operations from one pump state: continue an existing target, release
that target to the current tank state, and optionally track one source-bound
physical forecast endpoint.  This module only carries those operations through
the same first forecast interval and keeps the resulting responses together.

It deliberately does not accept raw :class:`ForecastEvidence`, reliability,
event probabilities, safety limits, scoring weights, or legacy action labels.
Physical wind information is present only through the source-bound forecast
trajectory.  The returned record is therefore a common input to a future
decision rule, not a decision, ranking, or performance claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fowt_platform.ballast_snapshot import BallastRuntimeAssembly
from .forecast_first_interval_lifecycle_rollout import (
    FirstForecastIntervalLifecycleRollout,
    rollout_first_forecast_interval_lifecycle,
)
from .forecast_platform_rhs_diagnostic import (
    CurrentPlatformRhsDiagnostic,
    ForecastPlatformRhsPointDiagnostic,
    diagnose_current_platform_rhs,
)
from .forecast_platform_trajectory import ForecastPlatformTrajectory
from .forecast_rhs_ballast_diagnostic import (
    ForecastRhsBallastDiagnostic,
    diagnose_forecast_horizon_rhs_ballast_redistributions,
    diagnose_forecast_rhs_ballast_redistribution,
)
from .forecast_rhs_ballast_endpoint_preview import (
    ForecastRhsBallastHorizonPreviews,
)
from .physical_option_rhs_comparison import (
    PhysicalTargetLifecycleRhsDiagnostic,
    diagnose_physical_target_lifecycle_rhs,
)
from .physical_target_lifecycle import (
    PhysicalTargetLifecycle,
    PhysicalTargetLifecycleFacts,
    PhysicalTargetLifecycleTrace,
)


@dataclass(frozen=True)
class FirstIntervalPhysicalDecisionInput:
    """Same-origin current-state and first-interval lifecycle facts.

    ``current_rhs`` records the complete frozen state equation at the current
    state, including the source-bound current rotor load and explicit current
    non-rotor loads. Every lifecycle response is a factual replay over the
    trajectory's first forecast interval. The fields keep current and future
    physical facts available at one boundary, without adding them, scoring
    them, or implying that any lifecycle should be selected.
    """

    trajectory: ForecastPlatformTrajectory
    horizon_rhs_ballast_diagnostics: tuple[ForecastRhsBallastDiagnostic, ...]
    horizon_endpoint_previews: ForecastRhsBallastHorizonPreviews
    lifecycle_facts: PhysicalTargetLifecycleFacts
    runtime_assembly: BallastRuntimeAssembly
    current_rhs: CurrentPlatformRhsDiagnostic
    continue_existing: FirstForecastIntervalLifecycleRollout
    release_to_current: FirstForecastIntervalLifecycleRollout
    continue_existing_state_conditioned_rhs: PhysicalTargetLifecycleRhsDiagnostic
    release_to_current_state_conditioned_rhs: PhysicalTargetLifecycleRhsDiagnostic
    new_track: FirstForecastIntervalLifecycleRollout | None = None
    new_track_state_conditioned_rhs: PhysicalTargetLifecycleRhsDiagnostic | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, ForecastPlatformTrajectory):
            raise TypeError("trajectory must be ForecastPlatformTrajectory")
        diagnostics = tuple(self.horizon_rhs_ballast_diagnostics)
        if len(diagnostics) != len(self.trajectory.steps):
            raise ValueError(
                "horizon_rhs_ballast_diagnostics must contain one record per forecast lead"
            )
        for index, diagnostic in enumerate(diagnostics):
            if not isinstance(diagnostic, ForecastRhsBallastDiagnostic):
                raise TypeError(
                    "horizon_rhs_ballast_diagnostics must contain ForecastRhsBallastDiagnostic values"
                )
            if diagnostic.rhs_point.trajectory is not self.trajectory:
                raise ValueError(
                    "horizon_rhs_ballast_diagnostics must use the shared forecast trajectory"
                )
            if diagnostic.lead_index != index:
                raise ValueError(
                    "horizon_rhs_ballast_diagnostics must retain contiguous forecast leads"
                )
            if not np.isclose(
                diagnostic.lead_time_s,
                self.trajectory.steps[index].end_time_s,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    "horizon_rhs_ballast_diagnostics must retain forecast lead times"
                )
        object.__setattr__(self, "horizon_rhs_ballast_diagnostics", diagnostics)
        if not isinstance(
            self.horizon_endpoint_previews,
            ForecastRhsBallastHorizonPreviews,
        ):
            raise TypeError(
                "horizon_endpoint_previews must be ForecastRhsBallastHorizonPreviews"
            )
        horizon_previews = self.horizon_endpoint_previews
        if horizon_previews.trajectory is not self.trajectory:
            raise ValueError(
                "horizon_endpoint_previews must use the shared forecast trajectory"
            )
        if len(horizon_previews.previews) != len(diagnostics):
            raise ValueError(
                "horizon_endpoint_previews must contain one record per forecast lead"
            )
        for index, diagnostic in enumerate(diagnostics):
            if horizon_previews.preview_for_lead(index).rhs_diagnostic is not diagnostic:
                raise ValueError(
                    "horizon_endpoint_previews must retain the supplied rhs diagnostics"
                )
        if not isinstance(self.lifecycle_facts, PhysicalTargetLifecycleFacts):
            raise TypeError("lifecycle_facts must be PhysicalTargetLifecycleFacts")
        if not isinstance(self.runtime_assembly, BallastRuntimeAssembly):
            raise TypeError("runtime_assembly must be BallastRuntimeAssembly")
        if not isinstance(self.current_rhs, CurrentPlatformRhsDiagnostic):
            raise TypeError("current_rhs must be CurrentPlatformRhsDiagnostic")
        if self.current_rhs.trajectory is not self.trajectory:
            raise ValueError("current_rhs must use the shared forecast trajectory")

        self._validate_response(
            response=self.continue_existing,
            trace=self.lifecycle_facts.continue_existing,
            lifecycle=PhysicalTargetLifecycle.CONTINUE_EXISTING,
            field_name="continue_existing",
        )
        shared_rhs_point = self._validate_state_conditioned_rhs(
            diagnostic=self.continue_existing_state_conditioned_rhs,
            response=self.continue_existing,
            trace=self.lifecycle_facts.continue_existing,
            field_name="continue_existing_state_conditioned_rhs",
        )
        if shared_rhs_point is not diagnostics[0].rhs_point:
            raise ValueError(
                "first-interval responses must use the first horizon ballast diagnostic"
            )
        self._validate_response(
            response=self.release_to_current,
            trace=self.lifecycle_facts.release_to_current,
            lifecycle=PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
            field_name="release_to_current",
        )
        self._validate_state_conditioned_rhs(
            diagnostic=self.release_to_current_state_conditioned_rhs,
            response=self.release_to_current,
            trace=self.lifecycle_facts.release_to_current,
            shared_rhs_point=shared_rhs_point,
            field_name="release_to_current_state_conditioned_rhs",
        )

        trace = self.lifecycle_facts.new_track
        if trace is None:
            if (
                self.new_track is not None
                or self.new_track_state_conditioned_rhs is not None
            ):
                raise ValueError(
                    "new_track response and rhs diagnostic require a new_track lifecycle fact"
                )
        else:
            if (
                self.new_track is None
                or self.new_track_state_conditioned_rhs is None
            ):
                raise ValueError(
                    "new_track lifecycle fact requires both response and rhs diagnostic"
                )
            self._validate_response(
                response=self.new_track,
                trace=trace,
                lifecycle=PhysicalTargetLifecycle.NEW_TRACK,
                field_name="new_track",
            )
            self._validate_state_conditioned_rhs(
                diagnostic=self.new_track_state_conditioned_rhs,
                response=self.new_track,
                trace=trace,
                shared_rhs_point=shared_rhs_point,
                field_name="new_track_state_conditioned_rhs",
            )
            if trace.source_preview is not horizon_previews.first_preview:
                raise ValueError(
                    "new_track must retain the first horizon endpoint preview"
                )

    def _validate_response(
        self,
        *,
        response: FirstForecastIntervalLifecycleRollout,
        trace: PhysicalTargetLifecycleTrace,
        lifecycle: PhysicalTargetLifecycle,
        field_name: str,
    ) -> None:
        if not isinstance(response, FirstForecastIntervalLifecycleRollout):
            raise TypeError(f"{field_name} must be FirstForecastIntervalLifecycleRollout")
        if response.trajectory is not self.trajectory:
            raise ValueError(f"{field_name} must use the shared forecast trajectory")
        if response.runtime_assembly is not self.runtime_assembly:
            raise ValueError(f"{field_name} must use the shared runtime assembly")
        if response.lifecycle_trace is not trace:
            raise ValueError(f"{field_name} must retain the matching lifecycle trace")
        if trace.lifecycle is not lifecycle:
            raise ValueError(f"{field_name} must retain the {lifecycle.value} operation")
        if trace.execution_start_time != self.lifecycle_facts.execution_start_time:
            raise ValueError(f"{field_name} must retain the shared execution start time")

    def _validate_state_conditioned_rhs(
        self,
        *,
        diagnostic: PhysicalTargetLifecycleRhsDiagnostic,
        response: FirstForecastIntervalLifecycleRollout,
        trace: PhysicalTargetLifecycleTrace,
        field_name: str,
        shared_rhs_point: ForecastPlatformRhsPointDiagnostic | None = None,
    ) -> ForecastPlatformRhsPointDiagnostic:
        """Bind one same-time RHS fact to its executed lifecycle response."""

        if not isinstance(diagnostic, PhysicalTargetLifecycleRhsDiagnostic):
            raise TypeError(f"{field_name} must be PhysicalTargetLifecycleRhsDiagnostic")
        if diagnostic.rhs_point.trajectory is not self.trajectory:
            raise ValueError(f"{field_name} must use the shared forecast trajectory")
        if shared_rhs_point is not None and diagnostic.rhs_point is not shared_rhs_point:
            raise ValueError(f"{field_name} must use the shared first rhs point")
        if diagnostic.runtime_assembly is not self.runtime_assembly:
            raise ValueError(f"{field_name} must use the shared runtime assembly")
        if diagnostic.lifecycle_trace is not trace:
            raise ValueError(f"{field_name} must retain the matching lifecycle trace")
        if not np.allclose(
            diagnostic.reachable_snapshot.actual_tank_masses_kg,
            response.reached_final_tank_masses_kg,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError(
                f"{field_name} reachable tank masses must match the lifecycle response"
            )
        return diagnostic.rhs_point

    def response_for(
        self,
        lifecycle: PhysicalTargetLifecycle | str,
    ) -> FirstForecastIntervalLifecycleRollout:
        """Return the factual first-interval response for one declared lifecycle.

        This is a lookup only.  It neither evaluates the returned response nor
        decides which lifecycle should be committed in the current period.
        """

        selected = PhysicalTargetLifecycle(lifecycle)
        if selected is PhysicalTargetLifecycle.CONTINUE_EXISTING:
            return self.continue_existing
        if selected is PhysicalTargetLifecycle.RELEASE_TO_CURRENT:
            return self.release_to_current
        if selected is PhysicalTargetLifecycle.NEW_TRACK:
            if self.new_track is None:
                raise ValueError("new_track is unavailable without a physical endpoint")
            return self.new_track
        raise AssertionError(f"unhandled lifecycle {selected!r}")


def assemble_first_interval_physical_decision_input(
    *,
    trajectory: ForecastPlatformTrajectory,
    lifecycle_facts: PhysicalTargetLifecycleFacts,
    runtime_assembly: BallastRuntimeAssembly,
    first_rhs_point: ForecastPlatformRhsPointDiagnostic,
    horizon_rhs_ballast_diagnostics: tuple[ForecastRhsBallastDiagnostic, ...]
    | None = None,
    horizon_endpoint_previews: ForecastRhsBallastHorizonPreviews | None = None,
) -> FirstIntervalPhysicalDecisionInput:
    """Replay the available target operations over one shared first interval.

    The function has no candidate fractions, no policy flags and no numerical
    preference.  It simply makes later decision work start from identical
    forecast timing, external loads, platform state and pump state.
    """

    if not isinstance(trajectory, ForecastPlatformTrajectory):
        raise TypeError("trajectory must be ForecastPlatformTrajectory")
    if not isinstance(lifecycle_facts, PhysicalTargetLifecycleFacts):
        raise TypeError("lifecycle_facts must be PhysicalTargetLifecycleFacts")
    if not isinstance(runtime_assembly, BallastRuntimeAssembly):
        raise TypeError("runtime_assembly must be BallastRuntimeAssembly")
    if not isinstance(first_rhs_point, ForecastPlatformRhsPointDiagnostic):
        raise TypeError("first_rhs_point must be ForecastPlatformRhsPointDiagnostic")
    if first_rhs_point.trajectory is not trajectory:
        raise ValueError("first_rhs_point must use the supplied forecast trajectory")
    if first_rhs_point.lead_index != 0:
        raise ValueError("first_rhs_point must refer to the first forecast lead")
    if horizon_rhs_ballast_diagnostics is None:
        generated_diagnostics = diagnose_forecast_horizon_rhs_ballast_redistributions(
            trajectory=trajectory
        )
        diagnostics = (
            diagnose_forecast_rhs_ballast_redistribution(rhs_point=first_rhs_point),
            *generated_diagnostics[1:],
        )
    else:
        diagnostics = tuple(horizon_rhs_ballast_diagnostics)
    if not diagnostics or first_rhs_point is not diagnostics[0].rhs_point:
        raise ValueError(
            "first_rhs_point must be the first record in horizon_rhs_ballast_diagnostics"
        )
    if horizon_endpoint_previews is None:
        raise ValueError(
            "horizon_endpoint_previews are required for first-interval physical input"
        )

    continuation = rollout_first_forecast_interval_lifecycle(
        trajectory=trajectory,
        lifecycle_trace=lifecycle_facts.continue_existing,
        runtime_assembly=runtime_assembly,
    )
    release = rollout_first_forecast_interval_lifecycle(
        trajectory=trajectory,
        lifecycle_trace=lifecycle_facts.release_to_current,
        runtime_assembly=runtime_assembly,
    )
    new_track = None
    if lifecycle_facts.new_track is not None:
        new_track = rollout_first_forecast_interval_lifecycle(
            trajectory=trajectory,
            lifecycle_trace=lifecycle_facts.new_track,
            runtime_assembly=runtime_assembly,
        )
    continuation_rhs = diagnose_physical_target_lifecycle_rhs(
        rhs_point=first_rhs_point,
        lifecycle_trace=lifecycle_facts.continue_existing,
        runtime_assembly=runtime_assembly,
    )
    release_rhs = diagnose_physical_target_lifecycle_rhs(
        rhs_point=first_rhs_point,
        lifecycle_trace=lifecycle_facts.release_to_current,
        runtime_assembly=runtime_assembly,
    )
    new_track_rhs = None
    if lifecycle_facts.new_track is not None:
        new_track_rhs = diagnose_physical_target_lifecycle_rhs(
            rhs_point=first_rhs_point,
            lifecycle_trace=lifecycle_facts.new_track,
            runtime_assembly=runtime_assembly,
        )
    return FirstIntervalPhysicalDecisionInput(
        trajectory=trajectory,
        horizon_rhs_ballast_diagnostics=diagnostics,
        horizon_endpoint_previews=horizon_endpoint_previews,
        lifecycle_facts=lifecycle_facts,
        runtime_assembly=runtime_assembly,
        current_rhs=diagnose_current_platform_rhs(trajectory=trajectory),
        continue_existing=continuation,
        release_to_current=release,
        continue_existing_state_conditioned_rhs=continuation_rhs,
        release_to_current_state_conditioned_rhs=release_rhs,
        new_track=new_track,
        new_track_state_conditioned_rhs=new_track_rhs,
    )


__all__ = [
    "FirstIntervalPhysicalDecisionInput",
    "assemble_first_interval_physical_decision_input",
]
