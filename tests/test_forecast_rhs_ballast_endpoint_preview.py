import ast
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from tests.test_forecast_platform_rhs_diagnostic import _trajectory
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
)
from wind_prediction.forecast_platform_rhs_diagnostic import (
    diagnose_forecast_platform_rhs,
)
from wind_prediction.forecast_rhs_ballast_diagnostic import (
    diagnose_forecast_horizon_rhs_ballast_redistributions,
    diagnose_forecast_rhs_ballast_redistribution,
)
from wind_prediction.forecast_rhs_ballast_endpoint_preview import (
    preview_forecast_rhs_ballast_horizon,
    preview_forecast_rhs_ballast_endpoint,
)
from wind_prediction.physical_endpoint_preview_binding import (
    validate_endpoint_preview_at_trajectory_lead,
)
from wind_prediction.physical_target_lifecycle import (
    PhysicalTargetLifecycle,
    trace_physical_target_lifecycles,
)


def _execution_config() -> ExecutionRolloutConfig:
    return ExecutionRolloutConfig(
        block_duration_s=5.0,
        water_density_kg_m3=1_000.0,
        max_pump_rate_m3_min=1.0,
        tank_capacity_kg=200.0,
        internal_step_s=1.0,
        stop_error_kg=0.0,
        restart_error_kg=0.0,
        min_on_s=0.0,
        min_off_s=0.0,
        near_target_hold_s=0.0,
        ramp_up_m3_min_per_s=np.inf,
        ramp_down_m3_min_per_s=np.inf,
        pump_rate_schedule_m3_min=((0.0, 1.0),),
    )


def _execution_state(trajectory) -> ExecutionRolloutState:
    masses = trajectory.platform_snapshot.actual_tank_masses_kg
    return ExecutionRolloutState(
        masses_kg=masses,
        target_masses_kg=masses,
        primary_target_kg=masses,
        pump_rates_m3_min=[0.0, 0.0, 0.0],
        pump_latched=[False, False, False],
        pump_off_elapsed_s=[100.0, 100.0, 100.0],
    )


def _preview():
    trajectory = _trajectory()
    rhs_point = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]
    diagnostic = diagnose_forecast_rhs_ballast_redistribution(rhs_point=rhs_point)
    state = _execution_state(trajectory)
    config = _execution_config()
    preview = preview_forecast_rhs_ballast_endpoint(
        rhs_diagnostic=diagnostic,
        execution_state=state,
        execution_config=config,
        execution_duration_s=rhs_point.lead_time_s,
        execution_start_time=trajectory.initial_state_time,
    )
    return trajectory, rhs_point, diagnostic, state, config, preview


def _horizon_previews():
    trajectory = _trajectory()
    diagnostics = diagnose_forecast_horizon_rhs_ballast_redistributions(
        trajectory=trajectory
    )
    state = _execution_state(trajectory)
    config = _execution_config()
    previews = preview_forecast_rhs_ballast_horizon(
        rhs_diagnostics=diagnostics,
        execution_state=state,
        execution_config=config,
        execution_start_time=trajectory.initial_state_time,
    )
    return trajectory, diagnostics, state, config, previews


class ForecastRhsBallastEndpointPreviewTests(unittest.TestCase):
    def test_derives_one_full_state_conditioned_endpoint_and_pump_preview(self):
        trajectory, rhs_point, diagnostic, state, config, preview = _preview()

        self.assertIs(preview.rhs_diagnostic, diagnostic)
        self.assertEqual(preview.lead_index, 0)
        self.assertEqual(preview.lead_time_s, rhs_point.lead_time_s)
        self.assertEqual(preview.execution_start_time, trajectory.initial_state_time)
        self.assertEqual(preview.endpoint_sample.fraction, 1.0)
        np.testing.assert_allclose(
            preview.target_tank_masses_kg,
            diagnostic.hypothetical_tank_masses_kg,
        )
        np.testing.assert_allclose(
            preview.reachability.execution_start_state.actual_masses_kg,
            state.actual_masses_kg,
        )
        self.assertEqual(preview.reachability.execution_config, config)
        np.testing.assert_allclose(
            preview.reachability.execution_step.requested_target_kg,
            diagnostic.hypothetical_tank_masses_kg,
        )

    def test_lifecycle_reuses_the_state_conditioned_pump_result(self):
        trajectory, _, diagnostic, state, config, preview = _preview()

        facts = trace_physical_target_lifecycles(
            execution_state=state,
            execution_config=config,
            execution_duration_s=preview.lead_time_s,
            execution_start_time=trajectory.initial_state_time,
            new_track_preview=preview,
        )

        self.assertIsNotNone(facts.new_track)
        self.assertIs(facts.new_track.source_preview, preview)
        self.assertIs(facts.new_track.execution_step, preview.reachability.execution_step)
        self.assertIs(facts.new_track.lifecycle, PhysicalTargetLifecycle.NEW_TRACK)
        np.testing.assert_allclose(
            facts.new_track.requested_target_masses_kg,
            diagnostic.hypothetical_tank_masses_kg,
        )

    def test_horizon_previews_keep_independent_endpoints_at_one_current_state(self):
        trajectory, diagnostics, state, config, horizon = _horizon_previews()

        self.assertIs(horizon.trajectory, trajectory)
        self.assertEqual(len(horizon.previews), len(trajectory.steps))
        self.assertEqual(
            horizon.execution_preview_duration_s,
            trajectory.steps[0].end_time_s,
        )
        self.assertIs(horizon.first_preview, horizon.preview_for_lead(0))
        for index, preview in enumerate(horizon.previews):
            self.assertIs(preview.rhs_diagnostic, diagnostics[index])
            self.assertEqual(preview.lead_index, index)
            self.assertEqual(preview.lead_time_s, trajectory.steps[index].end_time_s)
            self.assertEqual(
                preview.reachability.execution_duration_s,
                trajectory.steps[0].end_time_s,
            )
            self.assertIs(preview.reachability.execution_start_state, state)
            self.assertEqual(preview.reachability.execution_config, config)

    def test_rejects_wrong_origin_start_state_capacity_or_fraction(self):
        trajectory, _, diagnostic, state, config, preview = _preview()

        with self.assertRaisesRegex(ValueError, "rhs trajectory origin"):
            preview_forecast_rhs_ballast_endpoint(
                rhs_diagnostic=diagnostic,
                execution_state=state,
                execution_config=config,
                execution_duration_s=5.0,
                execution_start_time="2026-08-22T00:10:00",
            )
        wrong_state = ExecutionRolloutState(
            masses_kg=state.actual_masses_kg + [1.0, 0.0, 0.0],
            target_masses_kg=state.actual_masses_kg,
            primary_target_kg=state.actual_masses_kg,
            pump_rates_m3_min=[0.0, 0.0, 0.0],
            pump_latched=[False, False, False],
            pump_off_elapsed_s=[100.0, 100.0, 100.0],
        )
        with self.assertRaisesRegex(ValueError, "formed from execution_state"):
            preview_forecast_rhs_ballast_endpoint(
                rhs_diagnostic=diagnostic,
                execution_state=wrong_state,
                execution_config=config,
                execution_duration_s=5.0,
                execution_start_time=trajectory.initial_state_time,
            )
        with self.assertRaisesRegex(ValueError, "capacity must match"):
            preview_forecast_rhs_ballast_endpoint(
                rhs_diagnostic=diagnostic,
                execution_state=state,
                execution_config=replace(config, tank_capacity_kg=201.0),
                execution_duration_s=5.0,
                execution_start_time=trajectory.initial_state_time,
            )
        with self.assertRaisesRegex(ValueError, "full diagnostic endpoint"):
            replace(
                preview,
                endpoint_sample=replace(preview.endpoint_sample, fraction=0.5),
            )
        with self.assertRaisesRegex(ValueError, "full rhs diagnostic endpoint"):
            replace(
                preview,
                reachability=replace(
                    preview.reachability,
                    endpoint_sample=replace(
                        preview.reachability.endpoint_sample,
                        fraction=0.5,
                    ),
                ),
            )

    def test_rejects_a_preview_rebound_to_a_different_trajectory(self):
        trajectory, rhs_point, _, _, _, preview = _preview()
        other_trajectory = _trajectory()
        self.assertIsNot(other_trajectory, trajectory)

        with self.assertRaisesRegex(ValueError, "matching rhs trajectory"):
            validate_endpoint_preview_at_trajectory_lead(
                preview=preview,
                trajectory=other_trajectory,
                lead_index=rhs_point.lead_index,
                lead_time_s=rhs_point.lead_time_s,
                first_lead_only=True,
            )

    def test_module_excludes_event_scoring_and_legacy_policy_dependencies(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "forecast_rhs_ballast_endpoint_preview.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        forbidden_fragments = (
            "forecast_evidence",
            "controller",
            "candidate",
            "planner",
            "action_target",
            "policy",
            "event",
            "reliability",
        )
        self.assertFalse(
            any(
                any(fragment in module for fragment in forbidden_fragments)
                for module in imported_modules
            )
        )


if __name__ == "__main__":
    unittest.main()
