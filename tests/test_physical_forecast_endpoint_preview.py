import ast
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from fowt_platform import (
    PlatformMatrices,
    diagnose_generalized_load_forecast_ballast_redistribution,
    sample_ballast_endpoint_path,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
    ExecutionTargetOperation,
)
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.forecast_physical_load import (
    ForecastRotorLoadParameters,
    ForecastRotorOperatingState,
    assemble_forecast_generalized_rotor_loads,
)
from wind_prediction.physical_endpoint_reachability import (
    evaluate_physical_endpoint_reachability,
)
from wind_prediction.physical_forecast_endpoint_preview import (
    preview_physical_endpoint_fractions,
    trace_physical_forecast_endpoint,
)


def _load_assembly():
    return assemble_forecast_generalized_rotor_loads(
        forecast=ForecastEvidence(
            source="forecast_test",
            model_version="model_test",
            origin_time="2026-08-22T00:00:00",
            sample_period_s=600.0,
            uv_ms=np.array([[0.0, -10.0], [0.0, -12.0]]),
            lead_reliability=np.array([0.9, 0.8]),
            event_probs={},
        ),
        current_enu_downwind_air_velocity_mps=[0.0, -8.0],
        current_wind_source="current_observation_test",
        current_wind_observation_time="2026-08-22T00:00:00",
        parameters=ForecastRotorLoadParameters(
            air_density_kg_m3=1.225,
            rotor_radius_m=100.0,
            wind_reference_height_m=100.0,
            rotor_plane_height_m=100.0,
            power_law_shear_exponent=0.0,
            frozen_equilibrium_heading_rad=np.pi,
            rotor_force_application_point_from_reference_m=[0.0, 0.0, 120.0],
            downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
        ),
        operating_state=ForecastRotorOperatingState(
            current_thrust_coefficient=0.8,
            future_thrust_coefficients=[0.8, 0.8],
            current_platform_reference_velocity_platform_mps=[0.0, 0.0, 0.0],
            current_platform_angular_velocity_platform_radps=[0.0, 0.0, 0.0],
            future_platform_reference_velocities_platform_mps=np.zeros((2, 3)),
            future_platform_angular_velocities_platform_radps=np.zeros((2, 3)),
            future_platform_kinematics_mode="frozen_zero",
            future_platform_kinematics_source="endpoint_preview_frozen_platform",
        ),
    )


def _matrices() -> PlatformMatrices:
    restoring = np.eye(6)
    restoring[4, 4] = 1.0e10
    restoring[3, 3] = 1.0e10
    return PlatformMatrices(
        mass=np.eye(6),
        damping=np.zeros((6, 6)),
        hydrostatic_stiffness=restoring,
        mooring_stiffness=np.zeros((6, 6)),
    )


def _execution_config() -> ExecutionRolloutConfig:
    return ExecutionRolloutConfig(
        block_duration_s=30.0,
        water_density_kg_m3=1_000.0,
        max_pump_rate_m3_min=1.0,
        tank_capacity_kg=1.0e9,
        internal_step_s=1.0,
        stop_error_kg=0.0,
        restart_error_kg=0.0,
        min_on_s=0.0,
        min_off_s=0.0,
        near_target_hold_s=0.0,
        ramp_up_m3_min_per_s=np.inf,
        ramp_down_m3_min_per_s=np.inf,
        pump_rate_schedule_m3_min=((0.0, 1.0), (1.0e9, 1.0)),
    )


def _prepared_preview(lead_index: int = 0):
    assembly = _load_assembly()
    actual = np.full(3, 5.0e8)
    capacities = np.full(3, 1.0e9)
    profile = diagnose_generalized_load_forecast_ballast_redistribution(
        forecast=assembly.load_forecast,
        matrices=_matrices(),
        actual_tank_masses_kg=actual,
        tank_capacities_kg=capacities,
        tank_coordinates_m=[[-20.0, 0.0, -1.0], [10.0, 17.0, -1.0], [10.0, -17.0, -1.0]],
    )
    timed = profile[lead_index]
    sample = sample_ballast_endpoint_path(
        actual_tank_masses_kg=actual,
        diagnostic_target_tank_masses_kg=timed.diagnostic.allocation.target_tank_masses_kg,
        tank_capacities_kg=capacities,
        fractions=(0.5,),
    )[0]
    state = ExecutionRolloutState(
        masses_kg=actual,
        target_masses_kg=actual,
        primary_target_kg=actual,
        pump_rates_m3_min=[0.0, 0.0, 0.0],
        pump_latched=[False, False, False],
        pump_off_elapsed_s=[100.0, 100.0, 100.0],
    )
    reachability = evaluate_physical_endpoint_reachability(
        lead_time_s=timed.lead_time_s,
        endpoint_sample=sample,
        execution_state=state,
        execution_config=_execution_config(),
        execution_duration_s=30.0,
    )
    return assembly, profile, timed, sample, reachability


class PhysicalForecastEndpointPreviewTests(unittest.TestCase):
    def test_records_one_source_bound_physical_forecast_endpoint_preview(self):
        assembly, _, timed, sample, reachability = _prepared_preview()

        preview = trace_physical_forecast_endpoint(
            load_assembly=assembly,
            lead_index=0,
            timed_diagnostic=timed,
            endpoint_sample=sample,
            reachability=reachability,
            execution_start_time="2026-08-22T00:00:00",
        )

        self.assertEqual(preview.lead_time_s, 600.0)
        self.assertEqual(preview.execution_start_time, "2026-08-22T00:00:00")
        self.assertEqual(preview.input_identity.forecast_source, "forecast_test")
        self.assertEqual(
            preview.input_identity.current_wind_source,
            "current_observation_test",
        )
        np.testing.assert_allclose(
            preview.reachability.execution_step.requested_target_kg,
            sample.hypothetical_tank_masses_kg,
        )
        np.testing.assert_allclose(
            preview.timed_diagnostic.diagnostic.restoring_diagnostic.relative_pitch_roll_load_nm,
            (
                assembly.load_forecast.future_load_at(0)
                - assembly.load_forecast.current_generalized_load
            )[[4, 3]],
        )

    def test_rejects_mismatched_lead_or_endpoint(self):
        assembly, profile, timed, sample, reachability = _prepared_preview()
        with self.assertRaisesRegex(ValueError, "timed_diagnostic lead"):
            trace_physical_forecast_endpoint(
                load_assembly=assembly,
                lead_index=0,
                timed_diagnostic=profile[1],
                endpoint_sample=sample,
                reachability=reachability,
                execution_start_time="2026-08-22T00:00:00",
            )

    def test_rejects_execution_that_does_not_track_the_endpoint(self):
        assembly, _, timed, sample, reachability = _prepared_preview()
        wrong_target_step = replace(
            reachability.execution_step,
            requested_target_kg=sample.hypothetical_tank_masses_kg + 1.0,
        )
        with self.assertRaisesRegex(ValueError, "endpoint sample"):
            trace_physical_forecast_endpoint(
                load_assembly=assembly,
                lead_index=0,
                timed_diagnostic=timed,
                endpoint_sample=sample,
                reachability=replace(reachability, execution_step=wrong_target_step),
                execution_start_time="2026-08-22T00:00:00",
            )

        release_step = replace(
            reachability.execution_step,
            target_operation=ExecutionTargetOperation.RELEASE_TO_CURRENT,
        )
        with self.assertRaisesRegex(ValueError, "endpoint sample"):
            trace_physical_forecast_endpoint(
                load_assembly=assembly,
                lead_index=0,
                timed_diagnostic=timed,
                endpoint_sample=sample,
                reachability=replace(reachability, execution_step=release_step),
                execution_start_time="2026-08-22T00:00:00",
            )

    def test_rejects_timed_diagnostic_from_a_different_actual_tank_state(self):
        assembly, _, timed, sample, reachability = _prepared_preview()
        with self.assertRaisesRegex(ValueError, "diagnostic allocation"):
            replace(
                timed,
                actual_tank_masses_kg=(
                    timed.actual_tank_masses_kg + [1.0, 0.0, 0.0]
                ),
            )

        other_sample = sample_ballast_endpoint_path(
            actual_tank_masses_kg=np.full(3, 5.0e8),
            diagnostic_target_tank_masses_kg=timed.diagnostic.allocation.target_tank_masses_kg,
            tank_capacities_kg=np.full(3, 1.0e9),
            fractions=(1.0,),
        )[0]
        with self.assertRaisesRegex(ValueError, "preview the supplied endpoint_sample"):
            trace_physical_forecast_endpoint(
                load_assembly=assembly,
                lead_index=0,
                timed_diagnostic=timed,
                endpoint_sample=other_sample,
                reachability=reachability,
                execution_start_time="2026-08-22T00:00:00",
            )

    def test_rejects_a_preview_with_a_different_execution_start_time(self):
        assembly, _, timed, sample, reachability = _prepared_preview()
        with self.assertRaisesRegex(ValueError, "must match the forecast origin time"):
            trace_physical_forecast_endpoint(
                load_assembly=assembly,
                lead_index=0,
                timed_diagnostic=timed,
                endpoint_sample=sample,
                reachability=reachability,
                execution_start_time="2026-08-22T00:10:00",
            )

    def test_previews_explicit_new_target_fractions_without_selecting_one(self):
        assembly, _, timed, _, _ = _prepared_preview()
        actual = np.full(3, 5.0e8)
        state = ExecutionRolloutState(
            masses_kg=actual,
            target_masses_kg=actual,
            primary_target_kg=actual,
            pump_rates_m3_min=[0.0, 0.0, 0.0],
            pump_latched=[False, False, False],
            pump_off_elapsed_s=[100.0, 100.0, 100.0],
        )
        previews = preview_physical_endpoint_fractions(
            load_assembly=assembly,
            lead_index=0,
            timed_diagnostic=timed,
            actual_tank_masses_kg=actual,
            tank_capacities_kg=np.full(3, 1.0e9),
            fractions=(0.0, 0.5, 1.0),
            execution_state=state,
            execution_config=_execution_config(),
            execution_duration_s=30.0,
            execution_start_time="2026-08-22T00:00:00",
        )

        self.assertEqual([preview.endpoint_sample.fraction for preview in previews], [0.0, 0.5, 1.0])
        self.assertTrue(all(preview.lead_time_s == 600.0 for preview in previews))
        np.testing.assert_allclose(
            previews[0].endpoint_sample.hypothetical_tank_masses_kg,
            actual,
        )
        self.assertGreater(
            np.linalg.norm(previews[-1].endpoint_sample.mass_delta_from_actual_kg),
            0.0,
        )

    def test_fraction_previews_reject_a_different_execution_start_time(self):
        assembly, _, timed, _, _ = _prepared_preview()
        actual = np.full(3, 5.0e8)
        state = ExecutionRolloutState(
            masses_kg=actual,
            target_masses_kg=actual,
            primary_target_kg=actual,
            pump_rates_m3_min=[0.0, 0.0, 0.0],
            pump_latched=[False, False, False],
            pump_off_elapsed_s=[100.0, 100.0, 100.0],
        )

        with self.assertRaisesRegex(ValueError, "must match the forecast origin time"):
            preview_physical_endpoint_fractions(
                load_assembly=assembly,
                lead_index=0,
                timed_diagnostic=timed,
                actual_tank_masses_kg=actual,
                tank_capacities_kg=np.full(3, 1.0e9),
                fractions=(0.5,),
                execution_state=state,
                execution_config=_execution_config(),
                execution_duration_s=30.0,
                execution_start_time="2026-08-22T00:10:00",
            )

    def test_fraction_previews_reject_incompatible_actuator_start_or_capacity(self):
        assembly, _, timed, _, _ = _prepared_preview()
        actual = np.full(3, 5.0e8)
        state = ExecutionRolloutState(
            masses_kg=actual,
            target_masses_kg=actual,
            primary_target_kg=actual,
            pump_rates_m3_min=[0.0, 0.0, 0.0],
            pump_latched=[False, False, False],
            pump_off_elapsed_s=[100.0, 100.0, 100.0],
        )
        with self.assertRaisesRegex(ValueError, "physical diagnostic capacities"):
            preview_physical_endpoint_fractions(
                load_assembly=assembly,
                lead_index=0,
                timed_diagnostic=timed,
                actual_tank_masses_kg=actual,
                tank_capacities_kg=[1.0e9, 9.0e8, 1.0e9],
                fractions=(0.5,),
                execution_state=state,
                execution_config=_execution_config(),
                execution_duration_s=30.0,
                execution_start_time="2026-08-22T00:00:00",
            )

    def test_module_does_not_import_controller_or_candidate_ranking(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "physical_forecast_endpoint_preview.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        forbidden_fragments = ("controller", "candidate", "planner", "action_target")
        self.assertFalse(
            any(
                any(fragment in module for fragment in forbidden_fragments)
                for module in imported_modules
            )
        )


if __name__ == "__main__":
    unittest.main()
