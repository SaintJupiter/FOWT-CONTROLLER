import ast
from pathlib import Path
import unittest

import numpy as np

from fowt_platform import (
    BallastModelSnapshot,
    PlatformMatrices,
    diagnose_generalized_load_forecast_ballast_redistribution,
    sample_ballast_endpoint_path,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
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
from wind_prediction.physical_execution_compensation import (
    diagnose_actual_execution_compensation,
)
from wind_prediction.physical_forecast_endpoint_preview import (
    trace_physical_forecast_endpoint,
)


_ACTUAL = np.full(3, 5.0e8)
_CAPACITIES = np.full(3, 1.0e9)
_COORDINATES = np.array(
    [
        [-20.0, 0.0, -1.0],
        [10.0, 17.0, -1.0],
        [10.0, -17.0, -1.0],
    ]
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


def _snapshot(
    actual=_ACTUAL,
    *,
    capacities=_CAPACITIES,
    coordinates=_COORDINATES,
    gravity_m_s2=9.81,
) -> BallastModelSnapshot:
    actual = np.asarray(actual, dtype=float)
    return BallastModelSnapshot(
        actual_tank_masses_kg=actual,
        reference_tank_masses_kg=actual,
        tank_mass_deltas_kg=np.zeros(3),
        tank_capacities_kg=capacities,
        tank_coordinates_m=coordinates,
        gravity_m_s2=gravity_m_s2,
        matrices=_matrices(),
        incremental_ballast_load=np.zeros(6),
        runtime_provenance="physical_execution_compensation_test",
    )


def _load_assembly():
    return assemble_forecast_generalized_rotor_loads(
        forecast=ForecastEvidence(
            source="compensation_test_forecast",
            model_version="compensation_test_model",
            origin_time="2026-08-22T00:00:00",
            sample_period_s=600.0,
            uv_ms=np.array([[0.0, -10.0], [0.0, -12.0]]),
            lead_reliability=np.array([0.9, 0.8]),
            event_probs={},
        ),
        current_enu_downwind_air_velocity_mps=[0.0, -8.0],
        current_wind_source="compensation_test_current_observation",
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
            future_platform_kinematics_source="compensation_frozen_platform",
        ),
    )


def _execution_config(*, max_pump_rate_m3_min: float) -> ExecutionRolloutConfig:
    return ExecutionRolloutConfig(
        block_duration_s=30.0,
        water_density_kg_m3=1_000.0,
        max_pump_rate_m3_min=max_pump_rate_m3_min,
        tank_capacity_kg=1.0e9,
        internal_step_s=1.0,
        stop_error_kg=0.0,
        restart_error_kg=0.0,
        min_on_s=0.0,
        min_off_s=0.0,
        near_target_hold_s=0.0,
        ramp_up_m3_min_per_s=np.inf,
        ramp_down_m3_min_per_s=np.inf,
        pump_rate_schedule_m3_min=(
            (0.0, max_pump_rate_m3_min),
            (1.0e12, max_pump_rate_m3_min),
        ),
    )


def _preview(*, max_pump_rate_m3_min: float):
    assembly = _load_assembly()
    profile = diagnose_generalized_load_forecast_ballast_redistribution(
        forecast=assembly.load_forecast,
        matrices=_matrices(),
        actual_tank_masses_kg=_ACTUAL,
        tank_capacities_kg=_CAPACITIES,
        tank_coordinates_m=_COORDINATES,
    )
    timed = profile[0]
    endpoint = sample_ballast_endpoint_path(
        actual_tank_masses_kg=_ACTUAL,
        diagnostic_target_tank_masses_kg=timed.diagnostic.allocation.target_tank_masses_kg,
        tank_capacities_kg=_CAPACITIES,
        fractions=(0.5,),
    )[0]
    state = ExecutionRolloutState(
        masses_kg=_ACTUAL,
        target_masses_kg=_ACTUAL,
        primary_target_kg=_ACTUAL,
        pump_rates_m3_min=[0.0, 0.0, 0.0],
        pump_latched=[False, False, False],
        pump_off_elapsed_s=[100.0, 100.0, 100.0],
    )
    reachability = evaluate_physical_endpoint_reachability(
        lead_time_s=timed.lead_time_s,
        endpoint_sample=endpoint,
        execution_state=state,
        execution_config=_execution_config(
            max_pump_rate_m3_min=max_pump_rate_m3_min
        ),
        execution_duration_s=30.0,
    )
    return trace_physical_forecast_endpoint(
        load_assembly=assembly,
        lead_index=0,
        timed_diagnostic=timed,
        endpoint_sample=endpoint,
        reachability=reachability,
        execution_start_time=assembly.input_identity.forecast_origin_time,
    )


class PhysicalExecutionCompensationTests(unittest.TestCase):
    def test_uses_the_actual_reached_endpoint_when_pump_capacity_is_sufficient(self):
        preview = _preview(max_pump_rate_m3_min=1.0e8)
        self.assertTrue(preview.reachability.endpoint_reached)

        diagnostic = diagnose_actual_execution_compensation(
            preview=preview,
            platform_snapshot=_snapshot(),
        )

        np.testing.assert_allclose(
            diagnostic.execution_final_tank_masses_kg,
            preview.endpoint_sample.hypothetical_tank_masses_kg,
        )
        np.testing.assert_allclose(
            diagnostic.executed_ballast_pitch_roll_moment_nm,
            diagnostic.endpoint_ballast_pitch_roll_moment_nm,
        )
        np.testing.assert_allclose(
            diagnostic.remaining_endpoint_ballast_pitch_roll_moment_nm,
            [0.0, 0.0],
        )
        np.testing.assert_allclose(
            diagnostic.endpoint_ballast_pitch_roll_moment_nm,
            preview.endpoint_sample.fraction
            * preview.timed_diagnostic.diagnostic.allocation.achieved_pitch_roll_moment_nm,
            atol=1.0e-5,
        )
        self.assertAlmostEqual(diagnostic.executed_net_mass_delta_kg, 0.0)

    def test_reports_actual_partial_compensation_when_pump_is_rate_limited(self):
        preview = _preview(max_pump_rate_m3_min=1.0)
        self.assertFalse(preview.reachability.endpoint_reached)

        diagnostic = diagnose_actual_execution_compensation(
            preview=preview,
            platform_snapshot=_snapshot(),
        )

        self.assertGreater(
            np.linalg.norm(diagnostic.remaining_endpoint_ballast_pitch_roll_moment_nm),
            0.0,
        )
        self.assertFalse(
            np.allclose(
                diagnostic.executed_ballast_pitch_roll_moment_nm,
                diagnostic.endpoint_ballast_pitch_roll_moment_nm,
            )
        )
        np.testing.assert_allclose(
            diagnostic.relative_load_residual_after_execution_nm,
            diagnostic.relative_external_pitch_roll_load_nm
            + diagnostic.executed_ballast_pitch_roll_moment_nm,
        )

    def test_keeps_nonzero_transient_total_mass_change_explicit(self):
        preview = _preview(max_pump_rate_m3_min=1.0)
        diagnostic = diagnose_actual_execution_compensation(
            preview=preview,
            platform_snapshot=_snapshot(),
        )

        actual_delta = (
            diagnostic.execution_final_tank_masses_kg
            - diagnostic.execution_start_tank_masses_kg
        )
        self.assertNotAlmostEqual(float(np.sum(actual_delta)), 0.0)
        self.assertAlmostEqual(
            diagnostic.executed_net_mass_delta_kg,
            float(np.sum(actual_delta)),
        )
        self.assertAlmostEqual(
            diagnostic.executed_net_mass_delta_kg,
            preview.reachability.executed_net_mass_delta_kg,
        )

    def test_rejects_snapshot_from_a_different_actual_tank_state(self):
        preview = _preview(max_pump_rate_m3_min=1.0)
        mismatched_snapshot = _snapshot(_ACTUAL + [1.0, 0.0, 0.0])

        with self.assertRaisesRegex(ValueError, "snapshot actual tank masses"):
            diagnose_actual_execution_compensation(
                preview=preview,
                platform_snapshot=mismatched_snapshot,
            )

    def test_rejects_snapshot_with_different_geometry_gravity_or_capacities(self):
        preview = _preview(max_pump_rate_m3_min=1.0)

        mismatched_contexts = (
            (
                _snapshot(coordinates=_COORDINATES + [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                "tank coordinates",
            ),
            (_snapshot(gravity_m_s2=9.8), "gravity"),
            (_snapshot(capacities=_CAPACITIES + [0.0, 1.0, 0.0]), "tank capacities"),
        )
        for snapshot, message in mismatched_contexts:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    diagnose_actual_execution_compensation(
                        preview=preview,
                        platform_snapshot=snapshot,
                    )

    def test_module_does_not_import_controller_or_candidate_ranking(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "physical_execution_compensation.py"
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
