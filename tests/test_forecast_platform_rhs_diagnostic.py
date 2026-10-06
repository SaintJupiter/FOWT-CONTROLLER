import ast
from dataclasses import replace
import unittest

import numpy as np

from fowt_platform import BallastModelSnapshot, IncrementalState, PlatformMatrices
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.forecast_physical_load import (
    ForecastRotorLoadParameters,
    ForecastRotorOperatingState,
    assemble_forecast_generalized_rotor_loads,
)
from wind_prediction.forecast_platform_rhs_diagnostic import (
    CurrentPlatformRhsDiagnostic,
    ForecastPlatformRhsPointDiagnostic,
    diagnose_current_platform_rhs,
    diagnose_forecast_platform_rhs,
)
from wind_prediction.forecast_platform_trajectory import (
    ForecastTrajectoryEnvironment,
    rollout_frozen_forecast_platform_trajectory,
)


_ORIGIN_TIME = "2026-08-22T00:00:00"


def _matrices() -> PlatformMatrices:
    mass = np.diag([12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
    mass[0, 4] = mass[4, 0] = 0.4
    damping = np.diag([1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    damping[2, 3] = damping[3, 2] = 0.2
    stiffness = np.diag([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    stiffness[0, 4] = stiffness[4, 0] = 0.5
    return PlatformMatrices(
        mass=mass,
        damping=damping,
        hydrostatic_stiffness=stiffness,
        mooring_stiffness=np.zeros((6, 6)),
    )


def _trajectory(
    *,
    current_wave_load=None,
    current_other_load=None,
    operating_state=None,
    environment_origin_time=_ORIGIN_TIME,
):
    if current_wave_load is None:
        current_wave_load = np.zeros(6)
    if current_other_load is None:
        current_other_load = np.zeros(6)
    initial_state = IncrementalState(
        position=[0.4, 0.0, 0.0, -0.1, -0.2, 0.0],
        velocity=[0.0, 0.0, 0.3, -0.15, 0.05, 0.0],
    )
    evidence = ForecastEvidence(
        source="rhs_test",
        model_version="rhs_test_model",
        origin_time=_ORIGIN_TIME,
        sample_period_s=5.0,
        uv_ms=np.array([[0.0, -8.0], [2.0, -12.0]]),
        lead_reliability=np.array([0.9, 0.8]),
        event_probs={},
    )
    assembly = assemble_forecast_generalized_rotor_loads(
        forecast=evidence,
        current_enu_downwind_air_velocity_mps=[0.0, -6.0],
        current_wind_source="rhs_test_observation",
        current_wind_observation_time=_ORIGIN_TIME,
        parameters=ForecastRotorLoadParameters(
            air_density_kg_m3=1.225,
            rotor_radius_m=10.0,
            wind_reference_height_m=100.0,
            rotor_plane_height_m=100.0,
            power_law_shear_exponent=0.0,
            frozen_equilibrium_heading_rad=np.pi,
            rotor_force_application_point_from_reference_m=[0.0, 0.0, 20.0],
            downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
        ),
        operating_state=(
            ForecastRotorOperatingState(
                current_thrust_coefficient=0.8,
                future_thrust_coefficients=[0.8, 0.8],
                current_platform_reference_velocity_platform_mps=initial_state.velocity[:3],
                current_platform_angular_velocity_platform_radps=initial_state.velocity[3:],
                future_platform_reference_velocities_platform_mps=np.zeros((2, 3)),
                future_platform_angular_velocities_platform_radps=np.zeros((2, 3)),
                future_platform_kinematics_mode="frozen_zero",
                future_platform_kinematics_source="rhs_test_frozen_kinematics",
            )
            if operating_state is None
            else operating_state
        ),
    )
    matrices = _matrices()
    snapshot = BallastModelSnapshot(
        actual_tank_masses_kg=[100.0, 100.0, 100.0],
        reference_tank_masses_kg=[100.0, 100.0, 100.0],
        tank_mass_deltas_kg=[0.0, 0.0, 0.0],
        tank_capacities_kg=[200.0, 200.0, 200.0],
        tank_coordinates_m=[[-2.0, 0.0, -1.0], [1.0, 2.0, -1.0], [1.0, -2.0, -1.0]],
        gravity_m_s2=9.81,
        matrices=matrices,
        incremental_ballast_load=np.zeros(6),
        runtime_provenance="rhs_test_snapshot",
    )
    return rollout_frozen_forecast_platform_trajectory(
        load_assembly=assembly,
        platform_snapshot=snapshot,
        initial_state=initial_state,
        initial_state_time=_ORIGIN_TIME,
        environment=ForecastTrajectoryEnvironment(
            current_wave_load=current_wave_load,
            current_other_load=current_other_load,
            wave_interval_loads=np.zeros((2, 6)),
            other_interval_loads=np.zeros((2, 6)),
            provenance="rhs_test_zero_non_rotor_loads",
            **(
                {}
                if environment_origin_time is None
                else {
                    "forecast_origin_time": environment_origin_time,
                    "lead_times_s": [5.0, 10.0],
                }
            ),
        ),
    )


class ForecastPlatformRhsDiagnosticTests(unittest.TestCase):
    def test_current_rhs_uses_origin_loads_and_full_six_dof_equation(self):
        current_wave = np.array([0.0, 0.0, 1.5, -0.2, 0.4, 0.0])
        current_other = np.array([0.3, 0.0, 0.0, 0.0, -0.1, 0.2])
        trajectory = _trajectory(
            current_wave_load=current_wave,
            current_other_load=current_other,
        )

        diagnostic = diagnose_current_platform_rhs(trajectory=trajectory)

        self.assertIsInstance(diagnostic, CurrentPlatformRhsDiagnostic)
        self.assertIs(diagnostic.trajectory, trajectory)
        self.assertIs(
            diagnostic.posture_passive_load.platform_snapshot,
            trajectory.platform_snapshot,
        )
        self.assertIs(
            diagnostic.posture_passive_load.platform_state,
            trajectory.initial_state,
        )
        np.testing.assert_allclose(
            diagnostic.loads.wind,
            trajectory.load_assembly.current_rotor_load.generalized_load_platform,
        )
        np.testing.assert_allclose(diagnostic.loads.wave, current_wave)
        np.testing.assert_allclose(diagnostic.loads.other, current_other)
        np.testing.assert_allclose(
            diagnostic.dynamic_rhs_generalized_load,
            diagnostic.loads.total + diagnostic.passive_generalized_load,
        )
        np.testing.assert_allclose(
            trajectory.platform_snapshot.matrices.mass
            @ diagnostic.state_derivative.velocity_rate,
            diagnostic.dynamic_rhs_generalized_load,
        )

    def test_current_non_rotor_load_is_not_inferred_from_future_trajectory(self):
        reference = _trajectory()
        changed = _trajectory(
            current_wave_load=[0.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            current_other_load=[0.0, 0.0, 0.0, 0.0, -1.5, 0.0],
        )
        reference_current = diagnose_current_platform_rhs(trajectory=reference)
        changed_current = diagnose_current_platform_rhs(trajectory=changed)

        np.testing.assert_allclose(
            changed_current.dynamic_rhs_generalized_load
            - reference_current.dynamic_rhs_generalized_load,
            [0.0, 0.0, 0.0, 0.0, 6.0, 0.0],
        )
        for reference_step, changed_step in zip(reference.steps, changed.steps):
            np.testing.assert_allclose(
                reference_step.platform_step.loads.total,
                changed_step.platform_step.loads.total,
            )
            np.testing.assert_allclose(
                reference_step.end_state.position,
                changed_step.end_state.position,
            )
            np.testing.assert_allclose(
                reference_step.end_state.velocity,
                changed_step.end_state.velocity,
            )

    def test_current_rhs_rejects_unbound_environment_or_foreign_kinematics(self):
        unbound_environment = _trajectory(environment_origin_time=None)
        with self.assertRaisesRegex(
            ValueError,
            "requires environment forecast_origin_time",
        ):
            diagnose_current_platform_rhs(trajectory=unbound_environment)

        mismatched_operating_state = replace(
            _trajectory().load_assembly.operating_state,
            current_platform_reference_velocity_platform_mps=np.zeros(3),
            current_platform_angular_velocity_platform_radps=np.zeros(3),
        )
        mismatched_kinematics = _trajectory(
            operating_state=mismatched_operating_state,
        )
        with self.assertRaisesRegex(
            ValueError,
            "rotor operating-state kinematics must match",
        ):
            diagnose_current_platform_rhs(trajectory=mismatched_kinematics)

    def test_uses_the_matching_endpoint_state_and_full_six_dof_equation(self):
        trajectory = _trajectory()
        diagnostics = diagnose_forecast_platform_rhs(trajectory=trajectory)

        self.assertEqual(len(diagnostics), 2)
        for index, diagnostic in enumerate(diagnostics):
            step = trajectory.steps[index]
            self.assertEqual(diagnostic.lead_index, index)
            self.assertAlmostEqual(diagnostic.lead_time_s, step.end_time_s)
            self.assertIs(diagnostic.platform_state, step.end_state)
            self.assertIs(diagnostic.loads, step.platform_step.loads)
            expected_passive = (
                -trajectory.platform_snapshot.matrices.damping
                @ diagnostic.platform_state.velocity
                - trajectory.platform_snapshot.matrices.restoring_stiffness
                @ diagnostic.platform_state.position
            )
            np.testing.assert_allclose(
                diagnostic.passive_generalized_load,
                expected_passive,
            )
            np.testing.assert_allclose(
                diagnostic.dynamic_rhs_generalized_load,
                diagnostic.loads.total + expected_passive,
            )
            np.testing.assert_allclose(
                trajectory.platform_snapshot.matrices.mass
                @ diagnostic.state_derivative.velocity_rate,
                diagnostic.dynamic_rhs_generalized_load,
            )
            np.testing.assert_allclose(
                diagnostic.state_derivative.position_rate,
                diagnostic.platform_state.velocity,
            )

        first = diagnostics[0]
        diagonal_only_pitch = -(
            trajectory.platform_snapshot.matrices.damping[4, 4]
            * first.platform_state.velocity[4]
            + trajectory.platform_snapshot.matrices.restoring_stiffness[4, 4]
            * first.platform_state.position[4]
        )
        self.assertFalse(
            np.isclose(first.passive_generalized_load[4], diagonal_only_pitch)
        )

    def test_rejects_a_record_that_rebinds_an_endpoint_to_another_state_or_load(self):
        trajectory = _trajectory()
        diagnostic = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]
        with self.assertRaisesRegex(ValueError, "matching trajectory endpoint"):
            ForecastPlatformRhsPointDiagnostic(
                trajectory=trajectory,
                lead_index=0,
                lead_time_s=diagnostic.lead_time_s,
                platform_state=IncrementalState.zeros(),
                loads=diagnostic.loads,
                passive_generalized_load=diagnostic.passive_generalized_load,
                dynamic_rhs_generalized_load=diagnostic.dynamic_rhs_generalized_load,
                state_derivative=diagnostic.state_derivative,
            )

    def test_module_does_not_import_control_or_execution_logic(self):
        path = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "forecast_platform_rhs_diagnostic.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        forbidden_fragments = ("controller", "candidate", "planner", "execution")
        self.assertFalse(
            any(
                any(fragment in module for fragment in forbidden_fragments)
                for module in imported_modules
            )
        )


if __name__ == "__main__":
    unittest.main()
