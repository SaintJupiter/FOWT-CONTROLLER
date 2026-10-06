import ast
import unittest
from dataclasses import replace

import numpy as np

from fowt_platform import BallastModelSnapshot, IncrementalState, PlatformMatrices
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.forecast_physical_load import (
    ForecastRotorLoadParameters,
    ForecastRotorOperatingState,
    assemble_forecast_generalized_rotor_loads,
)


_ORIGIN_TIME = "2026-08-22T00:00:00"
from wind_prediction.forecast_platform_trajectory import (
    ForecastTrajectoryEnvironment,
    rollout_frozen_forecast_platform_trajectory,
)


def _assembly(*, future_uv=None, operating_state=None):
    if future_uv is None:
        future_uv = [[0.0, -10.0], [0.0, -20.0]]
    evidence = ForecastEvidence(
        source="trajectory_test",
        model_version="trajectory_test_model",
        origin_time=_ORIGIN_TIME,
        sample_period_s=10.0,
        uv_ms=np.asarray(future_uv, dtype=float),
        lead_reliability=np.array([0.9, 0.8]),
        event_probs={},
    )
    parameters = ForecastRotorLoadParameters(
        air_density_kg_m3=1.225,
        rotor_radius_m=10.0,
        wind_reference_height_m=100.0,
        rotor_plane_height_m=100.0,
        power_law_shear_exponent=0.0,
        frozen_equilibrium_heading_rad=np.pi,
        rotor_force_application_point_from_reference_m=[0.0, 0.0, 20.0],
        downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
    )
    operating = operating_state or ForecastRotorOperatingState(
        current_thrust_coefficient=0.8,
        future_thrust_coefficients=[0.8, 0.8],
        current_platform_reference_velocity_platform_mps=[0.0, 0.0, 0.0],
        current_platform_angular_velocity_platform_radps=[0.0, 0.0, 0.0],
        future_platform_reference_velocities_platform_mps=np.zeros((2, 3)),
        future_platform_angular_velocities_platform_radps=np.zeros((2, 3)),
        future_platform_kinematics_mode="frozen_zero",
        future_platform_kinematics_source="trajectory_test_frozen_kinematics",
    )
    return assemble_forecast_generalized_rotor_loads(
        forecast=evidence,
        current_enu_downwind_air_velocity_mps=[0.0, -5.0],
        current_wind_source="trajectory_test_observation",
        current_wind_observation_time="2026-08-22T00:00:00",
        parameters=parameters,
        operating_state=operating,
    )


def _snapshot(*, matrices=None) -> BallastModelSnapshot:
    matrices = matrices or PlatformMatrices(
        mass=np.diag([1.0e6] * 6),
        damping=np.zeros((6, 6)),
        hydrostatic_stiffness=np.zeros((6, 6)),
        mooring_stiffness=np.zeros((6, 6)),
    )
    masses = np.array([1.0e5, 1.0e5, 1.0e5])
    return BallastModelSnapshot(
        actual_tank_masses_kg=masses,
        reference_tank_masses_kg=masses,
        tank_mass_deltas_kg=np.zeros(3),
        tank_capacities_kg=np.full(3, 1.0e6),
        tank_coordinates_m=np.zeros((3, 3)),
        gravity_m_s2=9.81,
        matrices=matrices,
        incremental_ballast_load=np.zeros(6),
        runtime_provenance="trajectory_test_snapshot",
    )


def _environment():
    return ForecastTrajectoryEnvironment(
        current_wave_load=np.zeros(6),
        current_other_load=np.zeros(6),
        wave_interval_loads=np.zeros((2, 6)),
        other_interval_loads=np.zeros((2, 6)),
        provenance="synthetic_zero_non_rotor_loads",
    )


class ForecastPlatformTrajectoryTests(unittest.TestCase):
    def test_future_point_backward_hold_uses_each_endpoint_load_in_its_preceding_interval(self):
        assembly = _assembly()
        snapshot = _snapshot()
        initial = IncrementalState.zeros()

        trajectory = rollout_frozen_forecast_platform_trajectory(
            load_assembly=assembly,
            platform_snapshot=snapshot,
            initial_state=initial,
            initial_state_time=_ORIGIN_TIME,
            environment=_environment(),
        )

        self.assertEqual(len(trajectory.steps), 2)
        self.assertEqual(
            trajectory.interval_load_convention,
            "future_point_backward_hold",
        )
        self.assertIs(
            trajectory.steps[0].rotor_load,
            assembly.future_rotor_loads[0],
        )
        self.assertIs(
            trajectory.steps[1].rotor_load,
            assembly.future_rotor_loads[1],
        )
        self.assertAlmostEqual(trajectory.steps[0].duration_s, 10.0)
        self.assertAlmostEqual(trajectory.steps[1].duration_s, 10.0)
        self.assertGreater(
            trajectory.steps[1].end_state.position[0],
            trajectory.steps[0].end_state.position[0],
        )

    def test_requires_an_explicit_non_rotor_interval_schedule_with_matching_horizon(self):
        with self.assertRaisesRegex(ValueError, "one entry per physical forecast lead"):
            rollout_frozen_forecast_platform_trajectory(
                load_assembly=_assembly(),
                platform_snapshot=_snapshot(),
                initial_state=IncrementalState.zeros(),
                initial_state_time=_ORIGIN_TIME,
                environment=ForecastTrajectoryEnvironment(
                    current_wave_load=np.zeros(6),
                    current_other_load=np.zeros(6),
                    wave_interval_loads=np.zeros((1, 6)),
                    other_interval_loads=np.zeros((1, 6)),
                    provenance="incomplete_schedule",
                ),
            )

    def test_explicit_environment_binding_must_match_the_rotor_forecast_cycle(self):
        common = {
            "current_wave_load": np.zeros(6),
            "current_other_load": np.zeros(6),
            "wave_interval_loads": np.zeros((2, 6)),
            "other_interval_loads": np.zeros((2, 6)),
            "provenance": "bound_schedule",
        }
        with self.assertRaisesRegex(ValueError, "forecast_origin_time must match"):
            rollout_frozen_forecast_platform_trajectory(
                load_assembly=_assembly(),
                platform_snapshot=_snapshot(),
                initial_state=IncrementalState.zeros(),
                initial_state_time=_ORIGIN_TIME,
                environment=ForecastTrajectoryEnvironment(
                    **common,
                    forecast_origin_time="2026-08-22T00:00:10",
                    lead_times_s=[10.0, 20.0],
                ),
            )

        with self.assertRaisesRegex(ValueError, "lead_times_s must match"):
            rollout_frozen_forecast_platform_trajectory(
                load_assembly=_assembly(),
                platform_snapshot=_snapshot(),
                initial_state=IncrementalState.zeros(),
                initial_state_time=_ORIGIN_TIME,
                environment=ForecastTrajectoryEnvironment(
                    **common,
                    forecast_origin_time=_ORIGIN_TIME,
                    lead_times_s=[10.0, 21.0],
                ),
            )

    def test_environment_copies_input_loads_and_rollout_does_not_change_state(self):
        current_wave = np.zeros(6)
        current_other = np.zeros(6)
        wave = np.zeros((2, 6))
        other = np.zeros((2, 6))
        environment = ForecastTrajectoryEnvironment(
            current_wave_load=current_wave,
            current_other_load=current_other,
            wave_interval_loads=wave,
            other_interval_loads=other,
            provenance="copied_schedule",
        )
        initial = IncrementalState.zeros()
        before_position = initial.position.copy()
        before_velocity = initial.velocity.copy()
        rollout_frozen_forecast_platform_trajectory(
            load_assembly=_assembly(),
            platform_snapshot=_snapshot(),
            initial_state=initial,
            initial_state_time=_ORIGIN_TIME,
            environment=environment,
        )
        current_wave[:] = 0.5
        current_other[:] = 0.75
        wave[:] = 1.0
        other[:] = 2.0
        np.testing.assert_allclose(environment.current_wave_load, 0.0)
        np.testing.assert_allclose(environment.current_other_load, 0.0)
        np.testing.assert_allclose(environment.wave_interval_loads, 0.0)
        np.testing.assert_allclose(environment.other_interval_loads, 0.0)
        np.testing.assert_allclose(initial.position, before_position)
        np.testing.assert_allclose(initial.velocity, before_velocity)
        self.assertFalse(environment.wave_interval_loads.flags.writeable)

    def test_rejects_a_state_time_or_kinematics_mode_that_cannot_share_this_rollout(self):
        with self.assertRaisesRegex(ValueError, "initial_state_time must match"):
            rollout_frozen_forecast_platform_trajectory(
                load_assembly=_assembly(),
                platform_snapshot=_snapshot(),
                initial_state=IncrementalState.zeros(),
                initial_state_time="2026-08-22T00:00:10",
                environment=_environment(),
            )

        explicit_motion = replace(
            _assembly().operating_state,
            future_platform_reference_velocities_platform_mps=np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
            future_platform_kinematics_mode="explicit_per_lead",
            future_platform_kinematics_source="external_future_motion",
        )
        with self.assertRaisesRegex(ValueError, "only supports frozen_zero"):
            rollout_frozen_forecast_platform_trajectory(
                load_assembly=_assembly(operating_state=explicit_motion),
                platform_snapshot=_snapshot(),
                initial_state=IncrementalState.zeros(),
                initial_state_time=_ORIGIN_TIME,
                environment=_environment(),
            )

    def test_static_offset_remains_fixed_under_identical_held_forecast_loads(self):
        assembly = _assembly(future_uv=[[0.0, -10.0], [0.0, -10.0]])
        matrices = PlatformMatrices(
            mass=np.diag([1.0e6] * 6),
            damping=np.diag([2.0e4] * 6),
            hydrostatic_stiffness=np.diag([5.0e5] * 6),
            mooring_stiffness=np.diag([5.0e5] * 6),
        )
        snapshot = _snapshot(matrices=matrices)
        static_load = assembly.future_rotor_loads[0].generalized_load_platform
        initial = IncrementalState(
            position=np.linalg.solve(matrices.restoring_stiffness, static_load),
            velocity=np.zeros(6),
        )

        trajectory = rollout_frozen_forecast_platform_trajectory(
            load_assembly=assembly,
            platform_snapshot=snapshot,
            initial_state=initial,
            initial_state_time=_ORIGIN_TIME,
            environment=_environment(),
        )

        for step in trajectory.steps:
            np.testing.assert_allclose(step.end_state.position, initial.position, atol=1e-8)
            np.testing.assert_allclose(step.end_state.velocity, 0.0, atol=1e-8)

    def test_module_does_not_import_controller_or_execution_logic(self):
        path = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "forecast_platform_trajectory.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        forbidden_fragments = ("controller", "execution", "candidate", "planner")
        self.assertFalse(
            any(
                any(fragment in module for fragment in forbidden_fragments)
                for module in imported_modules
            )
        )


if __name__ == "__main__":
    unittest.main()
