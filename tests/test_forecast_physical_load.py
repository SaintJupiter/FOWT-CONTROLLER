import ast
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from fowt_platform import GeneralizedLoadForecast
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.forecast_physical_load import (
    FuturePlatformKinematicsMode,
    FutureRotorOrientationMode,
    ForecastRotorLoadParameters,
    ForecastRotorOperatingState,
    assemble_forecast_generalized_rotor_loads,
)


def _forecast(*, provides_future_preview=True) -> ForecastEvidence:
    return ForecastEvidence(
        source="test",
        model_version="test",
        origin_time="2026-08-22T00:00:00",
        sample_period_s=600.0,
        uv_ms=np.array([[0.0, -10.0], [0.0, -12.0]], dtype=float),
        lead_reliability=np.array([0.9, 0.8]),
        event_probs={"attention_event_0_20m": 0.8},
        provides_future_preview=provides_future_preview,
    )


def _parameters() -> ForecastRotorLoadParameters:
    return ForecastRotorLoadParameters(
        air_density_kg_m3=1.225,
        rotor_radius_m=100.0,
        wind_reference_height_m=100.0,
        rotor_plane_height_m=200.0,
        power_law_shear_exponent=1.0,
        frozen_equilibrium_heading_rad=np.pi,
        rotor_force_application_point_from_reference_m=[0.0, 0.0, 120.0],
        downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
    )


def _operating_state(*, future_coefficients=(0.8, 0.8)) -> ForecastRotorOperatingState:
    return ForecastRotorOperatingState(
        current_thrust_coefficient=0.8,
        future_thrust_coefficients=future_coefficients,
        current_platform_reference_velocity_platform_mps=[0.0, 0.0, 0.0],
        current_platform_angular_velocity_platform_radps=[0.0, 0.0, 0.0],
        future_platform_reference_velocities_platform_mps=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        future_platform_angular_velocities_platform_radps=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        future_platform_kinematics_mode="frozen_zero",
        future_platform_kinematics_source="test_frozen_platform",
    )


def _current_observation_kwargs() -> dict[str, object]:
    return {
        "current_wind_source": "test_current_observation",
        "current_wind_observation_time": "2026-08-22T00:00:00",
    }


class ForecastPhysicalLoadTests(unittest.TestCase):
    def test_height_transfer_assumptions_and_discrete_leads_are_preserved(self):
        parameters = _parameters()
        operating_state = _operating_state()
        assembly = assemble_forecast_generalized_rotor_loads(
            forecast=_forecast(),
            current_enu_downwind_air_velocity_mps=[0.0, -8.0],
            parameters=parameters,
            operating_state=operating_state,
            **_current_observation_kwargs(),
        )

        self.assertIs(assembly.parameters, parameters)
        self.assertIs(assembly.operating_state, operating_state)
        self.assertEqual(assembly.input_identity.forecast_source, "test")
        self.assertEqual(assembly.input_identity.forecast_model_version, "test")
        self.assertEqual(
            assembly.input_identity.forecast_origin_time,
            "2026-08-22T00:00:00",
        )
        self.assertEqual(
            assembly.input_identity.current_wind_source,
            "test_current_observation",
        )
        self.assertEqual(
            assembly.input_identity.current_wind_observation_time,
            "2026-08-22T00:00:00",
        )
        self.assertIs(
            assembly.operating_state.future_platform_kinematics_mode,
            FuturePlatformKinematicsMode.FROZEN_ZERO,
        )
        self.assertIs(
            assembly.input_identity.future_platform_kinematics_mode,
            FuturePlatformKinematicsMode.FROZEN_ZERO,
        )
        self.assertEqual(
            assembly.input_identity.future_platform_kinematics_source,
            "test_frozen_platform",
        )
        np.testing.assert_allclose(
            assembly.current_rotor_plane_enu_downwind_mps,
            [0.0, -16.0],
        )
        np.testing.assert_allclose(
            assembly.future_rotor_plane_enu_downwind_mps,
            [[0.0, -20.0], [0.0, -24.0]],
        )
        np.testing.assert_allclose(assembly.load_forecast.lead_times_s, [600.0, 1200.0])
        self.assertGreater(
            assembly.future_rotor_loads[1].normal_load.thrust_n,
            assembly.future_rotor_loads[0].normal_load.thrust_n,
        )

    def test_future_plus_600_second_load_is_not_used_as_current_load(self):
        assembly = assemble_forecast_generalized_rotor_loads(
            forecast=_forecast(),
            current_enu_downwind_air_velocity_mps=[0.0, -8.0],
            parameters=_parameters(),
            operating_state=_operating_state(),
            **_current_observation_kwargs(),
        )

        self.assertAlmostEqual(
            assembly.current_rotor_load.normal_load.thrust_n
            / assembly.future_rotor_loads[0].normal_load.thrust_n,
            (16.0 / 20.0) ** 2,
        )
        np.testing.assert_allclose(
            assembly.load_forecast.current_generalized_load,
            assembly.current_rotor_load.generalized_load_platform,
        )
        np.testing.assert_allclose(
            assembly.load_forecast.future_load_at(0),
            assembly.future_rotor_loads[0].generalized_load_platform,
        )

    def test_future_platform_motion_is_explicit_and_changes_the_corresponding_load(self):
        stationary = assemble_forecast_generalized_rotor_loads(
            forecast=_forecast(),
            current_enu_downwind_air_velocity_mps=[0.0, -8.0],
            parameters=_parameters(),
            operating_state=_operating_state(),
            **_current_observation_kwargs(),
        )
        state = _operating_state()
        moving_state = replace(
            state,
            future_platform_reference_velocities_platform_mps=[
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            future_platform_kinematics_mode="explicit_per_lead",
            future_platform_kinematics_source="synthetic_per_lead_motion",
        )
        moving = assemble_forecast_generalized_rotor_loads(
            forecast=_forecast(),
            current_enu_downwind_air_velocity_mps=[0.0, -8.0],
            parameters=_parameters(),
            operating_state=moving_state,
            **_current_observation_kwargs(),
        )

        self.assertLess(
            moving.future_rotor_loads[0].normal_load.thrust_n,
            stationary.future_rotor_loads[0].normal_load.thrust_n,
        )
        self.assertAlmostEqual(
            moving.future_rotor_loads[1].normal_load.thrust_n,
            stationary.future_rotor_loads[1].normal_load.thrust_n,
        )

    def test_future_kinematics_mode_rejects_hidden_motion_and_missing_source(self):
        with self.assertRaisesRegex(ValueError, "frozen_zero"):
            replace(
                _operating_state(),
                future_platform_reference_velocities_platform_mps=[
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            )
        with self.assertRaisesRegex(ValueError, "must be non-empty"):
            replace(
                _operating_state(),
                future_platform_kinematics_source=" ",
            )

    def test_frozen_current_orientation_rotates_the_normal_and_hub_about_yaw_axis(self):
        parameters = replace(
            _parameters(),
            frozen_equilibrium_heading_rad=0.0,
            rotor_force_application_point_from_reference_m=[25.0, -7.0, 120.0],
            nacelle_yaw_axis_point_from_reference_m=[5.0, -7.0, 100.0],
        )
        operating = replace(
            _operating_state(),
            current_nacelle_yaw_relative_platform_rad=np.pi / 2.0,
            future_rotor_orientation_source="synthetic_external_frozen_nacelle_yaw",
        )
        crosswind_forecast = replace(
            _forecast(),
            uv_ms=np.array([[-10.0, 0.0], [-12.0, 0.0]], dtype=float),
        )
        assembly = assemble_forecast_generalized_rotor_loads(
            forecast=crosswind_forecast,
            current_enu_downwind_air_velocity_mps=[-8.0, 0.0],
            parameters=parameters,
            operating_state=operating,
            **_current_observation_kwargs(),
        )

        self.assertIs(
            assembly.operating_state.future_rotor_orientation_mode,
            FutureRotorOrientationMode.FROZEN_CURRENT,
        )
        self.assertIs(
            assembly.input_identity.future_rotor_orientation_mode,
            FutureRotorOrientationMode.FROZEN_CURRENT,
        )
        self.assertEqual(
            assembly.input_identity.future_rotor_orientation_source,
            "synthetic_external_frozen_nacelle_yaw",
        )
        self.assertGreater(
            assembly.future_rotor_loads[1].normal_load.force_platform_n[1],
            0.0,
        )
        force = assembly.future_rotor_loads[0].normal_load.force_platform_n[1]
        np.testing.assert_allclose(
            assembly.future_rotor_loads[0].generalized_load_platform,
            [0.0, force, 0.0, -120.0 * force, 0.0, 5.0 * force],
            rtol=0.0,
            atol=1e-6,
        )

    def test_nonzero_yaw_requires_an_explicit_nacelle_yaw_axis(self):
        operating = replace(
            _operating_state(),
            current_nacelle_yaw_relative_platform_rad=np.pi / 2.0,
            future_rotor_orientation_source="synthetic_external_frozen_nacelle_yaw",
        )
        with self.assertRaisesRegex(ValueError, "nacelle_yaw_axis"):
            assemble_forecast_generalized_rotor_loads(
                forecast=replace(
                    _forecast(),
                    uv_ms=np.array([[-10.0, 0.0], [-12.0, 0.0]], dtype=float),
                ),
                current_enu_downwind_air_velocity_mps=[-8.0, 0.0],
                parameters=replace(_parameters(), frozen_equilibrium_heading_rad=0.0),
                operating_state=operating,
                **_current_observation_kwargs(),
            )

    def test_nonzero_yaw_requires_an_explicit_orientation_source(self):
        with self.assertRaisesRegex(ValueError, "explicit future_rotor_orientation_source"):
            operating = replace(
                _operating_state(),
                current_nacelle_yaw_relative_platform_rad=np.pi / 2.0,
            )
            assemble_forecast_generalized_rotor_loads(
                forecast=replace(
                    _forecast(),
                    uv_ms=np.array([[-10.0, 0.0], [-12.0, 0.0]], dtype=float),
                ),
                current_enu_downwind_air_velocity_mps=[-8.0, 0.0],
                parameters=replace(
                    _parameters(),
                    frozen_equilibrium_heading_rad=0.0,
                    nacelle_yaw_axis_point_from_reference_m=[0.0, 0.0, 100.0],
                ),
                operating_state=operating,
                **_current_observation_kwargs(),
            )

    def test_reversed_normal_flow_remains_an_explicit_orientation_boundary(self):
        reversed_forecast = replace(
            _forecast(),
            uv_ms=np.array([[0.0, 10.0], [0.0, 12.0]], dtype=float),
        )
        with self.assertRaisesRegex(ValueError, "future forecast lead 0"):
            assemble_forecast_generalized_rotor_loads(
                forecast=reversed_forecast,
                current_enu_downwind_air_velocity_mps=[0.0, -8.0],
                parameters=_parameters(),
                operating_state=_operating_state(),
                **_current_observation_kwargs(),
            )

    def test_assembly_rejects_an_identity_label_that_disagrees_with_operating_state(self):
        assembly = assemble_forecast_generalized_rotor_loads(
            forecast=_forecast(),
            current_enu_downwind_air_velocity_mps=[0.0, -8.0],
            parameters=_parameters(),
            operating_state=_operating_state(),
            **_current_observation_kwargs(),
        )
        inconsistent_identity = replace(
            assembly.input_identity,
            future_platform_kinematics_mode="explicit_per_lead",
            future_platform_kinematics_source="incorrect_manual_label",
        )
        with self.assertRaisesRegex(ValueError, "must match operating_state"):
            replace(assembly, input_identity=inconsistent_identity)

    def test_assembly_rejects_mismatched_load_grid_or_duplicate_load_records(self):
        assembly = assemble_forecast_generalized_rotor_loads(
            forecast=_forecast(),
            current_enu_downwind_air_velocity_mps=[0.0, -8.0],
            parameters=_parameters(),
            operating_state=_operating_state(),
            **_current_observation_kwargs(),
        )
        wrong_grid = GeneralizedLoadForecast(
            current_generalized_load=assembly.load_forecast.current_generalized_load,
            future_generalized_loads=assembly.load_forecast.future_generalized_loads,
            lead_times_s=[5.0, 10.0],
        )
        with self.assertRaisesRegex(ValueError, "lead times must match"):
            replace(assembly, load_forecast=wrong_grid)

        wrong_future = np.array(
            assembly.load_forecast.future_generalized_loads,
            dtype=float,
            copy=True,
        )
        wrong_future[0, 4] += 1.0
        inconsistent_loads = GeneralizedLoadForecast(
            current_generalized_load=assembly.load_forecast.current_generalized_load,
            future_generalized_loads=wrong_future,
            lead_times_s=assembly.load_forecast.lead_times_s,
        )
        with self.assertRaisesRegex(ValueError, "future generalized loads must match"):
            replace(assembly, load_forecast=inconsistent_loads)

    def test_confidence_and_event_values_do_not_scale_physical_loads(self):
        evidence = _forecast()
        low_confidence = replace(
            evidence,
            lead_reliability=np.array([0.0, 0.0]),
            event_probs={"attention_event_0_20m": 0.0},
        )
        common = dict(
            current_enu_downwind_air_velocity_mps=[0.0, -8.0],
            parameters=_parameters(),
            operating_state=_operating_state(),
            **_current_observation_kwargs(),
        )
        high = assemble_forecast_generalized_rotor_loads(forecast=evidence, **common)
        low = assemble_forecast_generalized_rotor_loads(
            forecast=low_confidence,
            **common,
        )

        np.testing.assert_allclose(
            high.load_forecast.current_generalized_load,
            low.load_forecast.current_generalized_load,
        )
        np.testing.assert_allclose(
            high.load_forecast.future_generalized_loads,
            low.load_forecast.future_generalized_loads,
        )

    def test_rejects_missing_preview_or_mismatched_operating_horizon(self):
        with self.assertRaisesRegex(ValueError, "future preview"):
            assemble_forecast_generalized_rotor_loads(
                forecast=_forecast(provides_future_preview=False),
                current_enu_downwind_air_velocity_mps=[0.0, -8.0],
                parameters=_parameters(),
                operating_state=_operating_state(),
                **_current_observation_kwargs(),
            )

    def test_rejects_current_observation_from_a_different_forecast_origin(self):
        with self.assertRaisesRegex(ValueError, "must match forecast_origin_time"):
            assemble_forecast_generalized_rotor_loads(
                forecast=_forecast(),
                current_enu_downwind_air_velocity_mps=[0.0, -8.0],
                current_wind_source="test_current_observation",
                current_wind_observation_time="2026-08-22T00:10:00",
                parameters=_parameters(),
                operating_state=_operating_state(),
            )
        with self.assertRaisesRegex(ValueError, "one finite non-negative"):
            assemble_forecast_generalized_rotor_loads(
                forecast=_forecast(),
                current_enu_downwind_air_velocity_mps=[0.0, -8.0],
                parameters=_parameters(),
                operating_state=_operating_state(future_coefficients=(0.8,)),
                **_current_observation_kwargs(),
            )

    def test_future_operating_inputs_are_copied_before_load_assembly(self):
        coefficients = np.array([0.8, 0.75], dtype=float)
        future_reference_velocities = np.zeros((2, 3), dtype=float)
        future_angular_velocities = np.zeros((2, 3), dtype=float)
        operating_state = ForecastRotorOperatingState(
            current_thrust_coefficient=0.8,
            future_thrust_coefficients=coefficients,
            current_platform_reference_velocity_platform_mps=[0.0, 0.0, 0.0],
            current_platform_angular_velocity_platform_radps=[0.0, 0.0, 0.0],
            future_platform_reference_velocities_platform_mps=future_reference_velocities,
            future_platform_angular_velocities_platform_radps=future_angular_velocities,
            future_platform_kinematics_mode="frozen_zero",
            future_platform_kinematics_source="copied_frozen_platform",
        )
        assembly = assemble_forecast_generalized_rotor_loads(
            forecast=_forecast(),
            current_enu_downwind_air_velocity_mps=[0.0, -8.0],
            parameters=_parameters(),
            operating_state=operating_state,
            **_current_observation_kwargs(),
        )
        before = np.array(assembly.load_forecast.future_generalized_loads, copy=True)

        coefficients[:] = 0.0
        future_reference_velocities[:] = 9.0
        future_angular_velocities[:] = 7.0

        np.testing.assert_allclose(
            assembly.operating_state.future_thrust_coefficients,
            [0.8, 0.75],
        )
        np.testing.assert_allclose(
            assembly.operating_state.future_platform_reference_velocities_platform_mps,
            np.zeros((2, 3)),
        )
        np.testing.assert_allclose(
            assembly.operating_state.future_platform_angular_velocities_platform_radps,
            np.zeros((2, 3)),
        )
        np.testing.assert_allclose(assembly.load_forecast.future_generalized_loads, before)
        self.assertFalse(assembly.operating_state.future_thrust_coefficients.flags.writeable)
        self.assertFalse(
            assembly.operating_state.future_platform_reference_velocities_platform_mps.flags.writeable
        )

    def test_module_does_not_import_controller_or_execution_logic(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "forecast_physical_load.py"
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
