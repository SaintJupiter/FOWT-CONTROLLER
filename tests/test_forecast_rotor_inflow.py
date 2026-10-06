from dataclasses import replace
import unittest

import numpy as np

from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.forecast_physical_load import ForecastRotorLoadParameters
from wind_prediction.forecast_physical_load import (
    assemble_forecast_generalized_rotor_loads,
)
from wind_prediction.forecast_rotor_inflow import (
    ForecastRotorKinematics,
    assemble_forecast_rotor_relative_inflow,
)
from fowt_platform.coordinates import downwind_normal_relative_wind_component


def _forecast(*, event_probs=None, reliability=None) -> ForecastEvidence:
    return ForecastEvidence(
        source="test_forecast",
        model_version="test_model",
        origin_time="2026-08-22T00:00:00",
        sample_period_s=600.0,
        uv_ms=np.array([[0.0, -10.0], [0.0, -12.0]], dtype=float),
        lead_reliability=(
            np.array([0.9, 0.8]) if reliability is None else np.asarray(reliability)
        ),
        event_probs={"attention_event_0_20m": 0.8}
        if event_probs is None
        else event_probs,
        provides_future_preview=True,
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


def _kinematics(*, future_velocities=None) -> ForecastRotorKinematics:
    return ForecastRotorKinematics(
        current_platform_reference_velocity_platform_mps=[0.0, 0.0, 0.0],
        current_platform_angular_velocity_platform_radps=[0.0, 0.0, 0.0],
        future_platform_reference_velocities_platform_mps=(
            np.zeros((2, 3)) if future_velocities is None else future_velocities
        ),
        future_platform_angular_velocities_platform_radps=np.zeros((2, 3)),
        future_platform_kinematics_mode=(
            "frozen_zero" if future_velocities is None else "explicit_per_lead"
        ),
        future_platform_kinematics_source="synthetic_future_kinematics",
    )


def _assemble(*, forecast=None, kinematics=None):
    return assemble_forecast_rotor_relative_inflow(
        forecast=_forecast() if forecast is None else forecast,
        current_enu_downwind_air_velocity_mps=[0.0, -8.0],
        current_wind_source="test_current_observation",
        current_wind_observation_time="2026-08-22T00:00:00",
        parameters=_parameters(),
        kinematics=_kinematics() if kinematics is None else kinematics,
    )


class ForecastRotorRelativeInflowTests(unittest.TestCase):
    def test_uses_same_height_transfer_and_signed_normal_direction_as_load_assembly(self):
        inflow = _assemble()

        self.assertAlmostEqual(inflow.current_normal_relative_inflow_mps, 16.0)
        np.testing.assert_allclose(
            inflow.future_normal_relative_inflows_mps,
            [20.0, 24.0],
        )
        self.assertFalse(inflow.future_normal_relative_inflows_mps.flags.writeable)

    def test_explicit_future_platform_motion_changes_only_the_corresponding_lead(self):
        inflow = _assemble(
            kinematics=_kinematics(
                future_velocities=np.array([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            )
        )

        np.testing.assert_allclose(
            inflow.future_normal_relative_inflows_mps,
            [16.0, 24.0],
        )

    def test_returns_reverse_flow_as_signed_fact_instead_of_flipping_or_clipping_it(self):
        reverse = replace(
            _forecast(),
            uv_ms=np.array([[0.0, 10.0], [0.0, 12.0]], dtype=float),
        )
        inflow = _assemble(forecast=reverse)

        np.testing.assert_allclose(
            inflow.future_normal_relative_inflows_mps,
            [-20.0, -24.0],
        )

    def test_event_and_reliability_values_do_not_scale_relative_physical_inflow(self):
        normal = _assemble()
        low_confidence = _assemble(
            forecast=_forecast(event_probs={"attention_event_0_20m": 0.0}, reliability=[0.0, 0.0])
        )

        self.assertAlmostEqual(
            normal.current_normal_relative_inflow_mps,
            low_confidence.current_normal_relative_inflow_mps,
        )
        np.testing.assert_allclose(
            normal.future_normal_relative_inflows_mps,
            low_confidence.future_normal_relative_inflows_mps,
        )

    def test_relative_inflow_matches_actual_load_path_with_per_lead_platform_motion(self):
        forecast = _forecast()
        parameters = _parameters()
        kinematics = _kinematics(
            future_velocities=np.array([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        )
        inflow = assemble_forecast_rotor_relative_inflow(
            forecast=forecast,
            current_enu_downwind_air_velocity_mps=[0.0, -8.0],
            current_wind_source="test_current_observation",
            current_wind_observation_time="2026-08-22T00:00:00",
            parameters=parameters,
            kinematics=kinematics,
        )
        operating_state = kinematics.to_operating_state(
            current_thrust_coefficient=0.8,
            future_thrust_coefficients=[0.7, 0.6],
        )
        assembly = assemble_forecast_generalized_rotor_loads(
            forecast=forecast,
            current_enu_downwind_air_velocity_mps=[0.0, -8.0],
            current_wind_source="test_current_observation",
            current_wind_observation_time="2026-08-22T00:00:00",
            parameters=parameters,
            operating_state=operating_state,
        )

        current_normal = downwind_normal_relative_wind_component(
            assembly.current_rotor_load.relative_air_velocity_platform_mps,
            parameters.downwind_rotor_normal_platform,
        )
        future_normal = np.asarray(
            [
                downwind_normal_relative_wind_component(
                    load.relative_air_velocity_platform_mps,
                    parameters.downwind_rotor_normal_platform,
                )
                for load in assembly.future_rotor_loads
            ]
        )
        self.assertAlmostEqual(
            current_normal,
            inflow.current_normal_relative_inflow_mps,
        )
        np.testing.assert_allclose(
            future_normal,
            inflow.future_normal_relative_inflows_mps,
        )

    def test_operating_state_binding_preserves_kinematics_and_requires_matching_horizon(self):
        kinematics = _kinematics()
        state = kinematics.to_operating_state(
            current_thrust_coefficient=0.8,
            future_thrust_coefficients=[0.7, 0.6],
        )

        np.testing.assert_allclose(state.future_thrust_coefficients, [0.7, 0.6])
        np.testing.assert_allclose(
            state.future_platform_reference_velocities_platform_mps,
            kinematics.future_platform_reference_velocities_platform_mps,
        )
        with self.assertRaisesRegex(ValueError, "one finite non-negative"):
            kinematics.to_operating_state(
                current_thrust_coefficient=0.8,
                future_thrust_coefficients=[0.7],
            )

    def test_current_wind_time_must_bind_to_forecast_origin(self):
        with self.assertRaisesRegex(ValueError, "must match forecast.origin_time"):
            assemble_forecast_rotor_relative_inflow(
                forecast=_forecast(),
                current_enu_downwind_air_velocity_mps=[0.0, -8.0],
                current_wind_source="test_current_observation",
                current_wind_observation_time="2026-08-22T00:10:00",
                parameters=_parameters(),
                kinematics=_kinematics(),
            )


if __name__ == "__main__":
    unittest.main()
