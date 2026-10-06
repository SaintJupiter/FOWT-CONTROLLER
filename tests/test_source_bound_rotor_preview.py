import unittest
from dataclasses import replace

import numpy as np

from fowt_platform import (
    IncrementalState,
    NominalBelowRatedRotorSchedule,
    RotorPerformanceTable,
)
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.forecast_physical_load import ForecastRotorLoadParameters
from wind_prediction.forecast_rotor_inflow import ForecastRotorKinematics
from wind_prediction.source_bound_rotor_preview import (
    RotorPreviewOperatingDomainError,
    assemble_source_bound_nominal_rotor_preview,
)


def _forecast() -> ForecastEvidence:
    return ForecastEvidence(
        source="test_lstm",
        model_version="test_model_v1",
        origin_time="2026-09-04 00:00:00",
        sample_period_s=600.0,
        uv_ms=np.array([[0.0, -7.0], [0.0, -8.0]]),
        event_probs={"change": 0.7},
        lead_reliability=np.array([0.9, 0.8]),
        provides_future_preview=True,
    )


def _parameters() -> ForecastRotorLoadParameters:
    return ForecastRotorLoadParameters(
        air_density_kg_m3=1.225,
        rotor_radius_m=100.0,
        wind_reference_height_m=100.0,
        rotor_plane_height_m=100.0,
        power_law_shear_exponent=0.0,
        frozen_equilibrium_heading_rad=np.pi,
        rotor_force_application_point_from_reference_m=[0.0, 0.0, 120.0],
        downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
    )


def _schedule() -> NominalBelowRatedRotorSchedule:
    return NominalBelowRatedRotorSchedule(
        cut_in_wind_speed_mps=3.0,
        rated_wind_speed_mps=10.74,
        operational_tip_speed_ratio=9.0,
        minimum_rotor_speed_rad_s=0.3,
        rated_rotor_speed_rad_s=1.0,
        minimum_pitch_rad=0.0,
        source_name="test_schedule",
        source_sha256="schedule_sha256",
    )


def _table() -> RotorPerformanceTable:
    return RotorPerformanceTable(
        pitch_deg=np.array([-5.0, 0.0, 10.0]),
        tip_speed_ratio=np.array([2.0, 9.0, 14.5]),
        wind_speed_metadata_mps=np.array([10.0]),
        cp=np.ones((3, 3)),
        ct=np.array(
            [
                [0.45, 0.50, 0.55],
                [0.55, 0.60, 0.65],
                [0.65, 0.70, 0.75],
            ]
        ),
        cq=np.ones((3, 3)),
        source_name="test_table",
        source_sha256="table_sha256",
    )


def _kinematics(state: IncrementalState) -> ForecastRotorKinematics:
    return ForecastRotorKinematics(
        current_platform_reference_velocity_platform_mps=state.velocity[:3],
        current_platform_angular_velocity_platform_radps=state.velocity[3:],
        future_platform_reference_velocities_platform_mps=np.zeros((2, 3)),
        future_platform_angular_velocities_platform_radps=np.zeros((2, 3)),
        future_platform_kinematics_mode="frozen_zero",
        future_platform_kinematics_source="test_explicit_frozen_future",
        current_nacelle_yaw_relative_platform_rad=0.0,
        future_rotor_orientation_mode="frozen_current",
        future_rotor_orientation_source="test_frozen_current_yaw",
    )


def _assemble(state: IncrementalState, forecast=None):
    return assemble_source_bound_nominal_rotor_preview(
        forecast=_forecast() if forecast is None else forecast,
        current_enu_downwind_wind_mps=[0.0, -6.0],
        current_wind_source="test_observation",
        current_wind_observation_time="2026-09-04 00:00:00",
        platform_state=state,
        kinematics=_kinematics(state),
        parameters=_parameters(),
        schedule=_schedule(),
        performance_table=_table(),
        operating_mode_source="test_declared_below_rated_generating",
    )


class SourceBoundRotorPreviewTests(unittest.TestCase):
    def test_future_domain_failure_has_typed_endpoint_evidence(self):
        forecast = replace(_forecast(), uv_ms=np.array([[0.0, -7.0], [0.0, -2.0]]))
        with self.assertRaises(RotorPreviewOperatingDomainError) as raised:
            _assemble(IncrementalState.zeros(), forecast)
        self.assertIsInstance(raised.exception, ValueError)
        self.assertEqual(raised.exception.unsupported_labels, ("future_2",))
        self.assertFalse(raised.exception.operating_domain["supported"])

    def test_current_domain_failure_is_distinct_from_future_failure(self):
        state = IncrementalState(position=np.zeros(6), velocity=np.array([5.0, 0, 0, 0, 0, 0]))
        with self.assertRaises(RotorPreviewOperatingDomainError) as raised:
            _assemble(state)
        self.assertEqual(raised.exception.unsupported_labels, ("current",))

    def test_first_interval_load_is_the_declared_endpoint_average(self):
        preview = _assemble(IncrementalState.zeros())

        expected = 0.5 * (
            preview.load_assembly.current_rotor_load.generalized_load_platform
            + preview.load_assembly.future_rotor_loads[0].generalized_load_platform
        )
        np.testing.assert_allclose(
            preview.first_interval_rotor_load.generalized_load_platform,
            expected,
        )
        self.assertEqual(
            preview.interval_load_mapping,
            "piecewise_linear_endpoint_average",
        )

    def test_current_platform_velocity_changes_current_rotor_load(self):
        stationary = _assemble(IncrementalState.zeros())
        moving_state = IncrementalState(
            position=np.zeros(6),
            velocity=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        moving = _assemble(moving_state)

        self.assertNotEqual(
            moving.load_assembly.current_rotor_load.normal_load.thrust_n,
            stationary.load_assembly.current_rotor_load.normal_load.thrust_n,
        )
        np.testing.assert_allclose(
            moving.load_assembly.future_rotor_loads[0].generalized_load_platform,
            stationary.load_assembly.future_rotor_loads[0].generalized_load_platform,
        )
        self.assertEqual(
            moving.source_record_sha256,
            stationary.source_record_sha256,
        )

    def test_rejects_kinematics_from_a_different_cycle_state(self):
        state = IncrementalState(
            position=np.zeros(6),
            velocity=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )

        with self.assertRaisesRegex(ValueError, "must match platform_state"):
            assemble_source_bound_nominal_rotor_preview(
                forecast=_forecast(),
                current_enu_downwind_wind_mps=[0.0, -6.0],
                current_wind_source="test_observation",
                current_wind_observation_time="2026-09-04 00:00:00",
                platform_state=state,
                kinematics=_kinematics(IncrementalState.zeros()),
                parameters=_parameters(),
                schedule=_schedule(),
                performance_table=_table(),
                operating_mode_source="test_declared_below_rated_generating",
            )


if __name__ == "__main__":
    unittest.main()
