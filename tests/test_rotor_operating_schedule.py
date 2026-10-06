from pathlib import Path
import unittest

import numpy as np

from fowt_platform.rotor_operating_schedule import (
    NOMINAL_BELOW_RATED_GENERATING,
    NominalBelowRatedRotorOperatingForecast,
    NominalBelowRatedRotorOperatingPoint,
    NominalBelowRatedRotorSchedule,
    assemble_nominal_below_rated_operating_forecast,
    inspect_nominal_below_rated_operating_inputs,
    load_rosco_nominal_below_rated_schedule_from_zip,
)
from fowt_platform.rotor_performance import RotorPerformanceTable


ROOT = Path(__file__).resolve().parents[1]
MODEL_ZIP = ROOT / (
    "references/fowt_reference_models/3_核心必用_官方模型文件包_IEAWindSystems_"
    "v1.1.16/IEAWindSystems_IEA-15-240-RWT_v1.1.16.zip"
)
MODEL_ARCHIVE_SHA256 = (
    "c96c6be345abf9440170764e2d9a2c201b5da6191192cef66537c5bc16805b80"
)
ROSCO_TUNING_MEMBER = (
    "IEA-15-240-RWT-1.1.16/OpenFAST/IEA-15-240-RWT-UMaineSemi/"
    "IEA-15-240-RWT-UMaineSemi_ROSCO.yaml"
)


def _schedule(*, archive_sha256=None):
    return NominalBelowRatedRotorSchedule(
        cut_in_wind_speed_mps=3.0,
        rated_wind_speed_mps=10.74,
        operational_tip_speed_ratio=9.0,
        minimum_rotor_speed_rad_s=0.523598775,
        rated_rotor_speed_rad_s=0.7916813478,
        minimum_pitch_rad=0.0,
        source_name="synthetic_schedule",
        source_sha256="synthetic_schedule_sha256",
        source_archive_sha256=archive_sha256,
    )


def _table(*, archive_sha256=None):
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
        source_name="synthetic_table",
        source_sha256="synthetic_table_sha256",
        source_archive_sha256=archive_sha256,
    )


def _nominal_mode_kwargs():
    return {
        "operating_mode": NOMINAL_BELOW_RATED_GENERATING,
        "operating_mode_source": "synthetic_declared_generating_fixture",
    }


class NominalBelowRatedRotorScheduleTests(unittest.TestCase):
    def test_tracks_tsr_then_obeys_declared_speed_limits(self):
        schedule = _schedule()

        self.assertAlmostEqual(
            schedule.rotor_speed_rad_s_at(
                normal_inflow_speed_mps=9.0,
                rotor_radius_m=120.0,
            ),
            9.0 * 9.0 / 120.0,
        )
        self.assertAlmostEqual(
            schedule.rotor_speed_rad_s_at(
                normal_inflow_speed_mps=5.0,
                rotor_radius_m=120.0,
            ),
            schedule.minimum_rotor_speed_rad_s,
        )
        self.assertAlmostEqual(
            schedule.rotor_speed_rad_s_at(
                normal_inflow_speed_mps=10.73,
                rotor_radius_m=120.0,
            ),
            schedule.rated_rotor_speed_rad_s,
        )

    def test_rejects_inflow_outside_declared_below_rated_window(self):
        schedule = _schedule()
        with self.assertRaisesRegex(ValueError, "at or below.*cut-in"):
            schedule.rotor_speed_rad_s_at(
                normal_inflow_speed_mps=3.0,
                rotor_radius_m=120.0,
            )
        with self.assertRaisesRegex(ValueError, "at or above.*rated"):
            schedule.rotor_speed_rad_s_at(
                normal_inflow_speed_mps=10.74,
                rotor_radius_m=120.0,
            )

    def test_table_domain_is_not_extrapolated_when_minimum_speed_raises_tsr(self):
        with self.assertRaisesRegex(ValueError, "tip_speed_ratio=.*outside"):
            assemble_nominal_below_rated_operating_forecast(
                schedule=_schedule(),
                performance_table=_table(),
                rotor_radius_m=120.0,
                current_normal_inflow_speed_mps=3.1,
                future_normal_inflow_speeds_mps=[5.0],
                **_nominal_mode_kwargs(),
            )

    def test_operating_domain_report_retains_low_inflow_rejection(self):
        report = inspect_nominal_below_rated_operating_inputs(
            schedule=_schedule(),
            performance_table=_table(),
            rotor_radius_m=120.0,
            current_normal_inflow_speed_mps=3.1,
            future_normal_inflow_speeds_mps=[9.0],
        )

        self.assertFalse(report["supported"])
        self.assertEqual(report["unsupported_record_count"], 1)
        self.assertIn("tip_speed_ratio", report["records"][0]["reason"])
        self.assertTrue(report["records"][1]["supported"])
        self.assertAlmostEqual(
            report["derived_nominal_inflow_range_mps"][0],
            120.0 * 0.523598775 / 14.5,
        )

    def test_resolves_current_and_future_coefficients_from_individual_inflows(self):
        resolved = assemble_nominal_below_rated_operating_forecast(
            schedule=_schedule(),
            performance_table=_table(),
            rotor_radius_m=120.0,
            current_normal_inflow_speed_mps=9.0,
            future_normal_inflow_speeds_mps=[5.0, 9.0],
            **_nominal_mode_kwargs(),
        )

        self.assertAlmostEqual(resolved.current.tip_speed_ratio, 9.0)
        self.assertAlmostEqual(resolved.current.thrust_coefficient, 0.60)
        self.assertGreater(resolved.future[0].tip_speed_ratio, 9.0)
        self.assertAlmostEqual(resolved.future[1].thrust_coefficient, 0.60)
        np.testing.assert_allclose(
            resolved.future_thrust_coefficients,
            [resolved.future[0].thrust_coefficient, 0.60],
        )
        self.assertFalse(resolved.future_thrust_coefficients.flags.writeable)

    def test_rejects_mixed_source_archives(self):
        schedule = _schedule(archive_sha256="schedule_archive")
        point = NominalBelowRatedRotorOperatingPoint(
            normal_inflow_speed_mps=9.0,
            pitch_deg=0.0,
            rotor_speed_rad_s=0.675,
            rotor_speed_rpm=6.445775,
            tip_speed_ratio=9.0,
            thrust_coefficient=0.60,
        )
        with self.assertRaisesRegex(ValueError, "same archive"):
            NominalBelowRatedRotorOperatingForecast(
                schedule=schedule,
                performance_table=_table(archive_sha256="table_archive"),
                current=point,
                future=(point,),
                **_nominal_mode_kwargs(),
            )

    def test_rejects_a_one_sided_archive_binding(self):
        point = NominalBelowRatedRotorOperatingPoint(
            normal_inflow_speed_mps=9.0,
            pitch_deg=0.0,
            rotor_speed_rad_s=0.675,
            rotor_speed_rpm=6.445775,
            tip_speed_ratio=9.0,
            thrust_coefficient=0.60,
        )
        with self.assertRaisesRegex(ValueError, "both bind the same"):
            NominalBelowRatedRotorOperatingForecast(
                schedule=_schedule(archive_sha256="bound_schedule_archive"),
                performance_table=_table(),
                current=point,
                future=(point,),
                **_nominal_mode_kwargs(),
            )

    def test_requires_an_explicit_nominal_generating_mode(self):
        with self.assertRaisesRegex(ValueError, "requires operating_mode"):
            assemble_nominal_below_rated_operating_forecast(
                schedule=_schedule(),
                performance_table=_table(),
                rotor_radius_m=120.0,
                current_normal_inflow_speed_mps=9.0,
                future_normal_inflow_speeds_mps=[9.0],
                operating_mode="inferred_from_wind_speed",
                operating_mode_source="synthetic_invalid_inference",
            )

    def test_reads_published_rosco_below_rated_setpoints_from_frozen_archive(self):
        schedule = load_rosco_nominal_below_rated_schedule_from_zip(
            MODEL_ZIP,
            ROSCO_TUNING_MEMBER,
            expected_archive_sha256=MODEL_ARCHIVE_SHA256,
        )

        self.assertEqual(schedule.source_archive_sha256, MODEL_ARCHIVE_SHA256)
        self.assertAlmostEqual(schedule.cut_in_wind_speed_mps, 3.0)
        self.assertAlmostEqual(schedule.rated_wind_speed_mps, 10.74)
        self.assertAlmostEqual(schedule.operational_tip_speed_ratio, 9.0)
        self.assertAlmostEqual(schedule.minimum_rotor_speed_rad_s, 0.523598775)
        self.assertAlmostEqual(schedule.rated_rotor_speed_rad_s, 0.7916813478)
        self.assertAlmostEqual(schedule.minimum_pitch_deg, 0.0)


if __name__ == "__main__":
    unittest.main()
