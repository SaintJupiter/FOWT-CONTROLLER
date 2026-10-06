import unittest

import numpy as np

from fowt_platform import BallastModelSnapshot, IncrementalState, PlatformMatrices
from wind_prediction.execution_rollout import ExecutionTargetOperation
from wind_prediction.posture_feedback_baseline import (
    PostureFeedbackBaselineDesign,
    decide_posture_feedback_target,
    research_posture_feedback_baseline_v1,
    research_posture_feedback_baseline_v2,
)


def _snapshot() -> BallastModelSnapshot:
    return BallastModelSnapshot(
        actual_tank_masses_kg=[100.0, 100.0, 100.0],
        reference_tank_masses_kg=[100.0, 100.0, 100.0],
        tank_mass_deltas_kg=[0.0, 0.0, 0.0],
        tank_capacities_kg=[200.0, 200.0, 200.0],
        tank_coordinates_m=[[-2.0, 0.0, -1.0], [1.0, 2.0, -1.0], [1.0, -2.0, -1.0]],
        gravity_m_s2=9.81,
        matrices=PlatformMatrices(
            mass=np.diag([8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
            damping=np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            hydrostatic_stiffness=np.diag([30.0, 31.0, 32.0, 33.0, 34.0, 35.0]),
            mooring_stiffness=np.zeros((6, 6)),
        ),
        incremental_ballast_load=np.zeros(6),
        runtime_provenance="posture_feedback_fixture",
    )


def _design(*, deadband_deg=(1.0, 0.8), damping=1.0):
    return PostureFeedbackBaselineDesign(
        identity="test_posture_feedback",
        status="test_only",
        deadband_pitch_roll_rad=np.deg2rad(deadband_deg),
        stiffness_feedback_fraction=1.0,
        damping_feedback_fraction=damping,
    )


class PostureFeedbackBaselineTests(unittest.TestCase):
    def test_inside_deadband_releases_to_measured_tank_state(self):
        decision = decide_posture_feedback_target(
            platform_snapshot=_snapshot(),
            platform_state=IncrementalState(
                position=[0.0, 0.0, 0.0, np.deg2rad(0.5), np.deg2rad(0.5), 0.0],
                velocity=np.zeros(6),
            ),
            design=_design(),
        )

        self.assertEqual(
            decision.execution_request.operation,
            ExecutionTargetOperation.RELEASE_TO_CURRENT,
        )
        self.assertEqual(decision.active_axes, (False, False))
        np.testing.assert_allclose(
            decision.requested_ballast_pitch_roll_moment_nm,
            [0.0, 0.0],
        )

    def test_positive_pitch_requests_negative_pitch_ballast_moment(self):
        decision = decide_posture_feedback_target(
            platform_snapshot=_snapshot(),
            platform_state=IncrementalState(
                position=[0.0, 0.0, 0.0, 0.0, np.deg2rad(2.0), 0.0],
                velocity=np.zeros(6),
            ),
            design=_design(damping=0.0),
        )

        self.assertEqual(
            decision.execution_request.operation,
            ExecutionTargetOperation.TRACK,
        )
        self.assertLess(decision.requested_ballast_pitch_roll_moment_nm[0], 0.0)
        self.assertAlmostEqual(decision.requested_ballast_pitch_roll_moment_nm[1], 0.0)
        np.testing.assert_allclose(
            decision.allocation.achieved_pitch_roll_moment_nm,
            decision.requested_ballast_pitch_roll_moment_nm,
        )

    def test_positive_roll_requests_negative_roll_ballast_moment(self):
        decision = decide_posture_feedback_target(
            platform_snapshot=_snapshot(),
            platform_state=IncrementalState(
                position=[0.0, 0.0, 0.0, np.deg2rad(2.0), 0.0, 0.0],
                velocity=np.zeros(6),
            ),
            design=_design(damping=0.0),
        )

        self.assertAlmostEqual(decision.requested_ballast_pitch_roll_moment_nm[0], 0.0)
        self.assertLess(decision.requested_ballast_pitch_roll_moment_nm[1], 0.0)

    def test_velocity_feedback_is_gated_by_posture_deadband(self):
        decision = decide_posture_feedback_target(
            platform_snapshot=_snapshot(),
            platform_state=IncrementalState(
                position=np.zeros(6),
                velocity=[0.0, 0.0, 0.0, 0.4, -0.3, 0.0],
            ),
            design=_design(),
        )

        np.testing.assert_allclose(
            decision.requested_ballast_pitch_roll_moment_nm,
            [0.0, 0.0],
        )

    def test_named_design_records_parameter_identity(self):
        record = research_posture_feedback_baseline_v1().as_dict()

        self.assertEqual(record["identity"], "research_posture_feedback_baseline_v1")
        self.assertEqual(len(record["parameter_sha256"]), 64)
        self.assertEqual(record["deadband_pitch_roll_deg"], [1.0, 0.8])

    def test_v2_baseline_is_explicitly_p_only(self):
        design = research_posture_feedback_baseline_v2()

        self.assertEqual(design.damping_feedback_fraction, 0.0)
        self.assertIn("p_only", design.status)

    def test_block_reachability_scales_target_without_changing_direction(self):
        unlimited = decide_posture_feedback_target(
            platform_snapshot=_snapshot(),
            platform_state=IncrementalState(
                position=[0.0, 0.0, 0.0, 0.0, np.deg2rad(8.0), 0.0],
                velocity=np.zeros(6),
            ),
            design=_design(damping=0.0),
        )
        limited = decide_posture_feedback_target(
            platform_snapshot=_snapshot(),
            platform_state=IncrementalState(
                position=[0.0, 0.0, 0.0, 0.0, np.deg2rad(8.0), 0.0],
                velocity=np.zeros(6),
            ),
            design=_design(damping=0.0),
            maximum_abs_tank_mass_change_per_block_kg=0.05,
        )

        unlimited_delta = (
            unlimited.execution_request.target_masses_kg
            - _snapshot().actual_tank_masses_kg
        )
        limited_delta = (
            limited.execution_request.target_masses_kg
            - _snapshot().actual_tank_masses_kg
        )
        self.assertLess(limited.block_reachability_scale, 1.0)
        self.assertLessEqual(np.max(np.abs(limited_delta)), 0.05 + 1.0e-12)
        np.testing.assert_allclose(
            limited_delta,
            limited.block_reachability_scale * unlimited_delta,
        )
        self.assertAlmostEqual(float(np.sum(limited_delta)), 0.0)


if __name__ == "__main__":
    unittest.main()
