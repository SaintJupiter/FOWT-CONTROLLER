import unittest

import numpy as np

from fowt_platform import (
    IncrementalLoads,
    IncrementalPlatformModel,
    IncrementalState,
    PlatformMatrices,
)
from wind_prediction.preview_mpc import PreviewBlockModel, PreviewState


class PreviewBlockModelTests(unittest.TestCase):
    def setUp(self):
        self.matrices = PlatformMatrices(
            mass=np.diag([10.0, 12.0, 14.0, 20.0, 22.0, 24.0]),
            damping=np.diag([1.0, 1.0, 2.0, 3.0, 3.0, 1.0]),
            hydrostatic_stiffness=np.diag([0.0, 0.0, 30.0, 40.0, 50.0, 0.0]),
            mooring_stiffness=np.diag([4.0, 5.0, 0.0, 0.0, 0.0, 6.0]),
        )
        modal_load = np.zeros((6, 2))
        modal_load[4, 0] = 8.0
        modal_load[3, 1] = 6.0
        self.model = PreviewBlockModel(
            matrices=self.matrices,
            generalized_load_per_mode_kg=modal_load,
            block_duration_s=10.0,
            sample_fractions=(0.25, 0.5, 0.75, 1.0),
        )

    def test_zero_modal_increment_matches_existing_exact_platform_step(self):
        platform = IncrementalState(
            position=np.array([0.1, 0.0, -0.2, 0.03, -0.02, 0.0]),
            velocity=np.array([0.0, 0.05, 0.0, 0.0, 0.01, 0.0]),
        )
        load = np.array([2.0, -1.0, 0.5, 4.0, -3.0, 0.0])

        preview = self.model.propagate_block(
            initial_state=PreviewState.from_current_platform(platform),
            modal_increment_kg=np.zeros(2),
            generalized_disturbance_load=load,
        ).endpoint
        direct = IncrementalPlatformModel(self.matrices).advance_frozen_step(
            platform,
            IncrementalLoads(
                wind=load,
                wave=np.zeros(6),
                ballast=np.zeros(6),
                other=np.zeros(6),
            ),
            duration_s=10.0,
        )

        np.testing.assert_allclose(preview.platform.position, direct.position, atol=1e-12)
        np.testing.assert_allclose(preview.platform.velocity, direct.velocity, atol=1e-12)
        np.testing.assert_allclose(preview.modal_redistribution_kg, np.zeros(2))

    def test_modal_increment_is_a_linear_ramp_and_reaches_endpoint(self):
        block = self.model.propagate_block(
            initial_state=PreviewState.from_current_platform(IncrementalState.zeros()),
            modal_increment_kg=np.array([40.0, -20.0]),
            generalized_disturbance_load=np.zeros(6),
        )

        np.testing.assert_allclose(
            block.state_vectors[:, 12:14],
            np.array(
                [
                    [10.0, -5.0],
                    [20.0, -10.0],
                    [30.0, -15.0],
                    [40.0, -20.0],
                ]
            ),
            atol=1e-12,
        )
        self.assertGreater(float(np.max(np.abs(block.pitch_rad))), 0.0)
        self.assertGreater(float(np.max(np.abs(block.roll_rad))), 0.0)

    def test_ramp_solution_matches_fine_piecewise_constant_integration(self):
        increment = np.array([30.0, -15.0])
        external = np.array([0.0, 0.0, 0.0, 2.0, -3.0, 0.0])
        exact = self.model.propagate_block(
            initial_state=PreviewState.from_current_platform(IncrementalState.zeros()),
            modal_increment_kg=increment,
            generalized_disturbance_load=external,
        ).endpoint

        fine_state = IncrementalState.zeros()
        steps = 1_000
        dt = 10.0 / steps
        modal_load = np.zeros((6, 2))
        modal_load[4, 0] = 8.0
        modal_load[3, 1] = 6.0
        for index in range(steps):
            midpoint_fraction = (index + 0.5) / steps
            load = external + modal_load @ (midpoint_fraction * increment)
            fine_state = IncrementalPlatformModel(self.matrices).advance_frozen_step(
                fine_state,
                IncrementalLoads(
                    wind=external,
                    wave=np.zeros(6),
                    ballast=load - external,
                    other=np.zeros(6),
                ),
                duration_s=dt,
            )

        np.testing.assert_allclose(exact.platform.position, fine_state.position, atol=2e-5)
        np.testing.assert_allclose(exact.platform.velocity, fine_state.velocity, atol=2e-5)

    def test_rollout_retains_within_block_posture_not_only_endpoints(self):
        trajectory = self.model.rollout(
            initial_state=PreviewState.from_current_platform(IncrementalState.zeros()),
            modal_increments_kg=np.array([[20.0, 0.0], [-20.0, 10.0]]),
            generalized_disturbance_loads=np.zeros((2, 6)),
        )

        self.assertEqual(len(trajectory.blocks), 2)
        self.assertGreater(trajectory.sampled_posture_energy(), 0.0)
        self.assertGreater(trajectory.maximum_dominant_tilt_rad, 0.0)
        np.testing.assert_allclose(
            trajectory.endpoint_states[-1].modal_redistribution_kg,
            [0.0, 10.0],
            atol=1e-12,
        )

    def test_default_600_second_block_checks_posture_every_five_seconds(self):
        modal_load = np.zeros((6, 2))
        model = PreviewBlockModel(
            matrices=self.matrices,
            generalized_load_per_mode_kg=modal_load,
            block_duration_s=600.0,
        )

        elapsed = model.sample_fractions * model.block_duration_s
        self.assertEqual(elapsed.size, 120)
        np.testing.assert_allclose(np.diff(np.concatenate(([0.0], elapsed))), 5.0)


if __name__ == "__main__":
    unittest.main()
