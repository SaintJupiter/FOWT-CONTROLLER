import unittest

import numpy as np

from fowt_platform import (
    advance_frozen_open_loop_step,
    BallastModelSnapshot,
    IncrementalLoads,
    IncrementalPlatformModel,
    IncrementalState,
    OpenLoopPlatformStep,
    PlatformMatrices,
    RotorGeneralizedLoad,
    RotorNormalLoad,
)


def _matrices(*, mass_x: float) -> PlatformMatrices:
    return PlatformMatrices(
        mass=np.diag([mass_x, 10.0, 10.0, 10.0, 10.0, 10.0]),
        damping=np.zeros((6, 6)),
        hydrostatic_stiffness=np.zeros((6, 6)),
        mooring_stiffness=np.zeros((6, 6)),
    )


def _snapshot(*, mass_x: float, ballast_load: np.ndarray) -> BallastModelSnapshot:
    masses = np.array([100.0, 100.0, 100.0])
    return BallastModelSnapshot(
        actual_tank_masses_kg=masses,
        reference_tank_masses_kg=masses,
        tank_mass_deltas_kg=np.zeros(3),
        tank_capacities_kg=np.full(3, 1.0e6),
        tank_coordinates_m=np.zeros((3, 3)),
        gravity_m_s2=9.81,
        matrices=_matrices(mass_x=mass_x),
        incremental_ballast_load=ballast_load,
        runtime_provenance="unit_test",
    )


def _rotor_load(generalized_load: np.ndarray) -> RotorGeneralizedLoad:
    return RotorGeneralizedLoad(
        relative_air_velocity_platform_mps=[5.0, 0.0, 0.0],
        normal_load=RotorNormalLoad(
            thrust_n=100.0,
            force_platform_n=np.array([100.0, 0.0, 0.0]),
        ),
        generalized_load_platform=generalized_load,
    )


class OpenLoopPlatformStepTests(unittest.TestCase):
    def test_each_explicit_load_source_enters_its_own_channel_once(self):
        rotor_generalized = np.array([10.0, 0.0, 0.0, 0.0, 20.0, 0.0])
        ballast = np.array([0.0, 0.0, -3.0, 4.0, 0.0, 0.0])
        wave = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 6.0])
        other = np.array([7.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = advance_frozen_open_loop_step(
            snapshot=_snapshot(mass_x=10.0, ballast_load=ballast),
            state=IncrementalState.zeros(),
            rotor_load=_rotor_load(rotor_generalized),
            wave_load=wave,
            other_load=other,
            duration_s=0.0,
        )

        np.testing.assert_allclose(result.loads.wind, rotor_generalized)
        np.testing.assert_allclose(result.loads.ballast, ballast)
        np.testing.assert_allclose(result.loads.wave, wave)
        np.testing.assert_allclose(result.loads.other, other)
        np.testing.assert_allclose(
            result.loads.total,
            rotor_generalized + ballast + wave + other,
        )

    def test_thin_step_matches_direct_frozen_model_advance(self):
        snapshot = _snapshot(
            mass_x=12.0,
            ballast_load=np.array([0.0, 0.0, -3.0, 0.0, 2.0, 0.0]),
        )
        state = IncrementalState.zeros()
        rotor = _rotor_load(np.array([10.0, 0.0, 0.0, 0.0, 15.0, 0.0]))
        wave = np.array([0.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        other = np.zeros(6)

        result = advance_frozen_open_loop_step(
            snapshot=snapshot,
            state=state,
            rotor_load=rotor,
            wave_load=wave,
            other_load=other,
            duration_s=0.5,
        )
        direct_loads = IncrementalLoads(
            wind=rotor.generalized_load_platform,
            wave=wave,
            ballast=snapshot.incremental_ballast_load,
            other=other,
        )
        direct = IncrementalPlatformModel(snapshot.matrices).advance_frozen_step(
            state,
            direct_loads,
            duration_s=0.5,
        )

        np.testing.assert_allclose(result.next_state.position, direct.position)
        np.testing.assert_allclose(result.next_state.velocity, direct.velocity)

    def test_step_uses_the_given_snapshot_matrix_and_ballast_load(self):
        rotor = _rotor_load(np.zeros(6))
        first = advance_frozen_open_loop_step(
            snapshot=_snapshot(mass_x=10.0, ballast_load=np.array([10.0, 0, 0, 0, 0, 0])),
            state=IncrementalState.zeros(),
            rotor_load=rotor,
            wave_load=np.zeros(6),
            other_load=np.zeros(6),
            duration_s=1.0,
        )
        second = advance_frozen_open_loop_step(
            snapshot=_snapshot(mass_x=20.0, ballast_load=np.array([10.0, 0, 0, 0, 0, 0])),
            state=IncrementalState.zeros(),
            rotor_load=rotor,
            wave_load=np.zeros(6),
            other_load=np.zeros(6),
            duration_s=1.0,
        )

        self.assertGreater(first.next_state.position[0], second.next_state.position[0])
        self.assertIsInstance(first, OpenLoopPlatformStep)


if __name__ == "__main__":
    unittest.main()
