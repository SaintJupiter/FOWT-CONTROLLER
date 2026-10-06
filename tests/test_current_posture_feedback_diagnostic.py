import ast
from pathlib import Path
import unittest

import numpy as np

from fowt_platform import (
    BallastModelSnapshot,
    IncrementalLoads,
    IncrementalPlatformModel,
    IncrementalState,
    PlatformMatrices,
    diagnose_current_posture_passive_load,
)


def _snapshot() -> BallastModelSnapshot:
    mass = np.diag([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    damping = np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    restoring = np.diag([30.0, 31.0, 32.0, 33.0, 34.0, 35.0])
    # Keep the full six-DOF multiplication visible: surge displacement and
    # heave velocity both contribute to the pitch component through coupling.
    restoring[4, 0] = restoring[0, 4] = 2.0
    damping[3, 2] = damping[2, 3] = 1.5
    return BallastModelSnapshot(
        actual_tank_masses_kg=[100.0, 100.0, 100.0],
        reference_tank_masses_kg=[100.0, 100.0, 100.0],
        tank_mass_deltas_kg=[0.0, 0.0, 0.0],
        tank_capacities_kg=[200.0, 200.0, 200.0],
        tank_coordinates_m=[[-2.0, 0.0, -1.0], [1.0, 2.0, -1.0], [1.0, -2.0, -1.0]],
        gravity_m_s2=9.81,
        matrices=PlatformMatrices(
            mass=mass,
            damping=damping,
            hydrostatic_stiffness=restoring,
            mooring_stiffness=np.zeros((6, 6)),
        ),
        incremental_ballast_load=np.zeros(6),
        runtime_provenance="current_posture_fixture",
    )


class CurrentPosturePassiveLoadDiagnosticTests(unittest.TestCase):
    def test_matches_the_passive_right_hand_side_of_the_frozen_linear_model(self):
        snapshot = _snapshot()
        state = IncrementalState(
            position=[1.5, 0.0, 0.0, 0.2, -0.3, 0.0],
            velocity=[0.0, 0.0, 0.4, -0.05, 0.06, 0.0],
        )

        diagnostic = diagnose_current_posture_passive_load(
            platform_snapshot=snapshot,
            platform_state=state,
        )
        derivative = IncrementalPlatformModel(snapshot.matrices).derivative(
            state,
            IncrementalLoads.zeros(),
        )
        expected_passive = (
            snapshot.matrices.mass @ derivative.velocity_rate
        )[[4, 3]]

        np.testing.assert_allclose(
            diagnostic.current_passive_pitch_roll_load_nm,
            expected_passive,
        )
        np.testing.assert_allclose(
            diagnostic.current_passive_pitch_roll_load_nm,
            diagnostic.current_restoring_pitch_roll_load_nm
            + diagnostic.current_damping_pitch_roll_load_nm,
        )
        self.assertAlmostEqual(
            diagnostic.current_restoring_pitch_roll_load_nm[0],
            -(34.0 * -0.3 + 2.0 * 1.5),
        )
        self.assertAlmostEqual(
            diagnostic.current_damping_pitch_roll_load_nm[1],
            -(5.0 * -0.05 + 1.5 * 0.4),
        )

    def test_zero_state_has_no_passive_pitch_roll_contribution(self):
        diagnostic = diagnose_current_posture_passive_load(
            platform_snapshot=_snapshot(),
            platform_state=IncrementalState.zeros(),
        )
        np.testing.assert_allclose(
            diagnostic.current_passive_pitch_roll_load_nm,
            [0.0, 0.0],
        )

    def test_requires_explicit_frozen_snapshot_and_incremental_state(self):
        with self.assertRaises(TypeError):
            diagnose_current_posture_passive_load(
                platform_snapshot=object(),
                platform_state=IncrementalState.zeros(),
            )
        with self.assertRaises(TypeError):
            diagnose_current_posture_passive_load(
                platform_snapshot=_snapshot(),
                platform_state=object(),
            )

    def test_module_does_not_import_controller_or_candidate_logic(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "fowt_platform"
            / "current_posture_feedback_diagnostic.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        forbidden_fragments = ("controller", "candidate", "planner", "action")
        self.assertFalse(
            any(
                any(fragment in module for fragment in forbidden_fragments)
                for module in imported_modules
            )
        )


if __name__ == "__main__":
    unittest.main()
