import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "validation"
    / "run_openfast_prescribed_wind_relaxation_release_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "openfast_prescribed_wind_relaxation_release_audit", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OpenFastPrescribedWindRelaxationReleaseAuditTests(unittest.TestCase):
    def test_adds_temporary_diagonal_to_existing_hydrodyn_add_blin(self):
        source = (
            "1 0 0 0 0 0 AddBLin - matrix\n"
            "0 2 0 0 0 0\n"
            "0 0 3 0 0 0\n"
            "0 0 0 4 0 0\n"
            "0 0 0 0 5 0\n"
            "0 0 0 0 0 6\n"
            "0 0 0 0 0 0 AddBQuad - next matrix\n"
        )

        updated = MODULE._add_hydrodyn_linear_damping(
            source, np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        )

        lines = updated.splitlines()
        matrix = np.array([[float(value) for value in line.split()[:6]] for line in lines[:6]])
        np.testing.assert_allclose(matrix, np.diag([11.0, 22.0, 33.0, 44.0, 55.0, 66.0]))
        self.assertIn("AddBLin", lines[0])
        self.assertEqual(lines[6], "0 0 0 0 0 0 AddBQuad - next matrix")

    def test_writes_all_six_platform_initial_conditions_with_state_units(self):
        source = "".join(
            f"0 {field} - initial state\n"
            for field in (
                "PtfmSurge",
                "PtfmSway",
                "PtfmHeave",
                "PtfmRoll",
                "PtfmPitch",
                "PtfmYaw",
            )
        )
        state = {
            "surge_m": 1.2,
            "sway_m": -2.3,
            "heave_m": -0.4,
            "roll_deg": 0.5,
            "pitch_deg": -1.6,
            "yaw_deg": 2.7,
        }

        updated = MODULE._set_platform_initial_state(source, state)

        for value, (_, field) in zip(
            (1.2, -2.3, -0.4, 0.5, -1.6, 2.7),
            MODULE.PLATFORM_STATE_FIELDS,
            strict=True,
        ):
            line = next(line for line in updated.splitlines() if field in line)
            self.assertAlmostEqual(float(line.split()[0]), value)

    def test_rejects_tail_output_that_does_not_retain_prescribed_rotor_state(self):
        columns = {
            "Time": np.array([0.0, 1.0, 2.0]),
            "BldPitch1": np.array([1.0, 1.0, 0.8]),
            "BldPitch2": np.array([1.0, 1.0, 1.0]),
            "BldPitch3": np.array([1.0, 1.0, 1.0]),
            "RotSpeed": np.array([7.55, 7.55, 7.55]),
        }

        with self.assertRaisesRegex(ValueError, "does not retain the prescribed blade pitch"):
            MODULE._prescribed_rotor_tail_output(
                columns, duration_s=2.0, tail_window_s=1.0
            )

    def test_rejects_tail_rotor_pitch_variation_with_a_correct_mean(self):
        columns = {
            "Time": np.array([0.0, 1.0]),
            "BldPitch1": np.array([0.9, 1.1]),
            "BldPitch2": np.array([1.0, 1.0]),
            "BldPitch3": np.array([1.0, 1.0]),
            "RotSpeed": np.array([7.55, 7.55]),
        }

        with self.assertRaisesRegex(ValueError, "does not retain the prescribed blade pitch"):
            MODULE._prescribed_rotor_tail_output(
                columns, duration_s=1.0, tail_window_s=1.0
            )

    def test_accepts_a_nondefault_prescribed_rotor_state(self):
        columns = {
            "Time": np.array([0.0, 1.0]),
            "BldPitch1": np.array([1.5, 1.5]),
            "BldPitch2": np.array([1.5, 1.5]),
            "BldPitch3": np.array([1.5, 1.5]),
            "RotSpeed": np.array([5.0, 5.0]),
        }

        output = MODULE._prescribed_rotor_tail_output(
            columns,
            duration_s=1.0,
            tail_window_s=1.0,
            prescribed_blade_pitch_deg=1.5,
            prescribed_rotor_speed_rpm=5.0,
        )

        self.assertEqual(output["blade_pitch_deg"], [1.5, 1.5, 1.5])
        self.assertEqual(output["rotor_speed_rpm"], 5.0)

    def test_transfers_relaxation_late_mean_to_fresh_release_phase(self):
        relaxation_state = {
            "surge_m": 11.0,
            "sway_m": 12.0,
            "heave_m": 13.0,
            "roll_deg": 14.0,
            "pitch_deg": 15.0,
            "yaw_deg": 16.0,
        }
        relaxation = {
            "late_window_statistics": {
                "second_window_mean_state_for_release_only": relaxation_state
            }
        }
        release = {"late_window_statistics": {}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "openfast"
            binary.touch()
            with (
                patch.object(
                    MODULE,
                    "_load_candidate_reference_state",
                    return_value={"surge_m": 0.1, "heave_m": -0.2, "pitch_deg": -0.3},
                ),
                patch.object(MODULE, "_sha256", return_value="f" * 64),
                patch.object(MODULE, "_run_phase", side_effect=[relaxation, release]) as run_phase,
            ):
                result = MODULE.run_audit(
                    work_dir=root / "work",
                    openfast_binary=binary,
                    duration_s=60.0,
                    output_step_s=1.0,
                    wind_speed_mps=5.0,
                    tail_window_s=20.0,
                    prescribed_blade_pitch_deg=1.5,
                    prescribed_rotor_speed_rpm=5.0,
                )

        relaxation_call, release_call = run_phase.call_args_list
        self.assertEqual(relaxation_call.kwargs["phase_root"].name, "relaxation")
        self.assertEqual(release_call.kwargs["phase_root"].name, "release")
        self.assertEqual(
            release_call.kwargs["initial_state"], relaxation_state
        )
        self.assertIsNone(release_call.kwargs["temporary_add_blin_diagonal"])
        self.assertEqual(relaxation_call.kwargs["prescribed_blade_pitch_deg"], 1.5)
        self.assertEqual(relaxation_call.kwargs["prescribed_rotor_speed_rpm"], 5.0)
        self.assertEqual(release_call.kwargs["prescribed_blade_pitch_deg"], 1.5)
        self.assertEqual(release_call.kwargs["prescribed_rotor_speed_rpm"], 5.0)
        self.assertEqual(result["reference"]["wind_speed_mps"], 5.0)
        self.assertEqual(result["reference"]["prescribed_blade_pitch_deg"], 1.5)
        self.assertEqual(result["reference"]["prescribed_rotor_speed_rpm"], 5.0)
        self.assertEqual(result["state_transfer"]["release_initial_state"], relaxation_state)


if __name__ == "__main__":
    unittest.main()
