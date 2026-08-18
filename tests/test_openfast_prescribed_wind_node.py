import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validation" / "run_openfast_prescribed_wind_node.py"
SPEC = importlib.util.spec_from_file_location("openfast_prescribed_wind_node", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OpenFastPrescribedWindNodeTests(unittest.TestCase):
    def test_loads_candidate_reference_state_only_when_source_archive_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "model.zip"
            archive.write_bytes(b"frozen model fixture")
            state_path = root / "reference_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_model": {
                            "archive_sha256": hashlib.sha256(
                                archive.read_bytes()
                            ).hexdigest()
                        },
                        "state": {
                            "surge_m": 0.4198,
                            "heave_m": -0.3503,
                            "pitch_deg": -1.452,
                        },
                    }
                ),
                encoding="utf-8",
            )

            state = MODULE._load_candidate_reference_state(
                state_path, model_zip=archive
            )

        self.assertEqual(
            state,
            {"surge_m": 0.4198, "heave_m": -0.3503, "pitch_deg": -1.452},
        )

    def test_rejects_candidate_reference_state_from_another_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "model.zip"
            archive.write_bytes(b"frozen model fixture")
            state_path = root / "reference_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_model": {"archive_sha256": "0" * 64},
                        "state": {
                            "surge_m": 0.4198,
                            "heave_m": -0.3503,
                            "pitch_deg": -1.452,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "archive SHA-256"):
                MODULE._load_candidate_reference_state(state_path, model_zip=archive)

    def test_rejects_incomplete_candidate_reference_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "model.zip"
            archive.write_bytes(b"frozen model fixture")
            state_path = root / "reference_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_model": {
                            "archive_sha256": hashlib.sha256(
                                archive.read_bytes()
                            ).hexdigest()
                        },
                        "state": {"surge_m": 0.4198, "heave_m": -0.3503},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "pitch_deg must be numeric"):
                MODULE._load_candidate_reference_state(state_path, model_zip=archive)

    def test_replaces_only_the_value_before_a_named_openfast_field(self):
        source = "0                      NumCrctn    - control corrections\n"

        updated = MODULE._replace_field(source, "NumCrctn", "2")

        self.assertEqual(updated, "2                      NumCrctn    - control corrections\n")

    def test_missing_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing OpenFAST field"):
            MODULE._replace_field("0 Value - example\n", "NumCrctn", "2")

    def test_reads_openfast_text_output_by_channel_name(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case.out"
            output.write_text(
                "OpenFAST test\n"
                "Time PtfmSurge RotThrust\n"
                "(s) (m) (kN)\n"
                "0.0 1.0 2.0\n"
                "1.0 3.0 4.0\n",
                encoding="utf-8",
            )

            columns = MODULE._read_openfast_columns(output)

        np.testing.assert_allclose(columns["Time"], [0.0, 1.0])
        np.testing.assert_allclose(columns["PtfmSurge"], [1.0, 3.0])
        np.testing.assert_allclose(columns["RotThrust"], [2.0, 4.0])

    def test_reads_openfast_text_output_units_by_channel_name(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case.out"
            output.write_text(
                "OpenFAST test\n"
                "Time RtFldFxh RtFldMxh\n"
                "(s) (kN) (kN-m)\n"
                "0.0 1.0 2.0\n",
                encoding="utf-8",
            )

            columns, units = MODULE._read_openfast_output(output)

        np.testing.assert_allclose(columns["RtFldFxh"], [1.0])
        self.assertEqual(units["RtFldMxh"], "(kN-m)")

    def test_rejects_partial_or_nonmonotonic_openfast_output_time_series(self):
        with self.assertRaisesRegex(ValueError, "does not reach the requested time horizon"):
            MODULE._assert_output_reaches_time_horizon(
                {"Time": np.array([0.0, 1.0, 2.0])},
                duration_s=5.0,
                output_step_s=1.0,
            )
        with self.assertRaisesRegex(ValueError, "must be strictly increasing"):
            MODULE._assert_output_reaches_time_horizon(
                {"Time": np.array([0.0, 1.0, 1.0, 2.0])},
                duration_s=2.0,
                output_step_s=1.0,
            )

    def test_accepts_openfast_output_that_reaches_requested_time_horizon(self):
        MODULE._assert_output_reaches_time_horizon(
            {"Time": np.array([0.0, 1.0, 2.0, 3.0])},
            duration_s=3.0,
            output_step_s=1.0,
        )
        MODULE._assert_output_reaches_time_horizon(
            {"Time": np.array([0.0, 1.0, 2.0, 3.0])},
            duration_s=3.7,
            output_step_s=1.0,
        )

    def test_recognizes_explicit_aerodyn_hub_load_scales(self):
        self.assertEqual(
            MODULE._hub_wrench_unit_scales(
                {
                    "RtFldFxh": "(N)",
                    "RtFldFyh": "(N)",
                    "RtFldFzh": "(N)",
                    "RtFldMxh": "(N-m)",
                    "RtFldMyh": "(N-m)",
                    "RtFldMzh": "(N-m)",
                }
            ),
            (1.0, 1.0),
        )
        self.assertEqual(
            MODULE._hub_wrench_unit_scales(
                {
                    "RtFldFxh": "(kN)",
                    "RtFldFyh": "(kN)",
                    "RtFldFzh": "(kN)",
                    "RtFldMxh": "(kN-m)",
                    "RtFldMyh": "(kN-m)",
                    "RtFldMzh": "(kN-m)",
                }
            ),
            (1_000.0, 1_000.0),
        )

    def test_rejects_unexpected_aerodyn_hub_load_units(self):
        with self.assertRaisesRegex(ValueError, "unexpected AeroDyn hub-force units"):
            MODULE._hub_wrench_unit_scales(
                {
                    "RtFldFxh": "(MN)",
                    "RtFldFyh": "(MN)",
                    "RtFldFzh": "(MN)",
                    "RtFldMxh": "(N-m)",
                    "RtFldMyh": "(N-m)",
                    "RtFldMzh": "(N-m)",
                }
            )

    def test_rejects_non_degree_hub_conversion_angle_units(self):
        with self.assertRaisesRegex(ValueError, "must use '\\(deg\\)'"):
            MODULE._assert_hub_conversion_angle_units(
                {
                    "Azimuth": "(rad)",
                    "PtfmRoll": "(deg)",
                    "PtfmPitch": "(deg)",
                    "PtfmYaw": "(deg)",
                    "NacYaw": "(deg)",
                }
            )

    def test_rejects_a_nonpositive_or_reversed_wind_pair_before_running_openfast(self):
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory) / "work"
            binary = Path(directory) / "openfast"
            binary.touch()
            with self.assertRaisesRegex(ValueError, "must exceed"):
                MODULE.run_audit(
                    work_dir=work_dir,
                    openfast_binary=binary,
                    duration_s=10.0,
                    output_step_s=1.0,
                    wind_speed_mps=0.1,
                    baseline_wind_speed_mps=0.1,
                )

    def test_passes_an_explicit_rotor_state_to_both_openfast_cases(self):
        case = {
            "second_half_mean": {"surge_m": 0.0, "heave_m": 0.0, "pitch_deg": 0.0},
            "aerodynamic_hub_load_about_platform_reference_second_half_mean": {
                "generalized_load_si": {"force_n": [0.0, 0.0, 0.0], "moment_nm": [0.0, 0.0, 0.0]}
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "openfast"
            binary.touch()
            with (
                unittest.mock.patch.object(
                    MODULE,
                    "_load_candidate_reference_state",
                    return_value={"surge_m": 0.0, "heave_m": 0.0, "pitch_deg": 0.0},
                ),
                unittest.mock.patch.object(MODULE, "_sha256", return_value="f" * 64),
                unittest.mock.patch.object(MODULE, "_run_case", side_effect=[case, case]) as run_case,
            ):
                result = MODULE.run_audit(
                    work_dir=root / "work",
                    openfast_binary=binary,
                    duration_s=10.0,
                    output_step_s=1.0,
                    wind_speed_mps=5.0,
                    prescribed_blade_pitch_deg=1.0,
                    prescribed_rotor_speed_rpm=5.0,
                )

        for call in run_case.call_args_list:
            self.assertEqual(call.kwargs["prescribed_blade_pitch_deg"], 1.0)
            self.assertEqual(call.kwargs["prescribed_rotor_speed_rpm"], 5.0)
        self.assertEqual(result["reference"]["prescribed_blade_pitch_deg"], 1.0)
        self.assertEqual(result["reference"]["prescribed_rotor_speed_rpm"], 5.0)

    def test_rejects_output_that_does_not_retain_the_prescribed_rotor_state(self):
        with self.assertRaisesRegex(ValueError, "does not retain the prescribed blade pitch"):
            MODULE._assert_prescribed_rotor_output(
                blade_pitch_deg=[0.9, 1.0, 1.0],
                rotor_speed_rpm=7.55,
            )
        with self.assertRaisesRegex(ValueError, "does not retain the prescribed rotor speed"):
            MODULE._assert_prescribed_rotor_output(
                blade_pitch_deg=[1.0, 1.0, 1.0],
                rotor_speed_rpm=7.54,
            )

    def test_rejects_tail_rotor_speed_variation_with_a_correct_mean(self):
        with self.assertRaisesRegex(ValueError, "does not retain the prescribed rotor speed"):
            MODULE._assert_prescribed_rotor_output(
                blade_pitch_deg=[
                    np.array([1.0, 1.0]),
                    np.array([1.0, 1.0]),
                    np.array([1.0, 1.0]),
                ],
                rotor_speed_rpm=np.array([4.9, 5.1]),
                prescribed_blade_pitch_deg=1.0,
                prescribed_rotor_speed_rpm=5.0,
            )

    def test_accepts_an_explicit_nondefault_prescribed_rotor_state(self):
        MODULE._assert_prescribed_rotor_output(
            blade_pitch_deg=[1.5, 1.5, 1.5],
            rotor_speed_rpm=5.0,
            prescribed_blade_pitch_deg=1.5,
            prescribed_rotor_speed_rpm=5.0,
        )


if __name__ == "__main__":
    unittest.main()
