import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "validation"
    / "run_openfast_prescribed_wind_lateral_release_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "openfast_prescribed_wind_lateral_release_audit", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _state(surge: float = 1.0, sway: float = 2.0, yaw: float = 3.0) -> dict[str, float]:
    return {
        "surge_m": surge,
        "sway_m": sway,
        "heave_m": -0.4,
        "roll_deg": 0.5,
        "pitch_deg": -0.6,
        "yaw_deg": yaw,
    }


def _phase(first: dict[str, float], second: dict[str, float]) -> dict[str, object]:
    def statistics(values: dict[str, float]) -> dict[str, dict[str, float]]:
        return {
            name: {"mean": value, "minimum": value, "maximum": value, "range": 0.0}
            for name, value in values.items()
        }

    return {
        "late_window_statistics": {
            "first_window_statistics": statistics(first),
            "second_window_statistics": statistics(second),
        }
    }


def _candidate_reference(wind_speed_mps: float = 10.74) -> dict[str, object]:
    return {
        "model_zip_sha256": MODULE._sha256(MODULE.MODEL_ZIP),
        "wind_speed_mps": wind_speed_mps,
        "prescribed_blade_pitch_deg": 1.0,
        "prescribed_rotor_speed_rpm": 7.55,
        "wave_and_current": "disabled",
    }


class OpenFastPrescribedWindLateralReleaseAuditTests(unittest.TestCase):
    def test_loads_only_the_explicit_nonreference_candidate_path(self):
        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / "audit.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "reference": _candidate_reference(),
                        "temporary_relaxation": {
                            "late_window_statistics": {
                                "second_window_mean_state_for_release_only": _state()
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            loaded = MODULE._load_candidate_state(audit_path)

        self.assertEqual(loaded, _state())

    def test_rejects_candidate_from_a_different_prescribed_wind_condition(self):
        with self.assertRaisesRegex(ValueError, "wind_speed_mps=9.0"):
            MODULE._validate_candidate_operating_condition(
                _candidate_reference(wind_speed_mps=9.0), wind_speed_mps=10.74
            )

    def test_rejects_candidate_with_nonmatching_rotor_condition(self):
        reference = _candidate_reference()
        reference["wave_and_current"] = "enabled"

        with self.assertRaisesRegex(ValueError, "wave_and_current='enabled'"):
            MODULE._validate_candidate_operating_condition(
                reference, wind_speed_mps=10.74
            )

    def test_changes_only_requested_lateral_state(self):
        changed = MODULE._perturb_state(_state(), field="sway_m", value=0.1)

        self.assertEqual(changed["sway_m"], 2.1)
        self.assertEqual(changed["yaw_deg"], 3.0)
        self.assertEqual(changed["pitch_deg"], -0.6)

    def test_rejects_zero_and_nonlateral_perturbations(self):
        with self.assertRaisesRegex(ValueError, "finite and nonzero"):
            MODULE._perturb_state(_state(), field="sway_m", value=0.0)
        with self.assertRaisesRegex(ValueError, "must be one of"):
            MODULE._perturb_state(_state(), field="surge_m", value=0.1)

    def test_reports_relative_tail_state_change(self):
        unperturbed = _phase(_state(sway=2.0), _state(sway=2.0))
        perturbed = _phase(_state(sway=2.1), _state(sway=2.03))

        result = MODULE._relative_tail_state_change(
            unperturbed=unperturbed, perturbed=perturbed
        )

        self.assertAlmostEqual(
            result["first_window_perturbed_minus_unperturbed"]["sway_m"], 0.1
        )
        self.assertAlmostEqual(
            result["second_window_perturbed_minus_unperturbed"]["sway_m"], 0.03
        )
        self.assertAlmostEqual(
            result["second_minus_first_relative_state"]["sway_m"], -0.07
        )

    def test_runs_two_fresh_original_damping_releases(self):
        candidate = _state()
        relaxation_audit = {
            "reference": _candidate_reference(),
            "temporary_relaxation": {
                "late_window_statistics": {
                    "second_window_mean_state_for_release_only": candidate
                }
            }
        }
        unperturbed = _phase(candidate, candidate)
        perturbed = _phase(_state(sway=2.1), _state(sway=2.05))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "openfast"
            binary.touch()
            audit_path = root / "relaxation-audit.json"
            audit_path.write_text(json.dumps(relaxation_audit), encoding="utf-8")
            with (
                patch.object(
                    MODULE,
                    "_sha256",
                    return_value=relaxation_audit["reference"]["model_zip_sha256"],
                ),
                patch.object(
                    MODULE, "_run_phase", side_effect=[unperturbed, perturbed]
                ) as run_phase,
            ):
                result = MODULE.run_audit(
                    work_dir=root / "work",
                    openfast_binary=binary,
                    relaxation_audit_path=audit_path,
                    duration_s=60.0,
                    output_step_s=1.0,
                    wind_speed_mps=10.74,
                    tail_window_s=20.0,
                    perturbation_field="sway_m",
                    perturbation_value=0.1,
                )

        base_call, perturbed_call = run_phase.call_args_list
        self.assertEqual(base_call.kwargs["phase_root"].name, "unperturbed")
        self.assertEqual(perturbed_call.kwargs["phase_root"].name, "perturbed")
        self.assertIsNone(base_call.kwargs["temporary_add_blin_diagonal"])
        self.assertIsNone(perturbed_call.kwargs["temporary_add_blin_diagonal"])
        self.assertEqual(base_call.kwargs["initial_state"], candidate)
        self.assertEqual(perturbed_call.kwargs["initial_state"]["sway_m"], 2.1)
        self.assertEqual(result["perturbation"]["field"], "sway_m")


if __name__ == "__main__":
    unittest.main()
