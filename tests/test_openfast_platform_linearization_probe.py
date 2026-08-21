import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validation" / "run_openfast_platform_linearization_probe.py"
SPEC = importlib.util.spec_from_file_location("openfast_platform_linearization_probe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OpenFastPlatformLinearizationProbeTests(unittest.TestCase):
    def test_configures_one_absolute_time_linearization(self):
        source = (
            "False Linearize - flag\n"
            "False CalcSteady - flag\n"
            "2 NLinTimes - count\n"
            "30, 60 LinTimes - values\n"
            "1 LinInputs - input selection\n"
            "1 LinOutputs - output selection\n"
            "False LinOutJac - jacobians\n"
            "False LinOutMod - modules\n"
            "3 CompMooring - mooring stays active\n"
        )
        updated = MODULE._configure_linearization_settings(
            source,
            linearization_time_s=60.0,
        )
        self.assertIn("True                   Linearize", updated)
        self.assertIn("False                  CalcSteady", updated)
        self.assertIn("1                      NLinTimes", updated)
        self.assertIn("60                     LinTimes", updated)
        self.assertIn("0                      LinInputs", updated)
        self.assertIn("1                      LinOutputs", updated)
        self.assertIn("3 CompMooring", updated)

    def test_rigid_stationary_configuration_disables_only_nonplatform_states(self):
        source = "\n".join(
            [
                "True FlapDOF1 - state",
                "True FlapDOF2 - state",
                "True EdgeDOF - state",
                "True DrTrDOF - state",
                "True GenDOF - state",
                "True YawDOF - state",
                "True TwFADOF1 - state",
                "True TwFADOF2 - state",
                "True TwSSDOF1 - state",
                "True TwSSDOF2 - state",
                "7.55 RotSpeed - state",
                "True PtfmPDOF - platform remains active",
            ]
        )
        updated = MODULE._configure_stationary_rigid_elastodyn(source)
        self.assertEqual(updated.count("False"), 10)
        self.assertIn("0.0                    RotSpeed", updated)
        self.assertIn("True PtfmPDOF", updated)

    def test_disables_second_order_wave_terms_for_linearization(self):
        source = "True WvDiffQTF - difference\nTrue WvSumQTF - summation\n"
        updated = MODULE._disable_second_order_wave_terms_for_linearization(source)
        self.assertIn("False                  WvDiffQTF", updated)
        self.assertIn("False                  WvSumQTF", updated)

    def test_configures_hydrodyn_paths_supported_by_first_probe(self):
        source = (
            "1 ExctnMod - excitation\n"
            "1 RdtnMod - radiation\n"
            "12 DiffQTF - second order\n"
            "1 PotMod - potential flow remains enabled\n"
        )
        updated = MODULE._configure_hydrodyn_for_local_linearization(source)
        self.assertIn("0                      ExctnMod", updated)
        self.assertIn("0                      RdtnMod", updated)
        self.assertIn("0                      DiffQTF", updated)
        self.assertIn("1 PotMod", updated)

    def test_collects_only_platform_pitch_and_roll_mentions(self):
        path = ROOT / ".tmp_linearization_mentions.lin"
        try:
            path.write_text(
                "PtfmPitch, PtfmPitch velocity\nOtherState\nPtfmRoll\n",
                encoding="utf-8",
            )
            self.assertEqual(
                MODULE._platform_linearization_mentions(path),
                ["PtfmPitch, PtfmPitch velocity", "PtfmRoll"],
            )
        finally:
            path.unlink(missing_ok=True)

    def test_extracts_only_high_level_linearization_dimensions(self):
        path = ROOT / ".tmp_linearization_summary.lin"
        try:
            path.write_text(
                "Number of continuous states:       18\n"
                "Number of discrete states:          0\n"
                "Number of inputs:                   0\n"
                "Number of outputs:                 12\n"
                "1 0 F 2 ED Platform pitch tilt rotation DOF (internal DOF index = DOF_P), rad\n"
                "2 0 F 2 ED Platform roll tilt rotation DOF (internal DOF index = DOF_R), rad\n"
                "3 0 F 0 ED PtfmPitch, (deg)\n",
                encoding="utf-8",
            )
            summary = MODULE._linearization_summary(path)
            self.assertEqual(summary["continuous_states"], 18)
            self.assertEqual(summary["discrete_states"], 0)
            self.assertEqual(summary["inputs"], 0)
            self.assertEqual(summary["outputs"], 12)
            self.assertEqual(len(summary["platform_state_lines"]), 2)
            self.assertEqual(
                summary["platform_pitch_roll_output_lines"],
                ["3 0 F 0 ED PtfmPitch, (deg)"],
            )
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
