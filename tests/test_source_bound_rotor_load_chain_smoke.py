import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validation" / "run_fowt_source_bound_rotor_chain_smoke.py"
SPEC = importlib.util.spec_from_file_location("source_bound_rotor_chain_smoke", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SourceBoundRotorLoadChainSmokeTests(unittest.TestCase):
    def test_routes_the_explicit_rotor_load_only_through_the_wind_channel(self):
        result = MODULE.run_smoke(
            wind_speed_mps=5.0,
            blade_pitch_deg=1.0,
            rotor_speed_rpm=5.0,
        )

        self.assertTrue(all(result["checks"].values()))
        self.assertGreater(result["source_bound_wind_load"]["surge_n"], 0.0)
        self.assertGreater(result["source_bound_wind_load"]["pitch_nm"], 0.0)
        self.assertIn("model_zip_sha256", result["source_bound_rotor_audit"]["source"])
        self.assertTrue(result["source_bound_rotor_audit"]["checks"]["all_checks_pass"])
        self.assertIn("translation_n", result["internal_static_balance_only"]["residual_max_abs"])
        self.assertIn(
            "rotation_radps2",
            result["internal_static_balance_only"]["balanced_acceleration_max_abs"],
        )
        self.assertFalse(result["boundaries"]["is_platform_dynamic_response_validation"])
        self.assertFalse(result["boundaries"]["is_static_offset_validation"])

    def test_rejects_a_failed_upstream_rotor_audit_before_assembling_the_platform(self):
        with patch.object(
            MODULE,
            "run_rotor_load_audit",
            return_value={"checks": {"all_checks_pass": False}},
        ):
            with self.assertRaisesRegex(ValueError, "did not pass"):
                MODULE.run_smoke()


if __name__ == "__main__":
    unittest.main()
