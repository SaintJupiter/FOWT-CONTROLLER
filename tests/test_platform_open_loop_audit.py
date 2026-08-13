import tempfile
import unittest
from pathlib import Path


from scripts.validation.run_platform_open_loop_audit import (
    AuditSettings,
    DEFAULT_THRUST_CURVE,
    LEGACY_DIR,
    run_audit,
)
from wind_prediction.input_files import resolve_mooring_stiffness_file


class PlatformOpenLoopAuditTests(unittest.TestCase):
    def test_short_audit_writes_traceable_outputs(self):
        stiffness_file = resolve_mooring_stiffness_file(
            search_dirs=(LEGACY_DIR / "data",),
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            summary = run_audit(
                output_dir=output_dir,
                stiffness_file=stiffness_file,
                thrust_curve_file=DEFAULT_THRUST_CURVE,
                platform_profile="default",
                settings=AuditSettings(
                    dt_s=0.2,
                    settle_duration_s=2.0,
                    decay_duration_s=2.0,
                    sample_interval_s=0.4,
                ),
            )

            self.assertTrue(all(summary["checks"].values()))
            self.assertEqual(summary["mooring_mode"], "TABLE_SIGNED")
            self.assertEqual(
                summary["evidence_level"],
                "internal_mathematical_consistency_not_physical_validation",
            )
            self.assertFalse(summary["absolute_static_equilibrium_solved"])
            for name in (
                "audit_summary.json",
                "load_channel_scan.csv",
                "zero_load_settle.csv",
                "free_decay_pitch.csv",
                "free_decay_roll.csv",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

    def test_incremental_profile_audits_signed_ballast_chain(self):
        stiffness_file = resolve_mooring_stiffness_file(
            search_dirs=(LEGACY_DIR / "data",),
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            summary = run_audit(
                output_dir=output_dir,
                stiffness_file=stiffness_file,
                thrust_curve_file=DEFAULT_THRUST_CURVE,
                platform_profile="research_incremental_v1",
                settings=AuditSettings(
                    dt_s=0.2,
                    settle_duration_s=2.0,
                    decay_duration_s=2.0,
                    sample_interval_s=0.4,
                ),
            )

            self.assertTrue(summary["ballast_increment_audit_applicable"])
            self.assertTrue(all(summary["checks"].values()))
            self.assertTrue(
                summary["checks"]["ballast_balanced_exchange_preserves_total_mass"]
            )
            self.assertTrue(
                summary["checks"]["ballast_balanced_exchange_moment_direction"]
            )
            self.assertTrue((output_dir / "ballast_increment_scan.csv").is_file())


if __name__ == "__main__":
    unittest.main()
