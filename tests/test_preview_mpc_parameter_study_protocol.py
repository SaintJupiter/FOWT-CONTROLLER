from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIRECTORY = ROOT / "scripts" / "validation"
if str(VALIDATION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIRECTORY))

from preview_mpc_parameter_study_protocol import (  # noqa: E402
    ParameterStudyCase,
    ParameterStudyObservation,
    ParameterStudyProtocol,
    ParameterStudyStage,
    freeze_candidate_registry,
    summarize_paired_candidate,
)
from run_preview_mpc_parameter_study import (  # noqa: E402
    _screening_configuration_sha256,
    default_parameter_study_protocol,
    dry_run,
    predeclared_screening_candidate_registry,
)
from preview_mpc_continuous_experiment import implementation_identity  # noqa: E402
from wind_prediction.preview_mpc_design import (  # noqa: E402
    research_preview_mpc_design_v2,
    research_preview_mpc_design_v3,
)


class PreviewMPCParameterStudyProtocolTests(unittest.TestCase):
    def test_default_protocol_separates_smoke_from_future_selection(self):
        protocol = default_parameter_study_protocol()

        self.assertTrue(protocol.stage("chain_smoke").readiness["ready"])
        self.assertFalse(
            protocol.stage("chain_smoke").permits_parameter_selection
        )
        self.assertTrue(protocol.stage("multi_regime_screening").readiness["ready"])
        self.assertTrue(
            protocol.stage("multi_regime_screening").permits_parameter_selection
        )
        self.assertTrue(protocol.stage("independent_confirmation").readiness["ready"])

    def test_candidate_identity_changes_without_changing_base_design(self):
        protocol = default_parameter_study_protocol()
        design = research_preview_mpc_design_v3()
        baseline = protocol.candidate(
            candidate_id="baseline",
            base_design=design,
        )
        variant = protocol.candidate(
            candidate_id="throughput_x2",
            base_design=design,
            priority_multipliers={"tank_throughput": 2.0},
        )

        self.assertNotEqual(baseline.parameter_sha256, variant.parameter_sha256)
        self.assertEqual(design.tank_throughput_priority, 5.0e-4)
        with self.assertRaises(ValueError):
            protocol.candidate(
                candidate_id="unsafe_limit_change",
                base_design=design,
                priority_multipliers={"running_posture": 2.0},
            )
        with self.assertRaises(ValueError):
            protocol.candidate(
                candidate_id="unsafe_slack_change",
                base_design=design,
                priority_multipliers={"posture_slack": 2.0},
            )
        with self.assertRaises(ValueError):
            protocol.candidate(
                candidate_id="unregistered_tank_movement_change",
                base_design=design,
                priority_multipliers={"tank_movement": 2.0},
            )

    def test_cases_cannot_be_reused_between_screening_and_confirmation(self):
        design = research_preview_mpc_design_v3()
        case = ParameterStudyCase(
            case_id="same_case",
            origin_time="2024-01-01 00:00:00",
            source_event_id="same_event",
            stratum="example",
            duration_h=6.0,
            selection_basis="precontrol_example",
        )
        with self.assertRaises(ValueError):
            ParameterStudyProtocol(
                study_id="invalid",
                base_design_identity=design.identity,
                base_design_parameter_sha256=design.as_dict()["parameter_sha256"],
                implementation_bundle_sha256="0" * 64,
                tunable_priorities=("tank_movement",),
                stages=(
                    ParameterStudyStage("smoke", "smoke", (), 1, 1, 1, False),
                    ParameterStudyStage(
                        "screen", "screening", (case,), 1, 1, 1, True
                    ),
                    ParameterStudyStage(
                        "confirm", "confirmation", (case,), 1, 1, 1, False
                    ),
                ),
            )

    def test_paired_summary_uses_all_cases_and_keeps_strata_visible(self):
        design = research_preview_mpc_design_v3()
        cases = (
            ParameterStudyCase(
                "a", "2024-01-01 00:00:00", "event_a", "rise", 6.0, "precontrol"
            ),
            ParameterStudyCase(
                "b",
                "2024-01-02 00:00:00",
                "event_b",
                "reversal",
                6.0,
                "precontrol",
            ),
        )
        protocol = ParameterStudyProtocol(
            study_id="paired_test",
            base_design_identity=design.identity,
            base_design_parameter_sha256=design.as_dict()["parameter_sha256"],
            implementation_bundle_sha256="0" * 64,
            tunable_priorities=("tank_movement",),
            stages=(
                ParameterStudyStage("smoke", "smoke", (), 1, 1, 1, False),
                ParameterStudyStage("screen", "screening", cases, 2, 2, 1, True),
                ParameterStudyStage(
                    "confirm", "confirmation", (), 1, 1, 1, False
                ),
            ),
        )
        baseline = protocol.candidate(candidate_id="base", base_design=design)
        comparison = protocol.candidate(
            candidate_id="candidate",
            base_design=design,
            priority_multipliers={"tank_movement": 2.0},
        )
        candidate_hashes = {
            baseline.candidate_id: baseline.parameter_sha256,
            comparison.candidate_id: comparison.parameter_sha256,
        }
        observations = tuple(
            ParameterStudyObservation(
                case_id=case_id,
                candidate_id=candidate,
                candidate_parameter_sha256=candidate_hashes[candidate],
                transferred_volume_m3=volume,
                aggregate_pump_active_time_s=volume * 10.0,
                dominant_tilt_rms_deg=rms,
                dominant_tilt_time_above_deg_s=(
                    (2.0, 10.0 + (1.0 if candidate == "candidate" else 0.0)),
                    (3.0, 2.0),
                ),
            )
            for case_id, candidate, volume, rms in (
                ("a", "base", 100.0, 2.0),
                ("a", "candidate", 90.0, 2.1),
                ("b", "base", 200.0, 3.0),
                ("b", "candidate", 220.0, 2.9),
            )
        )

        summary = summarize_paired_candidate(
            protocol=protocol,
            stage_name="screen",
            baseline_candidate_id="base",
            candidate_id="candidate",
            frozen_candidates=(baseline, comparison),
            observations=observations,
        )

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["pump_volume"]["improved_case_fraction"], 0.5)
        self.assertEqual(
            set(summary["pump_volume"]["by_stratum"]),
            {"rise", "reversal"},
        )
        self.assertEqual(
            summary["posture_threshold_time_change_s"]["2.0"]["median_change_s"],
            1.0,
        )
        self.assertIsNone(summary["automatic_scalar_score"])
        self.assertTrue(
            summary["execution_and_internal_constraint_records_valid"]
        )

    def test_paired_summary_is_invalid_when_the_common_baseline_fails(self):
        design = research_preview_mpc_design_v2()
        case = ParameterStudyCase(
            "case", "2024-01-01 00:00:00", "event", "rise", 6.0, "precontrol"
        )
        protocol = ParameterStudyProtocol(
            study_id="baseline_validity_test",
            base_design_identity=design.identity,
            base_design_parameter_sha256=design.as_dict()["parameter_sha256"],
            implementation_bundle_sha256="0" * 64,
            tunable_priorities=("tank_movement",),
            stages=(
                ParameterStudyStage("smoke", "smoke", (), 1, 1, 1, False),
                ParameterStudyStage("screen", "screening", (case,), 1, 1, 1, True),
                ParameterStudyStage("confirm", "confirmation", (), 1, 1, 1, False),
            ),
        )
        baseline = protocol.candidate(candidate_id="base", base_design=design)
        candidate = protocol.candidate(
            candidate_id="candidate",
            base_design=design,
            priority_multipliers={"tank_movement": 2.0},
        )
        observations = (
            ParameterStudyObservation(
                case_id="case",
                candidate_id="base",
                candidate_parameter_sha256=baseline.parameter_sha256,
                transferred_volume_m3=100.0,
                aggregate_pump_active_time_s=10.0,
                dominant_tilt_rms_deg=1.0,
                dominant_tilt_time_above_deg_s=((2.0, 1.0),),
                plan_failure_count=1,
            ),
            ParameterStudyObservation(
                case_id="case",
                candidate_id="candidate",
                candidate_parameter_sha256=candidate.parameter_sha256,
                transferred_volume_m3=90.0,
                aggregate_pump_active_time_s=9.0,
                dominant_tilt_rms_deg=1.0,
                dominant_tilt_time_above_deg_s=((2.0, 1.0),),
            ),
        )

        summary = summarize_paired_candidate(
            protocol=protocol,
            stage_name="screen",
            baseline_candidate_id="base",
            candidate_id="candidate",
            frozen_candidates=(baseline, candidate),
            observations=observations,
        )

        self.assertEqual(summary["validity"]["baseline_plan_failure_count"], 1)
        self.assertFalse(
            summary["execution_and_internal_constraint_records_valid"]
        )

    def test_overlapping_windows_are_rejected(self):
        design = research_preview_mpc_design_v2()
        cases = (
            ParameterStudyCase(
                "screen_case",
                "2024-01-01 00:00:00",
                "screen_event",
                "rise",
                6.0,
                "precontrol",
            ),
            ParameterStudyCase(
                "confirm_case",
                "2024-01-01 03:00:00",
                "confirm_event",
                "reversal",
                6.0,
                "precontrol",
            ),
        )
        with self.assertRaises(ValueError):
            ParameterStudyProtocol(
                study_id="overlap",
                base_design_identity=design.identity,
                base_design_parameter_sha256=design.as_dict()["parameter_sha256"],
                implementation_bundle_sha256="0" * 64,
                tunable_priorities=("tank_movement",),
                stages=(
                    ParameterStudyStage("smoke", "smoke", (), 1, 1, 1, False),
                    ParameterStudyStage(
                        "screen", "screening", (cases[0],), 1, 1, 1, True
                    ),
                    ParameterStudyStage(
                        "confirm", "confirmation", (cases[1],), 1, 1, 1, False
                    ),
                ),
            )

    def test_observation_parameter_hash_must_match_frozen_candidate(self):
        protocol = default_parameter_study_protocol()
        design = research_preview_mpc_design_v3()
        baseline = protocol.candidate(candidate_id="base", base_design=design)
        candidate = protocol.candidate(
            candidate_id="candidate",
            base_design=design,
            priority_multipliers={"tank_throughput": 2.0},
        )
        case = protocol.stage("chain_smoke").cases[0]
        observations = tuple(
            ParameterStudyObservation(
                case_id=case.case_id,
                candidate_id=item.candidate_id,
                candidate_parameter_sha256=(
                    "f" * 64 if item is candidate else item.parameter_sha256
                ),
                transferred_volume_m3=1.0,
                aggregate_pump_active_time_s=1.0,
                dominant_tilt_rms_deg=1.0,
                dominant_tilt_time_above_deg_s=((2.0, 0.0),),
            )
            for item in (baseline, candidate)
        )
        with self.assertRaises(ValueError):
            summarize_paired_candidate(
                protocol=protocol,
                stage_name="chain_smoke",
                baseline_candidate_id="base",
                candidate_id="candidate",
                frozen_candidates=(baseline, candidate),
                observations=observations,
            )

    def test_candidate_registry_has_stable_identity(self):
        protocol = default_parameter_study_protocol()
        design = research_preview_mpc_design_v3()
        baseline = protocol.candidate(candidate_id="baseline", base_design=design)

        first = freeze_candidate_registry((baseline,))
        second = freeze_candidate_registry((baseline,))

        self.assertEqual(
            first["candidate_registry_sha256"],
            second["candidate_registry_sha256"],
        )
        self.assertTrue(first["frozen_before_screening"])

    def test_dry_run_does_not_start_parameter_search(self):
        result = dry_run()

        self.assertFalse(result["is_parameter_search"])
        self.assertFalse(result["is_performance_result"])
        self.assertEqual(result["candidate_registry"]["candidate_count"], 3)
        self.assertEqual(
            result["candidate_registry"]["status"],
            "predeclared_registry_not_executed",
        )
        self.assertTrue(result["stage_readiness"]["chain_smoke"]["ready"])

    def test_predeclared_candidates_are_single_factor_and_lstm_bound(self):
        registry = predeclared_screening_candidate_registry()

        self.assertEqual(
            [item["candidate_id"] for item in registry["candidates"]],
            [
                "baseline_design",
                "tank_throughput_x2",
                "movement_change_x2",
            ],
        )
        self.assertEqual(
            registry["candidates"][1]["priority_multipliers"],
            {"tank_throughput": 2.0},
        )
        self.assertEqual(
            registry["candidates"][2]["priority_multipliers"],
            {"movement_change": 2.0},
        )
        self.assertEqual(
            registry["screening_forecast_variant"]["planner_forecast_mode"],
            "lstm",
        )
        self.assertEqual(
            registry["screening_forecast_variant"]["planner_horizon_blocks"],
            6,
        )
        self.assertEqual(len(registry["screening_configuration_sha256"]), 64)
        self.assertEqual(len(registry["runtime_source_artifact_sha256"]), 64)
        self.assertEqual(len(registry["screening_case_identity"]["case_registry_sha256"]), 64)

    def test_implementation_identity_covers_the_active_source_packages(self):
        identity = implementation_identity()

        self.assertEqual(len(identity["controller_platform_source_tree"]), 64)

    def test_screening_configuration_hash_binds_candidate_and_variant_context(self):
        registry = predeclared_screening_candidate_registry()
        protocol = default_parameter_study_protocol()
        baseline = registry["screening_configuration_sha256"]
        changed_candidate = _screening_configuration_sha256(
            candidate_registry_sha256="f" * 64,
            forecast_variant=registry["screening_forecast_variant"],
            screening_case_identity=registry["screening_case_identity"],
            base_design_parameter_sha256=protocol.base_design_parameter_sha256,
            implementation_bundle_sha256=protocol.implementation_bundle_sha256,
        )
        changed_variant = _screening_configuration_sha256(
            candidate_registry_sha256=registry["candidate_registry_sha256"],
            forecast_variant={
                **registry["screening_forecast_variant"],
                "planner_horizon_blocks": 1,
            },
            screening_case_identity=registry["screening_case_identity"],
            base_design_parameter_sha256=protocol.base_design_parameter_sha256,
            implementation_bundle_sha256=protocol.implementation_bundle_sha256,
        )
        changed_cases = _screening_configuration_sha256(
            candidate_registry_sha256=registry["candidate_registry_sha256"],
            forecast_variant=registry["screening_forecast_variant"],
            screening_case_identity={
                **registry["screening_case_identity"],
                "stage10_cases_file_sha256": "f" * 64,
            },
            base_design_parameter_sha256=protocol.base_design_parameter_sha256,
            implementation_bundle_sha256=protocol.implementation_bundle_sha256,
        )
        self.assertNotEqual(baseline, changed_candidate)
        self.assertNotEqual(baseline, changed_variant)
        self.assertNotEqual(baseline, changed_cases)


if __name__ == "__main__":
    unittest.main()
