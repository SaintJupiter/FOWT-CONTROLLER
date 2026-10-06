from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIRECTORY = ROOT / "scripts" / "validation"
if str(VALIDATION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIRECTORY))

from preview_mpc_parameter_study_protocol import ParameterCandidate  # noqa: E402
import run_preview_mpc_stage10_screening as screening_module  # noqa: E402
from run_preview_mpc_stage10_screening import _observation_from_result  # noqa: E402


class PreviewMPCStage10ScreeningTests(unittest.TestCase):
    def test_observation_uses_actual_execution_and_posture_aggregates(self):
        candidate = ParameterCandidate(
            candidate_id="baseline_design",
            parent_design_identity="research_preview_mpc_design_v3",
            priority_multipliers=(),
            parameter_sha256="a" * 64,
        )
        result = {
            "completed_requested_window": True,
            "preview_mpc_design_parameter_sha256": "a" * 64,
            "aggregate": {
                "transferred_volume_m3": 12.5,
                "aggregate_pump_active_time_s": 80.0,
                "pump_start_count": 3,
                "dominant_tilt_rms_deg": 1.25,
                "dominant_tilt_time_above_deg_s": {
                    "1.5": 60.0,
                    "2.0": 30.0,
                    "3.0": 12.0,
                    "5.0": 1.0,
                },
                "plan_failure_count": 0,
                "fallback_count": 0,
                "selected_precheck_failure_count": 0,
                "no_safe_precheck_candidate_count": 0,
                "actual_posture_limit_failure_count": 0,
            },
        }

        observation = _observation_from_result(
            case_id="stage10_example",
            candidate=candidate,
            result=result,
        )

        self.assertEqual(observation.transferred_volume_m3, 12.5)
        self.assertEqual(observation.aggregate_pump_active_time_s, 80.0)
        self.assertEqual(observation.pump_start_count, 3)
        self.assertEqual(observation.dominant_tilt_rms_deg, 1.25)
        self.assertEqual(dict(observation.dominant_tilt_time_above_deg_s)[3.0], 12.0)

    def test_all_cases_are_preflighted_before_any_candidate_run(self):
        candidate_ids = (
            "baseline_design",
            "terminal_posture_x2",
            "tank_throughput_x2",
            "movement_change_x2",
        )
        registry = {
            "screening_case_identity": "frozen-cases",
            "candidate_registry_sha256": "a" * 64,
            "screening_configuration_sha256": "b" * 64,
            "runtime_source_artifact_sha256": "c" * 64,
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "parent_design_identity": "research_preview_mpc_design_v3",
                    "priority_multipliers": {},
                    "parameter_sha256": str(index + 1) * 64,
                }
                for index, candidate_id in enumerate(candidate_ids)
            ],
        }
        cases = (
            SimpleNamespace(
                case_id="supported-case",
                origin_time="2024-01-01 00:00:00",
                stratum="a",
            ),
            SimpleNamespace(
                case_id="unsupported-case",
                origin_time="2024-01-01 06:00:00",
                stratum="b",
            ),
        )
        stage = SimpleNamespace(
            readiness={"ready": True, "issues": []},
            cases=cases,
            as_dict=lambda: {"name": screening_module.STAGE_NAME},
        )
        protocol = Mock()
        protocol.stage.return_value = stage
        source_records = {
            case.case_id: [{"case_id": case.case_id}] for case in cases
        }

        def prepare(*, origin, **_):
            case_id = (
                "supported-case"
                if origin.hour == 0
                else "unsupported-case"
            )
            return source_records[case_id]

        def preflight(*, source_records, **_):
            supported = source_records[0]["case_id"] == "supported-case"
            return {
                "supported": supported,
                "checked_source_record_count": 1,
                "unsupported_source_record_count": 0 if supported else 1,
                "checks": [],
            }

        with (
            patch.object(
                screening_module,
                "read_and_verify_screening_registry",
                return_value=registry,
            ),
            patch.object(screening_module, "default_parameter_study_protocol", return_value=protocol),
            patch.object(screening_module, "screening_case_identity", return_value="frozen-cases"),
            patch.object(
                screening_module,
                "assemble_volturnus_static_restoring_aligned_runtime_assembly",
                return_value=SimpleNamespace(gravity_m_s2=9.81),
            ),
            patch.object(screening_module, "ThreeTankDifferentialModes", return_value=object()),
            patch.object(
                screening_module,
                "prepare_real_lstm_preview_resources",
                return_value=object(),
            ),
            patch.object(screening_module, "prepare_source_records", side_effect=prepare) as prepare_records,
            patch.object(screening_module, "preflight_source_operating_domain", side_effect=preflight) as preflight_records,
            patch.object(screening_module, "run_variant") as run_variant,
        ):
            result = screening_module.run_stage10_screening(
                registry_path=ROOT / "registry.json"
            )

        self.assertEqual(prepare_records.call_count, 2)
        self.assertEqual(preflight_records.call_count, 2)
        run_variant.assert_not_called()
        self.assertEqual(result["status"], "source_operating_domain_rejected")
        self.assertFalse(result["boundaries"]["candidate_runs_started"])
        self.assertEqual(result["unsupported_case_ids"], ["unsupported-case"])
        self.assertEqual(result["run_count"], 0)


if __name__ == "__main__":
    unittest.main()
