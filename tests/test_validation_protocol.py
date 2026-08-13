from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from scripts.validation.check_production_learned_lstm_6h_gate import (
    GateConfigError,
    validate_gate_config,
)
from wind_prediction.validation_protocol import (
    CaseTimeInterval,
    ControlValidationProtocol,
    ParameterFreezeManifest,
    StageMetrics,
    STAGE_GATES,
    evaluate_validation_stage,
    find_case_interval_overlaps,
    validate_algorithm_physical_configs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_NAMES = (
    "posture_feedback",
    "no_future_prediction",
    "prediction_assisted",
)


def passing_metrics(case_count: int) -> StageMetrics:
    return StageMetrics(
        case_count=case_count,
        pump_reduction_vs_posture_pct=20.0,
        pump_reduction_vs_no_future_pct=15.0,
        improved_case_count=max(2, int(case_count * 0.8)),
        pump_reduction_ci_lower_pct=10.0,
        class_pump_reduction_pct={"stable": 8.0, "gusty": 7.0},
        time_over_2deg_delta_pp=2.0,
        time_over_3deg_delta_pp=1.0,
        time_over_5deg_delta_pp=0.05,
        new_over_7p5deg_case_count=0,
        max_case_over_7p5deg_increase_s=30.0,
        time_over_10deg_increase_s=0.0,
        pitch_rms_increase_pct=5.0,
        roll_rms_increase_pct=5.0,
        pitch_p95_increase_pct=5.0,
        roll_p95_increase_pct=5.0,
        preview_actual_pump_mismatch_pct=0.5,
        preview_actual_target_mismatch_pct=0.5,
    )


def frozen_manifest(
    *, tuning_stages: tuple[int, ...] = (1, 3, 10, 20)
) -> ParameterFreezeManifest:
    return ParameterFreezeManifest(
        parameter_set_id="planner-v2-frozen",
        frozen=True,
        frozen_after_stage=20,
        tuning_stage_case_counts=tuning_stages,
        config_sha256="a" * 64,
    )


def physical_configs(*, prediction_pump: str = "engineered_minimal") -> dict:
    common = {"duration_s": 21600, "pump_profile": "engineered_minimal"}
    return {
        "posture_feedback": dict(common),
        "no_future_prediction": dict(common),
        "prediction_assisted": {
            "duration_s": 21600,
            "pump_profile": prediction_pump,
        },
    }


def staged_payload(
    *, stage: int = 1, duration_s: int = 21600, cases: int | None = None
) -> dict:
    case_count = stage if cases is None else cases
    base = datetime(2026, 1, 1)
    intervals = []
    for index in range(case_count):
        start = base.replace(day=1 + index)
        end = start.replace(hour=6)
        intervals.append(
            {
                "case_id": f"case_{index}",
                "start": start.isoformat(),
                "end": end.isoformat(),
            }
        )
    staged = {
        "stage_case_count": stage,
        "case_intervals": intervals,
        "algorithm_physical_configs": physical_configs(),
    }
    if stage == 30:
        staged["parameter_manifest"] = {
            "parameter_set_id": "planner-v2-frozen",
            "frozen": True,
            "frozen_after_stage": 20,
            "tuning_stage_case_counts": [1, 3, 10, 20],
            "config_sha256": "a" * 64,
        }
    return {
        "schema_version": "staged.v2",
        "casebook_command": [
            "python3",
            "scripts/analysis/run_prediction_primary_casebook.py",
            "--out-dir",
            "out",
            "--duration-s",
            str(duration_s),
        ],
        "expected_cases": [f"case_{index}" for index in range(case_count)],
        "staged_validation": staged,
    }


def load_payload(payload: dict) -> ControlValidationProtocol:
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name) / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    protocol = ControlValidationProtocol.load(path, repo_root=REPO_ROOT)
    # Keep the temporary directory alive for callers using source_path.
    protocol.__dict__.get("_unused")
    directory.cleanup()
    return protocol


class ValidationProtocolTests(unittest.TestCase):
    def test_repository_legacy_protocol_remains_readable(self) -> None:
        protocol = ControlValidationProtocol.load(
            "configs/control_chain_smoke_gate_v2.json",
            repo_root=REPO_ROOT,
        )
        self.assertTrue(protocol.runner_argv()[1].endswith("run_prediction_primary_casebook.py"))
        self.assertEqual(
            protocol.summary_csv,
            REPO_ROOT
            / "outputs"
            / "wind_prediction"
            / "control_chain_smoke_gate_v2"
            / "casebook_summary.csv",
        )
        self.assertIsNone(protocol.staged_validation)

    def test_noncanonical_runner_and_missing_output_are_rejected(self) -> None:
        payload = staged_payload()
        payload["casebook_command"][1] = "other.py"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "casebook runner"):
                ControlValidationProtocol.load(path, repo_root=REPO_ROOT)

        payload = staged_payload()
        del payload["casebook_command"][2:4]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "--out-dir"):
                ControlValidationProtocol.load(path, repo_root=REPO_ROOT)

    def test_case_interval_overlap_uses_half_open_intervals(self) -> None:
        intervals = (
            CaseTimeInterval("a", datetime(2026, 1, 1, 0), datetime(2026, 1, 1, 6)),
            CaseTimeInterval("b", datetime(2026, 1, 1, 5), datetime(2026, 1, 1, 11)),
            CaseTimeInterval("c", datetime(2026, 1, 1, 11), datetime(2026, 1, 1, 17)),
        )
        self.assertEqual(find_case_interval_overlaps(intervals), (("a", "b"),))

    def test_staged_protocol_requires_each_interval_to_be_six_hours(self) -> None:
        payload = staged_payload()
        payload["staged_validation"]["case_intervals"][0]["end"] = (
            "2026-01-01T05:59:59"
        )
        protocol = load_payload(payload)
        self.assertIn(
            "case_interval_not_six_hours",
            [issue.code for issue in protocol.manifest_issues()],
        )
        with self.assertRaisesRegex(ValueError, "21600s"):
            protocol.runner_argv()

    def test_staged_command_must_also_request_six_hours(self) -> None:
        protocol = load_payload(staged_payload(duration_s=7200))
        self.assertIn(
            "staged_duration_not_six_hours",
            [issue.code for issue in protocol.manifest_issues()],
        )
        with self.assertRaisesRegex(ValueError, "--duration-s 21600"):
            protocol.runner_argv()

    def test_staged_protocol_requires_all_case_intervals(self) -> None:
        payload = staged_payload(stage=3)
        payload["staged_validation"]["case_intervals"].pop()
        protocol = load_payload(payload)
        self.assertIn(
            "case_interval_missing",
            [issue.code for issue in protocol.manifest_issues()],
        )

    def test_physical_configs_are_required_and_must_match(self) -> None:
        issues = validate_algorithm_physical_configs(
            physical_configs(prediction_pump="different")
        )
        self.assertEqual([issue.code for issue in issues], ["physical_config_mismatch"])

        payload = staged_payload()
        payload["staged_validation"]["algorithm_physical_configs"] = {}
        protocol = load_payload(payload)
        self.assertIn(
            "physical_configs_missing",
            [issue.code for issue in protocol.manifest_issues()],
        )

    def test_valid_staged_protocol_can_build_runner(self) -> None:
        protocol = load_payload(staged_payload(stage=3))
        self.assertEqual(protocol.manifest_issues(), ())
        self.assertIn("21600", protocol.runner_argv())

    def test_protocol_is_readable_but_cannot_run_more_than_thirty_cases(self) -> None:
        payload = staged_payload(stage=30, cases=31)
        payload["staged_validation"]["stage_case_count"] = 30
        protocol = load_payload(payload)
        codes = [issue.code for issue in protocol.manifest_issues()]
        self.assertIn("experiment_case_limit_exceeded", codes)
        with self.assertRaisesRegex(ValueError, "cannot exceed 30"):
            protocol.runner_argv()

    def test_only_five_stage_sizes_are_supported(self) -> None:
        self.assertEqual(tuple(STAGE_GATES), (1, 3, 10, 20, 30))
        for count in (2, 5, 50, 150):
            with self.subTest(case_count=count):
                with self.assertRaisesRegex(ValueError, "unsupported validation stage"):
                    evaluate_validation_stage(StageMetrics(case_count=count))

    def test_one_case_checks_integrity_and_safety_but_not_performance(self) -> None:
        result = evaluate_validation_stage(passing_metrics(1))
        self.assertTrue(result.integrity_passed)
        self.assertTrue(result.safety_passed)
        self.assertIsNone(result.performance_passed)
        self.assertEqual(result.evidence_level, "structural_smoke")

    def test_three_cases_are_diagnostic_only(self) -> None:
        result = evaluate_validation_stage(passing_metrics(3))
        self.assertTrue(result.integrity_passed)
        self.assertTrue(result.safety_passed)
        self.assertIsNone(result.performance_passed)
        self.assertEqual(result.evidence_level, "cross_case_diagnostic")

    def test_ten_cases_are_the_minimum_performance_screen(self) -> None:
        result = evaluate_validation_stage(passing_metrics(10))
        self.assertTrue(result.performance_passed)
        self.assertEqual(result.evidence_level, "minimum_directional_screen")

    def test_all_five_stages_accept_complete_passing_evidence(self) -> None:
        for count in (1, 3, 10, 20, 30):
            with self.subTest(case_count=count):
                result = evaluate_validation_stage(
                    passing_metrics(count),
                    parameter_manifest=frozen_manifest() if count == 30 else None,
                )
                self.assertTrue(result.passed, result.issues)

    def test_development_stage_does_not_encode_a_fixed_reduction_target(self) -> None:
        raw = passing_metrics(10)
        result = evaluate_validation_stage(
            StageMetrics(
                **{
                    **raw.__dict__,
                    "pump_reduction_vs_posture_pct": -4.0,
                    "pump_reduction_vs_no_future_pct": -2.0,
                    "improved_case_count": 1,
                }
            )
        )
        self.assertTrue(result.integrity_passed)
        self.assertTrue(result.safety_passed)
        self.assertTrue(result.performance_passed)
        self.assertTrue(result.passed)
        self.assertNotIn(
            "pump_reduction_below_stage_evidence",
            [issue.code for issue in result.issues],
        )

    def test_safety_failure_is_separate_from_result_amplitude(self) -> None:
        raw = passing_metrics(20)
        result = evaluate_validation_stage(
            StageMetrics(**{**raw.__dict__, "time_over_10deg_increase_s": 1.0})
        )
        self.assertTrue(result.integrity_passed)
        self.assertFalse(result.safety_passed)
        self.assertTrue(result.performance_passed)
        self.assertFalse(result.passed)

    def test_integrity_failure_is_separate_from_safety_and_result(self) -> None:
        raw = passing_metrics(20)
        result = evaluate_validation_stage(
            StageMetrics(
                **{**raw.__dict__, "preview_actual_target_mismatch_pct": 1.1}
            )
        )
        self.assertFalse(result.integrity_passed)
        self.assertTrue(result.safety_passed)
        self.assertTrue(result.performance_passed)
        self.assertFalse(result.passed)

    def test_missing_safety_metric_fails_the_safety_layer(self) -> None:
        raw = passing_metrics(3)
        result = evaluate_validation_stage(
            StageMetrics(**{**raw.__dict__, "roll_p95_increase_pct": None})
        )
        self.assertFalse(result.safety_passed)
        self.assertIn("metric_missing", [issue.code for issue in result.errors])

    def test_over_thirty_percent_result_requires_audit_but_is_not_a_target(self) -> None:
        raw = passing_metrics(20)
        result = evaluate_validation_stage(
            StageMetrics(
                **{**raw.__dict__, "pump_reduction_vs_posture_pct": 31.0}
            )
        )
        self.assertTrue(result.passed)
        self.assertIn(
            "pump_reduction_above_audit_threshold",
            [issue.code for issue in result.warnings],
        )

    def test_thirty_case_stage_requires_frozen_untuned_evaluation(self) -> None:
        result = evaluate_validation_stage(passing_metrics(30))
        self.assertFalse(result.integrity_passed)
        self.assertIn("parameter_manifest_missing", [i.code for i in result.errors])

        invalid = ParameterFreezeManifest(
            parameter_set_id="candidate",
            frozen=False,
            frozen_after_stage=30,
            tuning_stage_case_counts=(1, 3, 10, 20, 30),
            config_sha256="b" * 64,
        )
        result = evaluate_validation_stage(
            passing_metrics(30), parameter_manifest=invalid
        )
        codes = {issue.code for issue in result.errors}
        self.assertIn("parameters_not_frozen", codes)
        self.assertIn("invalid_parameter_freeze_stage", codes)
        self.assertIn("evaluation_results_used_for_tuning", codes)

    def test_parameter_manifest_rejects_tuning_after_declared_freeze(self) -> None:
        manifest = ParameterFreezeManifest(
            parameter_set_id="premature-freeze",
            frozen=True,
            frozen_after_stage=10,
            tuning_stage_case_counts=(1, 3, 10, 20),
            config_sha256="c" * 64,
        )
        result = evaluate_validation_stage(
            passing_metrics(30), parameter_manifest=manifest
        )
        self.assertFalse(result.integrity_passed)
        self.assertIn(
            "post_freeze_tuning_recorded",
            [issue.code for issue in result.errors],
        )

    def test_thirty_case_manifest_requires_a_real_config_hash(self) -> None:
        manifest = ParameterFreezeManifest(
            parameter_set_id="planner-v2",
            frozen=True,
            frozen_after_stage=20,
            tuning_stage_case_counts=(1, 3, 10, 20),
            config_sha256="not-a-sha256",
        )
        result = evaluate_validation_stage(
            passing_metrics(30), parameter_manifest=manifest
        )
        self.assertFalse(result.integrity_passed)
        self.assertIn(
            "parameter_identity_missing",
            [issue.code for issue in result.errors],
        )

    def test_thirty_case_stage_requires_positive_ci_and_reports_class_limits(self) -> None:
        raw = passing_metrics(30)
        result = evaluate_validation_stage(
            StageMetrics(
                **{
                    **raw.__dict__,
                    "pump_reduction_ci_lower_pct": 0.0,
                    "class_pump_reduction_pct": {"stable": 8.0, "gusty": -0.1},
                }
            ),
            parameter_manifest=frozen_manifest(),
        )
        self.assertFalse(result.performance_passed)
        codes = {issue.code for issue in result.errors}
        self.assertIn("pump_reduction_ci_not_positive", codes)
        self.assertNotIn("class_pump_reduction_negative", codes)
        self.assertIn(
            "class_pump_reduction_negative",
            {issue.code for issue in result.warnings},
        )

    def test_thirty_case_stage_accepts_small_but_supported_reduction(self) -> None:
        raw = passing_metrics(30)
        result = evaluate_validation_stage(
            StageMetrics(
                **{
                    **raw.__dict__,
                    "pump_reduction_vs_posture_pct": 0.4,
                    "pump_reduction_vs_no_future_pct": 0.2,
                    "improved_case_count": 10,
                    "pump_reduction_ci_lower_pct": 0.01,
                }
            ),
            parameter_manifest=frozen_manifest(),
        )
        self.assertTrue(result.performance_passed)
        self.assertTrue(result.passed)

    def test_thirty_case_result_only_reports_expansion_evidence(self) -> None:
        result = evaluate_validation_stage(
            passing_metrics(30), parameter_manifest=frozen_manifest()
        )
        self.assertTrue(result.supports_expanded_validation)
        self.assertIsNone(
            evaluate_validation_stage(passing_metrics(20)).supports_expanded_validation
        )

    def test_submitted_v1_result_is_not_encoded_as_a_gate(self) -> None:
        self.assertNotIn("19.85", repr(STAGE_GATES))

    def test_production_learned_lstm_six_hour_gate_preflight(self) -> None:
        report = validate_gate_config(repo_root=REPO_ROOT)
        self.assertEqual(report.command[report.command.index("--duration-s") + 1], "21600")
        self.assertEqual(
            report.command[report.command.index("--forecast-source") + 1],
            "learned",
        )
        self.assertIsInstance(report.run_identity_present, bool)

    def test_production_gate_rejects_oracle_forecast(self) -> None:
        source = REPO_ROOT / "configs" / "production_learned_lstm_6h_gate_v1.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        command = payload["casebook_command"]
        command[command.index("--forecast-source") + 1] = "oracle"
        payload["gate_contract"]["forecast_source"] = "oracle"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(GateConfigError, "learned"):
                validate_gate_config(path, repo_root=REPO_ROOT)

    def test_production_gate_rejects_short_duration(self) -> None:
        source = REPO_ROOT / "configs" / "production_learned_lstm_6h_gate_v1.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        command = payload["casebook_command"]
        command[command.index("--duration-s") + 1] = "7200"
        payload["gate_contract"]["duration_s"] = 7200
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(GateConfigError, "21600"):
                validate_gate_config(path, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
