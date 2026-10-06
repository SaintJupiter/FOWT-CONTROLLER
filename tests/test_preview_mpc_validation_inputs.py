from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import wind_prediction


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIRECTORY = ROOT / "scripts" / "validation"
if str(VALIDATION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIRECTORY))

from preview_mpc_experiment_runtime import (  # noqa: E402
    LOCAL_ROTATIONAL_EQUIVALENT_DAMPING_RATIO,
    assemble_planner_load_forecast,
    bind_source_identity,
    research_runtime_assembly,
    validate_oracle_endpoint_times,
)
from wind_prediction.preview_mpc import (  # noqa: E402
    MAX_MASS_RECONSTRUCTION_VIOLATION_KG,
    MAX_SCALED_RECONSTRUCTION_VIOLATION,
)
from wind_prediction.preview_mpc_design import (  # noqa: E402
    research_preview_mpc_design_v4,
)
from real_lstm_preview_fixture import (  # noqa: E402
    RealLstmWindRecord,
    planner_forecast_evidence_from_record,
)
import preview_mpc_continuous_experiment as comparison_module  # noqa: E402
from fowt_platform import (  # noqa: E402
    GeneralizedLoadForecast,
    RotorGeneralizedLoad,
    RotorNormalLoad,
)
from wind_prediction.forecast_evidence import ForecastEvidence  # noqa: E402


def _rotor_load(generalized_load: np.ndarray) -> RotorGeneralizedLoad:
    values = np.asarray(generalized_load, dtype=float)
    force = values[:3]
    return RotorGeneralizedLoad(
        relative_air_velocity_platform_mps=np.array([6.0, 0.0, 0.0]),
        normal_load=RotorNormalLoad(
            thrust_n=float(np.linalg.norm(force)),
            force_platform_n=force,
        ),
        generalized_load_platform=values,
    )


def _source_preview(
    current: np.ndarray,
    future: np.ndarray,
    *,
    planner_forecast_mode: str = "lstm",
) -> SimpleNamespace:
    lead_times = np.arange(1, len(future) + 1, dtype=float) * 600.0
    current_load = _rotor_load(current)
    future_loads = tuple(_rotor_load(row) for row in future)
    preview = SimpleNamespace(
        forecast=SimpleNamespace(
            origin_time="2024-01-01 00:00:00",
            model_version="fixture_lstm_v1",
            sample_period_s=600.0,
            metadata={"planner_forecast_mode": planner_forecast_mode},
        ),
        load_assembly=SimpleNamespace(
            current_rotor_load=current_load,
            future_rotor_loads=future_loads,
            load_forecast=GeneralizedLoadForecast(
                current_generalized_load=current,
                future_generalized_loads=future,
                lead_times_s=lead_times,
            ),
        ),
    )
    preview.source_record_payload = lambda: {
        "forecast": {
            "uv_ms": np.asarray(future, dtype=float).tolist(),
        },
        "current_observation": {
            "generalized_load_fixture": np.asarray(current, dtype=float).tolist(),
        },
    }
    return preview


class PreviewMPCValidationInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.current = np.arange(6, dtype=float)
        self.future = np.vstack(
            [self.current + float(index) for index in range(1, 7)]
        )
        self.source_preview = _source_preview(self.current, self.future)

    def test_package_advertises_only_the_active_preview_mpc_surface(self):
        self.assertIn("PreviewMPCApplication", wind_prediction.__all__)
        self.assertIn("PreviewMPCControlCycleResult", wind_prediction.__all__)
        self.assertNotIn("PhysicalLifecycleSelectionFact", wind_prediction.__all__)
        self.assertIsNotNone(wind_prediction.PhysicalLifecycleSelectionFact)

    def test_active_mpc_entry_does_not_import_the_legacy_physical_smoke(self):
        source = (
            VALIDATION_DIRECTORY
            / "preview_mpc_experiment_runtime.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("run_real_lstm_physical_cycle_smoke", source)
        self.assertNotIn("run_smoke(", source)

    def test_runtime_roll_pitch_damping_matches_declared_local_ratio(self):
        matrices = research_runtime_assembly().base_matrices

        for index in (3, 4):
            ratio = matrices.damping[index, index] / (
                2.0
                * np.sqrt(
                    matrices.mass[index, index]
                    * matrices.restoring_stiffness[index, index]
                )
            )
            self.assertAlmostEqual(
                ratio,
                LOCAL_ROTATIONAL_EQUIVALENT_DAMPING_RATIO,
                places=12,
            )

    def test_v4_mass_reconstruction_tolerance_matches_scaled_solver_bound(self):
        design = research_preview_mpc_design_v4()
        implied_mass_bound = (
            MAX_SCALED_RECONSTRUCTION_VIOLATION
            * design.tank_movement_objective_scale_kg
        )

        self.assertGreaterEqual(
            MAX_MASS_RECONSTRUCTION_VIOLATION_KG,
            0.95 * implied_mass_bound,
        )
        self.assertLessEqual(
            MAX_MASS_RECONSTRUCTION_VIOLATION_KG,
            implied_mass_bound,
        )

    def test_lstm_source_digest_changes_with_consumed_future_record(self) -> None:
        baseline = bind_source_identity(
            source_preview=self.source_preview,
            planner_forecast_mode="lstm",
            lead_times_s=np.array([600.0, 1200.0]),
            oracle_records=None,
        )
        changed_future = self.future.copy()
        changed_future[0] += 1.0
        alternative = bind_source_identity(
            source_preview=_source_preview(self.current, changed_future),
            planner_forecast_mode="lstm",
            lead_times_s=np.array([600.0, 1200.0]),
            oracle_records=None,
        )

        self.assertNotEqual(
            baseline.source_record_sha256,
            alternative.source_record_sha256,
        )
        self.assertTrue(baseline.uses_future_information)

    def test_persistence_source_digest_ignores_unused_lstm_future(self) -> None:
        baseline = bind_source_identity(
            source_preview=_source_preview(
                self.current,
                self.future,
                planner_forecast_mode="persistence",
            ),
            planner_forecast_mode="persistence",
            lead_times_s=np.array([600.0, 1200.0]),
            oracle_records=None,
        )
        alternative = bind_source_identity(
            source_preview=_source_preview(
                self.current,
                np.full((6, 6), 1.0e12),
                planner_forecast_mode="persistence",
            ),
            planner_forecast_mode="persistence",
            lead_times_s=np.array([600.0, 1200.0]),
            oracle_records=None,
        )

        self.assertEqual(
            baseline.source_record_sha256,
            alternative.source_record_sha256,
        )
        self.assertFalse(baseline.uses_future_information)

    def test_lstm_horizon_is_sliced_without_changing_values(self) -> None:
        forecast = assemble_planner_load_forecast(
            source_preview=self.source_preview,
            planner_forecast_mode="lstm",
            planner_horizon_blocks=1,
        )

        self.assertEqual(forecast.horizon_steps, 1)
        np.testing.assert_allclose(forecast.future_generalized_loads, self.future[:1])
        np.testing.assert_allclose(forecast.lead_times_s, [600.0])

    def test_persistence_uses_the_loads_already_converted_from_its_wind_sequence(self) -> None:
        forecast = assemble_planner_load_forecast(
            source_preview=_source_preview(
                self.current,
                self.future,
                planner_forecast_mode="persistence",
            ),
            planner_forecast_mode="persistence",
            planner_horizon_blocks=6,
        )

        self.assertEqual(forecast.horizon_steps, 6)
        np.testing.assert_allclose(
            forecast.future_generalized_loads,
            self.future,
        )

    def test_current_observation_uses_its_preconverted_one_block_preview(self) -> None:
        forecast = assemble_planner_load_forecast(
            source_preview=_source_preview(
                self.current,
                self.future,
                planner_forecast_mode="current_observation",
            ),
            planner_forecast_mode="current_observation",
            planner_horizon_blocks=1,
        )

        self.assertEqual(forecast.horizon_steps, 1)
        np.testing.assert_allclose(
            forecast.future_generalized_loads,
            self.future[:1],
        )

    def test_current_observation_reference_rejects_other_horizons(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one control block"):
            assemble_planner_load_forecast(
                source_preview=_source_preview(
                    self.current,
                    self.future,
                    planner_forecast_mode="current_observation",
                ),
                planner_forecast_mode="current_observation",
                planner_horizon_blocks=2,
            )

    def test_variant_suite_exposes_a_reactive_reference_contrast(self) -> None:
        variants = {
            name: (mode, horizon)
            for name, mode, horizon in comparison_module.VARIANTS
        }
        self.assertEqual(
            variants["current_observation_1block"],
            ("current_observation", 1),
        )
        self.assertIn(
            ("current_observation_1block", "lstm_6block"),
            comparison_module.CONTRASTS,
        )
        self.assertEqual(
            comparison_module.CONTRAST_INTERPRETATIONS[
                ("persistence_6block", "lstm_6block")
            ],
            "future_prediction_content_effect_with_identical_six_block_preview_"
            "structure",
        )

    def test_failed_diagnostic_does_not_remove_complete_deployable_contrasts(self):
        by_name = {
            name: {
                "completed_requested_window": name != "recorded_oracle_6block"
            }
            for name, _, _ in comparison_module.VARIANTS
        }

        pairs = comparison_module._completed_contrast_pairs(by_name)

        self.assertIn(
            ("persistence_6block", "lstm_6block"),
            pairs,
        )
        self.assertNotIn(
            ("lstm_6block", "recorded_oracle_6block"),
            pairs,
        )

        cases = [
            {
                "completed_requested_window_by_variant": {
                    name: name != "recorded_oracle_6block"
                    for name, _, _ in comparison_module.VARIANTS
                }
            }
        ]
        self.assertEqual(
            len(
                comparison_module._complete_cases_for_contrast(
                    cases,
                    "persistence_6block",
                    "lstm_6block",
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                comparison_module._complete_cases_for_contrast(
                    cases,
                    "lstm_6block",
                    "recorded_oracle_6block",
                )
            ),
            0,
        )

    def test_partial_prefix_cannot_enter_an_economic_contrast(self):
        with self.assertRaisesRegex(ValueError, "complete requested-window"):
            comparison_module._contrast(
                {"aggregate_eligible_for_economic_comparison": False},
                {"aggregate_eligible_for_economic_comparison": True},
            )

    def test_invalid_horizon_and_mode_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assemble_planner_load_forecast(
                source_preview=self.source_preview,
                planner_forecast_mode="lstm",
                planner_horizon_blocks=0,
            )
        with self.assertRaises(ValueError):
            assemble_planner_load_forecast(
                source_preview=self.source_preview,
                planner_forecast_mode="unknown",  # type: ignore[arg-type]
                planner_horizon_blocks=6,
            )

    def test_planner_load_forecast_requires_matching_mode_provenance(self) -> None:
        missing_mode = _source_preview(self.current, self.future)
        missing_mode.forecast.metadata.clear()
        with self.assertRaisesRegex(ValueError, "must declare planner_forecast_mode"):
            assemble_planner_load_forecast(
                source_preview=missing_mode,
                planner_forecast_mode="persistence",
                planner_horizon_blocks=6,
            )

        with self.assertRaisesRegex(ValueError, "does not match"):
            assemble_planner_load_forecast(
                source_preview=self.source_preview,
                planner_forecast_mode="persistence",
                planner_horizon_blocks=6,
            )

    def test_recorded_oracle_uses_the_common_source_preview_load_conversion(self) -> None:
        forecast = assemble_planner_load_forecast(
            source_preview=_source_preview(
                self.current,
                self.future,
                planner_forecast_mode="recorded_oracle",
            ),
            planner_forecast_mode="recorded_oracle",
            planner_horizon_blocks=3,
        )

        np.testing.assert_allclose(forecast.future_generalized_loads, self.future[:3])
        np.testing.assert_allclose(forecast.lead_times_s, [600.0, 1200.0, 1800.0])

    def test_forecast_modes_select_wind_before_common_load_conversion(self) -> None:
        def record(origin: str, current: np.ndarray, future: np.ndarray) -> RealLstmWindRecord:
            return RealLstmWindRecord(
                forecast=ForecastEvidence(
                    source="fixture_lstm",
                    model_version="fixture_v1",
                    origin_time=origin,
                    sample_period_s=600.0,
                    uv_ms=future,
                    lead_reliability=np.ones(len(future)),
                ),
                current_enu_downwind_wind_mps=current,
            )

        common = np.array([7.0, -1.0])
        source = record(
            "2024-01-01 00:00:00",
            common,
            np.repeat(common[None, :], 3, axis=0),
        )
        oracle = tuple(
            record(
                f"2024-01-01 00:{10 * (index + 1):02d}:00",
                common,
                np.repeat(common[None, :], 3, axis=0),
            )
            for index in range(3)
        )
        selected = [
            planner_forecast_evidence_from_record(
                source_record=source,
                planner_forecast_mode=mode,
                horizon_blocks=3,
                oracle_records=oracle if mode == "recorded_oracle" else None,
            ).uv_ms
            for mode in ("lstm", "persistence", "recorded_oracle")
        ]

        np.testing.assert_allclose(selected[0], selected[1])
        np.testing.assert_allclose(selected[1], selected[2])

    def test_oracle_endpoint_times_reject_a_shifted_record(self) -> None:
        records = tuple(
            {"recorded_observation_time": f"2024-01-01 0{index}:10:00"}
            for index in range(3)
        )
        with self.assertRaisesRegex(ValueError, "not aligned"):
            validate_oracle_endpoint_times(
                origin_time="2024-01-01 00:00:00",
                oracle_records=records,
                lead_times_s=np.array([600.0, 1200.0, 1800.0]),
            )

    def test_source_records_reuse_one_explicit_resource_bundle(self) -> None:
        resources = SimpleNamespace(device="cpu")
        observed_origins = []

        def infer(**kwargs):
            observed_origins.append(kwargs["origin"])
            return SimpleNamespace(
                resources=kwargs["resources"],
                forecast=SimpleNamespace(origin_time=kwargs["origin"]),
                source_record_sha256=str(kwargs["origin"]),
            )

        with patch.object(
            comparison_module,
            "infer_real_lstm_wind_record",
            side_effect=infer,
        ) as infer_record:
            records = comparison_module.prepare_source_records(
                origin=comparison_module.datetime(2024, 1, 1),
                device="cpu",
                cycle_count=2,
                lookahead_count=1,
                resources=resources,
            )

        self.assertEqual(infer_record.call_count, 3)
        self.assertEqual(len(observed_origins), 3)
        self.assertTrue(all(record.resources is resources for record in records))

    def test_source_record_prefix_is_prepared_before_the_statistical_window(self) -> None:
        resources = SimpleNamespace(device="cpu")
        observed_origins = []

        def infer(**kwargs):
            observed_origins.append(kwargs["origin"])
            return SimpleNamespace(
                forecast=SimpleNamespace(origin_time=kwargs["origin"]),
                source_record_sha256=str(kwargs["origin"]),
            )

        with patch.object(
            comparison_module,
            "infer_real_lstm_wind_record",
            side_effect=infer,
        ):
            records = comparison_module.prepare_source_records(
                origin=comparison_module.datetime(2024, 1, 1, 0, 0),
                device="cpu",
                cycle_count=2,
                lookahead_count=1,
                warmup_block_count=2,
                resources=resources,
            )

        self.assertEqual(len(records), 5)
        self.assertEqual(
            observed_origins,
            [
                comparison_module.datetime(2023, 12, 31, 23, 40),
                comparison_module.datetime(2023, 12, 31, 23, 50),
                comparison_module.datetime(2024, 1, 1, 0, 0),
                comparison_module.datetime(2024, 1, 1, 0, 10),
                comparison_module.datetime(2024, 1, 1, 0, 20),
            ],
        )

    def test_preflight_reads_the_prepared_records_without_rerunning_inference(self):
        records = [
            SimpleNamespace(
                forecast=SimpleNamespace(origin_time="2024-01-01 00:00:00"),
                source_record_sha256="a" * 64,
            ),
            SimpleNamespace(
                forecast=SimpleNamespace(origin_time="2024-01-01 00:10:00"),
                source_record_sha256="b" * 64,
            ),
        ]
        domains = [
            {
                "checked_input_scope": "current_and_forecast",
                "supported": True,
                "unsupported_record_count": 0,
                "records": [{"label": "current", "supported": True}],
                "derived_nominal_inflow_range_mps": [4.0, 10.0],
            },
            {
                "checked_input_scope": "current_only",
                "supported": False,
                "unsupported_record_count": 1,
                "records": [{"label": "current", "supported": False}],
                "derived_nominal_inflow_range_mps": [4.0, 10.0],
            },
        ]

        with (
            patch.object(
                comparison_module,
                "inspect_real_lstm_wind_record_operating_domain",
                side_effect=domains,
            ) as inspect,
            patch.object(comparison_module, "infer_real_lstm_wind_record") as infer,
        ):
            report = comparison_module.preflight_source_operating_domain(
                source_records=records,
                resources=SimpleNamespace(),
                origin=comparison_module.datetime(2024, 1, 1),
                cycle_count=1,
            )

        infer.assert_not_called()
        self.assertEqual(inspect.call_count, 2)
        self.assertTrue(inspect.call_args_list[0].kwargs["include_future"])
        self.assertFalse(inspect.call_args_list[1].kwargs["include_future"])
        self.assertFalse(report["supported"])
        self.assertEqual(report["checked_source_record_count"], 2)
        self.assertEqual(report["unsupported_source_record_count"], 1)
        self.assertEqual(
            report["checks"][1]["source_record_scope"],
            "current_only",
        )

    def test_objective_breakdown_aggregation_preserves_components_and_blocks(self):
        def row(scale: float) -> dict[str, object]:
            return {
                "planner_objective_breakdown": {
                    "running_posture_by_block": [1.0 * scale, 2.0 * scale],
                    "running_posture": 3.0 * scale,
                    "terminal_posture": 4.0 * scale,
                    "tank_movement_by_block": [5.0 * scale, 6.0 * scale],
                    "tank_movement": 11.0 * scale,
                    "tank_throughput_by_block": [0.0, 0.0],
                    "tank_throughput": 0.0,
                    "movement_change_by_block": [7.0 * scale, 8.0 * scale],
                    "movement_change": 15.0 * scale,
                    "posture_slack": 0.0,
                    "numerical_regularization": 0.0,
                    "constant_offset": 2.0 * scale,
                    "absolute_total": 33.0 * scale,
                    "reduced_total": 31.0 * scale,
                }
            }

        first = comparison_module._aggregate_objective_breakdown(
            [row(1.0), row(2.0)]
        )
        second = comparison_module._aggregate_objective_breakdown([row(3.0)])
        combined = comparison_module._combine_objective_breakdowns([first, second])

        self.assertEqual(combined["cycle_count"], 3)
        self.assertEqual(combined["component_sum"]["running_posture"], 18.0)
        self.assertEqual(
            combined["per_block_sum"]["tank_movement_by_block"],
            [30.0, 36.0],
        )
        self.assertEqual(combined["absolute_total_sum"], 198.0)
        self.assertEqual(combined["reduced_total_sum"], 186.0)


if __name__ == "__main__":
    unittest.main()
