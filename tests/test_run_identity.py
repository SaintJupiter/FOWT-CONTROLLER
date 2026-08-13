from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from wind_prediction.run_identity import (
    MISSING,
    RepositoryState,
    RunIdentity,
    build_run_identity,
)
from wind_prediction.run_protocol import (
    build_casebook_run_identity,
    build_casebook_run_protocol,
    write_run_identity,
)


FIXED_REPOSITORY = RepositoryState(
    git_commit="a" * 40,
    git_dirty=False,
)
FIXED_DEPENDENCIES = {
    "numpy": "2.3.1",
    "torch": "2.7.1",
}


class RunIdentityTests(unittest.TestCase):
    def _build(
        self,
        root: Path,
        *,
        effective_config: dict | None = None,
        model_files: dict[str, Path] | None = None,
        dataset_files: dict[str, Path] | None = None,
        created_at_utc: str = "2026-08-11T12:00:00Z",
    ) -> RunIdentity:
        return build_run_identity(
            repo_root=root,
            effective_config=(
                effective_config
                if effective_config is not None
                else {"planner": {"horizon": 12}}
            ),
            model_files=model_files or {},
            dataset_files=dataset_files or {},
            output_directory="outputs/replay-a",
            control_profile="profile-v1",
            forecast_source="learned",
            forecast_model_version="lstm-v3",
            created_at_utc=created_at_utc,
            repository_state=FIXED_REPOSITORY,
            python_version="3.12.13",
            dependency_versions=FIXED_DEPENDENCIES,
        )

    def test_hashes_are_stable_for_equivalent_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.bin"
            dataset = root / "metadata.json"
            model.write_bytes(b"fixed model")
            dataset.write_text('{"split":"test"}', encoding="utf-8")

            first = self._build(
                root,
                effective_config={"planner": {"horizon": 12, "gain": 0.4}},
                model_files={"weights": model},
                dataset_files={"metadata": dataset},
            )
            second = self._build(
                root,
                effective_config={"planner": {"gain": 0.4, "horizon": 12}},
                model_files={"weights": model},
                dataset_files={"metadata": dataset},
                created_at_utc="2026-08-11T12:05:00Z",
            )

            self.assertEqual(
                first.effective_config_sha256,
                second.effective_config_sha256,
            )
            self.assertEqual(first.run_id, second.run_id)
            self.assertNotEqual(first.created_at_utc, second.created_at_utc)
            json.dumps(first.to_dict(), allow_nan=False)

    def test_input_content_change_changes_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.bin"
            model.write_bytes(b"model version one")
            first = self._build(root, model_files={"weights": model})

            model.write_bytes(b"model version two")
            second = self._build(root, model_files={"weights": model})

            self.assertNotEqual(
                first.input_files["model"]["weights"].sha256,
                second.input_files["model"]["weights"].sha256,
            )
            self.assertNotEqual(first.run_id, second.run_id)

    def test_missing_files_are_recorded_instead_of_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = self._build(
                root,
                model_files={"weights": root / "missing-model.bin"},
                dataset_files={"metadata": root / "missing-metadata.json"},
            )

            missing_model = identity.input_files["model"]["weights"]
            missing_dataset = identity.input_files["dataset"]["metadata"]
            self.assertEqual(missing_model.status, MISSING)
            self.assertEqual(missing_model.sha256, MISSING)
            self.assertEqual(missing_dataset.status, MISSING)
            self.assertEqual(missing_dataset.sha256, MISSING)
            self.assertIn("weights", identity.to_dict()["input_files"]["model"])
            self.assertIn("metadata", identity.to_dict()["input_files"]["dataset"])


@dataclass(frozen=True)
class _PlannerConfigStub:
    pressure_norm_cap: float = 2.5
    envelope_use_discount: bool = True
    envelope_barrier_active: bool = False
    posture_hold_barrier_active: bool = False
    posture_state_residual_active: bool = False
    posture_state_gain: float = 0.0


class CasebookRunIdentityTests(unittest.TestCase):
    def _fixture(self, root: Path):
        model_dir = root / "model"
        dataset_dir = root / "dataset"
        config_dir = root / "configs"
        model_dir.mkdir()
        dataset_dir.mkdir()
        config_dir.mkdir()
        for filename, content in (
            ("lstm_best.pt", b"checkpoint"),
            ("lstm_config.json", b'{"model_type":"lstm"}'),
            ("lstm_event_thresholds.json", b'{"thresholds":{}}'),
        ):
            (model_dir / filename).write_bytes(content)
        (dataset_dir / "metadata.json").write_text('{"split":"test"}', encoding="utf-8")
        (dataset_dir / "scaler_train.json").write_text('{"scale":1}', encoding="utf-8")
        planner_runtime = config_dir / "planner_runtime_v2.json"
        planner_runtime.write_text('{"schema_version":"planner_runtime.v2"}', encoding="utf-8")
        cases_csv = root / "cases.csv"
        cases_csv.write_text("case_id,timestamp\ncase_1,2020-01-01\n", encoding="utf-8")
        stiffness = root / "stiffness.xlsx"
        stiffness.write_bytes(b"stiffness")
        args = Namespace(
            cases_csv=str(cases_csv),
            case_ids="",
            dataset_dir=str(dataset_dir),
            model_dir=str(model_dir),
            stiffness_file=str(stiffness),
            planner_runtime_config=str(planner_runtime),
            duration_s=21600.0,
            skip_figures=True,
            primary_label="prediction_primary",
            primary_control_profile="profile-v2",
            forecast_source="learned",
            replay_split="test",
            primary_only=False,
            reactive_primary_only=False,
            primary_safety_profile="default",
            primary_scale=1.0,
            event_reset_mode="action",
            primary_hold_target_mode="current",
            pump_suppression=False,
            forecast_advised_economy_suppression_only=False,
            relief_medium_cap=False,
            hold_relief_debt=False,
            recovery_mode=False,
            hold_comfort_release=False,
            out_dir=str(root / "outputs" / "casebook"),
        )
        cfg = _PlannerConfigStub()
        protocol = build_casebook_run_protocol(
            args=args,
            cfg=cfg,
            repo_root=root,
            out_dir=root / "outputs" / "casebook",
            dataset_dir=dataset_dir,
            closed_pump_profile="baseline",
            primary_pump_profile="matched",
            primary_scale=1.0,
            forecast_source_effective="lstm_dual_head_preview",
            completed_cases=150,
            elapsed_s=123.0,
            created_at_utc="2026-08-11T12:00:00Z",
        )
        return args, cfg, protocol, dataset_dir

    def test_casebook_sidecar_contains_required_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, cfg, protocol, dataset_dir = self._fixture(root)
            identity = build_casebook_run_identity(
                protocol=protocol,
                args=args,
                planner_config=cfg,
                repo_root=root,
                out_dir=root / "outputs" / "casebook",
                dataset_dir=dataset_dir,
                forecast_source_effective="lstm_dual_head_preview",
                created_at_utc=protocol.created_at_utc,
                repository_state=FIXED_REPOSITORY,
                python_version="3.12.13",
                dependency_versions=FIXED_DEPENDENCIES,
            )

            self.assertEqual(identity.control_profile, "profile-v2")
            self.assertEqual(identity.forecast_source, "learned")
            self.assertEqual(identity.forecast_model_version, "lstm_dual_head_preview")
            self.assertEqual(
                identity.input_files["model"]["forecast_lstm_best.pt"].status,
                "present",
            )
            self.assertEqual(identity.input_files["dataset"]["metadata.json"].status, "present")
            self.assertEqual(
                identity.input_files["configuration"]["planner_runtime_config"].status,
                "present",
            )
            self.assertEqual(identity.input_files["cases"]["cases_csv"].status, "present")
            self.assertEqual(identity.input_files["plant"]["stiffness_file"].status, "present")
            self.assertEqual(len(identity.effective_config_sha256), 64)
            self.assertEqual(len(identity.run_id), 64)

            identity_path = write_run_identity(
                root / "outputs" / "casebook" / "run_identity.json",
                identity,
            )
            payload = json.loads(identity_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], identity.run_id)

    def test_result_updates_do_not_change_stable_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, cfg, protocol, dataset_dir = self._fixture(root)
            first = build_casebook_run_identity(
                protocol=protocol,
                args=args,
                planner_config=cfg,
                repo_root=root,
                out_dir=root / "outputs" / "casebook",
                dataset_dir=dataset_dir,
                forecast_source_effective="lstm_dual_head_preview",
                created_at_utc="2026-08-11T12:00:00Z",
                repository_state=FIXED_REPOSITORY,
                python_version="3.12.13",
                dependency_versions=FIXED_DEPENDENCIES,
            )
            changed_result_protocol = build_casebook_run_protocol(
                args=args,
                cfg=cfg,
                repo_root=root,
                out_dir=root / "outputs" / "casebook",
                dataset_dir=dataset_dir,
                closed_pump_profile="baseline",
                primary_pump_profile="matched",
                primary_scale=1.0,
                forecast_source_effective="lstm_dual_head_preview",
                completed_cases=1,
                issue_count=2,
                elapsed_s=999.0,
                created_at_utc="2026-08-11T13:00:00Z",
            )
            second = build_casebook_run_identity(
                protocol=changed_result_protocol,
                args=args,
                planner_config=cfg,
                repo_root=root,
                out_dir=root / "outputs" / "casebook",
                dataset_dir=dataset_dir,
                forecast_source_effective="lstm_dual_head_preview",
                created_at_utc="2026-08-11T13:00:00Z",
                repository_state=FIXED_REPOSITORY,
                python_version="3.12.13",
                dependency_versions=FIXED_DEPENDENCIES,
            )

            self.assertEqual(first.run_id, second.run_id)
            self.assertEqual(protocol.to_dict()["schema_version"], "run_protocol.v1")
            self.assertNotIn("run_id", protocol.to_dict())

    def test_planner_runtime_content_change_changes_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, cfg, protocol, dataset_dir = self._fixture(root)

            def build() -> RunIdentity:
                return build_casebook_run_identity(
                    protocol=protocol,
                    args=args,
                    planner_config=cfg,
                    repo_root=root,
                    out_dir=root / "outputs" / "casebook",
                    dataset_dir=dataset_dir,
                    forecast_source_effective="lstm_dual_head_preview",
                    created_at_utc="2026-08-11T12:00:00Z",
                    repository_state=FIXED_REPOSITORY,
                    python_version="3.12.13",
                    dependency_versions=FIXED_DEPENDENCIES,
                )

            first = build()
            Path(args.planner_runtime_config).write_text(
                '{"schema_version":"planner_runtime.v2","lead_reliability":true}',
                encoding="utf-8",
            )
            second = build()

            self.assertNotEqual(
                first.input_files["configuration"]["planner_runtime_config"].sha256,
                second.input_files["configuration"]["planner_runtime_config"].sha256,
            )
            self.assertNotEqual(first.run_id, second.run_id)


if __name__ == "__main__":
    unittest.main()
