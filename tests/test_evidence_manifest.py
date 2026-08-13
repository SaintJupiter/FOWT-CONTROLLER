from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from wind_prediction.evidence_manifest import (
    build_evidence_manifest,
    verify_evidence_manifest,
)


class EvidenceManifestTests(unittest.TestCase):
    def _repo(self, root: Path) -> None:
        subprocess.run(("git", "init", "-q", str(root)), check=True)
        subprocess.run(("git", "-C", str(root), "config", "user.email", "test@example.com"), check=True)
        subprocess.run(("git", "-C", str(root), "config", "user.name", "Test User"), check=True)

    def _commit(self, root: Path, *paths: str) -> None:
        subprocess.run(("git", "-C", str(root), "add", "--", *paths), check=True)
        subprocess.run(("git", "-C", str(root), "commit", "-qm", "fixture"), check=True)

    def _source_declaration(self, **file_set_overrides: object) -> dict[str, object]:
        file_set: dict[str, object] = {
            "id": "source",
            "storage_class": "git",
            "paths": ["source.py"],
        }
        file_set.update(file_set_overrides)
        return {
            "schema_version": "evidence_freeze_declaration.v1",
            "purpose": "test",
            "dependency_names": [],
            "file_sets": [file_set],
        }

    def test_manifest_id_ignores_capture_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
            self._commit(root, "source.py")
            declaration = self._source_declaration()
            first = build_evidence_manifest(declaration, repo_root=root, created_at_utc="2026-08-13T00:00:00Z")
            second = build_evidence_manifest(declaration, repo_root=root, created_at_utc="2026-08-13T01:00:00Z")
            self.assertEqual(first["manifest_id"], second["manifest_id"])
            self.assertEqual(first["validation"]["status"], "passed")

    def test_untracked_git_file_keeps_candidate_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "README.md").write_text("fixture\n", encoding="utf-8")
            self._commit(root, "README.md")
            (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
            manifest = build_evidence_manifest(self._source_declaration(), repo_root=root)
            self.assertEqual(manifest["validation"]["status"], "incomplete")
            self.assertEqual(manifest["file_sets"][0]["files"][0]["git_state"], "untracked")

    def test_required_unmatched_glob_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "README.md").write_text("fixture\n", encoding="utf-8")
            self._commit(root, "README.md")
            declaration = self._source_declaration(paths=[], globs=["tests/test_*.py"])
            manifest = build_evidence_manifest(declaration, repo_root=root)
            self.assertEqual(manifest["validation"]["status"], "failed")
            self.assertEqual(manifest["validation"]["unmatched_globs"], ["source:tests/test_*.py"])

    def test_json_assertion_failure_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "evidence.json").write_text(json.dumps({"ok": False}), encoding="utf-8")
            self._commit(root, "source.py", "evidence.json")
            declaration = self._source_declaration()
            declaration["evidence_assertions"] = [
                {"id": "chain", "path": "evidence.json", "json_checks": [{"pointer": "/ok", "equals": True}]}
            ]
            manifest = build_evidence_manifest(declaration, repo_root=root)
            self.assertEqual(manifest["validation"]["status"], "failed")
            self.assertEqual(manifest["validation"]["failed_assertions"], ["chain"])

    def test_changed_file_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            source = root / "source.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            self._commit(root, "source.py")
            manifest = build_evidence_manifest(self._source_declaration(), repo_root=root)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            result = verify_evidence_manifest(manifest, repo_root=root)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(any(item["field"] == "sha256" for item in result["record_mismatches"]))

    def test_file_list_tampering_cannot_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
            self._commit(root, "source.py")
            manifest = build_evidence_manifest(self._source_declaration(), repo_root=root)
            manifest["file_sets"][0]["files"] = []
            with self.assertRaisesRegex(ValueError, "do not match expanded_paths"):
                verify_evidence_manifest(manifest, repo_root=root)

    def test_validation_tampering_invalidates_manifest_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "README.md").write_text("fixture\n", encoding="utf-8")
            self._commit(root, "README.md")
            (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
            manifest = build_evidence_manifest(self._source_declaration(), repo_root=root)
            self.assertEqual(manifest["validation"]["status"], "incomplete")
            manifest["validation"]["status"] = "passed"
            result = verify_evidence_manifest(manifest, repo_root=root)
            self.assertFalse(result["manifest_id_valid"])
            self.assertEqual(result["status"], "failed")

    def test_verify_preserves_recorded_incomplete_freeze_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "README.md").write_text("fixture\n", encoding="utf-8")
            self._commit(root, "README.md")
            (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
            manifest = build_evidence_manifest(self._source_declaration(), repo_root=root)
            result = verify_evidence_manifest(manifest, repo_root=root)
            self.assertEqual(result["integrity_status"], "passed")
            self.assertEqual(result["freeze_status"], "incomplete")
            self.assertEqual(result["status"], "incomplete")

    def test_paths_cannot_escape_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "README.md").write_text("fixture\n", encoding="utf-8")
            self._commit(root, "README.md")
            declaration = self._source_declaration(storage_class="external", paths=["../outside.txt"])
            with self.assertRaisesRegex(ValueError, "within the repository"):
                build_evidence_manifest(declaration, repo_root=root)

    def test_duplicate_file_set_ids_are_rejected(self) -> None:
        declaration = self._source_declaration()
        declaration["file_sets"].append(copy.deepcopy(declaration["file_sets"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate file set id"):
            build_evidence_manifest(declaration, repo_root=Path.cwd())

    def test_invalid_storage_class_is_rejected(self) -> None:
        declaration = self._source_declaration(storage_class="gti")
        with self.assertRaisesRegex(ValueError, "invalid storage_class"):
            build_evidence_manifest(declaration, repo_root=Path.cwd())

    def test_duplicate_paths_across_file_sets_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
            self._commit(root, "source.py")
            declaration = self._source_declaration()
            declaration["file_sets"].append(
                {"id": "duplicate", "storage_class": "git", "paths": ["source.py"]}
            )
            with self.assertRaisesRegex(ValueError, "appears in multiple file sets"):
                build_evidence_manifest(declaration, repo_root=root)

    def test_empty_required_file_set_is_rejected(self) -> None:
        declaration = self._source_declaration(paths=[])
        with self.assertRaisesRegex(ValueError, "at least one path or glob"):
            build_evidence_manifest(declaration, repo_root=Path.cwd())

    def test_assertion_comparison_is_type_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "evidence.json").write_text(json.dumps({"ok": 1}), encoding="utf-8")
            self._commit(root, "source.py", "evidence.json")
            declaration = self._source_declaration()
            declaration["evidence_assertions"] = [
                {"id": "typed", "path": "evidence.json", "json_checks": [{"pointer": "/ok", "equals": True}]}
            ]
            manifest = build_evidence_manifest(declaration, repo_root=root)
            self.assertEqual(manifest["validation"]["failed_assertions"], ["typed"])

    def test_assertion_requires_at_least_one_check(self) -> None:
        declaration = self._source_declaration()
        declaration["evidence_assertions"] = [{"id": "empty", "path": "source.py", "json_checks": []}]
        with self.assertRaisesRegex(ValueError, "at least one json_check"):
            build_evidence_manifest(declaration, repo_root=Path.cwd())

    def test_glob_expansion_change_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "tests").mkdir()
            (root / "tests" / "test_one.py").write_text("VALUE = 1\n", encoding="utf-8")
            self._commit(root, "tests/test_one.py")
            declaration = self._source_declaration(paths=[], globs=["tests/test_*.py"])
            manifest = build_evidence_manifest(declaration, repo_root=root)
            (root / "tests" / "test_two.py").write_text("VALUE = 2\n", encoding="utf-8")
            result = verify_evidence_manifest(manifest, repo_root=root)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["expansion_mismatches"][0]["file_set"], "source")

    def test_test_run_is_loaded_from_result_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            result_record = {
                "scope": "unit",
                "command": "python -m unittest",
                "run": 3,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "exit_code": 0,
            }
            (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "test_result.json").write_text(json.dumps(result_record), encoding="utf-8")
            self._commit(root, "source.py", "test_result.json")
            declaration = self._source_declaration(paths=["source.py", "test_result.json"])
            declaration["test_run"] = {"result_path": "test_result.json"}
            manifest = build_evidence_manifest(declaration, repo_root=root)
            self.assertEqual(manifest["test_run"]["result"]["run"], 3)
            self.assertEqual(manifest["test_run"]["status"], "incomplete")
            self.assertEqual(manifest["validation"]["status"], "incomplete")

    def test_test_result_change_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            result_path = root / "test_result.json"
            result_path.write_text(
                json.dumps({"scope": "unit", "command": "test", "run": 1, "failures": 0, "errors": 0, "skipped": 0, "exit_code": 0}),
                encoding="utf-8",
            )
            (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
            self._commit(root, "source.py", "test_result.json")
            declaration = self._source_declaration(paths=["source.py", "test_result.json"])
            declaration["test_run"] = {"result_path": "test_result.json"}
            manifest = build_evidence_manifest(declaration, repo_root=root)
            result_path.write_text(
                json.dumps({"scope": "unit", "command": "test", "run": 2, "failures": 0, "errors": 0, "skipped": 0, "exit_code": 0}),
                encoding="utf-8",
            )
            result = verify_evidence_manifest(manifest, repo_root=root)
            self.assertFalse(result["test_run_match"])
            self.assertEqual(result["status"], "failed")

    def test_optional_file_removal_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            source = root / "source.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            self._commit(root, "source.py")
            declaration = self._source_declaration(required=False)
            manifest = build_evidence_manifest(declaration, repo_root=root)
            source.unlink()
            result = verify_evidence_manifest(manifest, repo_root=root)
            self.assertEqual(result["optional_missing"], ["source.py"])
            self.assertEqual(result["status"], "failed")

    def test_optional_unmatched_glob_is_a_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "README.md").write_text("fixture\n", encoding="utf-8")
            self._commit(root, "README.md")
            declaration = self._source_declaration(
                required=False,
                paths=[],
                globs=["optional/*.json"],
            )
            manifest = build_evidence_manifest(declaration, repo_root=root)
            self.assertEqual(manifest["validation"]["status"], "complete_with_warnings")
            self.assertEqual(
                manifest["validation"]["optional_unmatched_globs"],
                ["source:optional/*.json"],
            )

    def test_unicode_git_path_is_recognized_as_tracked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "中文.txt").write_text("fixture\n", encoding="utf-8")
            self._commit(root, "中文.txt")
            declaration = self._source_declaration(paths=["中文.txt"])
            manifest = build_evidence_manifest(declaration, repo_root=root)
            self.assertEqual(manifest["file_sets"][0]["files"][0]["git_state"], "tracked_clean")
            self.assertEqual(manifest["validation"]["status"], "passed")

    def test_file_set_order_does_not_change_manifest_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "one.py").write_text("ONE = 1\n", encoding="utf-8")
            (root / "two.py").write_text("TWO = 2\n", encoding="utf-8")
            self._commit(root, "one.py", "two.py")
            declaration = {
                "schema_version": "evidence_freeze_declaration.v1",
                "purpose": "test",
                "dependency_names": [],
                "file_sets": [
                    {"id": "one", "storage_class": "git", "paths": ["one.py"]},
                    {"id": "two", "storage_class": "git", "paths": ["two.py"]},
                ],
            }
            reversed_declaration = copy.deepcopy(declaration)
            reversed_declaration["file_sets"].reverse()
            first = build_evidence_manifest(declaration, repo_root=root)
            second = build_evidence_manifest(reversed_declaration, repo_root=root)
            self.assertEqual(first["manifest_id"], second["manifest_id"])

    def test_approved_manifest_id_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
            self._commit(root, "source.py")
            manifest = build_evidence_manifest(self._source_declaration(), repo_root=root)
            result = verify_evidence_manifest(
                manifest,
                repo_root=root,
                expected_manifest_id="0" * 64,
            )
            self.assertFalse(result["approved_manifest_id_match"])
            self.assertEqual(result["status"], "failed")

    def test_optional_file_records_cannot_be_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
            self._commit(root, "source.py")
            manifest = build_evidence_manifest(
                self._source_declaration(required=False),
                repo_root=root,
            )
            manifest["file_sets"][0]["files"] = []
            with self.assertRaisesRegex(ValueError, "do not match expanded_paths"):
                verify_evidence_manifest(manifest, repo_root=root)

    def test_duplicate_declared_paths_are_rejected(self) -> None:
        declaration = self._source_declaration(paths=["source.py", "source.py"])
        with self.assertRaisesRegex(ValueError, "duplicate declared path"):
            build_evidence_manifest(declaration, repo_root=Path.cwd())

    def test_path_and_glob_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._repo(root)
            (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
            self._commit(root, "source.py")
            declaration = self._source_declaration(globs=["*.py"])
            with self.assertRaisesRegex(ValueError, "also matched by a glob"):
                build_evidence_manifest(declaration, repo_root=root)


if __name__ == "__main__":
    unittest.main()
