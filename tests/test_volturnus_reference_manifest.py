import hashlib
import json
import re
import unittest
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "configs"
    / "reference_platforms"
    / "volturnus_s_openfast_v1_1_16.json"
)


class VolturnusReferenceManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.model_zip = ROOT / cls.manifest["sources"]["model_zip"]["path"]

    @classmethod
    def _member_text(cls, member_key):
        member = cls.manifest["archive_members"][member_key]
        with zipfile.ZipFile(cls.model_zip) as archive:
            return archive.read(member).decode("utf-8")

    @staticmethod
    def _named_scalar(text, field):
        pattern = re.compile(
            rf"^\s*([-+0-9.Ee]+)\s+{re.escape(field)}(?:\s|$)",
            re.MULTILINE,
        )
        match = pattern.search(text)
        if match is None:
            raise AssertionError(f"missing source field {field}")
        return float(match.group(1))

    def test_manifest_is_evidence_only(self):
        self.assertEqual(
            self.manifest["identity"]["status"],
            "evidence_only_not_yet_a_controller_plant",
        )
        self.assertEqual(
            self.manifest["manifest_scope"]["role"],
            "source_facts_only_no_runtime_assembly",
        )
        self.assertEqual(
            self.manifest["manifest_scope"]["runtime_assembly_status"],
            "not_defined_pending_user_confirmation",
        )

    def test_source_hashes_match_local_frozen_evidence(self):
        for source in self.manifest["sources"].values():
            path = ROOT / source["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, source["sha256"])

    def test_direct_model_values_are_reextracted_from_openfast_member(self):
        values = self.manifest["direct_model_values"]
        elastodyn = self._member_text("elastodyn")
        hydrodyn = self._member_text("hydrodyn")

        self.assertEqual(
            values["platform_mass_kg"],
            self._named_scalar(elastodyn, "PtfmMass"),
        )
        np.testing.assert_array_equal(
            values["platform_center_of_mass_m"],
            [
                self._named_scalar(elastodyn, "PtfmCMxt"),
                self._named_scalar(elastodyn, "PtfmCMyt"),
                self._named_scalar(elastodyn, "PtfmCMzt"),
            ],
        )
        np.testing.assert_array_equal(
            np.diag(values["platform_inertia_about_platform_cm_kg_m2"]),
            [
                self._named_scalar(elastodyn, "PtfmRIner"),
                self._named_scalar(elastodyn, "PtfmPIner"),
                self._named_scalar(elastodyn, "PtfmYIner"),
            ],
        )
        self.assertEqual(
            values["displaced_volume_m3"],
            self._named_scalar(hydrodyn, "PtfmVol0"),
        )
        self.assertEqual(self._named_scalar(hydrodyn, "WAMITULEN"), 1.0)

    def test_wamit_matrices_are_reextracted_and_redimensionalized(self):
        values = self.manifest["direct_model_values"]
        rho = 1025.0
        gravity = 9.81

        hydrostatic_source = np.zeros((6, 6), dtype=float)
        for line in self._member_text("wamit_hydrostatic").splitlines():
            fields = line.split()
            if len(fields) != 3:
                continue
            i, j, value = int(fields[0]), int(fields[1]), float(fields[2])
            hydrostatic_source[i - 1, j - 1] = value * rho * gravity
        hydrostatic_manifest = np.asarray(
            values["hydrostatic_hull_stiffness_about_prp"], dtype=float
        )
        np.testing.assert_allclose(
            hydrostatic_manifest,
            hydrostatic_source,
            rtol=self.manifest["value_provenance"][
                "hydrostatic_hull_stiffness_about_prp"
            ]["relative_transcription_tolerance"],
            atol=1.0,
        )

        added_mass_source = np.zeros((6, 6), dtype=float)
        for line in self._member_text("wamit_added_mass").splitlines():
            fields = line.split()
            if len(fields) < 4 or float(fields[0]) != 0.0:
                continue
            i, j, value = int(fields[1]), int(fields[2]), float(fields[3])
            added_mass_source[i - 1, j - 1] = value * rho
        added_mass_source = 0.5 * (added_mass_source + added_mass_source.T)
        drop_threshold = np.max(np.abs(added_mass_source)) * 1e-5
        added_mass_source[np.abs(added_mass_source) < drop_threshold] = 0.0
        added_mass_manifest = np.asarray(
            values["infinite_frequency_added_mass_about_prp"], dtype=float
        )
        np.testing.assert_allclose(
            added_mass_manifest,
            added_mass_source,
            rtol=self.manifest["value_provenance"][
                "infinite_frequency_added_mass_about_prp"
            ]["relative_transcription_tolerance"],
            atol=1.0,
        )

    def test_auxiliary_wamit_matrices_are_reextracted(self):
        lines = self._member_text("wamit_second_order_force_control").splitlines()
        rigid_mass = np.asarray(
            [[float(value) for value in lines[index].split()] for index in range(5, 11)]
        )
        mooring = np.asarray(
            [[float(value) for value in lines[index].split()] for index in range(19, 25)]
        )
        auxiliary = self.manifest["auxiliary_wamit_second_order_values"]

        np.testing.assert_array_equal(
            auxiliary["whole_system_rigid_body_mass_matrix_about_prp"],
            rigid_mass,
        )
        np.testing.assert_array_equal(
            auxiliary["linearized_mooring_stiffness_about_prp"],
            mooring,
        )

    def test_source_matrix_transcriptions_have_expected_numerical_properties(self):
        values = self.manifest["direct_model_values"]
        added_mass = np.asarray(
            values["infinite_frequency_added_mass_about_prp"],
            dtype=float,
        )
        rigid_mass = np.asarray(
            self.manifest["auxiliary_wamit_second_order_values"][
                "whole_system_rigid_body_mass_matrix_about_prp"
            ],
            dtype=float,
        )
        hydrostatic = np.asarray(
            values["hydrostatic_hull_stiffness_about_prp"],
            dtype=float,
        )
        mooring = np.asarray(
            self.manifest["auxiliary_wamit_second_order_values"][
                "linearized_mooring_stiffness_about_prp"
            ],
            dtype=float,
        )

        for matrix in (added_mass, rigid_mass):
            np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=1e-9)
        for raw_matrix in (hydrostatic, mooring):
            asymmetry = np.max(np.abs(raw_matrix - raw_matrix.T))
            dominant_stiffness = np.max(np.abs(raw_matrix))
            self.assertLess(asymmetry / dominant_stiffness, 1e-3)
        self.assertGreater(np.min(np.linalg.eigvalsh(rigid_mass)), 0.0)
        self.assertEqual(
            self.manifest["representation_groups"]["whole_system_rigid_body_properties"]["assembly_status"],
            "source_fact_not_selected",
        )
        self.assertEqual(
            self.manifest["representation_groups"]["hydrodynamic_added_mass"]["assembly_status"],
            "source_fact_not_selected",
        )

    def test_active_tank_values_are_not_labeled_direct_mapping(self):
        boundary = self.manifest["active_ballast_boundary"]
        self.assertEqual(
            boundary["local_three_point_tank_geometry_status"],
            "research_assumption_not_direct_mapping",
        )
        self.assertEqual(
            boundary["local_variable_pump_schedule_status"],
            "research_assumption_not_public_reference_parameter",
        )


if __name__ == "__main__":
    unittest.main()
