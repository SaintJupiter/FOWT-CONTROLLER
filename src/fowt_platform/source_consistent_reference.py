"""Static reference assembly derived from one OpenFAST source set.

The historical :mod:`fowt_platform.reference` candidate intentionally keeps
first-order WAMIT matrices separate from an auxiliary second-order ``.frc``
representation.  It remains available for reproducibility, but its rigid-body
mass properties and its hydrostatic matrix do not describe one verified object
scope.  This module is deliberately narrower: it derives the mass inventory,
vertical centre of mass and static restoring terms from the frozen ElastoDyn
and first-order WAMIT inputs only.

It is a static pitch/roll reference, not a complete six-degree-of-freedom
plant.  In particular, it supplies no mooring model, no empirical damping and
no full turbine rigid-body inertia approximation.  Those pieces will be added
only after their source and reference point are independently established.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import zipfile

import numpy as np

from .ballast import (
    BallastMassProperties,
    rigid_body_mass_matrix_about_reference,
    weight_stiffness_about_reference,
)
from .ballast_snapshot import BallastRuntimeAssembly
from .incremental import PlatformMatrices
from .reference import load_volturnus_reference_components


def _readonly_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"value must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("value must contain only finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _repository_root(path: Path) -> Path:
    for parent in (path, *path.parents):
        if (parent / "src").is_dir() and (parent / "configs").is_dir():
            return parent
    raise ValueError(f"could not find repository root above {path}")


def _named_scalar(text: str, field: str) -> float:
    pattern = re.compile(
        rf"^\s*([-+0-9.Ee]+)\s+{re.escape(field)}(?:\s|$)",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"OpenFAST input is missing {field}")
    return float(match.group(1))


def _numeric_table_after_heading(
    text: str,
    *,
    heading: str,
    columns: int,
) -> np.ndarray:
    """Read a numeric OpenFAST table while skipping its unit row."""

    reading = False
    rows: list[list[float]] = []
    for line in text.splitlines():
        if re.match(rf"\s*{re.escape(heading)}\b", line):
            reading = True
            continue
        if not reading:
            continue
        fields = line.split()
        if not fields:
            continue
        try:
            row = [float(value) for value in fields[:columns]]
        except ValueError:
            if rows:
                break
            continue
        if len(row) != columns:
            if rows:
                break
            continue
        rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"OpenFAST table after {heading} has fewer than two rows")
    return np.asarray(rows, dtype=float)


def _hydrostatic_from_wamit(
    text: str,
    *,
    seawater_density_kg_m3: float,
    gravity_m_s2: float,
) -> np.ndarray:
    matrix = np.zeros((6, 6), dtype=float)
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            row, column = int(fields[0]), int(fields[1])
            coefficient = float(fields[2])
        except ValueError:
            continue
        if not 1 <= row <= 6 or not 1 <= column <= 6:
            raise ValueError("WAMIT hydrostatic index is outside 1..6")
        matrix[row - 1, column - 1] = (
            coefficient * seawater_density_kg_m3 * gravity_m_s2
        )
    result = 0.5 * (matrix + matrix.T)
    result.setflags(write=False)
    return result


def _infinite_frequency_added_mass_from_wamit(
    text: str,
    *,
    seawater_density_kg_m3: float,
) -> np.ndarray:
    matrix = np.zeros((6, 6), dtype=float)
    found = False
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            period = float(fields[0])
            row, column = int(fields[1]), int(fields[2])
            coefficient = float(fields[3])
        except ValueError:
            continue
        if period != 0.0:
            continue
        if not 1 <= row <= 6 or not 1 <= column <= 6:
            raise ValueError("WAMIT added-mass index is outside 1..6")
        matrix[row - 1, column - 1] = coefficient * seawater_density_kg_m3
        found = True
    if not found:
        raise ValueError("WAMIT file has no infinite-frequency added-mass entries")
    result = 0.5 * (matrix + matrix.T)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class StaticMassItem:
    """One mass contribution used only in the static vertical inventory."""

    name: str
    mass_kg: float
    vertical_position_m: float
    source_description: str

    def __post_init__(self) -> None:
        mass = float(self.mass_kg)
        vertical_position = float(self.vertical_position_m)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("mass_kg must be finite and positive")
        if not np.isfinite(vertical_position):
            raise ValueError("vertical_position_m must be finite")
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "vertical_position_m", vertical_position)


@dataclass(frozen=True)
class SourceConsistentStaticReference:
    """ElastoDyn/WAMIT static reference about the SWL platform reference point."""

    model_version: str
    reference_point: str
    archive_sha256: str
    gravity_m_s2: float
    seawater_density_kg_m3: float
    mass_items: tuple[StaticMassItem, ...]
    total_mass_kg: float
    vertical_center_of_mass_m: float
    hydrostatic_stiffness: Any
    infinite_frequency_added_mass: Any

    def __post_init__(self) -> None:
        total_mass = float(self.total_mass_kg)
        vertical_center = float(self.vertical_center_of_mass_m)
        gravity = float(self.gravity_m_s2)
        seawater_density = float(self.seawater_density_kg_m3)
        if not np.isfinite(total_mass) or total_mass <= 0.0:
            raise ValueError("total_mass_kg must be finite and positive")
        if not np.isfinite(vertical_center):
            raise ValueError("vertical_center_of_mass_m must be finite")
        if not np.isfinite(gravity) or gravity <= 0.0:
            raise ValueError("gravity_m_s2 must be finite and positive")
        if not np.isfinite(seawater_density) or seawater_density <= 0.0:
            raise ValueError("seawater_density_kg_m3 must be finite and positive")
        if not self.mass_items:
            raise ValueError("mass_items must not be empty")
        inventory_mass = sum(item.mass_kg for item in self.mass_items)
        if not np.isclose(total_mass, inventory_mass, rtol=1.0e-12, atol=1.0e-6):
            raise ValueError("total_mass_kg must equal the listed mass inventory")
        object.__setattr__(self, "total_mass_kg", total_mass)
        object.__setattr__(self, "vertical_center_of_mass_m", vertical_center)
        object.__setattr__(self, "gravity_m_s2", gravity)
        object.__setattr__(self, "seawater_density_kg_m3", seawater_density)
        object.__setattr__(
            self,
            "hydrostatic_stiffness",
            _readonly_array(self.hydrostatic_stiffness, (6, 6)),
        )
        object.__setattr__(
            self,
            "infinite_frequency_added_mass",
            _readonly_array(self.infinite_frequency_added_mass, (6, 6)),
        )

    @property
    def weight_stiffness(self) -> np.ndarray:
        """Return small-angle gravity stiffness using the source-derived CG."""

        return weight_stiffness_about_reference(
            total_mass_kg=self.total_mass_kg,
            center_of_mass_m=[0.0, 0.0, self.vertical_center_of_mass_m],
            gravity_m_s2=self.gravity_m_s2,
        )

    @property
    def restoring_stiffness_without_mooring(self) -> np.ndarray:
        """Return hull hydrostatics plus the same-source gravity contribution."""

        result = self.hydrostatic_stiffness + self.weight_stiffness
        result = 0.5 * (result + result.T)
        result.setflags(write=False)
        return result

    @property
    def pitch_restoring_stiffness_without_mooring_nm_per_rad(self) -> float:
        """Effective pitch stiffness after allowing hydrostatic heave coupling."""

        reduced = self.restoring_stiffness_without_mooring[np.ix_((2, 4), (2, 4))]
        unit_moment = np.array([0.0, 1.0], dtype=float)
        compliance = float(np.linalg.solve(reduced, unit_moment)[1])
        if not np.isfinite(compliance) or compliance <= 0.0:
            raise ValueError("static pitch compliance must be finite and positive")
        return 1.0 / compliance


def load_volturnus_source_consistent_static_reference(
    manifest_path: str | Path,
    *,
    verify_archive_hash: bool = True,
) -> SourceConsistentStaticReference:
    """Build a narrow static reference without using the auxiliary ``.frc`` file.

    Blade mass is represented at the rotor apex for the vertical first-moment
    inventory.  The precone/shaft projection of the blade mass centre changes
    the assembled vertical CG by only millimetres here, whereas the public
    input does not provide a complete six-DOF rotor inertia tensor.  This
    deliberate lumping is suitable for the static gravity check only and must
    not be promoted to a complete dynamic rigid-body representation.
    """

    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["identity"].get("status") != "evidence_only_not_yet_a_controller_plant":
        raise ValueError("expected an evidence-only source manifest")
    if payload["manifest_scope"].get("role") != "source_facts_only_no_runtime_assembly":
        raise ValueError("expected an evidence-only source manifest")

    root = _repository_root(path.resolve())
    source = payload["sources"]["model_zip"]
    archive_path = root / source["path"]
    expected_hash = str(source["sha256"])
    if verify_archive_hash:
        actual_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("frozen OpenFAST archive hash does not match the manifest")

    members = payload["archive_members"]
    required = (
        "elastodyn",
        "elastodyn_tower",
        "elastodyn_blade",
        "hydrodyn",
        "openfast_fst",
        "wamit_hydrostatic",
        "wamit_added_mass",
    )
    missing = [name for name in required if name not in members]
    if missing:
        raise ValueError(f"manifest is missing archive members: {', '.join(missing)}")

    with zipfile.ZipFile(archive_path) as archive:
        def read_member(name: str) -> str:
            return archive.read(members[name]).decode("utf-8")

        fst = read_member("openfast_fst")
        gravity_m_s2 = _named_scalar(fst, "Gravity")
        seawater_density_kg_m3 = _named_scalar(fst, "WtrDens")
        elastodyn = read_member("elastodyn")
        tower = read_member("elastodyn_tower")
        blade = read_member("elastodyn_blade")
        hydrodyn = read_member("hydrodyn")
        hydrostatic = _hydrostatic_from_wamit(
            read_member("wamit_hydrostatic"),
            seawater_density_kg_m3=seawater_density_kg_m3,
            gravity_m_s2=gravity_m_s2,
        )
        added_mass = _infinite_frequency_added_mass_from_wamit(
            read_member("wamit_added_mass"),
            seawater_density_kg_m3=seawater_density_kg_m3,
        )

    platform_reference_z = _named_scalar(elastodyn, "PtfmRefzt")
    if not np.isclose(platform_reference_z, 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "this static reference expects the ElastoDyn platform reference at SWL"
        )
    wamit_length_m = _named_scalar(hydrodyn, "WAMITULEN")
    if not np.isclose(wamit_length_m, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "this static reference only supports the frozen WAMITULEN=1 m input"
        )

    tower_height = _named_scalar(elastodyn, "TowerHt")
    tower_base_height = _named_scalar(elastodyn, "TowerBsHt")
    tower_length = tower_height - tower_base_height
    if tower_length <= 0.0:
        raise ValueError("TowerHt must exceed TowerBsHt")
    tower_table = _numeric_table_after_heading(
        tower,
        heading="HtFract",
        columns=4,
    )
    tower_fraction = tower_table[:, 0]
    tower_density = tower_table[:, 1]
    tower_z = tower_base_height + tower_fraction * tower_length
    tower_mass = float(np.trapezoid(tower_density, tower_z))
    tower_first_moment = float(np.trapezoid(tower_density * tower_z, tower_z))

    hub_radius = _named_scalar(elastodyn, "HubRad")
    tip_radius = _named_scalar(elastodyn, "TipRad")
    if tip_radius <= hub_radius:
        raise ValueError("TipRad must exceed HubRad")
    blade_table = _numeric_table_after_heading(
        blade,
        heading="BlFract",
        columns=6,
    )
    blade_span = hub_radius + blade_table[:, 0] * (tip_radius - hub_radius)
    blade_density = blade_table[:, 3]
    blade_mass = float(np.trapezoid(blade_density, blade_span))

    shaft_tilt_rad = np.deg2rad(_named_scalar(elastodyn, "ShftTilt"))
    rotor_apex_z = (
        tower_height
        + _named_scalar(elastodyn, "Twr2Shft")
        + _named_scalar(elastodyn, "OverHang") * np.sin(shaft_tilt_rad)
    )
    number_of_blades = int(_named_scalar(elastodyn, "NumBl"))
    if number_of_blades <= 0:
        raise ValueError("NumBl must be positive")

    mass_items = (
        StaticMassItem(
            name="platform",
            mass_kg=_named_scalar(elastodyn, "PtfmMass"),
            vertical_position_m=_named_scalar(elastodyn, "PtfmCMzt"),
            source_description="ElastoDyn PtfmMass and PtfmCMzt",
        ),
        StaticMassItem(
            name="tower",
            mass_kg=tower_mass,
            vertical_position_m=tower_first_moment / tower_mass,
            source_description="ElastoDyn tower mass-density table",
        ),
        StaticMassItem(
            name="nacelle",
            mass_kg=_named_scalar(elastodyn, "NacMass"),
            vertical_position_m=tower_height + _named_scalar(elastodyn, "NacCMzn"),
            source_description="ElastoDyn NacMass and NacCMzn",
        ),
        StaticMassItem(
            name="yaw_bearing",
            mass_kg=_named_scalar(elastodyn, "YawBrMass"),
            vertical_position_m=tower_height,
            source_description="ElastoDyn YawBrMass at the tower top",
        ),
        StaticMassItem(
            name="hub",
            mass_kg=_named_scalar(elastodyn, "HubMass"),
            vertical_position_m=rotor_apex_z,
            source_description="ElastoDyn HubMass at the rotor apex",
        ),
        StaticMassItem(
            name="blades",
            mass_kg=number_of_blades * blade_mass,
            vertical_position_m=rotor_apex_z,
            source_description=(
                "ElastoDyn blade mass-density table, statically lumped at the rotor apex"
            ),
        ),
    )
    total_mass = sum(item.mass_kg for item in mass_items)
    vertical_center = sum(
        item.mass_kg * item.vertical_position_m for item in mass_items
    ) / total_mass
    return SourceConsistentStaticReference(
        model_version=str(payload["identity"]["model_version"]),
        reference_point=str(payload["identity"]["reference_point"]),
        archive_sha256=expected_hash,
        gravity_m_s2=gravity_m_s2,
        seawater_density_kg_m3=seawater_density_kg_m3,
        mass_items=mass_items,
        total_mass_kg=total_mass,
        vertical_center_of_mass_m=vertical_center,
        hydrostatic_stiffness=hydrostatic,
        infinite_frequency_added_mass=added_mass,
    )


def assemble_volturnus_static_restoring_aligned_runtime_assembly(
    manifest_path: str | Path,
    damping: Any,
    *,
    verify_archive_hash: bool = True,
) -> BallastRuntimeAssembly:
    """Build the bounded runtime candidate with corrected static restoring.

    The frozen OpenFAST files provide a source-consistent inventory for total
    mass, vertical centre of mass, hull hydrostatics and added mass.  They do
    not provide a complete whole-turbine rigid-body inertia tensor in the same
    form.  This assembly therefore keeps the auxiliary second-order inertia
    tensor and the local linearized mooring matrix explicit, while replacing
    the historical mixed mass/CG and static-restoring baseline.

    It provides the current bounded runtime candidate used by the short-chain
    and static checks. Its zero-ballast static pitch restoring has been checked
    against the MoorDyn-enabled OpenFAST endpoint. The provenance deliberately
    identifies the remaining auxiliary dynamic terms; it must not be described
    as a wholly source-consistent or fully validated six-DOF model.
    """

    static_reference = load_volturnus_source_consistent_static_reference(
        manifest_path,
        verify_archive_hash=verify_archive_hash,
    )
    auxiliary_reference = load_volturnus_reference_components(manifest_path)
    inertia_about_reference = auxiliary_reference.whole_system_rigid_body_mass[
        3:, 3:
    ]
    reference_properties = BallastMassProperties(
        total_mass_kg=static_reference.total_mass_kg,
        center_of_mass_m=np.array(
            [0.0, 0.0, static_reference.vertical_center_of_mass_m],
            dtype=float,
        ),
        inertia_about_reference_kg_m2=inertia_about_reference,
    )
    rigid_body_mass = rigid_body_mass_matrix_about_reference(
        total_mass_kg=reference_properties.total_mass_kg,
        center_of_mass_m=reference_properties.center_of_mass_m,
        inertia_about_reference_kg_m2=(
            reference_properties.inertia_about_reference_kg_m2
        ),
    )
    return BallastRuntimeAssembly(
        base_matrices=PlatformMatrices(
            mass=rigid_body_mass + static_reference.infinite_frequency_added_mass,
            damping=damping,
            hydrostatic_stiffness=static_reference.hydrostatic_stiffness,
            mooring_stiffness=0.5
            * (
                auxiliary_reference.mooring_stiffness
                + auxiliary_reference.mooring_stiffness.T
            ),
            weight_stiffness=static_reference.weight_stiffness,
        ),
        reference_mass_properties=reference_properties,
        gravity_m_s2=static_reference.gravity_m_s2,
        provenance="static_restoring_aligned_with_aux_frc_inertia_mooring",
    )
