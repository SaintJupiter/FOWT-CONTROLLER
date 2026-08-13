"""Read-only reference-platform components and explicit low-order assembly."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .ballast import (
    BallastMassProperties,
    compute_incremental_ballast_mass_properties,
    rigid_body_mass_matrix_about_reference,
    weight_stiffness_about_reference,
)
from .incremental import PlatformMatrices


def _readonly_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"reference value must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("reference value must contain only finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _symmetrized(
    name: str,
    value: np.ndarray,
    *,
    maximum_relative_asymmetry: float,
) -> np.ndarray:
    scale = max(float(np.linalg.norm(value, ord="fro")), 1.0)
    relative_asymmetry = float(np.linalg.norm(value - value.T, ord="fro")) / scale
    if relative_asymmetry > maximum_relative_asymmetry:
        raise ValueError(
            f"{name} relative asymmetry {relative_asymmetry:.3e} exceeds "
            f"the allowed {maximum_relative_asymmetry:.3e}"
        )
    result = 0.5 * (value + value.T)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ReferenceVerticalBalanceEvidence:
    """Rounded report values used only to audit the nominal vertical balance."""

    total_system_mass_kg: float
    displaced_volume_m3: float
    seawater_density_kg_m3: float
    gravity_m_s2: float
    mooring_vertical_pretension_n: float

    @property
    def residual_n(self) -> float:
        buoyancy = (
            self.seawater_density_kg_m3
            * self.gravity_m_s2
            * self.displaced_volume_m3
        )
        weight = self.total_system_mass_kg * self.gravity_m_s2
        return buoyancy - weight - self.mooring_vertical_pretension_n


@dataclass(frozen=True)
class VolturnusReferenceComponents:
    """Source components kept separate until one assembly is selected."""

    model_version: str
    reference_point: str
    platform_only_mass_kg: float
    platform_only_center_of_mass: Any
    platform_only_inertia_about_cm: Any
    whole_system_mass_kg: float
    whole_system_center_of_mass: Any
    whole_system_rigid_body_mass: Any
    added_mass: Any
    hydrostatic_stiffness: Any
    mooring_stiffness: Any
    vertical_balance_evidence: ReferenceVerticalBalanceEvidence

    def __post_init__(self) -> None:
        for name in ("platform_only_mass_kg", "whole_system_mass_kg"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "platform_only_center_of_mass",
            _readonly_array(self.platform_only_center_of_mass, (3,)),
        )
        object.__setattr__(
            self,
            "platform_only_inertia_about_cm",
            _readonly_array(self.platform_only_inertia_about_cm, (3, 3)),
        )
        object.__setattr__(
            self,
            "whole_system_center_of_mass",
            _readonly_array(self.whole_system_center_of_mass, (3,)),
        )
        for name in (
            "whole_system_rigid_body_mass",
            "added_mass",
            "hydrostatic_stiffness",
            "mooring_stiffness",
        ):
            object.__setattr__(
                self,
                name,
                _readonly_array(getattr(self, name), (6, 6)),
            )

    def assemble_whole_system_candidate(self, damping: Any) -> PlatformMatrices:
        """Assemble a reviewable whole-system low-order matrix candidate."""

        if damping is None:
            raise ValueError("damping must be provided explicitly")
        return PlatformMatrices(
            mass=self.whole_system_rigid_body_mass + self.added_mass,
            damping=damping,
            hydrostatic_stiffness=_symmetrized(
                "hydrostatic_stiffness",
                self.hydrostatic_stiffness,
                maximum_relative_asymmetry=1.0e-4,
            ),
            mooring_stiffness=_symmetrized(
                "mooring_stiffness",
                self.mooring_stiffness,
                maximum_relative_asymmetry=1.0e-4,
            ),
            weight_stiffness=weight_stiffness_about_reference(
                total_mass_kg=self.whole_system_mass_kg,
                center_of_mass_m=self.whole_system_center_of_mass,
            ),
        )

    def mass_properties_with_ballast_deltas(
        self,
        *,
        tank_mass_deltas_kg: Any,
        tank_coordinates_m: Any,
    ) -> BallastMassProperties:
        """Apply research-tank mass changes to the public working reference."""

        return compute_incremental_ballast_mass_properties(
            reference_mass_kg=self.whole_system_mass_kg,
            reference_center_of_mass_m=self.whole_system_center_of_mass,
            reference_inertia_about_reference_kg_m2=(
                self.whole_system_rigid_body_mass[3:, 3:]
            ),
            tank_mass_deltas_kg=tank_mass_deltas_kg,
            tank_coordinates_m=tank_coordinates_m,
        )

    def rigid_body_mass_with_ballast_deltas(
        self,
        *,
        tank_mass_deltas_kg: Any,
        tank_coordinates_m: Any,
    ) -> np.ndarray:
        """Update the source rigid-body matrix without changing its zero delta."""

        reference_reconstruction = rigid_body_mass_matrix_about_reference(
            total_mass_kg=self.whole_system_mass_kg,
            center_of_mass_m=self.whole_system_center_of_mass,
            inertia_about_reference_kg_m2=(
                self.whole_system_rigid_body_mass[3:, 3:]
            ),
        )
        updated_properties = self.mass_properties_with_ballast_deltas(
            tank_mass_deltas_kg=tank_mass_deltas_kg,
            tank_coordinates_m=tank_coordinates_m,
        )
        updated_reconstruction = rigid_body_mass_matrix_about_reference(
            total_mass_kg=updated_properties.total_mass_kg,
            center_of_mass_m=updated_properties.center_of_mass_m,
            inertia_about_reference_kg_m2=(
                updated_properties.inertia_about_reference_kg_m2
            ),
        )
        result = self.whole_system_rigid_body_mass + (
            updated_reconstruction - reference_reconstruction
        )
        result = 0.5 * (result + result.T)
        if np.min(np.linalg.eigvalsh(result)) <= 0.0:
            raise ValueError(
                "ballast changes must preserve a positive-definite rigid-body mass"
            )
        result.setflags(write=False)
        return result

    def assemble_ballast_adjusted_candidate(
        self,
        *,
        damping: Any,
        tank_mass_deltas_kg: Any,
        tank_coordinates_m: Any,
    ) -> PlatformMatrices:
        """Assemble the low-order candidate at an explicit ballast state."""

        if damping is None:
            raise ValueError("damping must be provided explicitly")
        rigid_body_mass = self.rigid_body_mass_with_ballast_deltas(
            tank_mass_deltas_kg=tank_mass_deltas_kg,
            tank_coordinates_m=tank_coordinates_m,
        )
        updated_properties = self.mass_properties_with_ballast_deltas(
            tank_mass_deltas_kg=tank_mass_deltas_kg,
            tank_coordinates_m=tank_coordinates_m,
        )
        return PlatformMatrices(
            mass=rigid_body_mass + self.added_mass,
            damping=damping,
            hydrostatic_stiffness=_symmetrized(
                "hydrostatic_stiffness",
                self.hydrostatic_stiffness,
                maximum_relative_asymmetry=1.0e-4,
            ),
            mooring_stiffness=_symmetrized(
                "mooring_stiffness",
                self.mooring_stiffness,
                maximum_relative_asymmetry=1.0e-4,
            ),
            weight_stiffness=weight_stiffness_about_reference(
                total_mass_kg=updated_properties.total_mass_kg,
                center_of_mass_m=updated_properties.center_of_mass_m,
            ),
        )


def load_volturnus_reference_components(
    manifest_path: str | Path,
) -> VolturnusReferenceComponents:
    """Load source facts without silently promoting the manifest to a plant."""

    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = payload["identity"]
    if identity.get("status") != "evidence_only_not_yet_a_controller_plant":
        raise ValueError("expected an evidence-only source manifest")
    if payload["manifest_scope"].get("role") != "source_facts_only_no_runtime_assembly":
        raise ValueError("expected an evidence-only source manifest")

    direct = payload["direct_model_values"]
    auxiliary = payload["auxiliary_wamit_second_order_values"]
    balance = payload["report_static_balance_evidence"]
    return VolturnusReferenceComponents(
        model_version=str(identity["model_version"]),
        reference_point=str(identity["reference_point"]),
        platform_only_mass_kg=float(direct["platform_mass_kg"]),
        platform_only_center_of_mass=direct["platform_center_of_mass_m"],
        platform_only_inertia_about_cm=direct[
            "platform_inertia_about_platform_cm_kg_m2"
        ],
        whole_system_mass_kg=float(auxiliary["whole_system_mass_kg_rounded"]),
        whole_system_center_of_mass=auxiliary["whole_system_center_of_mass_m"],
        whole_system_rigid_body_mass=auxiliary[
            "whole_system_rigid_body_mass_matrix_about_prp"
        ],
        added_mass=direct["infinite_frequency_added_mass_about_prp"],
        hydrostatic_stiffness=direct["hydrostatic_hull_stiffness_about_prp"],
        mooring_stiffness=auxiliary["linearized_mooring_stiffness_about_prp"],
        vertical_balance_evidence=ReferenceVerticalBalanceEvidence(
            total_system_mass_kg=float(balance["total_system_mass_kg"]),
            displaced_volume_m3=float(balance["displaced_volume_m3"]),
            seawater_density_kg_m3=float(balance["seawater_density_kg_m3"]),
            gravity_m_s2=float(balance["gravity_m_s2"]),
            mooring_vertical_pretension_n=float(
                balance["total_mooring_vertical_pretension_n"]
            ),
        ),
    )
