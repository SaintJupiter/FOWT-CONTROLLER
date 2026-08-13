"""Compatibility imports for the frozen legacy platform chain.

New code should import these helpers from :mod:`fowt_platform`.  The legacy
plant still uses this historical module path, so it remains as a thin alias
without carrying a second implementation.
"""

from fowt_platform.ballast import (
    BallastMassProperties,
    compute_ballast_mass_properties,
    compute_incremental_ballast_mass_properties,
)

__all__ = [
    "BallastMassProperties",
    "compute_ballast_mass_properties",
    "compute_incremental_ballast_mass_properties",
]
