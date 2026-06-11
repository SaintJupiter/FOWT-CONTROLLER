"""Profile registry gate for prediction-primary casebook entry points."""

from __future__ import annotations

import json
import os
from pathlib import Path


CASEBOOK_PROFILE_REGISTRY_RELATIVE = Path("configs") / "casebook_profile_registry_v1.json"
PROFILE_GATE_DIRECT_ALLOWED_GROUPS = (
    "production",
    "diagnostic",
    "historical_diagnostic",
)
PROFILE_GATE_LOCKED_GROUPS = (
    "isolated_experiment",
    "prune_candidate",
)
PROFILE_GATE_GROUPS = (
    *PROFILE_GATE_DIRECT_ALLOWED_GROUPS,
    *PROFILE_GATE_LOCKED_GROUPS,
)
ISOLATED_PROFILE_UNLOCK_ENV = "FOWT_ALLOW_ISOLATED_PROFILE"


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def load_casebook_profile_registry(registry_path: Path) -> dict[str, set[str]]:
    if not registry_path.exists():
        raise FileNotFoundError(f"profile registry missing: {registry_path}")
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    groups: dict[str, set[str]] = {}
    for group in PROFILE_GATE_GROUPS:
        values = raw.get(group, [])
        if not isinstance(values, list):
            raise TypeError(f"registry group {group!r} must be a list")
        groups[group] = {str(value) for value in values}
    return groups


def profile_registry_group(profile: str, registry: dict[str, set[str]]) -> str | None:
    hits = [group for group, profiles in registry.items() if profile in profiles]
    if len(hits) > 1:
        raise ValueError(f"profile {profile!r} appears in multiple registry groups: {hits}")
    return hits[0] if hits else None


def _display_path(path: Path, repo_root: Path | None) -> str:
    if repo_root is None:
        return str(path)
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def enforce_casebook_profile_gate(
    profile: str,
    *,
    allow_isolated_profile: bool,
    registry_path: Path,
    repo_root: Path | None = None,
) -> str:
    registry = load_casebook_profile_registry(registry_path)
    group = profile_registry_group(str(profile), registry)
    registry_label = _display_path(registry_path, repo_root)
    if group is None:
        raise ValueError(
            f"Primary control profile {profile!r} is not registered in "
            f"{registry_label}. Add it to production, diagnostic, "
            "historical_diagnostic, isolated_experiment, or prune_candidate "
            "before running it."
        )
    if group in PROFILE_GATE_DIRECT_ALLOWED_GROUPS:
        return group
    if group in PROFILE_GATE_LOCKED_GROUPS:
        unlocked = bool(allow_isolated_profile) or env_bool(
            ISOLATED_PROFILE_UNLOCK_ENV,
            False,
        )
        if unlocked:
            return group
        raise ValueError(
            f"Primary control profile {profile!r} is classified as {group!r}; "
            "it is locked out of the normal controller entry. Re-run with "
            "--allow-isolated-profile only for an explicit isolated experiment."
        )
    raise ValueError(f"Unsupported profile registry group {group!r} for {profile!r}.")
