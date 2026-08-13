"""Content-addressed identity records for reproducible experiment runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "run_identity.v1"
MISSING = "missing"
DEFAULT_DEPENDENCIES = (
    "joblib",
    "lightgbm",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
)
PathInputs = Mapping[str, str | Path] | Iterable[str | Path] | str | Path


@dataclass(frozen=True)
class RepositoryState:
    git_commit: str
    git_dirty: bool


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    status: str
    sha256: str


@dataclass(frozen=True)
class RunIdentity:
    """JSON-serializable identity for one effective experiment definition."""

    schema_version: str
    created_at_utc: str
    git_commit: str
    git_dirty: bool
    python_version: str
    dependency_versions: dict[str, str]
    control_profile: str
    forecast_source: str
    forecast_model_version: str
    input_files: dict[str, dict[str, FileFingerprint]]
    effective_config_sha256: str
    output_directory: str
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value independent of mapping insertion order."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_repository_state(repo_root: str | Path) -> RepositoryState:
    """Read the current commit and worktree dirty marker from a Git repository."""

    root = Path(repo_root).expanduser().resolve()

    def run_git(*args: str) -> str:
        try:
            completed = subprocess.run(
                ("git", "-C", str(root), *args),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            raise RuntimeError(
                f"cannot read Git state for {root}: {str(detail).strip()}"
            ) from exc
        return completed.stdout.strip()

    commit = run_git("rev-parse", "HEAD")
    dirty = bool(run_git("status", "--porcelain=v1", "--untracked-files=all"))
    return RepositoryState(git_commit=commit, git_dirty=dirty)


def collect_dependency_versions(
    dependency_names: Iterable[str] = DEFAULT_DEPENDENCIES,
) -> dict[str, str]:
    """Return every requested dependency, explicitly marking absent packages."""

    versions: dict[str, str] = {}
    for name in sorted({str(item) for item in dependency_names}):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = MISSING
    return versions


def _display_path(path: str | Path, repo_root: Path) -> str:
    candidate = Path(path).expanduser()
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (repo_root / candidate).resolve()
    )
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return str(resolved)


def _named_paths(
    paths: PathInputs,
) -> dict[str, str | Path]:
    if isinstance(paths, Mapping):
        return {str(name): path for name, path in paths.items()}
    if isinstance(paths, (str, Path)):
        paths = (paths,)

    named: dict[str, str | Path] = {}
    for index, path in enumerate(paths):
        candidate = Path(path)
        label = candidate.name or f"input_{index}"
        while label in named:
            label = f"{index}_{label}"
        named[label] = path
    return named


def fingerprint_files(
    paths: PathInputs,
    *,
    repo_root: str | Path,
) -> dict[str, FileFingerprint]:
    """Fingerprint every declared path without dropping missing files."""

    root = Path(repo_root).expanduser().resolve()
    fingerprints: dict[str, FileFingerprint] = {}
    for label, supplied_path in sorted(_named_paths(paths).items()):
        candidate = Path(supplied_path).expanduser()
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (root / candidate).resolve()
        )
        display_path = _display_path(supplied_path, root)
        if not resolved.is_file():
            fingerprints[label] = FileFingerprint(
                path=display_path,
                status=MISSING,
                sha256=MISSING,
            )
            continue
        fingerprints[label] = FileFingerprint(
            path=display_path,
            status="present",
            sha256=sha256_file(resolved),
        )
    return fingerprints


def _utc_timestamp(value: datetime | str | None) -> str:
    if value is None:
        instant = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        instant = value
    else:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        instant = datetime.fromisoformat(normalized)
    if instant.tzinfo is None:
        raise ValueError("created_at_utc must include a timezone")
    return instant.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def assemble_run_identity(
    *,
    created_at_utc: datetime | str,
    repository_state: RepositoryState,
    python_version: str,
    dependency_versions: Mapping[str, str],
    control_profile: str,
    forecast_source: str,
    forecast_model_version: str,
    input_files: Mapping[str, Mapping[str, FileFingerprint]],
    effective_config_sha256: str,
    output_directory: str,
) -> RunIdentity:
    """Purely assemble a stable identity from already captured run facts."""

    normalized_inputs = {
        str(group): {
            str(label): fingerprint
            for label, fingerprint in sorted(files.items())
        }
        for group, files in sorted(input_files.items())
    }
    material = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": repository_state.git_commit,
        "git_dirty": repository_state.git_dirty,
        "python_version": str(python_version),
        "dependency_versions": {
            str(name): str(version)
            for name, version in sorted(dependency_versions.items())
        },
        "control_profile": str(control_profile),
        "forecast_source": str(forecast_source),
        "forecast_model_version": str(forecast_model_version),
        "input_files": {
            group: {
                label: asdict(fingerprint)
                for label, fingerprint in files.items()
            }
            for group, files in normalized_inputs.items()
        },
        "effective_config_sha256": str(effective_config_sha256),
        "output_directory": str(output_directory),
    }
    return RunIdentity(
        schema_version=SCHEMA_VERSION,
        created_at_utc=_utc_timestamp(created_at_utc),
        git_commit=repository_state.git_commit,
        git_dirty=repository_state.git_dirty,
        python_version=str(python_version),
        dependency_versions=material["dependency_versions"],
        control_profile=str(control_profile),
        forecast_source=str(forecast_source),
        forecast_model_version=str(forecast_model_version),
        input_files=normalized_inputs,
        effective_config_sha256=str(effective_config_sha256),
        output_directory=str(output_directory),
        run_id=sha256_json(material),
    )


def build_run_identity(
    *,
    repo_root: str | Path,
    effective_config: Mapping[str, Any],
    model_files: PathInputs,
    dataset_files: PathInputs,
    additional_file_groups: Mapping[str, PathInputs] | None = None,
    output_directory: str | Path,
    control_profile: str,
    forecast_source: str,
    forecast_model_version: str,
    created_at_utc: datetime | str | None = None,
    repository_state: RepositoryState | None = None,
    python_version: str | None = None,
    dependency_versions: Mapping[str, str] | None = None,
    dependency_names: Iterable[str] = DEFAULT_DEPENDENCIES,
) -> RunIdentity:
    """Capture run inputs and return a serializable, content-addressed identity.

    The UTC timestamp records when capture occurred but is intentionally excluded
    from ``run_id`` so identical experiment definitions keep the same identity.
    This function never creates the output directory or writes a manifest.
    """

    root = Path(repo_root).expanduser().resolve()
    files = {
        "model": fingerprint_files(model_files, repo_root=root),
        "dataset": fingerprint_files(dataset_files, repo_root=root),
    }
    for group, paths in sorted((additional_file_groups or {}).items()):
        group_name = str(group)
        if group_name in files:
            raise ValueError(f"duplicate run identity file group: {group_name}")
        files[group_name] = fingerprint_files(paths, repo_root=root)
    return assemble_run_identity(
        created_at_utc=created_at_utc or datetime.now(timezone.utc),
        repository_state=repository_state or collect_repository_state(root),
        python_version=python_version or platform.python_version(),
        dependency_versions=(
            dict(dependency_versions)
            if dependency_versions is not None
            else collect_dependency_versions(dependency_names)
        ),
        control_profile=control_profile,
        forecast_source=forecast_source,
        forecast_model_version=forecast_model_version,
        input_files=files,
        effective_config_sha256=sha256_json(effective_config),
        output_directory=_display_path(output_directory, root),
    )
