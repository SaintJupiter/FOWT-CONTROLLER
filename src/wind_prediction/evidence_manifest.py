"""Content-addressed manifests for frozen research evidence.

The manifest is deliberately stricter than a file checksum list.  It binds the
declared scope, expanded globs, repository state, evidence assertions, and test
record so a candidate cannot become "passed" by editing bookkeeping fields.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence


DECLARATION_SCHEMA_VERSION = "evidence_freeze_declaration.v1"
MANIFEST_SCHEMA_VERSION = "evidence_freeze_manifest.v1"
STORAGE_CLASSES = frozenset({"git", "external"})
FINAL_STATUSES = frozenset({"passed", "complete_with_warnings", "incomplete", "failed"})
DEFAULT_DEPENDENCIES = (
    "joblib",
    "lightgbm",
    "numpy",
    "openpyxl",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(repo_root: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo_root), "-c", "core.quotepath=false", *args),
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository_inventory(repo_root: Path) -> dict[str, Any]:
    tracked = {item for item in _run_git(repo_root, "ls-files", "-z").split("\0") if item}
    ignored = set(
        item
        for item in _run_git(
            repo_root, "ls-files", "-z", "--others", "--ignored", "--exclude-standard"
        ).split("\0")
        if item
    )
    statuses: dict[str, str] = {}
    status_text = _run_git(
        repo_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    entries = [item for item in status_text.split("\0") if item]
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if len(entry) < 4:
            continue
        code = entry[:2]
        statuses[entry[3:]] = code
        if "R" in code or "C" in code:
            index += 1
    return {
        "branch": _run_git(repo_root, "branch", "--show-current"),
        "head": _run_git(repo_root, "rev-parse", "HEAD"),
        "worktree_dirty": bool(status_text),
        "tracked": tracked,
        "ignored": ignored,
        "statuses": statuses,
    }


def _git_state(path: str, repository: Mapping[str, Any]) -> str:
    if path in repository["tracked"]:
        return "tracked_modified" if path in repository["statuses"] else "tracked_clean"
    if path in repository["ignored"]:
        return "ignored"
    if repository["statuses"].get(path) == "??":
        return "untracked"
    return "outside_git"


def _declared_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"evidence paths must stay within the repository: {value!r}")
    resolved = (repo_root / path).resolve()
    if not resolved.is_relative_to(repo_root):
        raise ValueError(f"evidence path escapes the repository: {value!r}")
    return resolved


def _strict_equal(actual: Any, expected: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_declaration(declaration: Mapping[str, Any]) -> None:
    if declaration.get("schema_version") != DECLARATION_SCHEMA_VERSION:
        raise ValueError("unsupported evidence declaration schema_version")
    _require_string(declaration.get("purpose"), "purpose")
    file_sets = declaration.get("file_sets")
    if not isinstance(file_sets, list) or not file_sets:
        raise ValueError("file_sets must be a non-empty array")

    set_ids: set[str] = set()
    for index, file_set in enumerate(file_sets):
        if not isinstance(file_set, Mapping):
            raise ValueError(f"file_sets[{index}] must be an object")
        set_id = _require_string(file_set.get("id"), f"file_sets[{index}].id")
        if set_id in set_ids:
            raise ValueError(f"duplicate file set id: {set_id}")
        set_ids.add(set_id)
        storage_class = file_set.get("storage_class")
        if storage_class not in STORAGE_CLASSES:
            raise ValueError(f"invalid storage_class for {set_id}: {storage_class!r}")
        _require_string(file_set.get("storage_id", "repository"), f"{set_id}.storage_id")
        paths = file_set.get("paths", [])
        globs = file_set.get("globs", [])
        if not isinstance(paths, list) or not isinstance(globs, list):
            raise ValueError(f"paths and globs must be arrays for {set_id}")
        if not paths and not globs:
            raise ValueError(f"file set {set_id} must declare at least one path or glob")
        normalized_paths = [Path(str(item)).as_posix() for item in paths]
        normalized_globs = [str(item) for item in globs]
        if len(normalized_paths) != len(set(normalized_paths)):
            raise ValueError(f"duplicate declared path in file set {set_id}")
        if len(normalized_globs) != len(set(normalized_globs)):
            raise ValueError(f"duplicate declared glob in file set {set_id}")
        expected_count = file_set.get("expected_count")
        if expected_count is not None and (
            not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count < 0
        ):
            raise ValueError(f"expected_count must be a non-negative integer for {set_id}")

    assertion_ids: set[str] = set()
    for index, assertion in enumerate(declaration.get("evidence_assertions", [])):
        if not isinstance(assertion, Mapping):
            raise ValueError(f"evidence_assertions[{index}] must be an object")
        assertion_id = _require_string(assertion.get("id"), f"evidence_assertions[{index}].id")
        if assertion_id in assertion_ids:
            raise ValueError(f"duplicate assertion id: {assertion_id}")
        assertion_ids.add(assertion_id)
        _require_string(assertion.get("path"), f"{assertion_id}.path")
        checks = assertion.get("json_checks")
        if not isinstance(checks, list) or not checks:
            raise ValueError(f"assertion {assertion_id} must contain at least one json_check")
        for check in checks:
            if not isinstance(check, Mapping) or "equals" not in check:
                raise ValueError(f"invalid json_check in assertion {assertion_id}")
            _require_string(check.get("pointer"), f"{assertion_id}.pointer")

    test_run = declaration.get("test_run")
    if test_run is not None:
        if not isinstance(test_run, Mapping):
            raise ValueError("test_run must be an object")
        _require_string(test_run.get("result_path"), "test_run.result_path")


def _validate_manifest_structure(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported evidence manifest schema_version")
    _require_string(manifest.get("purpose"), "purpose")
    if not isinstance(manifest.get("repository"), Mapping):
        raise ValueError("repository must be an object")
    if not isinstance(manifest.get("environment"), Mapping):
        raise ValueError("environment must be an object")
    file_sets = manifest.get("file_sets")
    if not isinstance(file_sets, list) or not file_sets:
        raise ValueError("file_sets must be a non-empty array")
    if not isinstance(manifest.get("validation"), Mapping):
        raise ValueError("validation must be an object")
    if manifest["validation"].get("status") not in FINAL_STATUSES:
        raise ValueError("invalid validation status")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for file_set in file_sets:
        set_id = _require_string(file_set.get("id"), "file_set.id")
        if set_id in seen_ids:
            raise ValueError(f"duplicate file set id: {set_id}")
        seen_ids.add(set_id)
        if file_set.get("storage_class") not in STORAGE_CLASSES:
            raise ValueError(f"invalid storage_class for {set_id}")
        files = file_set.get("files")
        if not isinstance(files, list):
            raise ValueError(f"files must be an array for {set_id}")
        expanded_paths = file_set.get("expanded_paths")
        if not isinstance(expanded_paths, list):
            raise ValueError(f"expanded_paths must be an array for {set_id}")
        file_paths = [record.get("path") for record in files]
        if len(file_paths) != len(set(file_paths)):
            raise ValueError(f"duplicate file record in file set {set_id}")
        if sorted(file_paths) != sorted(expanded_paths):
            raise ValueError(f"file records do not match expanded_paths for {set_id}")
        for record in files:
            path = _require_string(record.get("path"), f"{set_id}.files.path")
            if path in seen_paths:
                raise ValueError(f"evidence path appears in multiple file sets: {path}")
            seen_paths.add(path)


def _expand_paths(
    repo_root: Path,
    file_set: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    paths: set[str] = set()
    explicit_paths: set[str] = set()
    for supplied in file_set.get("paths", []):
        value = str(supplied)
        _declared_path(repo_root, value)
        normalized = Path(value).as_posix()
        explicit_paths.add(normalized)
        paths.add(normalized)
    unmatched: list[str] = []
    for pattern in file_set.get("globs", []):
        _declared_path(repo_root, str(pattern))
        matches = sorted(
            path.relative_to(repo_root).as_posix()
            for path in repo_root.glob(str(pattern))
            if path.is_file()
        )
        if not matches:
            unmatched.append(str(pattern))
        overlap = explicit_paths.intersection(matches)
        if overlap:
            raise ValueError(
                f"declared path is also matched by a glob: {sorted(overlap)}"
            )
        paths.update(matches)
    return sorted(paths), unmatched


def _file_record(
    repo_root: Path,
    relative_path: str,
    *,
    role: str,
    required: bool,
    storage_class: str,
    repository: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = _declared_path(repo_root, relative_path)
    record: dict[str, Any] = {
        "path": relative_path,
        "role": role,
        "required": required,
        "status": "missing",
        "size_bytes": None,
        "sha256": None,
        "git_state": _git_state(relative_path, repository) if storage_class == "git" else "not_applicable",
        "content_git_oid": None,
    }
    if not candidate.is_file():
        return record
    record.update(
        status="present",
        size_bytes=int(candidate.stat().st_size),
        sha256=sha256_file(candidate),
    )
    if storage_class == "git":
        record["content_git_oid"] = _run_git(repo_root, "hash-object", "--", relative_path)
    return record


def _environment(dependency_names: Sequence[str] = DEFAULT_DEPENDENCIES) -> dict[str, Any]:
    dependencies: dict[str, str] = {}
    for name in sorted(set(dependency_names)):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = "missing"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": dependencies,
    }


def _json_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer!r}")
    current = document
    for token in pointer[1:].split("/"):
        key = token.replace("~1", "/").replace("~0", "~")
        current = current[int(key)] if isinstance(current, list) else current[key]
    return current


def _evaluate_assertions(
    repo_root: Path,
    assertions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for assertion in assertions:
        path = str(assertion["path"])
        required = bool(assertion.get("required", True))
        checks_spec = assertion.get("json_checks", [])
        if not checks_spec:
            raise ValueError(f"assertion {assertion.get('id')} must contain at least one json_check")
        candidate = _declared_path(repo_root, path)
        record: dict[str, Any] = {
            "id": str(assertion["id"]),
            "path": path,
            "required": required,
            "status": "missing",
            "checks": [],
        }
        if not candidate.is_file():
            records.append(record)
            continue
        try:
            document = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            records.append(record)
            continue
        checks: list[dict[str, Any]] = []
        for check in checks_spec:
            pointer = str(check["pointer"])
            expected = check["equals"]
            try:
                actual = _json_pointer(document, pointer)
                passed = _strict_equal(actual, expected)
                error = None
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                actual = None
                passed = False
                error = str(exc)
            checks.append(
                {
                    "pointer": pointer,
                    "expected": expected,
                    "actual": actual,
                    "passed": passed,
                    "error": error,
                }
            )
        record["checks"] = checks
        record["status"] = "passed" if checks and all(item["passed"] for item in checks) else "failed"
        records.append(record)
    return records


def _assertion_specs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": record["id"],
            "path": record["path"],
            "required": record.get("required", True),
            "json_checks": [
                {"pointer": check["pointer"], "equals": check["expected"]}
                for check in record.get("checks", [])
            ],
        }
        for record in records
    ]


def _evaluate_test_run(repo_root: Path, specification: Mapping[str, Any]) -> dict[str, Any]:
    result_path = str(specification["result_path"])
    result_file = _declared_path(repo_root, result_path)
    required = bool(specification.get("required", True))
    record: dict[str, Any] = {
        "required": required,
        "result_path": result_path,
        "result_sha256": None,
        "result": None,
        "log_path": specification.get("log_path"),
        "log_sha256": None,
        "status": "missing",
        "reason": "test result file is missing",
        "evidence_level": "archived_test_record_not_execution_attestation",
    }
    if not result_file.is_file():
        return record
    try:
        result = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        record.update(status="failed", reason=str(exc))
        return record
    required_fields = {"scope", "command", "run", "failures", "errors", "skipped", "exit_code"}
    if not required_fields.issubset(result):
        missing = sorted(required_fields.difference(result))
        record.update(status="failed", reason=f"missing result fields: {missing}")
        return record
    record.update(result_sha256=sha256_file(result_file), result=result)
    if any(
        not isinstance(result[field], int) or isinstance(result[field], bool)
        for field in ("run", "failures", "errors", "skipped", "exit_code")
    ):
        record.update(status="failed", reason="test counters and exit_code must be integers")
        return record
    if result["run"] <= 0 or result["failures"] or result["errors"] or result["exit_code"]:
        record.update(status="failed", reason="test result does not report a successful non-empty run")
        return record

    log_path = specification.get("log_path")
    if log_path:
        log_file = _declared_path(repo_root, str(log_path))
        if log_file.is_file():
            record.update(log_sha256=sha256_file(log_file), status="passed", reason="")
        else:
            record.update(status="incomplete", reason="declared raw test log is missing")
    else:
        record.update(status="incomplete", reason="raw test log is not declared")
    return record


def _test_run_spec(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "required": record.get("required", True),
        "result_path": record["result_path"],
        "log_path": record.get("log_path"),
    }


def _manifest_identity_material(manifest: Mapping[str, Any]) -> dict[str, Any]:
    material = copy.deepcopy(dict(manifest))
    material.pop("manifest_id", None)
    material.pop("created_at_utc", None)
    material.get("repository", {}).pop("branch", None)
    for file_set in material.get("file_sets", []):
        for field in ("declared_paths", "declared_globs", "expanded_paths"):
            file_set[field] = sorted(file_set.get(field, []))
        file_set["files"] = sorted(file_set.get("files", []), key=lambda item: item["path"])
    material["file_sets"] = sorted(material.get("file_sets", []), key=lambda item: item["id"])
    for assertion in material.get("evidence_assertions", []):
        assertion["checks"] = sorted(assertion.get("checks", []), key=lambda item: item["pointer"])
    material["evidence_assertions"] = sorted(
        material.get("evidence_assertions", []), key=lambda item: item["id"]
    )
    return material


def _validation_status(
    *,
    required_missing: Sequence[str],
    optional_missing: Sequence[str],
    unmatched_globs: Sequence[str],
    optional_unmatched_globs: Sequence[str],
    count_mismatches: Sequence[str],
    dirty_git_paths: Sequence[str],
    failed_assertions: Sequence[str],
    unresolved_external_storage: Sequence[str],
    test_run_status: str | None,
) -> str:
    if required_missing or unmatched_globs or count_mismatches or failed_assertions or test_run_status == "failed":
        return "failed"
    if dirty_git_paths or unresolved_external_storage or test_run_status == "incomplete":
        return "incomplete"
    if optional_missing or optional_unmatched_globs:
        return "complete_with_warnings"
    return "passed"


def build_evidence_manifest(
    declaration: Mapping[str, Any],
    *,
    repo_root: str | Path,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a candidate manifest without mutating declared evidence."""

    _validate_declaration(declaration)
    root = Path(repo_root).expanduser().resolve()
    repository = _repository_inventory(root)
    file_sets: list[dict[str, Any]] = []
    required_missing: list[str] = []
    optional_missing: list[str] = []
    unmatched_globs: list[str] = []
    optional_unmatched_globs: list[str] = []
    count_mismatches: list[str] = []
    dirty_git_paths: list[str] = []
    unresolved_external_storage: list[str] = []
    seen_paths: dict[str, str] = {}

    for raw_set in declaration["file_sets"]:
        set_id = str(raw_set["id"])
        storage_class = str(raw_set["storage_class"])
        required = bool(raw_set.get("required", True))
        role = str(raw_set.get("role", "evidence"))
        paths, unmatched = _expand_paths(root, raw_set)
        target_unmatched = unmatched_globs if required else optional_unmatched_globs
        target_unmatched.extend(f"{set_id}:{pattern}" for pattern in unmatched)
        expected_count = raw_set.get("expected_count")
        if expected_count is not None and len(paths) != expected_count:
            count_mismatches.append(f"{set_id}:expected={expected_count}:actual={len(paths)}")
        for path in paths:
            if path in seen_paths:
                raise ValueError(
                    f"evidence path appears in multiple file sets: {path} "
                    f"({seen_paths[path]}, {set_id})"
                )
            seen_paths[path] = set_id
        files = [
            _file_record(
                root,
                path,
                role=role,
                required=required,
                storage_class=storage_class,
                repository=repository,
            )
            for path in paths
        ]
        for file_record in files:
            if file_record["status"] == "missing":
                (required_missing if required else optional_missing).append(file_record["path"])
            if storage_class == "git" and file_record["git_state"] != "tracked_clean":
                dirty_git_paths.append(file_record["path"])

        storage_locator = raw_set.get("storage_locator")
        if storage_class == "external" and (
            not isinstance(storage_locator, str)
            or not storage_locator.strip()
            or storage_locator.startswith("pending:")
        ):
            unresolved_external_storage.append(set_id)
        file_sets.append(
            {
                "id": set_id,
                "storage_class": storage_class,
                "storage_id": str(raw_set.get("storage_id", "repository")),
                "storage_locator": storage_locator,
                "required": required,
                "role": role,
                "declared_paths": [str(item) for item in raw_set.get("paths", [])],
                "declared_globs": [str(item) for item in raw_set.get("globs", [])],
                "expected_count": expected_count,
                "expanded_paths": paths,
                "bundle_sha256": sha256_json(files),
                "files": files,
            }
        )

    assertions = _evaluate_assertions(root, declaration.get("evidence_assertions", []))
    failed_assertions = [
        record["id"]
        for record in assertions
        if record["required"] and record["status"] != "passed"
    ]
    test_run = None
    if declaration.get("test_run") is not None:
        test_run = _evaluate_test_run(root, declaration["test_run"])
    test_run_status = None if test_run is None else str(test_run["status"])
    declared_scope_clean = not dirty_git_paths
    validation_status = _validation_status(
        required_missing=required_missing,
        optional_missing=optional_missing,
        unmatched_globs=unmatched_globs,
        optional_unmatched_globs=optional_unmatched_globs,
        count_mismatches=count_mismatches,
        dirty_git_paths=dirty_git_paths,
        failed_assertions=failed_assertions,
        unresolved_external_storage=unresolved_external_storage,
        test_run_status=test_run_status,
    )

    instant = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        if created_at_utc is None
        else created_at_utc
    )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": "",
        "created_at_utc": instant,
        "purpose": str(declaration["purpose"]),
        "status_note": str(declaration.get("status_note", "")),
        "repository": {
            "branch": repository["branch"],
            "head": repository["head"],
            "worktree_dirty": repository["worktree_dirty"],
            "declared_scope_clean": declared_scope_clean,
        },
        "environment": _environment(declaration.get("dependency_names", DEFAULT_DEPENDENCIES)),
        "file_sets": file_sets,
        "test_run": test_run,
        "evidence_assertions": assertions,
        "validation": {
            "status": validation_status,
            "required_missing": sorted(required_missing),
            "optional_missing": sorted(optional_missing),
            "unmatched_globs": sorted(unmatched_globs),
            "optional_unmatched_globs": sorted(optional_unmatched_globs),
            "count_mismatches": sorted(count_mismatches),
            "dirty_git_paths": sorted(set(dirty_git_paths)),
            "failed_assertions": sorted(failed_assertions),
            "unresolved_external_storage": sorted(unresolved_external_storage),
            "test_run_status": test_run_status,
        },
    }
    manifest["manifest_id"] = sha256_json(_manifest_identity_material(manifest))
    _validate_manifest_structure(manifest)
    return manifest


def verify_evidence_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path,
    expected_manifest_id: str | None = None,
) -> dict[str, Any]:
    """Recompute manifest integrity against the current repository and evidence."""

    _validate_manifest_structure(manifest)
    root = Path(repo_root).expanduser().resolve()
    repository = _repository_inventory(root)
    computed_manifest_id = sha256_json(_manifest_identity_material(manifest))
    manifest_id_valid = manifest.get("manifest_id") == computed_manifest_id
    approved_manifest_id_match = (
        True
        if expected_manifest_id is None
        else manifest.get("manifest_id") == expected_manifest_id
    )
    repository_match = {
        "branch": repository["branch"] == manifest["repository"].get("branch"),
        "head": repository["head"] == manifest["repository"].get("head"),
        "worktree_dirty": repository["worktree_dirty"] == manifest["repository"].get("worktree_dirty"),
    }
    required_missing: list[str] = []
    optional_missing: list[str] = []
    mismatches: list[dict[str, Any]] = []
    bundle_mismatches: list[str] = []
    expansion_mismatches: list[dict[str, Any]] = []
    unmatched_globs: list[str] = []
    optional_unmatched_globs: list[str] = []
    count_mismatches: list[str] = []
    dirty_git_paths: list[str] = []
    unresolved_external_storage: list[str] = []

    for file_set in manifest["file_sets"]:
        set_id = str(file_set["id"])
        stored_files = file_set["files"]
        if sha256_json(stored_files) != file_set.get("bundle_sha256"):
            bundle_mismatches.append(set_id)
        expansion_spec = {
            "paths": file_set.get("declared_paths", []),
            "globs": file_set.get("declared_globs", []),
        }
        expanded_paths, unmatched = _expand_paths(root, expansion_spec)
        target_unmatched = unmatched_globs if file_set.get("required", True) else optional_unmatched_globs
        target_unmatched.extend(f"{set_id}:{pattern}" for pattern in unmatched)
        expected_count = file_set.get("expected_count")
        if expected_count is not None and len(expanded_paths) != expected_count:
            count_mismatches.append(f"{set_id}:expected={expected_count}:actual={len(expanded_paths)}")
        if expanded_paths != file_set.get("expanded_paths", []) or unmatched:
            expansion_mismatches.append(
                {
                    "file_set": set_id,
                    "expected": file_set.get("expanded_paths", []),
                    "actual": expanded_paths,
                    "unmatched_globs": unmatched,
                }
            )
        for stored in stored_files:
            path = str(stored["path"])
            current = _file_record(
                root,
                path,
                role=str(stored.get("role", file_set.get("role", "evidence"))),
                required=bool(stored.get("required", file_set.get("required", True))),
                storage_class=str(file_set["storage_class"]),
                repository=repository,
            )
            if current["status"] == "missing":
                if stored.get("status") != "missing":
                    mismatches.append(
                        {"path": path, "field": "status", "expected": stored.get("status"), "actual": "missing"}
                    )
                (required_missing if current["required"] else optional_missing).append(path)
                continue
            for field in ("status", "size_bytes", "sha256", "git_state", "content_git_oid"):
                if current.get(field) != stored.get(field):
                    mismatches.append(
                        {"path": path, "field": field, "expected": stored.get(field), "actual": current.get(field)}
                    )
            if file_set["storage_class"] == "git" and current["git_state"] != "tracked_clean":
                dirty_git_paths.append(path)
        locator = file_set.get("storage_locator")
        if file_set["storage_class"] == "external" and (
            not isinstance(locator, str) or not locator.strip() or locator.startswith("pending:")
        ):
            unresolved_external_storage.append(set_id)

    assertion_specs = _assertion_specs(manifest.get("evidence_assertions", []))
    assertions = _evaluate_assertions(root, assertion_specs)
    failed_assertions = [
        record["id"]
        for record in assertions
        if record["required"] and record["status"] != "passed"
    ]
    test_run = None
    if manifest.get("test_run") is not None:
        test_run = _evaluate_test_run(root, _test_run_spec(manifest["test_run"]))
    test_run_match = test_run == manifest.get("test_run")
    test_run_status = None if test_run is None else str(test_run["status"])

    recorded_validation = manifest["validation"]
    recomputed_freeze_status = _validation_status(
        required_missing=required_missing,
        optional_missing=optional_missing,
        unmatched_globs=unmatched_globs,
        optional_unmatched_globs=optional_unmatched_globs,
        count_mismatches=count_mismatches,
        dirty_git_paths=dirty_git_paths,
        failed_assertions=failed_assertions,
        unresolved_external_storage=unresolved_external_storage,
        test_run_status=test_run_status,
    )
    recomputed_validation = {
        "status": recomputed_freeze_status,
        "required_missing": sorted(required_missing),
        "optional_missing": sorted(optional_missing),
        "unmatched_globs": sorted(unmatched_globs),
        "optional_unmatched_globs": sorted(optional_unmatched_globs),
        "count_mismatches": sorted(count_mismatches),
        "dirty_git_paths": sorted(set(dirty_git_paths)),
        "failed_assertions": sorted(failed_assertions),
        "unresolved_external_storage": sorted(unresolved_external_storage),
        "test_run_status": test_run_status,
    }
    validation_match = dict(recorded_validation) == recomputed_validation
    integrity_ok = (
        manifest_id_valid
        and approved_manifest_id_match
        and all(repository_match.values())
        and not required_missing
        and not mismatches
        and not bundle_mismatches
        and not expansion_mismatches
        and not failed_assertions
        and test_run_match
        and validation_match
    )
    integrity_status = "passed" if integrity_ok else "failed"
    overall_status = recomputed_freeze_status if integrity_ok else "failed"
    return {
        "manifest_id": manifest.get("manifest_id"),
        "manifest_id_valid": manifest_id_valid,
        "approved_manifest_id_match": approved_manifest_id_match,
        "repository_match": repository_match,
        "integrity_status": integrity_status,
        "freeze_status": recomputed_freeze_status,
        "status": overall_status,
        "validation_match": validation_match,
        "test_run_match": test_run_match,
        "missing": sorted(set(required_missing)),
        "optional_missing": sorted(set(optional_missing)),
        "record_mismatches": sorted(mismatches, key=lambda item: (item["path"], item["field"])),
        "bundle_mismatches": sorted(bundle_mismatches),
        "expansion_mismatches": expansion_mismatches,
        "failed_assertions": sorted(failed_assertions),
    }
