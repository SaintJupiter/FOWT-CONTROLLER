#!/usr/bin/env python3
"""Verify that a casebook run_protocol.json matches an expected run identity.

This is a small pre-plot/pre-interpretation guard. It intentionally depends only
on the Python standard library so it can run inside the project environment
before any plotting or analysis imports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _parse_value(raw: str) -> Any:
    text = raw.strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _get_path(payload: dict[str, Any], dotted_path: str) -> Any:
    cur: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted_path)
        cur = cur[part]
    return cur


def _parse_expectation(spec: str) -> tuple[str, Any]:
    if "=" not in spec:
        raise ValueError(f"Expectation must be PATH=VALUE, got: {spec!r}")
    path, raw_value = spec.split("=", 1)
    path = path.strip()
    if not path:
        raise ValueError(f"Expectation path is empty in: {spec!r}")
    return path, _parse_value(raw_value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if a casebook run_protocol.json does not match the expected "
            "controller identity/configuration."
        )
    )
    parser.add_argument("protocol", type=Path, help="Path to run_protocol.json")
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=(
            "Required JSON value, e.g. "
            "--expect identity.primary_control_profile=rawenv_holdpause_barrier_reliefcap_adaptive_v1"
        ),
    )
    parser.add_argument(
        "--require-timeseries",
        action="store_true",
        help="Require outputs.timeseries_dir to exist and contain *_timeseries.csv files.",
    )
    args = parser.parse_args(argv)

    if not args.protocol.exists():
        print(f"FAIL: protocol not found: {args.protocol}", file=sys.stderr)
        return 2

    payload = json.loads(args.protocol.read_text(encoding="utf-8"))
    failures: list[str] = []

    for spec in args.expect:
        path, expected = _parse_expectation(spec)
        try:
            actual = _get_path(payload, path)
        except KeyError:
            failures.append(f"{path}: missing, expected {expected!r}")
            continue
        if actual != expected:
            failures.append(f"{path}: actual {actual!r}, expected {expected!r}")

    if args.require_timeseries:
        ts_raw = _get_path(payload, "outputs.timeseries_dir")
        ts_dir = (args.protocol.parent / ts_raw).resolve() if not Path(ts_raw).is_absolute() else Path(ts_raw)
        # run_protocol paths are usually relative to the repo cwd, not to the
        # protocol directory. Fall back to cwd-relative when protocol-relative
        # resolution does not exist.
        if not ts_dir.exists():
            cwd_relative = Path(ts_raw)
            if cwd_relative.exists():
                ts_dir = cwd_relative
        if not ts_dir.exists():
            failures.append(f"outputs.timeseries_dir: missing directory {ts_raw!r}")
        elif not any(ts_dir.glob("*_timeseries.csv")):
            failures.append(f"outputs.timeseries_dir: no *_timeseries.csv files in {ts_dir}")

    if failures:
        print("FAIL: run protocol does not match expected casebook identity:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("PASS: run protocol matches expected casebook identity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
