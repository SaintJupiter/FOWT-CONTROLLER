#!/usr/bin/env python3
"""Check a three-case control-chain smoke summary without running simulations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "control_chain_smoke_gate_v1.json"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(raw: Any, default: float = math.nan) -> float:
    try:
        if raw in ("", None):
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def _equal_config(actual: str, expected: Any) -> bool:
    if isinstance(expected, int):
        return int(round(_to_float(actual, -999999.0))) == int(expected)
    if isinstance(expected, float):
        return abs(_to_float(actual) - float(expected)) <= 1e-9
    return str(actual) == str(expected)


def _row_by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("case_id", "")): row for row in rows}


def evaluate(summary_csv: Path, config: dict[str, Any]) -> dict[str, Any]:
    rows = _read_rows(summary_csv)
    by_case = _row_by_case(rows)
    failures: list[str] = []
    warnings: list[str] = []

    expected_cases = [str(case) for case in config.get("expected_cases", [])]
    missing = [case for case in expected_cases if case not in by_case]
    extra = sorted(set(by_case) - set(expected_cases))
    if missing:
        failures.append(f"missing expected cases: {missing}")
    if extra:
        warnings.append(f"extra cases present: {extra}")

    required_config = config.get("required_config", {})
    for case in expected_cases:
        row = by_case.get(case)
        if row is None:
            continue
        for key, expected in required_config.items():
            if key not in row:
                failures.append(f"{case}: missing config column {key}")
                continue
            if not _equal_config(row.get(key, ""), expected):
                failures.append(f"{case}: {key}={row.get(key)!r}, expected {expected!r}")

    for case in expected_cases:
        row = by_case.get(case)
        if row is None:
            continue
        for col in config.get("required_zero_ratio_columns", []):
            value = _to_float(row.get(str(col), "0"), 0.0)
            if abs(value) > 1e-9:
                failures.append(f"{case}: {col} must be 0, got {value:g}")

    thresholds = config.get("case_thresholds", {})
    for case, case_thresholds in thresholds.items():
        row = by_case.get(str(case))
        if row is None:
            continue
        for key, limit in case_thresholds.items():
            if not key.endswith("_max"):
                continue
            column = key[:-4]
            value = _to_float(row.get(column))
            if not math.isfinite(value):
                failures.append(f"{case}: {column} is not finite")
            elif value > float(limit):
                failures.append(f"{case}: {column}={value:g} exceeds max {float(limit):g}")

    ref_cfg = config.get("reference_tolerance", {})
    ref_path_raw = config.get("reference_summary_csv")
    if ref_cfg.get("enabled", False) and ref_path_raw:
        ref_path = _resolve(str(ref_path_raw))
        if not ref_path.exists():
            warnings.append(f"reference summary not found: {ref_path}")
        else:
            ref_rows = _row_by_case(_read_rows(ref_path))
            for case in expected_cases:
                row = by_case.get(case)
                ref = ref_rows.get(case)
                if row is None or ref is None:
                    continue
                for column in (
                    "primary_pump_work_m3",
                    "primary_pitch_p95",
                    "primary_roll_p95",
                    "primary_safety_fallback_ratio",
                ):
                    limit = ref_cfg.get(f"{column}_abs_max")
                    if limit is None:
                        continue
                    delta = abs(_to_float(row.get(column)) - _to_float(ref.get(column)))
                    if not math.isfinite(delta):
                        failures.append(f"{case}: reference delta for {column} is not finite")
                    elif delta > float(limit):
                        failures.append(
                            f"{case}: |delta {column}|={delta:g} exceeds {float(limit):g}"
                        )

    return {
        "ok": not failures,
        "summary_csv": str(summary_csv),
        "case_count": len(rows),
        "expected_cases": expected_cases,
        "failures": failures,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--summary-csv",
        default="outputs/wind_prediction/phase2b_3case_clean_defaults_no_overlay/casebook_summary.csv",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_json(_resolve(args.config))
    result = evaluate(_resolve(args.summary_csv), config)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        status = "PASS" if result["ok"] else "FAIL"
        print(f"[{status}] control-chain smoke gate: {result['summary_csv']}")
        for warning in result["warnings"]:
            print(f"[WARN] {warning}")
        for failure in result["failures"]:
            print(f"[FAIL] {failure}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

