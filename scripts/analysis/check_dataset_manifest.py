#!/usr/bin/env python3
"""Validate a wind-preview dataset manifest before replay or training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from wind_prediction.dataset_manifest import DatasetManifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1",
        help="Dataset directory containing metadata.json, scaler_train.json, arrays, and sample_index.csv.gz.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "validation", "test"),
        help="Split to validate.",
    )
    parser.add_argument(
        "--canonical-path",
        default="",
        help="Optional canonical replay observations CSV.GZ. Default uses FINO1 platform sibling path.",
    )
    parser.add_argument(
        "--no-array-shapes",
        action="store_true",
        help="Skip mmap array shape checks and only check metadata/path contract.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text report.",
    )
    return parser.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def main() -> None:
    args = parse_args()
    dataset_dir = _resolve(str(args.dataset_dir))
    manifest = DatasetManifest.load(dataset_dir)
    canonical_path = (
        _resolve(str(args.canonical_path))
        if str(args.canonical_path).strip()
        else manifest.default_fino1_canonical_path()
    )
    report = manifest.validate(
        str(args.split),
        require_replay_arrays=True,
        require_sample_index=True,
        canonical_path=canonical_path,
        check_array_shapes=not bool(args.no_array_shapes),
    )
    payload = {
        "ok": bool(report.ok),
        "dataset_dir": str(report.dataset_dir),
        "split": str(report.split),
        "checked_paths": [str(p) for p in report.checked_paths],
        "array_shapes": {k: list(v) for k, v in report.array_shapes.items()},
        "errors": list(report.errors),
        "warnings": list(report.warnings),
    }
    if bool(args.json):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        status = "PASS" if report.ok else "FAIL"
        print(f"[{status}] dataset manifest: {report.dataset_dir} split={report.split}")
        if payload["array_shapes"]:
            print("array shapes:")
            for key, shape in sorted(payload["array_shapes"].items()):
                print(f"  - {key}: {tuple(shape)}")
        if payload["checked_paths"]:
            print("checked paths:")
            for path in payload["checked_paths"]:
                print(f"  - {path}")
        if report.warnings:
            print("warnings:")
            for item in report.warnings:
                print(f"  - {item}")
        if report.errors:
            print("errors:")
            for item in report.errors:
                print(f"  - {item}")
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
