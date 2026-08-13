#!/usr/bin/env python3
"""Build or verify a content-addressed research evidence manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wind_prediction.evidence_manifest import (  # noqa: E402
    build_evidence_manifest,
    verify_evidence_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--declaration", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument(
        "--expected-manifest-id",
        required=True,
        help="Previously approved 64-character manifest identity.",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    args = parse_args()
    if args.command == "build":
        declaration_path = _resolve(args.declaration)
        output_path = _resolve(args.output)
        declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
        manifest = build_evidence_manifest(declaration, repo_root=REPO_ROOT)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(output_path)
        print(manifest["validation"]["status"])
        return 0 if manifest["validation"]["status"] == "passed" else 2

    manifest_path = _resolve(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = verify_evidence_manifest(
        manifest,
        repo_root=REPO_ROOT,
        expected_manifest_id=args.expected_manifest_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
