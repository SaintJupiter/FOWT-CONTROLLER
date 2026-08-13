#!/usr/bin/env python3
"""Run one versioned controller-validation protocol and check its outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wind_prediction.validation_protocol import ControlValidationProtocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="configs/control_chain_smoke_gate_v2.json",
        help="versioned validation protocol JSON",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="check existing outputs without rerunning the simulation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = ControlValidationProtocol.load(args.protocol, repo_root=REPO_ROOT)
    if not args.check_only:
        subprocess.run(
            protocol.runner_argv(),
            cwd=REPO_ROOT,
            check=True,
        )
    completed = subprocess.run(
        protocol.checker_argv(),
        cwd=REPO_ROOT,
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
