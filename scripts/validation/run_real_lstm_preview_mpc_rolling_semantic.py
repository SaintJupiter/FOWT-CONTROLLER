#!/usr/bin/env python3
"""CLI for the real-LSTM preview-MPC rolling semantic check."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from preview_mpc_experiment_runtime import run_rolling_semantic_check
from wind_prediction.replay_dataset import TIMESTAMP_FMT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--origin",
        help="test-split replay origin in YYYY-MM-DD HH:MM:SS",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cycle-count",
        type=int,
        default=1,
        help="number of consecutive ten-minute cycles, from 1 to 36",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    origin = None if args.origin is None else datetime.strptime(args.origin, TIMESTAMP_FMT)
    payload = json.dumps(
        run_rolling_semantic_check(
            origin=origin,
            device=args.device,
            cycle_count=args.cycle_count,
        ),
        ensure_ascii=False,
        indent=2,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
